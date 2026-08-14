from __future__ import annotations

import argparse
import json
import math
import os
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .._common.dataset import MAPDataset
from ..model.map import MAPModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MAP over the prepared data contract")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument(
        "--regime", choices=("unprofiled_drug", "unseen_combination"), required=True
    )
    parser.add_argument("--populations", nargs="+")
    parser.add_argument("--se-ckpt", required=True)
    parser.add_argument("--esm-embeddings", required=True)
    parser.add_argument("--mapkg-ckpt", required=True)
    parser.add_argument("--mapkg-vocab", required=True)
    parser.add_argument("--static-token-cache", required=True)
    parser.add_argument("--preparation-config")
    parser.add_argument("--resume")
    parser.add_argument("--set-size", type=int, default=24)
    parser.add_argument("--num-gene-tokens", type=int, default=2048)
    parser.add_argument("--hvg-dim", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--hvg-loss-weight", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=100_000)
    parser.add_argument("--warmup-steps", type=int, default=10_000)
    parser.add_argument("--eval-every-steps", type=int, default=100)
    parser.add_argument("--early-stopping-patience", type=int, default=1_000)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--samples-per-epoch", type=int)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp-dtype", choices=("bf16", "fp16"), default="bf16")
    return parser.parse_args()


def setup_distributed() -> tuple[int, int, int]:
    if "RANK" not in os.environ:
        return 0, 1, 0
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
    else:
        local_rank = 0
        dist.init_process_group("gloo")
    return rank, world_size, local_rank


def seed_everything(seed: int, rank: int) -> None:
    effective = int(seed) + int(rank)
    random.seed(effective)
    np.random.seed(effective)
    torch.manual_seed(effective)
    torch.cuda.manual_seed_all(effective)


def precision(device: torch.device, dtype: torch.dtype):
    return (
        torch.autocast(device_type="cuda", dtype=dtype)
        if device.type == "cuda"
        else nullcontext()
    )


def cosine_schedule(optimizer, warmup_steps: int, total_steps: int, minimum_ratio=0.1):
    def scale(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(minimum_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, scale)


def compute_loss(pred_embedding, true_embedding, pred_hvg, true_hvg, hvg_weight):
    embedding_loss = nn.functional.mse_loss(
        pred_embedding.float().mean(1), true_embedding.float().mean(1)
    )
    expression_loss = nn.functional.mse_loss(
        pred_hvg.float().mean(1), true_hvg.float().mean(1)
    )
    return (
        embedding_loss + hvg_weight * expression_loss,
        embedding_loss,
        expression_loss,
    )


def move_batch(batch: dict, device: torch.device):
    return (
        batch["control_gene_ids"].to(device, non_blocking=True),
        batch["control_expressions"].to(device, non_blocking=True),
        batch["condition_embeddings"].to(device, non_blocking=True),
        batch["condition_hvg_vectors"].to(device, non_blocking=True),
        list(batch["drug_smiles"]),
        batch["drug_conc"].to(device, non_blocking=True, dtype=torch.float32),
    )


@torch.no_grad()
def validate(model, loader, device, amp_dtype, hvg_weight) -> dict[str, float]:
    model.eval()
    totals = torch.zeros(4, device=device, dtype=torch.float64)
    for batch in loader:
        genes, expression, true_embedding, true_hvg, smiles, doses = move_batch(
            batch, device
        )
        with precision(device, amp_dtype):
            pred_embedding, pred_hvg = model(genes, expression, smiles, doses)
            loss, embedding_loss, expression_loss = compute_loss(
                pred_embedding, true_embedding, pred_hvg, true_hvg, hvg_weight
            )
        batch_size = genes.shape[0]
        totals += torch.tensor(
            [
                loss.item() * batch_size,
                embedding_loss.item() * batch_size,
                expression_loss.item() * batch_size,
                batch_size,
            ],
            device=device,
            dtype=torch.float64,
        )
    if dist.is_initialized():
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    count = max(float(totals[3].item()), 1.0)
    model.train()
    return {
        "loss": float(totals[0].item() / count),
        "embedding_loss": float(totals[1].item() / count),
        "expression_loss": float(totals[2].item() / count),
    }


def checkpoint_payload(model, optimizer, scheduler, args, epoch, step, best, best_step):
    raw_model = model.module if isinstance(model, DDP) else model
    return {
        "format": "map_method_4_4_v1",
        "epoch": int(epoch),
        "global_step": int(step),
        "best_validation_loss": float(best),
        "best_step": int(best_step),
        "pert_model_state_dict": raw_model.pert_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "args": vars(args),
    }


def save_checkpoint(path: Path, *args) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint_payload(*args), temporary)
    os.replace(temporary, path)


def load_checkpoint(path: str | Path, model, optimizer, scheduler) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "map_method_4_4_v1":
        raise ValueError("Only Method 4.4 checkpoints are supported")
    raw_model = model.module if isinstance(model, DDP) else model
    raw_model.pert_model.load_state_dict(
        checkpoint["pert_model_state_dict"], strict=True
    )
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint


def main() -> None:
    args = parse_args()
    if min(args.set_size, args.batch_size, args.gradient_accumulation_steps) <= 0:
        raise ValueError("set_size, batch_size and gradient accumulation must be positive")
    if args.lr <= 0 or args.hvg_loss_weight < 0:
        raise ValueError("lr must be positive and hvg_loss_weight non-negative")
    if args.max_steps <= 0 or args.epochs <= 0 or not 0 <= args.warmup_steps < args.max_steps:
        raise ValueError("Invalid epochs/max_steps/warmup_steps")
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume:
        raise FileExistsError(f"Run directory is not empty: {output_dir}")

    rank, world_size, local_rank = setup_distributed()
    if world_size != 4 and rank == 0:
        print(f"Method 4.4 used four GPUs; current world_size={world_size}", flush=True)
    seed_everything(args.seed, rank)
    device = torch.device(
        f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    )
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16

    common = {
        "data_dir": args.data_dir,
        "regime": args.regime,
        "set_size": args.set_size,
        "seed": args.seed,
        "populations": args.populations,
        "split_file": args.split_file,
    }
    training = MAPDataset(
        split=args.train_split,
        training=True,
        samples_per_epoch=args.samples_per_epoch,
        **common,
    )
    validation = MAPDataset(split=args.val_split, training=False, **common)
    train_sampler = DistributedSampler(
        training, num_replicas=world_size, rank=rank, shuffle=True,
        seed=args.seed, drop_last=True,
    )
    validation_sampler = DistributedSampler(
        validation, num_replicas=world_size, rank=rank, shuffle=False,
        drop_last=False,
    )
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(training, sampler=train_sampler, **loader_options)
    validation_loader = DataLoader(validation, sampler=validation_sampler, **loader_options)

    model = MAPModel(
        se_checkpoint=args.se_ckpt,
        esm_embeddings=args.esm_embeddings,
        mapkg_checkpoint=args.mapkg_ckpt,
        mapkg_vocab=args.mapkg_vocab,
        static_token_cache=args.static_token_cache,
        num_gene_tokens=args.num_gene_tokens,
        hvg_dim=args.hvg_dim,
    ).to(device)
    if world_size > 1:
        model = DDP(
            model, device_ids=[local_rank], output_device=local_rank,
            broadcast_buffers=False,
        )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = Adam(trainable, lr=args.lr)
    steps_per_epoch = max(1, len(train_loader) // args.gradient_accumulation_steps)
    total_steps = min(args.max_steps, args.epochs * steps_per_epoch)
    scheduler = cosine_schedule(optimizer, args.warmup_steps, total_steps)
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=device.type == "cuda" and args.amp_dtype == "fp16",
    )

    start_epoch = 0
    global_step = 0
    best_loss = float("inf")
    best_step = 0
    if args.resume:
        checkpoint = load_checkpoint(args.resume, model, optimizer, scheduler)
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["global_step"])
        best_loss = float(checkpoint["best_validation_loss"])
        best_step = int(checkpoint["best_step"])

    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        preparation = (
            json.loads(Path(args.preparation_config).read_text(encoding="utf-8"))
            if args.preparation_config else {}
        )
        config = {
            **vars(args),
            "model": "map",
            "run_name": output_dir.name,
            "split_file": str(training.split_file),
            "split_id": training.split_id,
            "split_rule": training.split_manifest.get("rule"),
            "split_seed": training.split_manifest.get("seed"),
            "world_size": world_size,
            "effective_batch_size": (
                args.batch_size * args.gradient_accumulation_steps * world_size
            ),
            "preparation_id": preparation.get("preparation_id"),
            "preparation": preparation,
        }
        (output_dir / "run_config.json").write_text(
            json.dumps(config, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(
            f"parameters total={sum(p.numel() for p in model.parameters()):,} "
            f"trainable={sum(p.numel() for p in trainable):,}; "
            f"train_conditions={len(training.condition_ids):,} "
            f"val_conditions={len(validation.condition_ids):,}",
            flush=True,
        )

    optimizer.zero_grad(set_to_none=True)
    stop = False
    for epoch in range(start_epoch, args.epochs):
        training.set_epoch(epoch)
        train_sampler.set_epoch(epoch)
        model.train()
        for batch_index, batch in enumerate(train_loader):
            should_step = (
                (batch_index + 1) % args.gradient_accumulation_steps == 0
                or batch_index + 1 == len(train_loader)
            )
            sync_context = (
                model.no_sync()
                if isinstance(model, DDP) and not should_step
                else torch.enable_grad()
            )
            genes, expression, true_embedding, true_hvg, smiles, doses = move_batch(
                batch, device
            )
            with sync_context:
                with precision(device, amp_dtype):
                    pred_embedding, pred_hvg = model(genes, expression, smiles, doses)
                    loss, _, _ = compute_loss(
                        pred_embedding, true_embedding, pred_hvg, true_hvg,
                        args.hvg_loss_weight,
                    )
                scaler.scale(loss / args.gradient_accumulation_steps).backward()
            if not should_step:
                continue
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1

            if global_step % args.eval_every_steps == 0:
                metrics = validate(
                    model, validation_loader, device, amp_dtype,
                    args.hvg_loss_weight,
                )
                improved = metrics["loss"] < best_loss - args.min_delta
                if improved:
                    best_loss, best_step = metrics["loss"], global_step
                if rank == 0:
                    print(
                        f"epoch={epoch} step={global_step} train_loss={loss.item():.6f} "
                        f"val_loss={metrics['loss']:.6f} "
                        f"val_embedding={metrics['embedding_loss']:.6f} "
                        f"val_hvg={metrics['expression_loss']:.6f}",
                        flush=True,
                    )
                    if improved:
                        save_checkpoint(
                            output_dir / "best.pt", model, optimizer, scheduler,
                            args, epoch, global_step, best_loss, best_step,
                        )
                stop = global_step - best_step >= args.early_stopping_patience
                stop_tensor = torch.tensor(int(stop), device=device)
                if dist.is_initialized():
                    dist.broadcast(stop_tensor, src=0)
                stop = bool(stop_tensor.item())
            if global_step >= args.max_steps or stop:
                break

        if rank == 0:
            save_checkpoint(
                output_dir / "last.pt", model, optimizer, scheduler,
                args, epoch, global_step, best_loss, best_step,
            )
        if dist.is_initialized():
            dist.barrier()
        if global_step >= args.max_steps or stop:
            break

    if rank == 0:
        reason = "early_stopping" if stop else "max_steps_or_epochs"
        print(
            f"training complete: reason={reason}, step={global_step}, "
            f"best_step={best_step}, best_val_loss={best_loss:.6f}",
            flush=True,
        )
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
