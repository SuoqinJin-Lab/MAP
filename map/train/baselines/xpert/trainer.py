from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from ..common import (
    atomic_torch_save,
    build_loaders,
    distributed_mean,
    finish_distributed,
    move_batch,
    raw_model,
    seed_everything,
    setup_distributed,
    write_run_config,
)
from .model import XPert


def add_arguments(parser) -> None:
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--attention-heads", type=int, default=8)
    parser.add_argument("--attention-dropout", type=float, default=0.1)
    parser.add_argument("--hidden-dropout", type=float, default=0.1)
    parser.add_argument("--cell-input-dropout", type=float, default=0.1)
    parser.add_argument("--drug-input-dropout", type=float, default=0.1)
    parser.add_argument("--treated-structure", default="CA+SA+SA+CA")
    parser.add_argument("--control-structure", default="SA+SA+SA+SA")
    parser.add_argument("--expression-bins", type=int, default=10)
    parser.add_argument("--expression-min", type=float, default=0.0)
    parser.add_argument("--expression-max", type=float, default=10.0)
    parser.add_argument("--treated-weight", type=float, default=0.2)
    parser.add_argument("--control-weight", type=float, default=0.003)
    parser.add_argument("--delta-weight", type=float, default=0.2)
    parser.add_argument("--correlation-weight", type=float, default=1.0)
    parser.add_argument("--scheduler-milestone", type=int, default=40)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)


def _model_options(args) -> dict:
    names = (
        "hidden_size",
        "attention_heads",
        "attention_dropout",
        "hidden_dropout",
        "cell_input_dropout",
        "drug_input_dropout",
        "treated_structure",
        "control_structure",
        "expression_bins",
        "expression_min",
        "expression_max",
        "treated_weight",
        "control_weight",
        "delta_weight",
        "correlation_weight",
    )
    return {name: getattr(args, name) for name in names}


def build_model(data_dir, hvg_dim: int, populations, **options) -> XPert:
    return XPert(data_dir, hvg_dim, populations, **options)


def load_model(checkpoint, data_dir, hvg_dim: int, populations, device) -> XPert:
    configuration = dict(checkpoint.get("model_configuration", {}))
    configuration.pop("model", None)
    configuration.pop("hvg_dim", None)
    model = build_model(data_dir, hvg_dim, populations, **configuration)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device)


@torch.no_grad()
def _validate(model, loader, device) -> tuple[float, float]:
    model.eval()
    total = torch.zeros(4, device=device, dtype=torch.float64)
    for batch in loader:
        batch = move_batch(batch, device)
        output = model(batch)
        value, metrics = raw_model(model).loss(output)
        total += torch.tensor(
            [value.item(), 1.0, metrics["correlation"].item(), 1.0],
            device=device,
            dtype=torch.float64,
        )
    loss = distributed_mean(total[:2])
    correlation = distributed_mean(total[2:])
    model.train()
    return loss, correlation


def _checkpoint(model, optimizer, scheduler, args, epoch, step, best, best_step):
    implementation = raw_model(model)
    return {
        "format": "map_baseline_v2",
        "model": "xpert",
        "epoch": int(epoch),
        "global_step": int(step),
        "best_validation_loss": float(best),
        "best_step": int(best_step),
        "model_state_dict": implementation.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "args": vars(args),
        "model_configuration": implementation.configuration(),
    }


def train(args) -> None:
    if min(args.hidden_size, args.attention_heads, args.scheduler_milestone) <= 0 or args.expression_bins < 2:
        raise ValueError("XPert dimensions and schedules must be positive")
    if args.hidden_size % args.attention_heads:
        raise ValueError("XPert hidden_size must be divisible by attention_heads")
    if not args.expression_min < args.expression_max:
        raise ValueError("XPert expression_min must be smaller than expression_max")
    if not 0 < args.scheduler_factor <= 1 or args.weight_decay < 0:
        raise ValueError("XPert scheduler_factor or weight_decay is invalid")
    rank, world, local, device = setup_distributed()
    seed_everything(args.seed, rank)
    training, _, train_sampler, train_loader, validation_loader = build_loaders(
        args, rank, world
    )
    shapes = json.loads((Path(args.data_dir) / "materialized_shapes.json").read_text())
    hvg_dim = int(next(iter(shapes.values()))["hvg_dim"])
    model = build_model(args.data_dir, hvg_dim, args.populations, **_model_options(args)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: 1.0 if epoch < args.scheduler_milestone else args.scheduler_factor,
    )
    start_epoch = global_step = best_step = 0
    best = float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if checkpoint.get("format") != "map_baseline_v2" or checkpoint.get("model") != "xpert":
            raise ValueError("Resume checkpoint is not an XPert checkpoint")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["global_step"])
        best = float(checkpoint["best_validation_loss"])
        best_step = int(checkpoint["best_step"])
    if world > 1:
        model = DDP(model, device_ids=[local], output_device=local, broadcast_buffers=False)
    output_dir = Path(args.output_dir)
    if rank == 0:
        write_run_config(output_dir, {
            **vars(args),
            "model": "xpert",
            "run_name": output_dir.name,
            "world_size": world,
            "split_id": training.split_id,
            "model_configuration": raw_model(model).configuration(),
        })
    stop = False
    last_epoch = max(start_epoch - 1, 0)
    for epoch in range(start_epoch, args.epochs):
        last_epoch = epoch
        training.set_epoch(epoch)
        train_sampler.set_epoch(epoch)
        model.train()
        for batch in train_loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            value, _ = raw_model(model).loss(model(batch))
            value.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            global_step += 1
            if global_step % args.eval_every_steps == 0:
                validation_loss, _ = _validate(model, validation_loader, device)
                if validation_loss < best:
                    best, best_step = validation_loss, global_step
                    if rank == 0:
                        atomic_torch_save(
                            _checkpoint(model, optimizer, scheduler, args, epoch, global_step, best, best_step),
                            output_dir / "best.pt",
                        )
                if rank == 0:
                    atomic_torch_save(
                        _checkpoint(model, optimizer, scheduler, args, epoch, global_step, best, best_step),
                        output_dir / "last.pt",
                    )
                if global_step - best_step >= args.early_stopping_patience:
                    stop = True
                    break
            if global_step >= args.max_steps:
                stop = True
                break
        scheduler.step()
        if stop:
            break
    if not (output_dir / "best.pt").is_file():
        best, _ = _validate(model, validation_loader, device)
        best_step = global_step
        if rank == 0:
            atomic_torch_save(
                _checkpoint(model, optimizer, scheduler, args, last_epoch, global_step, best, best_step),
                output_dir / "best.pt",
            )
    if rank == 0:
        atomic_torch_save(
            _checkpoint(model, optimizer, scheduler, args, last_epoch, global_step, best, best_step),
            output_dir / "last.pt",
        )
    finish_distributed()


__all__ = ["add_arguments", "build_model", "load_model", "train"]
