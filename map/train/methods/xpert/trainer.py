from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from ..common import (
    atomic_torch_save,
    build_loaders,
    EarlyStopping,
    finish_distributed,
    move_batch,
    precision,
    raw_model,
    seed_everything,
    synchronized_loss,
    setup_distributed,
    write_run_config,
)
from .model import XPert
from ..utils import method_data_fields


def add_arguments(parser) -> None:
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--attention-heads", type=int, default=8)
    parser.add_argument("--attention-dropout", type=float, default=0.1)
    parser.add_argument("--hidden-dropout", type=float, default=0.1)
    parser.add_argument("--cell-input-dropout", type=float, default=0.1)
    parser.add_argument("--drug-input-dropout", type=float, default=0.1)
    parser.add_argument(
        "--input-mode", choices=("official", "validation"), default="official"
    )
    parser.add_argument("--use-gene-position-embedding", action="store_true")
    parser.add_argument(
        "--context-tokens",
        choices=("none", "dose", "dose_time"),
        default="dose",
    )
    parser.add_argument(
        "--attention-padding-mode",
        choices=("masked", "official_unmasked"),
        default="masked",
    )
    parser.add_argument("--include-cell-context", action="store_true")
    parser.add_argument(
        "--loss-reduction-scale",
        choices=("none", "sample_count"),
        default="sample_count",
    )
    parser.add_argument("--treated-structure", default="CA+SA+SA+CA")
    parser.add_argument("--control-structure", default="SA+SA+SA+SA")
    parser.add_argument("--expression-bins", type=int, default=128)
    parser.add_argument("--expression-min", type=float)
    parser.add_argument("--expression-max", type=float)
    parser.add_argument("--treated-weight", type=float, default=0.2)
    parser.add_argument("--control-weight", type=float, default=0.003)
    parser.add_argument("--delta-weight", type=float, default=0.2)
    parser.add_argument("--correlation-weight", type=float, default=1.0)
    parser.add_argument("--initial-epochs", type=int, default=70)
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
        "input_mode",
        "use_gene_position_embedding",
        "context_tokens",
        "attention_padding_mode",
        "include_cell_context",
        "loss_reduction_scale",
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


def build_model(data_dir, hvg_dim: int, populations, *, material_dir=None, **options) -> XPert:
    return XPert(data_dir, hvg_dim, populations, material_dir=material_dir, **options)


def load_model(checkpoint, data_dir, hvg_dim: int, populations, device, *, material_dir=None) -> XPert:
    configuration = dict(checkpoint.get("model_configuration", {}))
    configuration.pop("model", None)
    configuration.pop("hvg_dim", None)
    configuration.pop("gene_dim", None)
    configuration.pop("full_gene_dim", None)
    model = build_model(data_dir, hvg_dim, populations, material_dir=material_dir, **configuration)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device)


def _checkpoint(model, optimizer, scheduler, args, epoch, step, early_stopping=None):
    implementation = raw_model(model)
    return {
        "format": "map_method_v2",
        "model": "xpert",
        "epoch": int(epoch),
        "global_step": int(step),
        "model_state_dict": implementation.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "args": vars(args),
        "model_configuration": implementation.configuration(),
        "early_stopping": (
            early_stopping.state_dict() if early_stopping is not None else None
        ),
    }


def train(args) -> None:
    if min(args.hidden_size, args.attention_heads, args.scheduler_milestone) <= 0 or args.expression_bins < 2:
        raise ValueError("XPert dimensions and schedules must be positive")
    if args.initial_epochs < 0:
        raise ValueError("XPert initial_epochs must be non-negative")
    if args.hidden_size % args.attention_heads:
        raise ValueError("XPert hidden_size must be divisible by attention_heads")
    if (args.expression_min is None) != (args.expression_max is None):
        raise ValueError("XPert expression_min and expression_max must be set together")
    if args.expression_min is not None and not args.expression_min < args.expression_max:
        raise ValueError("XPert expression_min must be smaller than expression_max")
    if not 0 < args.scheduler_factor <= 1 or args.weight_decay < 0:
        raise ValueError("XPert scheduler_factor or weight_decay is invalid")
    rank, world, local, device = setup_distributed()
    seed_everything(args.seed, rank)
    training, train_sampler, train_loader = build_loaders(
        args,
        rank,
        world,
        fields=method_data_fields("xpert", input_mode=args.input_mode),
        dataset_kwargs={"sampling_mode": "cell_abundance"},
    )
    shapes = json.loads((Path(args.data_dir) / "materialized_shapes.json").read_text())
    hvg_dim = int(next(iter(shapes.values()))["hvg_dim"])
    model = build_model(args.data_dir, hvg_dim, args.populations, material_dir=args.material_dir, **_model_options(args)).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        fused=device.type == "cuda",
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: 1.0 if epoch < args.scheduler_milestone else args.scheduler_factor,
    )
    start_epoch = global_step = 0
    early_stopping = EarlyStopping(
        args.early_stopping_patience, args.early_stopping_min_delta
    )
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if checkpoint.get("format") not in {"map_method_v2", "map_baseline_v2"} or checkpoint.get("model") != "xpert":
            raise ValueError("Resume checkpoint is not an XPert checkpoint")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["global_step"])
        early_stopping.load_state_dict(checkpoint.get("early_stopping"))
    if world > 1:
        model = DDP(
            model,
            device_ids=[local],
            output_device=local,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
            static_graph=True,
        )
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
    last_epoch = max(start_epoch - 1, 0)
    stop_reason = "max_steps_or_epochs"
    for epoch in range(start_epoch, args.epochs):
        last_epoch = epoch
        training.set_epoch(epoch)
        train_sampler.set_epoch(epoch)
        model.train()
        for batch in train_loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with precision(device, args.amp_dtype):
                value, _ = raw_model(model).loss(
                    model(batch), initial_phase=epoch < args.initial_epochs
                )
            value.backward()
            optimizer.step()
            global_step += 1
            stop_requested = early_stopping.update(
                synchronized_loss(value.detach(), device)
            )
            if stop_requested or global_step >= args.max_steps:
                if stop_requested:
                    stop_reason = "early_stopping"
                break
        scheduler.step()
        if rank == 0:
            if (epoch + 1) % args.checkpoint_every_epochs == 0:
                payload = _checkpoint(
                    model, optimizer, scheduler, args, epoch, global_step,
                    early_stopping,
                )
                atomic_torch_save(
                    payload,
                    output_dir / "checkpoints" / f"epoch_{epoch + 1:04d}.pt",
                )
        if early_stopping.stopped or global_step >= args.max_steps:
            break
    if rank == 0:
        atomic_torch_save(
            _checkpoint(
                model, optimizer, scheduler, args, last_epoch, global_step,
                early_stopping,
            ),
            output_dir / "last.pt",
        )
    finish_distributed()


__all__ = ["add_arguments", "build_model", "load_model", "train"]
