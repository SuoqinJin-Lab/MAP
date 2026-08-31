from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from ..common import (
    atomic_torch_save,
    build_loaders,
    finish_distributed,
    move_batch,
    precision,
    raw_model,
    seed_everything,
    setup_distributed,
    write_run_config,
)
from .model import PRnet
from ..utils import method_data_fields


def add_arguments(parser) -> None:
    parser.add_argument("--weight-decay", type=float, default=1e-8)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[128])
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--adaptor-sizes", type=int, nargs="+", default=[128])
    parser.add_argument("--condition-dim", type=int, default=64)
    parser.add_argument("--noise-dim", type=int, default=10)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--loss-type", choices=("gaussian", "mse"), default="gaussian")
    parser.add_argument("--prediction-samples", type=int, default=3)


def _model_options(args) -> dict:
    return {
        "hidden_sizes": tuple(args.hidden_sizes),
        "latent_dim": args.latent_dim,
        "adaptor_sizes": tuple(args.adaptor_sizes),
        "condition_dim": args.condition_dim,
        "noise_dim": args.noise_dim,
        "dropout": args.dropout,
        "prediction_samples": args.prediction_samples,
    }


def build_model(data_dir, hvg_dim: int, populations, *, material_dir=None, **options) -> PRnet:
    return PRnet(data_dir, hvg_dim, populations, material_dir=material_dir, **options)


def load_model(checkpoint, data_dir, hvg_dim: int, populations, device, *, material_dir=None) -> PRnet:
    configuration = dict(checkpoint.get("model_configuration", {}))
    configuration.pop("model", None)
    configuration.pop("hvg_dim", None)
    for key in ("hidden_sizes", "adaptor_sizes"):
        if key in configuration:
            configuration[key] = tuple(configuration[key])
    model = build_model(data_dir, hvg_dim, populations, material_dir=material_dir, **configuration)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device)


def _loss(model: PRnet, output: dict, batch: dict, loss_type: str) -> torch.Tensor:
    if loss_type == "mse":
        return F.mse_loss(output["mean"].float(), batch["condition_hvg_vectors"].float())
    return model.loss(output, batch)


def _checkpoint(model, optimizer, args, epoch, step) -> dict:
    implementation = raw_model(model)
    return {
        "format": "map_method_v2",
        "model": "prnet",
        "epoch": int(epoch),
        "global_step": int(step),
        "model_state_dict": implementation.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
        "model_configuration": implementation.configuration(),
    }


def train(args) -> None:
    if args.set_size * args.batch_size < 2:
        raise ValueError("PRnet BatchNorm requires set_size * batch_size >= 2")
    if (
        min(
            *args.hidden_sizes,
            *args.adaptor_sizes,
            args.latent_dim,
            args.condition_dim,
            args.noise_dim,
            args.prediction_samples,
        )
        <= 0
    ):
        raise ValueError("PRnet dimensions and prediction_samples must be positive")
    if not 0 <= args.dropout < 1 or args.weight_decay < 0:
        raise ValueError("PRnet dropout must be in [0, 1) and weight_decay non-negative")
    rank, world, local, device = setup_distributed()
    seed_everything(args.seed, rank)
    training, train_sampler, train_loader = build_loaders(
        args, rank, world, fields=method_data_fields("prnet")
    )
    shapes = json.loads((Path(args.data_dir) / "materialized_shapes.json").read_text())
    hvg_dim = int(next(iter(shapes.values()))["hvg_dim"])
    model = build_model(
        args.data_dir, hvg_dim, args.populations, material_dir=args.material_dir, **_model_options(args)
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    start_epoch = global_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if checkpoint.get("format") not in {"map_method_v2", "map_baseline_v2"} or checkpoint.get("model") != "prnet":
            raise ValueError("Resume checkpoint is not a PRnet method checkpoint")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["global_step"])
    if world > 1:
        model = DDP(
            model,
            device_ids=[local],
            output_device=local,
            broadcast_buffers=False,
        )

    output = Path(args.output_dir)
    if rank == 0:
        write_run_config(
            output,
            {
                **vars(args),
                "model": "prnet",
                "run_name": output.name,
                "world_size": world,
                "split_id": training.split_id,
                "model_configuration": raw_model(model).configuration(),
            },
        )

    last_epoch = max(start_epoch - 1, 0)
    for epoch in range(start_epoch, args.epochs):
        last_epoch = epoch
        training.set_epoch(epoch)
        train_sampler.set_epoch(epoch)
        model.train()
        for batch in train_loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with precision(device, args.amp_dtype):
                output_batch = model(batch)
                value = _loss(raw_model(model), output_batch, batch, args.loss_type)
            value.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            global_step += 1

            if global_step >= args.max_steps:
                break
        if rank == 0:
            payload = _checkpoint(model, optimizer, args, epoch, global_step)
            atomic_torch_save(payload, output / "last.pt")
            if (epoch + 1) % args.checkpoint_every_epochs == 0:
                atomic_torch_save(
                    payload, output / "checkpoints" / f"epoch_{epoch + 1:04d}.pt"
                )
        if global_step >= args.max_steps:
            break
    if rank == 0:
        atomic_torch_save(
            _checkpoint(model, optimizer, args, last_epoch, global_step),
            output / "last.pt",
        )
    finish_distributed()


__all__ = ["add_arguments", "build_model", "load_model", "train"]
