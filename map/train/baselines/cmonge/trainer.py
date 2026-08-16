from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from ..common import (
    atomic_torch_save,
    build_loaders,
    finish_distributed,
    move_batch,
    raw_model,
    seed_everything,
    setup_distributed,
    write_run_config,
)
from .model import CMonge
from ..utils import baseline_data_fields


def add_arguments(parser) -> None:
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--ae-lr", type=float, default=1e-4)
    parser.add_argument("--ae-weight-decay", type=float, default=1e-5)
    parser.add_argument("--ae-epochs", type=int, default=50)
    parser.add_argument("--ae-batch-size", type=int, default=256)
    parser.add_argument("--ae-width", type=int, default=512)
    parser.add_argument("--latent-dim", type=int, default=50)
    parser.add_argument("--context-dim", type=int, default=50)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[64, 64, 64, 64])
    parser.add_argument("--fitting-epsilon", type=float, default=1.0)
    parser.add_argument("--regularizer-epsilon", type=float, default=1e-2)
    parser.add_argument("--monge-gap-weight", type=float, default=1e-2)


def _model_options(args) -> dict:
    return {
        "ae_width": args.ae_width,
        "latent_dim": args.latent_dim,
        "context_dim": args.context_dim,
        "hidden_sizes": tuple(args.hidden_sizes),
        "fitting_epsilon": args.fitting_epsilon,
        "regularizer_epsilon": args.regularizer_epsilon,
        "monge_gap_weight": args.monge_gap_weight,
    }


def build_model(data_dir, hvg_dim: int, populations, **options) -> CMonge:
    return CMonge(data_dir, hvg_dim, populations, **options)


def load_model(checkpoint, data_dir, hvg_dim: int, populations, device) -> CMonge:
    configuration = dict(checkpoint.get("model_configuration", {}))
    for key in ("model", "hvg_dim", "descriptor_dim"):
        configuration.pop(key, None)
    if "hidden_sizes" in configuration:
        configuration["hidden_sizes"] = tuple(configuration["hidden_sizes"])
    model = build_model(data_dir, hvg_dim, populations, **configuration)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.freeze_autoencoder()
    return model.to(device)


def _pretrain_autoencoder(model, loader, sampler, dataset, args, device) -> None:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.ae_lr,
        weight_decay=args.ae_weight_decay,
        fused=device.type == "cuda",
    )
    for epoch in range(args.ae_epochs):
        dataset.set_epoch(epoch)
        sampler.set_epoch(epoch)
        model.train()
        for batch in loader:
            values = torch.cat(
                (batch["control_hvg_vectors"], batch["condition_hvg_vectors"]), dim=1
            ).reshape(-1, raw_model(model).hvg_dim)
            values = values.to(device, non_blocking=True).float()
            for chunk in values.split(args.ae_batch_size):
                optimizer.zero_grad(set_to_none=True)
                reconstruction = model(chunk, stage="autoencoder")
                loss = raw_model(model).autoencoder_loss(reconstruction, chunk)
                loss.backward()
                optimizer.step()


def _checkpoint(model, optimizer, args, epoch, step) -> dict:
    implementation = raw_model(model)
    return {
        "format": "map_baseline_v2",
        "model": "cmonge",
        "epoch": int(epoch),
        "global_step": int(step),
        "autoencoder_pretrained": True,
        "model_state_dict": implementation.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
        "model_configuration": implementation.configuration(),
    }


def train(args) -> None:
    dimensions = (
        args.ae_epochs,
        args.ae_batch_size,
        args.ae_width,
        args.latent_dim,
        args.context_dim,
        *args.hidden_sizes,
    )
    if min(dimensions) <= 0:
        raise ValueError("CMonge dimensions and autoencoder schedule must be positive")
    if min(args.lr, args.ae_lr, args.fitting_epsilon, args.regularizer_epsilon) <= 0:
        raise ValueError("CMonge learning rates and epsilon values must be positive")
    if min(args.weight_decay, args.ae_weight_decay, args.monge_gap_weight) < 0:
        raise ValueError("CMonge regularization values must be non-negative")
    shapes = json.loads((Path(args.data_dir) / "materialized_shapes.json").read_text())
    hvg_dimensions = {int(value["hvg_dim"]) for value in shapes.values()}
    if hvg_dimensions != {2000}:
        raise ValueError(
            "CMonge requires exactly 2000 HVGs; "
            f"materialized dimensions={sorted(hvg_dimensions)}"
        )
    hvg_dim = 2000
    rank, world, local, device = setup_distributed()
    seed_everything(args.seed, rank)
    training, train_sampler, train_loader = build_loaders(
        args, rank, world, fields=baseline_data_fields("cmonge")
    )
    model = build_model(
        args.data_dir, hvg_dim, args.populations, **_model_options(args)
    ).to(device)
    start_epoch = global_step = 0
    checkpoint = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if checkpoint.get("format") != "map_baseline_v2" or checkpoint.get("model") != "cmonge":
            raise ValueError("Resume checkpoint is not a CMonge checkpoint")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    if checkpoint is None:
        ae_model = (
            DDP(
                model,
                device_ids=[local],
                output_device=local,
                gradient_as_bucket_view=True,
                static_graph=True,
            )
            if world > 1
            else model
        )
        _pretrain_autoencoder(
            ae_model, train_loader, train_sampler, training, args, device
        )
        pretrained_model = raw_model(ae_model)
        if world > 1:
            del ae_model
        model = pretrained_model
    model.freeze_autoencoder()
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
        fused=device.type == "cuda",
    )
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["global_step"])
    if world > 1:
        model = DDP(
            model,
            device_ids=[local],
            output_device=local,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
            static_graph=True,
        )

    output = Path(args.output_dir)
    if rank == 0:
        write_run_config(output, {
            **vars(args),
            "model": "cmonge",
            "run_name": output.name,
            "world_size": world,
            "split_id": training.split_id,
            "model_configuration": raw_model(model).configuration(),
        })

    last_epoch = max(start_epoch - 1, 0)
    for epoch in range(start_epoch, args.epochs):
        last_epoch = epoch
        training.set_epoch(epoch + args.ae_epochs)
        train_sampler.set_epoch(epoch + args.ae_epochs)
        model.train()
        raw_model(model).autoencoder.eval()
        for batch in train_loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            value, _ = raw_model(model).loss(model(batch), batch)
            value.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            global_step += 1
            if global_step >= args.max_steps:
                break
        if rank == 0:
            if (epoch + 1) % args.checkpoint_every_epochs == 0:
                payload = _checkpoint(model, optimizer, args, epoch, global_step)
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
