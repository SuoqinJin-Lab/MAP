from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
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
from .model import PRnet


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


def build_model(data_dir, hvg_dim: int, populations, **options) -> PRnet:
    return PRnet(data_dir, hvg_dim, populations, **options)


def load_model(checkpoint, data_dir, hvg_dim: int, populations, device) -> PRnet:
    configuration = dict(checkpoint.get("model_configuration", {}))
    configuration.pop("model", None)
    configuration.pop("hvg_dim", None)
    for key in ("hidden_sizes", "adaptor_sizes"):
        if key in configuration:
            configuration[key] = tuple(configuration[key])
    model = build_model(data_dir, hvg_dim, populations, **configuration)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device)


def _loss(model: PRnet, output: dict, batch: dict, loss_type: str) -> torch.Tensor:
    if loss_type == "mse":
        return F.mse_loss(output["mean"].float(), batch["condition_hvg_vectors"].float())
    return model.loss(output, batch)


@torch.no_grad()
def _validate(model, loader, device, loss_type: str) -> dict[str, float]:
    model.eval()
    total = torch.zeros(4, device=device, dtype=torch.float64)
    implementation = raw_model(model)
    for batch in loader:
        batch = move_batch(batch, device)
        output = model(batch)
        value = _loss(implementation, output, batch, loss_type)
        predicted_delta = output["mean"].mean(dim=1) - batch[
            "control_hvg_vectors"
        ].float().mean(dim=1)
        observed_delta = batch["condition_hvg_vectors"].float().mean(dim=1) - batch[
            "control_hvg_vectors"
        ].float().mean(dim=1)
        predicted_delta = predicted_delta - predicted_delta.mean(dim=1, keepdim=True)
        observed_delta = observed_delta - observed_delta.mean(dim=1, keepdim=True)
        denominator = (
            predicted_delta.square().sum(dim=1)
            * observed_delta.square().sum(dim=1)
        ).sqrt()
        valid = denominator > 1e-8
        correlations = (
            (predicted_delta[valid] * observed_delta[valid]).sum(dim=1)
            / denominator[valid]
            if valid.any()
            else denominator.new_empty(0)
        )
        total += torch.tensor(
            [value.item(), 1.0, correlations.sum().item(), correlations.numel()],
            device=device,
            dtype=torch.float64,
        )
    loss = distributed_mean(total[:2])
    pearson = distributed_mean(total[2:])
    model.train()
    return {"loss": loss, "pearson_delta": pearson}


def _checkpoint(
    model, optimizer, args, epoch, step, best_score, best_loss, best_step
) -> dict:
    implementation = raw_model(model)
    return {
        "format": "map_baseline_v2",
        "model": "prnet",
        "epoch": int(epoch),
        "global_step": int(step),
        "best_validation_loss": float(best_loss),
        "best_validation_pearson": float(best_score),
        "best_step": int(best_step),
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
    training, _, train_sampler, train_loader, validation_loader = build_loaders(
        args, rank, world
    )
    shapes = json.loads((Path(args.data_dir) / "materialized_shapes.json").read_text())
    hvg_dim = int(next(iter(shapes.values()))["hvg_dim"])
    model = build_model(
        args.data_dir, hvg_dim, args.populations, **_model_options(args)
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    start_epoch = global_step = best_step = 0
    best_score = -float("inf")
    best_loss = float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if checkpoint.get("format") != "map_baseline_v2" or checkpoint.get("model") != "prnet":
            raise ValueError("Resume checkpoint is not a PRnet baseline checkpoint")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["global_step"])
        best_loss = float(checkpoint["best_validation_loss"])
        best_score = float(checkpoint.get("best_validation_pearson", -float("inf")))
        best_step = int(checkpoint["best_step"])
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
            output_batch = model(batch)
            value = _loss(raw_model(model), output_batch, batch, args.loss_type)
            value.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            global_step += 1

            if global_step % args.eval_every_steps == 0:
                validation = _validate(model, validation_loader, device, args.loss_type)
                if validation["pearson_delta"] > best_score:
                    best_score = validation["pearson_delta"]
                    best_loss = validation["loss"]
                    best_step = global_step
                    if rank == 0:
                        atomic_torch_save(
                            _checkpoint(
                                model, optimizer, args, epoch, global_step,
                                best_score, best_loss, best_step,
                            ),
                            output / "best.pt",
                        )
                if rank == 0:
                    atomic_torch_save(
                        _checkpoint(
                            model, optimizer, args, epoch, global_step,
                            best_score, best_loss, best_step,
                        ),
                        output / "last.pt",
                    )
                if global_step - best_step >= args.early_stopping_patience:
                    stop = True
                    break
            if global_step >= args.max_steps:
                stop = True
                break
        if stop:
            break

    if not (output / "best.pt").is_file():
        validation = _validate(model, validation_loader, device, args.loss_type)
        best_score = validation["pearson_delta"]
        best_loss = validation["loss"]
        best_step = global_step
        if rank == 0:
            atomic_torch_save(
                _checkpoint(
                    model, optimizer, args, last_epoch, global_step,
                    best_score, best_loss, best_step,
                ),
                output / "best.pt",
            )
    if rank == 0:
        atomic_torch_save(
            _checkpoint(
                model, optimizer, args, last_epoch, global_step,
                best_score, best_loss, best_step,
            ),
            output / "last.pt",
        )
    finish_distributed()


__all__ = ["add_arguments", "build_model", "load_model", "train"]
