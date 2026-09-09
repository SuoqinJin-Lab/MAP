from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from ..common import (
    apply_resume_state,
    atomic_torch_save,
    build_checkpoint,
    build_loaders,
    EarlyStopping,
    finish_distributed,
    load_hvg_dim,
    move_batch,
    precision,
    raw_model,
    save_epoch_checkpoint,
    seed_everything,
    synchronized_loss,
    setup_distributed,
    validate_resume_checkpoint,
    wrap_ddp,
    write_method_config,
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


def _checkpoint(model, optimizer, args, epoch, step, early_stopping=None) -> dict:
    return build_checkpoint(
        model_name="prnet", model=model, epoch=epoch, step=step, args=args,
        optimizer=optimizer, early_stopping=early_stopping,
    )


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
    hvg_dim = load_hvg_dim(args.data_dir)
    model = build_model(
        args.data_dir, hvg_dim, args.populations, material_dir=args.material_dir, **_model_options(args)
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    start_epoch = global_step = 0
    early_stopping = EarlyStopping(
        args.early_stopping_patience, args.early_stopping_min_delta
    )
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        validate_resume_checkpoint(checkpoint, "prnet")
        start_epoch, global_step = apply_resume_state(
            model, optimizer, None, early_stopping, checkpoint
        )
    model = wrap_ddp(model, local=local, world=world)

    output = Path(args.output_dir)
    if rank == 0:
        write_method_config(
            output, args, model="prnet", world=world,
            training=training,
            model_configuration=raw_model(model).configuration(),
        )

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
                output_batch = model(batch)
                value = _loss(raw_model(model), output_batch, batch, args.loss_type)
            value.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            global_step += 1
            stop_requested = early_stopping.update(
                synchronized_loss(value.detach(), device)
            )

            if stop_requested or global_step >= args.max_steps:
                if stop_requested:
                    stop_reason = "early_stopping"
                break
        if rank == 0:
            save_epoch_checkpoint(
                output,
                lambda: _checkpoint(
                    model, optimizer, args, epoch, global_step, early_stopping
                ),
                epoch=epoch,
                every=args.checkpoint_every_epochs,
                last_every_epoch=True,
            )
        if early_stopping.stopped or global_step >= args.max_steps:
            break
    if rank == 0:
        atomic_torch_save(
            _checkpoint(
                model, optimizer, args, last_epoch, global_step, early_stopping
            ),
            output / "last.pt",
        )
    finish_distributed()


__all__ = ["add_arguments", "build_model", "load_model", "train"]
