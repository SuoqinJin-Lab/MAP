from __future__ import annotations

import math
from pathlib import Path

import torch

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
from .model import CMonge
from ..utils import method_data_fields


def add_arguments(parser) -> None:
    parser.add_argument(
        "--drug-representation", choices=("rdkit", "moa"), default="rdkit",
        help="Drug context used by CMonge; MoA is required for unseen-combination runs.",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--ae-lr", type=float, default=1e-4)
    parser.add_argument("--ae-weight-decay", type=float, default=1e-6)
    parser.add_argument("--ae-epochs", type=int, default=50)
    parser.add_argument("--ae-batch-size", type=int, default=256)
    parser.add_argument("--ae-width", type=int, default=512)
    parser.add_argument("--ae-depth", type=int, default=2)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--context-dim", type=int, default=64)
    parser.add_argument(
        "--hidden-sizes", type=int, nargs="+", default=[256, 256, 256, 256]
    )
    parser.add_argument("--fitting-epsilon", type=float, default=2.0)
    parser.add_argument("--regularizer-epsilon", type=float, default=5e-2)
    parser.add_argument("--monge-gap-weight", type=float, default=5e-3)
    parser.add_argument("--transport-lr-min-ratio", type=float, default=1e-2)
    parser.add_argument("--ae-warmup-steps", type=int, default=100)
    parser.add_argument("--ae-end-lr", type=float, default=1e-5)


def _model_options(args) -> dict:
    return {
        "drug_representation": args.drug_representation,
        "ae_width": args.ae_width,
        "ae_depth": args.ae_depth,
        "latent_dim": args.latent_dim,
        "context_dim": args.context_dim,
        "hidden_sizes": tuple(args.hidden_sizes),
        "fitting_epsilon": args.fitting_epsilon,
        "regularizer_epsilon": args.regularizer_epsilon,
        "monge_gap_weight": args.monge_gap_weight,
    }


def build_model(data_dir, hvg_dim: int, populations, *, material_dir=None, **options) -> CMonge:
    return CMonge(data_dir, hvg_dim, populations, material_dir=material_dir, **options)


def load_model(checkpoint, data_dir, hvg_dim: int, populations, device, *, material_dir=None) -> CMonge:
    configuration = dict(checkpoint.get("model_configuration", {}))
    for key in ("model", "hvg_dim", "descriptor_dim"):
        configuration.pop(key, None)
    # Checkpoints written before the two-layer AE change used one hidden layer.
    configuration.setdefault("ae_depth", 1)
    if "hidden_sizes" in configuration:
        configuration["hidden_sizes"] = tuple(configuration["hidden_sizes"])
    model = build_model(data_dir, hvg_dim, populations, material_dir=material_dir, **configuration)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.freeze_autoencoder()
    return model.to(device)


def _cosine_scheduler(optimizer, total_steps: int, min_ratio: float):
    total_steps = max(int(total_steps), 1)
    min_ratio = float(min_ratio)
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: min_ratio
        + (1.0 - min_ratio)
        * (1.0 + math.cos(math.pi * min(int(step), total_steps) / total_steps))
        / 2.0,
    )


def _warmup_cosine_scheduler(
    optimizer, total_steps: int, warmup_steps: int, end_ratio: float
):
    total_steps = max(int(total_steps), 1)
    warmup_steps = min(max(int(warmup_steps), 0), total_steps)
    end_ratio = float(end_ratio)

    def lr_lambda(step: int) -> float:
        step = min(max(int(step), 0), total_steps)
        if warmup_steps and step < warmup_steps:
            return step / warmup_steps
        decay_steps = max(total_steps - warmup_steps, 1)
        progress = (step - warmup_steps) / decay_steps
        return end_ratio + (1.0 - end_ratio) * (1.0 + math.cos(math.pi * progress)) / 2.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def _pretrain_autoencoder(
    model, loader, sampler, dataset, args, device,
    early_stopping: EarlyStopping,
) -> EarlyStopping:
    implementation = raw_model(model)
    optimizer = torch.optim.AdamW(
        implementation.autoencoder.parameters(),
        lr=args.ae_lr,
        weight_decay=args.ae_weight_decay,
        fused=device.type == "cuda",
    )
    chunks_per_batch = math.ceil(
        2 * int(args.batch_size) * int(args.set_size) / int(args.ae_batch_size)
    )
    total_steps = int(args.ae_epochs) * len(loader) * chunks_per_batch
    scheduler = _warmup_cosine_scheduler(
        optimizer,
        total_steps,
        args.ae_warmup_steps,
        args.ae_end_lr / args.ae_lr,
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
                with precision(device, args.amp_dtype):
                    reconstruction = model(chunk, stage="autoencoder")
                    loss = raw_model(model).autoencoder_loss(reconstruction, chunk)
                loss.backward()
                optimizer.step()
                scheduler.step()
                if early_stopping.update(synchronized_loss(loss.detach(), device)):
                    break
            if early_stopping.stopped:
                break
        if early_stopping.stopped:
            break
    return early_stopping


def _checkpoint(model, optimizer, scheduler, args, epoch, step, early_stopping=None, autoencoder_early_stopping=None) -> dict:
    return build_checkpoint(
        model_name="cmonge", model=model, epoch=epoch, step=step, args=args,
        optimizer=optimizer, scheduler=scheduler, early_stopping=early_stopping,
        extra={
            "autoencoder_pretrained": True,
            "autoencoder_early_stopping": (
                autoencoder_early_stopping.state_dict()
                if autoencoder_early_stopping is not None else None
            ),
        },
    )


def train(args) -> None:
    if args.regime == "unseen_combination" and args.drug_representation != "moa":
        raise ValueError(
            "CMonge unseen-combination training follows the paper and requires "
            "--drug-representation moa"
        )
    dimensions = (
        args.ae_epochs,
        args.ae_batch_size,
        args.ae_width,
        args.ae_depth,
        args.latent_dim,
        args.context_dim,
        *args.hidden_sizes,
    )
    if min(dimensions) <= 0:
        raise ValueError("CMonge dimensions and autoencoder schedule must be positive")
    if min(args.lr, args.ae_lr, args.fitting_epsilon, args.regularizer_epsilon) <= 0:
        raise ValueError("CMonge learning rates and epsilon values must be positive")
    if not 0 < args.transport_lr_min_ratio <= 1:
        raise ValueError("transport_lr_min_ratio must be in (0, 1]")
    if args.ae_warmup_steps < 0 or args.ae_end_lr <= 0 or args.ae_end_lr > args.ae_lr:
        raise ValueError("Invalid AE warmup or end learning rate")
    if min(args.weight_decay, args.ae_weight_decay, args.monge_gap_weight) < 0:
        raise ValueError("CMonge regularization values must be non-negative")
    hvg_dim = load_hvg_dim(args.data_dir, required=2000)
    rank, world, local, device = setup_distributed()
    seed_everything(args.seed, rank)
    training, train_sampler, train_loader = build_loaders(
        args, rank, world, fields=method_data_fields("cmonge")
    )
    model = build_model(
        args.data_dir, hvg_dim, args.populations, material_dir=args.material_dir, **_model_options(args)
    ).to(device)
    start_epoch = global_step = 0
    early_stopping = EarlyStopping(
        args.early_stopping_patience, args.early_stopping_min_delta
    )
    autoencoder_early_stopping = EarlyStopping(
        args.early_stopping_patience, args.early_stopping_min_delta
    )
    checkpoint = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        validate_resume_checkpoint(checkpoint, "cmonge")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    if checkpoint is None:
        ae_model = wrap_ddp(
            model, local=local, world=world,
            gradient_as_bucket_view=True, static_graph=True,
        )
        _pretrain_autoencoder(
            ae_model, train_loader, train_sampler, training, args, device,
            autoencoder_early_stopping,
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
    total_steps = min(int(args.max_steps), int(args.epochs) * len(train_loader))
    scheduler = _cosine_scheduler(
        optimizer, total_steps, args.transport_lr_min_ratio
    )
    if checkpoint is not None:
        start_epoch, global_step = apply_resume_state(
            model, optimizer, scheduler, early_stopping, checkpoint,
        )
        # Method-specific extra state (written via build_checkpoint(extra=...)).
        if "autoencoder_early_stopping" in checkpoint:
            autoencoder_early_stopping.load_state_dict(
                checkpoint["autoencoder_early_stopping"]
            )
        if "scheduler_state_dict" not in checkpoint:
            scheduler.step(global_step)
    model = wrap_ddp(
        model, local=local, world=world,
        gradient_as_bucket_view=True, static_graph=True,
    )

    output = Path(args.output_dir)
    if rank == 0:
        write_method_config(
            output, args, model="cmonge", world=world,
            training=training,
            model_configuration=raw_model(model).configuration(),
        )

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
            with precision(device, args.amp_dtype):
                value, _ = raw_model(model).loss(model(batch), batch)
            value.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            global_step += 1
            stop_requested = early_stopping.update(
                synchronized_loss(value.detach(), device)
            )
            if stop_requested or global_step >= args.max_steps:
                break
        if rank == 0:
            save_epoch_checkpoint(
                output,
                lambda: _checkpoint(
                    model, optimizer, scheduler, args, epoch, global_step,
                    early_stopping, autoencoder_early_stopping,
                ),
                epoch=epoch,
                every=args.checkpoint_every_epochs,
            )
        if early_stopping.stopped or global_step >= args.max_steps:
            break
    if rank == 0:
        atomic_torch_save(
            _checkpoint(
                model, optimizer, scheduler, args, last_epoch, global_step,
                early_stopping, autoencoder_early_stopping,
            ),
            output / "last.pt",
        )
    finish_distributed()


__all__ = ["add_arguments", "build_model", "load_model", "train"]
