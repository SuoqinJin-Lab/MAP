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
    precision,
    raw_model,
    seed_everything,
    setup_distributed,
    write_run_config,
)
from .model import ChemCPA
from ..utils import method_data_fields


def add_arguments(parser) -> None:
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--autoencoder-width", type=int, default=512)
    parser.add_argument("--autoencoder-depth", type=int, default=4)
    parser.add_argument("--adversary-width", type=int, default=128)
    parser.add_argument("--adversary-depth", type=int, default=3)
    parser.add_argument("--adversary-steps", type=int, default=3)
    parser.add_argument("--reg-adversary", type=float, default=5.0)
    parser.add_argument("--reg-adversary-cov", type=float, default=1.0)
    parser.add_argument("--penalty-adversary", type=float, default=3.0)
    parser.add_argument("--dosers-width", type=int, default=64)
    parser.add_argument("--dosers-depth", type=int, default=2)
    parser.add_argument("--doser-type", choices=("logsigm", "sigm", "mlp"), default="logsigm")
    parser.add_argument("--embedding-encoder-width", type=int, default=512)
    parser.add_argument("--embedding-encoder-depth", type=int, default=0)
    parser.add_argument("--adversary-lr", type=float, default=3e-4)
    parser.add_argument("--dosers-lr", type=float, default=1e-3)
    parser.add_argument("--adversary-wd", type=float, default=1e-4)
    parser.add_argument("--dosers-wd", type=float, default=1e-7)
    parser.add_argument("--step-size-lr", type=int, default=45)


def _model_options(args) -> dict:
    return {
        "dim": args.dim,
        "autoencoder_width": args.autoencoder_width,
        "autoencoder_depth": args.autoencoder_depth,
        "adversary_width": args.adversary_width,
        "adversary_depth": args.adversary_depth,
        "adversary_steps": args.adversary_steps,
        "reg_adversary": args.reg_adversary,
        "reg_adversary_cov": args.reg_adversary_cov,
        "penalty_adversary": args.penalty_adversary,
        "dosers_width": args.dosers_width,
        "dosers_depth": args.dosers_depth,
        "doser_type": args.doser_type,
        "embedding_encoder_width": args.embedding_encoder_width,
        "embedding_encoder_depth": args.embedding_encoder_depth,
    }


def build_model(data_dir, hvg_dim: int, populations, *, material_dir=None, **options) -> ChemCPA:
    return ChemCPA(data_dir, hvg_dim, populations, material_dir=material_dir, **options)


def load_model(checkpoint, data_dir, hvg_dim: int, populations, device, *, material_dir=None) -> ChemCPA:
    configuration = dict(checkpoint.get("model_configuration", {}))
    configuration.pop("model", None)
    configuration.pop("hvg_dim", None)
    model = build_model(data_dir, hvg_dim, populations, material_dir=material_dir, **configuration)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device)


def _optimizers(model: ChemCPA, args):
    compert = model.compert
    autoencoder_parameters = [
        *compert.encoder.parameters(),
        *compert.decoder.parameters(),
        *compert.drug_embedding_encoder.parameters(),
        *compert.population_embeddings.parameters(),
    ]
    adversary_parameters = [
        *compert.adversary_drugs.parameters(),
        *compert.adversary_populations.parameters(),
    ]
    autoencoder = torch.optim.Adam(
        autoencoder_parameters, lr=args.lr, weight_decay=args.weight_decay
    )
    adversaries = torch.optim.Adam(
        adversary_parameters,
        lr=args.adversary_lr,
        weight_decay=args.adversary_wd,
    )
    dosers = torch.optim.Adam(
        compert.dosers.parameters(), lr=args.dosers_lr, weight_decay=args.dosers_wd
    )
    schedulers = {
        "autoencoder": torch.optim.lr_scheduler.StepLR(
            autoencoder, step_size=args.step_size_lr, gamma=0.9
        ),
        "adversaries": torch.optim.lr_scheduler.StepLR(
            adversaries, step_size=args.step_size_lr, gamma=0.9
        ),
        "dosers": torch.optim.lr_scheduler.StepLR(
            dosers, step_size=args.step_size_lr, gamma=0.9
        ),
    }
    return {
        "autoencoder": autoencoder,
        "adversaries": adversaries,
        "dosers": dosers,
    }, schedulers


def _adversary_losses(model: ChemCPA, output: dict) -> tuple[torch.Tensor, torch.Tensor]:
    compert = model.compert
    drug_targets = torch.zeros_like(output["drug_logits"])
    drug_targets.scatter_(1, output["drug_index"].unsqueeze(1), 1.0)
    drug = compert.drug_adversary_loss(output["drug_logits"], drug_targets)
    population = compert.population_adversary_loss(
        output["population_logits"], output["population_index"]
    )
    return drug, population


def _gradient_penalty(output: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
    gradient = torch.autograd.grad(
        output.sum(), latent, create_graph=True, retain_graph=True
    )[0]
    return gradient.square().mean()


def _checkpoint(model, optimizers, schedulers, args, epoch, step) -> dict:
    implementation = raw_model(model)
    return {
        "format": "map_method_v2",
        "model": "chemcpa",
        "epoch": int(epoch),
        "global_step": int(step),
        "model_state_dict": implementation.state_dict(),
        "optimizer_state_dicts": {
            name: optimizer.state_dict() for name, optimizer in optimizers.items()
        },
        "scheduler_state_dicts": {
            name: scheduler.state_dict() for name, scheduler in schedulers.items()
        },
        "args": vars(args),
        "model_configuration": implementation.configuration(),
    }


def train(args) -> None:
    if args.set_size * args.batch_size < 2:
        raise ValueError("chemCPA BatchNorm requires set_size * batch_size >= 2")
    if min(
        args.dim,
        args.autoencoder_width,
        args.autoencoder_depth,
        args.adversary_width,
        args.adversary_depth,
        args.adversary_steps,
        args.dosers_width,
        args.dosers_depth,
        args.step_size_lr,
    ) <= 0:
        raise ValueError("chemCPA dimensions, depths and schedules must be positive")
    if min(args.adversary_lr, args.dosers_lr) <= 0:
        raise ValueError("chemCPA optimizer learning rates must be positive")
    if min(args.weight_decay, args.adversary_wd, args.dosers_wd) < 0:
        raise ValueError("chemCPA weight decays must be non-negative")
    rank, world, local, device = setup_distributed()
    seed_everything(args.seed, rank)
    training, train_sampler, train_loader = build_loaders(
        args, rank, world, fields=method_data_fields("chemcpa")
    )
    shapes = json.loads((Path(args.data_dir) / "materialized_shapes.json").read_text())
    hvg_dim = int(next(iter(shapes.values()))["hvg_dim"])
    model = build_model(
        args.data_dir, hvg_dim, args.populations, material_dir=args.material_dir, **_model_options(args)
    ).to(device)
    optimizers, schedulers = _optimizers(model, args)
    start_epoch = global_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if checkpoint.get("format") not in {"map_method_v2", "map_baseline_v2"} or checkpoint.get("model") != "chemcpa":
            raise ValueError("Resume checkpoint is not a chemCPA method checkpoint")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        for name, optimizer in optimizers.items():
            optimizer.load_state_dict(checkpoint["optimizer_state_dicts"][name])
        for name, scheduler in schedulers.items():
            scheduler.load_state_dict(checkpoint["scheduler_state_dicts"][name])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["global_step"])
    if world > 1:
        model = DDP(
            model,
            device_ids=[local],
            output_device=local,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )

    output_dir = Path(args.output_dir)
    if rank == 0:
        write_run_config(
            output_dir,
            {
                **vars(args),
                "model": "chemcpa",
                "run_name": output_dir.name,
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
            implementation = raw_model(model)
            adversary_step = (
                global_step % implementation.hparams["adversary_steps"] == 0
            )
            with precision(device, args.amp_dtype):
                output = model(batch, adversary_only=adversary_step)
                drug_adversary, population_adversary = _adversary_losses(
                    implementation, output
                )
                if adversary_step:
                    penalty = _gradient_penalty(
                        output["drug_logits"], output["latent_basal"]
                    ) + _gradient_penalty(
                        output["population_logits"], output["latent_basal"]
                    )
                    adversary_loss = (
                        drug_adversary
                        + population_adversary
                        + implementation.hparams["penalty_adversary"] * penalty
                    )
                else:
                    reconstruction = implementation.loss(output, batch)
                    autoencoder_loss = (
                        reconstruction
                        - implementation.hparams["reg_adversary"] * drug_adversary
                        - implementation.hparams["reg_adversary_cov"]
                        * population_adversary
                    )
            if adversary_step:
                optimizers["adversaries"].zero_grad(set_to_none=True)
                adversary_loss.backward()
                optimizers["adversaries"].step()
            else:
                optimizers["autoencoder"].zero_grad(set_to_none=True)
                optimizers["dosers"].zero_grad(set_to_none=True)
                autoencoder_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [
                        parameter
                        for group in optimizers["autoencoder"].param_groups
                        for parameter in group["params"]
                    ],
                    1.0,
                )
                torch.nn.utils.clip_grad_norm_(
                    [
                        parameter
                        for group in optimizers["dosers"].param_groups
                        for parameter in group["params"]
                    ],
                    1.0,
                )
                optimizers["autoencoder"].step()
                optimizers["dosers"].step()
            global_step += 1

            if global_step >= args.max_steps:
                break
        for scheduler in schedulers.values():
            scheduler.step()
        if rank == 0:
            payload = _checkpoint(
                model, optimizers, schedulers, args, epoch, global_step
            )
            atomic_torch_save(payload, output_dir / "last.pt")
            if (epoch + 1) % args.checkpoint_every_epochs == 0:
                atomic_torch_save(
                    payload,
                    output_dir / "checkpoints" / f"epoch_{epoch + 1:04d}.pt",
                )
        if global_step >= args.max_steps:
            break
    if rank == 0:
        atomic_torch_save(
            _checkpoint(model, optimizers, schedulers, args, last_epoch, global_step),
            output_dir / "last.pt",
        )
    finish_distributed()


__all__ = ["add_arguments", "build_model", "load_model", "train"]
