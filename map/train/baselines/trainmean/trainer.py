from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ...._common.dataset import MAPDataset
from ..common import atomic_torch_save, move_batch, seed_everything, write_run_config
from .model import TrainMean


def add_arguments(parser) -> None:
    del parser


def build_model(data_dir, hvg_dim: int, populations, **options) -> TrainMean:
    del data_dir, options
    return TrainMean(hvg_dim, populations)


def load_model(checkpoint, data_dir, hvg_dim: int, populations, device) -> TrainMean:
    model = build_model(data_dir, hvg_dim, populations)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device)


def _dataset(args, split: str) -> MAPDataset:
    return MAPDataset(
        args.data_dir,
        args.regime,
        split,
        set_size=args.set_size,
        seed=args.seed,
        training=False,
        populations=args.populations,
        split_file=args.split_file,
    )


@torch.no_grad()
def _fit_vector(loader, device: torch.device) -> tuple[torch.Tensor, int]:
    total = None
    conditions = 0
    for batch in loader:
        values = batch["condition_hvg_vectors"].to(device).float().mean(dim=1)
        batch_sum = values.sum(dim=0)
        total = batch_sum if total is None else total + batch_sum
        conditions += values.shape[0]
    if total is None or conditions == 0:
        raise ValueError("The selected training split contains no conditions")
    return total / conditions, conditions


@torch.no_grad()
def _validation_loss(model: TrainMean, loader, device: torch.device) -> float:
    total = count = 0.0
    for batch in loader:
        batch = move_batch(batch, device)
        value = model.loss(model(batch), batch)
        total += value.item()
        count += 1
    return total / max(count, 1.0)


def train(args) -> None:
    seed_everything(args.seed)
    device = torch.device("cpu")
    training = _dataset(args, args.train_split)
    validation = _dataset(args, args.val_split)
    options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "shuffle": False,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(training, **options)
    validation_loader = DataLoader(validation, **options)
    shapes = json.loads((Path(args.data_dir) / "materialized_shapes.json").read_text())
    hvg_dim = int(next(iter(shapes.values()))["hvg_dim"])
    model = TrainMean(hvg_dim, args.populations).to(device)
    mean_expression, condition_count = _fit_vector(train_loader, device)
    model.fit(mean_expression)
    validation_loss = _validation_loss(model, validation_loader, device)
    output = Path(args.output_dir)
    write_run_config(
        output,
        {
            **vars(args),
            "model": "trainmean",
            "run_name": output.name,
            "world_size": 1,
            "split_id": training.split_id,
            "training_conditions": condition_count,
            "model_configuration": model.configuration(),
        },
    )
    checkpoint = {
        "format": "map_baseline_v2",
        "model": "trainmean",
        "epoch": 0,
        "global_step": int(condition_count),
        "best_validation_loss": float(validation_loss),
        "best_step": int(condition_count),
        "model_state_dict": model.state_dict(),
        "args": vars(args),
        "model_configuration": model.configuration(),
    }
    atomic_torch_save(checkpoint, output / "best.pt")
    atomic_torch_save(checkpoint, output / "last.pt")


__all__ = ["add_arguments", "build_model", "load_model", "train"]
