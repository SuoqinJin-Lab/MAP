from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ...._common.dataset import MAPDataset
from ..common import atomic_torch_save, seed_everything, write_run_config
from .model import TrainMean
from ..utils import method_data_fields


def add_arguments(parser) -> None:
    del parser


def build_model(data_dir, hvg_dim: int, populations, *, material_dir=None, **options) -> TrainMean:
    del data_dir, material_dir, options
    return TrainMean(hvg_dim, populations)


def load_model(checkpoint, data_dir, hvg_dim: int, populations, device, *, material_dir=None) -> TrainMean:
    model = build_model(data_dir, hvg_dim, populations, material_dir=material_dir)
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
        fields=method_data_fields("trainmean"),
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


def train(args) -> None:
    seed_everything(args.seed)
    device = torch.device("cpu")
    training = _dataset(args, args.train_split)
    options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "shuffle": False,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(training, **options)
    shapes = json.loads((Path(args.data_dir) / "materialized_shapes.json").read_text())
    hvg_dim = int(next(iter(shapes.values()))["hvg_dim"])
    model = TrainMean(hvg_dim, args.populations).to(device)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if checkpoint.get("format") not in {"map_method_v2", "map_baseline_v2"} or checkpoint.get("model") != "trainmean":
            raise ValueError("Resume checkpoint is not a TrainMean checkpoint")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        condition_count = int(checkpoint["global_step"])
    else:
        mean_expression, condition_count = _fit_vector(train_loader, device)
        model.fit(mean_expression)
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
            "early_stopping": {
                "applicable": False,
                "reason": "closed_form_fit",
            },
        },
    )
    checkpoint = {
        "format": "map_method_v2",
        "model": "trainmean",
        "epoch": 0,
        "global_step": int(condition_count),
        "model_state_dict": model.state_dict(),
        "args": vars(args),
        "model_configuration": model.configuration(),
        "early_stopping": {
            "applicable": False,
            "reason": "closed_form_fit",
        },
    }
    atomic_torch_save(checkpoint, output / "last.pt")
    atomic_torch_save(checkpoint, output / "checkpoints" / "epoch_0001.pt")


__all__ = ["add_arguments", "build_model", "load_model", "train"]
