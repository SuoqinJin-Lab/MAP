from __future__ import annotations

from importlib import import_module


MODEL_REGISTRY = ("prnet", "chemcpa", "trainmean", "crisp", "xpert", "cmonge")


def _module(name: str):
    name = str(name).casefold()
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown baseline model: {name}")
    return import_module(f"{__package__}.{name}")


def build_model(name: str, data_dir, hvg_dim: int, populations):
    return _module(name).build_model(data_dir, hvg_dim, populations)


def load_model(name: str, checkpoint: dict, data_dir, hvg_dim: int, populations, device):
    return _module(name).load_model(checkpoint, data_dir, hvg_dim, populations, device)


def add_model_arguments(name: str, parser) -> None:
    _module(name).add_arguments(parser)


def train_model(name: str, args) -> None:
    _module(name).train(args)


__all__ = [
    "MODEL_REGISTRY",
    "add_model_arguments",
    "build_model",
    "load_model",
    "train_model",
]
