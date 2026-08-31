"""One registry for every trainable method."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module


@dataclass(frozen=True)
class MethodSpec:
    name: str
    shared_artifacts: tuple[str, ...]
    artifacts: tuple[str, ...] = ()
    split_artifacts: tuple[str, ...] = ()
    frozen_assets: tuple[str, ...] = ()
    supported_regimes: tuple[str, ...] = ("unprofiled_drug", "unseen_combination")


_SHARED = ("cell_metadata", "hvg_expression")
METHOD_SPECS = (
    MethodSpec("map", _SHARED, ("state_inputs", "knowledge_tokens", "condition_embeddings"), frozen_assets=("se600m", "esm2", "mapkg")),
    MethodSpec("prnet", _SHARED, ("drug_fcfp4",)),
    MethodSpec("chemcpa", _SHARED, ("drug_ecfp4",)),
    MethodSpec("trainmean", _SHARED),
    MethodSpec("crisp", _SHARED, ("drug_rdkit2d", "control_means", "deg_masks")),
    MethodSpec("xpert", _SHARED, ("drug_unimol", "graph_assets", "expression_bins")),
    MethodSpec("cmonge", _SHARED, ("drug_rdkit2d",), ("drug_moa",)),
)
METHOD_REGISTRY = tuple(spec.name for spec in METHOD_SPECS)
_BY_NAME = {spec.name: spec for spec in METHOD_SPECS}


def method_spec(name: str) -> MethodSpec:
    key = str(name).casefold()
    try:
        return _BY_NAME[key]
    except KeyError as error:
        raise ValueError(f"Unknown method {name!r}; choose from {METHOD_REGISTRY}") from error


def _module(name: str):
    method_spec(name)
    return import_module(f"map.train.methods.{str(name).casefold()}.trainer")


def add_model_arguments(name: str, parser) -> None:
    if str(name).casefold() == "map":
        return
    _module(name).add_arguments(parser)


def build_model(name: str, data_dir, hvg_dim: int, populations, *, material_dir=None, **options):
    return _module(name).build_model(
        data_dir, hvg_dim, populations, material_dir=material_dir, **options
    )


def load_model(name: str, checkpoint: dict, data_dir, hvg_dim: int, populations, device, *, material_dir=None):
    return _module(name).load_model(
        checkpoint, data_dir, hvg_dim, populations, device, material_dir=material_dir
    )


def train_model(name: str, args) -> None:
    _module(name).train(args)


def method_runner(name: str):
    name = str(name).casefold()
    if name == "map":
        return import_module("map.train.methods.map.trainer").run
    from .engine import train_method
    return lambda paths, **kwargs: train_method(paths, model=name, **kwargs)


__all__ = [
    "METHOD_REGISTRY", "METHOD_SPECS", "MethodSpec",
    "add_model_arguments", "build_model", "load_model", "method_runner",
    "method_spec", "train_model",
]
