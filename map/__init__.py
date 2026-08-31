"""MAP tools organized as preprocessing, preparation, training and evaluation."""

from importlib import import_module

from ._common.feedback import StageResult
from ._common.paths import DatasetPaths, experiment_paths


def __getattr__(name):
    if name in {"preprocess", "preparation", "train", "eval"}:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(name)

__all__ = [
    "preprocess",
    "preparation",
    "train",
    "eval",
    "DatasetPaths",
    "StageResult",
    "experiment_paths",
]
