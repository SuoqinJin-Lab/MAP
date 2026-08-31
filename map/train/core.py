from __future__ import annotations

from .._common.paths import DatasetPaths
from .methods.map.engine import METHOD44
from .methods.registry import METHOD_REGISTRY, method_runner


def run(
    paths: DatasetPaths,
    *,
    regime: str,
    split_file,
    run_name: str | None = None,
    model: str = "map",
    **kwargs,
):
    model = str(model).casefold()
    if model not in METHOD_REGISTRY:
        choices = ", ".join(METHOD_REGISTRY)
        raise ValueError(f"Unknown training model {model!r}; choose from {choices}")
    return method_runner(model)(
        paths, regime=regime, split_file=split_file, run_name=run_name, **kwargs
    )
