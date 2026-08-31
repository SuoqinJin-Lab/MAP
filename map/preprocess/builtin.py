"""Ready-to-use dataset handlers.

These factories contain native-layout knowledge.  The common preprocessing
pipeline never imports a dataset name and only receives the returned handler.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .handlers.anndata import AnnDataHandler
from .handlers.atlas import AtlasHandler
from .handlers.tahoe import TahoeHandler


def _roots(storage: str | Path):
    root = Path(storage)
    return root / "raw_datasets", root / "projects", root / "frozen_models"


def tahoe(*, storage: str | Path = "storage", directory: str = "Tahoe-100M"):
    raw, projects, frozen = _roots(storage)
    return TahoeHandler(
        raw / directory,
        projects,
        frozen_models=frozen,
        dataset="Tahoe-100M",
    )


def sciplex(
    *, storage: str | Path = "storage", directory: str = "SciPlex3", **schema: Any
):
    raw, projects, frozen = _roots(storage)
    return AtlasHandler(
        raw / directory,
        projects,
        frozen_models=frozen,
        dataset="SciPlex3",
        kind="sciplex",
        population_label="cell_line_ids",
        **schema,
    )


def nips(
    *, storage: str | Path = "storage", directory: str = "OP3", **schema: Any
):
    raw, projects, frozen = _roots(storage)
    return AtlasHandler(
        raw / directory,
        projects,
        frozen_models=frozen,
        dataset="OP3",
        kind="nips",
        population_label="cell_type_ids",
        **schema,
    )


def anndata(
    *,
    dataset: str,
    source: str | Path,
    storage: str | Path = "storage",
    **schema: Any,
):
    _, projects, frozen = _roots(storage)
    return AnnDataHandler(
        source,
        projects,
        frozen_models=frozen,
        dataset=dataset,
        **schema,
    )


__all__ = ["anndata", "nips", "sciplex", "tahoe"]
