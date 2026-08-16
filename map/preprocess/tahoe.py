from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from .core import (
    filter_conditions as _filter_conditions,
    materialize_selection,
    select_hvg as _select_hvg,
)
from .selection import DataSelection
from .sources.tahoe import TahoeSource


DATASET_NAME = "Tahoe-100M"


def _root(storage: str | Path) -> Path:
    return Path(storage)


def _source(storage: str | Path) -> TahoeSource:
    root = _root(storage)
    return TahoeSource(
        root / "raw_datasets" / DATASET_NAME,
        root / "projects",
        frozen_models=root / "frozen_models",
        dataset=DATASET_NAME,
    )


def _selection(project_name: str, storage: str | Path) -> DataSelection:
    root = _root(storage)
    record = root / "projects" / str(project_name) / "preprocess.json"
    if not record.is_file():
        raise FileNotFoundError(
            f"Preprocess project is missing: {record}; run fetch_cell_line() first"
        )
    payload = json.loads(record.read_text(encoding="utf-8"))
    return DataSelection.open(
        payload["selection_cache"],
        root / "frozen_models",
        projects=root / "projects",
    )


def statistics(
    *, storage: str | Path = "storage", batch_size: int = 8192, refresh: bool = False
):
    return _source(storage).watch_data(batch_size=batch_size, refresh=refresh)


def fetch_cell_line(
    cell_lines: Sequence[str],
    *,
    project_name: str,
    storage: str | Path = "storage",
    batch_size: int = 8192,
) -> DataSelection:
    requested = tuple(dict.fromkeys(str(value) for value in cell_lines))
    record = _root(storage) / "projects" / str(project_name) / "preprocess.json"
    if record.is_file():
        selection = _selection(project_name, storage)
        if selection.populations != requested:
            raise FileExistsError(
                f"Project {project_name!r} is already bound to different cell lines"
            )
        return selection
    selection = _source(storage).fetch_cell_line(
        requested, project_name=project_name, batch_size=batch_size
    )
    selection.reserve_project(project_name)
    return selection


def select_hvg(
    project_name: str,
    *,
    storage: str | Path = "storage",
    n_top_genes: int = 2_000,
    workers: int = 8,
):
    return _select_hvg(
        _selection(project_name, storage),
        n_top_genes=n_top_genes,
        workers=workers,
    )


def filter_conditions(
    project_name: str,
    *,
    storage: str | Path = "storage",
    min_cells: int = 500,
    max_cells: int = 5_000,
    seed: int = 42,
    workers: int = 8,
    overwrite: bool = False,
):
    return _filter_conditions(
        _selection(project_name, storage),
        min_cells=min_cells,
        max_cells=max_cells,
        seed=seed,
        workers=workers,
        overwrite=overwrite,
    )


def materialize(
    project_name: str,
    *,
    storage: str | Path = "storage",
    workers: int = 8,
    target_sum: float = 10_000,
    pad_length: int = 2_048,
    overwrite: bool = False,
):
    return materialize_selection(
        _selection(project_name, storage),
        project_name=project_name,
        workers=workers,
        target_sum=target_sum,
        pad_length=pad_length,
        overwrite=overwrite,
    )


__all__ = [
    "statistics", "fetch_cell_line", "filter_conditions", "select_hvg", "materialize"
]
