from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .core import materialize_selection, select_hvg as _select_hvg
from .selection import DataSelection
from .sources.atlas import NativeAtlasSource


DATASET_NAME = "OP3"
SOURCE_DIRECTORY = "OP3"


def _root(storage: str | Path) -> Path:
    return Path(storage)


def _source(storage: str | Path, **options) -> NativeAtlasSource:
    root = _root(storage)
    return NativeAtlasSource(
        root / "raw_datasets" / SOURCE_DIRECTORY,
        root / "projects",
        frozen_models=root / "frozen_models",
        dataset=DATASET_NAME,
        kind="nips",
        population_label="cell_type_ids",
        **options,
    )


def _selection(project_name: str, storage: str | Path) -> DataSelection:
    root = _root(storage)
    workspace = root / "projects" / str(project_name)
    if not (workspace / "preprocess.json").is_file():
        raise FileNotFoundError(
            f"Preprocess project is missing: {workspace}; run fetch_cell_type() first"
        )
    return DataSelection.open(
        workspace, root / "frozen_models", projects=root / "projects"
    )


def statistics(
    *,
    storage: str | Path = "storage",
    smiles_map: str | Path | None = None,
    refresh: bool = False,
    cell_type_key: str | None = None,
    perturbation_key: str | None = None,
    dose_key: str | None = None,
    smiles_key: str | None = None,
    control_key: str | None = None,
    group_key: str | None = None,
):
    return _source(
        storage,
        smiles_map=smiles_map,
        population=cell_type_key,
        drug=perturbation_key,
        dose=dose_key,
        smiles=smiles_key,
        control=control_key,
        group=group_key,
    ).watch_data(refresh=refresh)


def fetch_cell_type(
    cell_types: Sequence[str],
    *,
    project_name: str,
    storage: str | Path = "storage",
    smiles_map: str | Path | None = None,
) -> DataSelection:
    return _source(storage, smiles_map=smiles_map).fetch_populations(
        cell_types, project_name=project_name
    )


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


def materialize(
    project_name: str,
    *,
    storage: str | Path = "storage",
    workers: int = 8,
    target_sum: float = 10_000,
    num_gene_tokens: int = 2_048,
    overwrite: bool = False,
):
    return materialize_selection(
        _selection(project_name, storage),
        project_name=project_name,
        workers=workers,
        target_sum=target_sum,
        num_gene_tokens=num_gene_tokens,
        overwrite=overwrite,
    )


__all__ = [
    "statistics",
    "fetch_cell_type",
    "select_hvg",
    "materialize",
]
