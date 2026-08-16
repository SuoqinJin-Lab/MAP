from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Sequence

from .core import (
    filter_conditions as _filter_conditions,
    materialize_selection,
    select_hvg as _select_hvg,
)
from .selection import DataSelection
from .sources.atlas import NativeAtlasSource


DATASET_NAME = "SciPlex3"
SOURCE_DIRECTORY = "SciPlex3"


def _root(storage: str | Path) -> Path:
    return Path(storage)


def _source(storage: str | Path, **options) -> NativeAtlasSource:
    root = _root(storage)
    return NativeAtlasSource(
        root / "raw_datasets" / SOURCE_DIRECTORY,
        root / "projects",
        frozen_models=root / "frozen_models",
        dataset=DATASET_NAME,
        kind="sciplex",
        population_label="cell_line_ids",
        **options,
    )


def _selection(project_name: str, storage: str | Path) -> DataSelection:
    root = _root(storage)
    workspace = root / "projects" / str(project_name)
    if not (workspace / "preprocess.json").is_file():
        raise FileNotFoundError(
            f"Preprocess project is missing: {workspace}; run fetch_cell_line() first"
        )
    return DataSelection.open(
        workspace, root / "frozen_models", projects=root / "projects"
    )


def export_rds(
    *,
    storage: str | Path = "storage",
    input_file: str | Path | None = None,
    overwrite: bool = False,
) -> dict:
    source = _root(storage) / "raw_datasets" / SOURCE_DIRECTORY
    if input_file is None:
        candidates = sorted(source.glob("*.RDS")) + sorted(source.glob("*.rds"))
        if len(candidates) != 1:
            raise FileNotFoundError(
                f"Expected one SciPlex3 RDS under {source}; found {len(candidates)}"
            )
        input_path = candidates[0]
    else:
        input_path = Path(input_file)
        if not input_path.is_absolute():
            input_path = source / input_path
    output = source / "exported"
    marker = output / "rds_export_manifest.json"
    if marker.is_file() and not overwrite:
        return json.loads(marker.read_text(encoding="utf-8"))
    if output.exists() and any(output.iterdir()):
        if not overwrite:
            raise FileExistsError(f"SciPlex3 export directory is not empty: {output}")
        shutil.rmtree(output)
    script = Path(__file__).resolve().parents[2] / "scripts" / "export_sciplex3_rds.R"
    if shutil.which("Rscript") is None:
        raise RuntimeError("Rscript is required to export the SciPlex3 RDS")
    subprocess.run(
        ["Rscript", str(script), str(input_path), str(output)], check=True
    )
    from scipy.io import mminfo

    matrix = next(
        path for path in (output / "matrix.mtx.gz", output / "matrix.mtx")
        if path.is_file()
    )
    genes, cells = mminfo(matrix)[:2]
    payload = {
        "input_rds": str(input_path.resolve()),
        "cells": int(cells),
        "genes": int(genes),
        "matrix": str(matrix.resolve()),
        "cell_metadata": str((output / "cell_metadata.tsv.gz").resolve()),
        "gene_metadata": str((output / "gene_metadata.tsv.gz").resolve()),
    }
    marker.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def statistics(
    *,
    storage: str | Path = "storage",
    smiles_map: str | Path | None = None,
    refresh: bool = False,
    cell_line_key: str | None = None,
    perturbation_key: str | None = None,
    dose_key: str | None = None,
    smiles_key: str | None = None,
    control_key: str | None = None,
    group_key: str | None = None,
    gene_symbol_key: str | None = None,
):
    return _source(
        storage,
        smiles_map=smiles_map,
        population=cell_line_key,
        drug=perturbation_key,
        dose=dose_key,
        smiles=smiles_key,
        control=control_key,
        group=group_key,
        gene_symbol=gene_symbol_key,
    ).watch_data(refresh=refresh)


def fetch_cell_line(
    cell_lines: Sequence[str],
    *,
    project_name: str,
    storage: str | Path = "storage",
    smiles_map: str | Path | None = None,
) -> DataSelection:
    return _source(storage, smiles_map=smiles_map).fetch_populations(
        cell_lines, project_name=project_name
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
    "export_rds",
    "statistics",
    "fetch_cell_line",
    "filter_conditions",
    "select_hvg",
    "materialize",
]
