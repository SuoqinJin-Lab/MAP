from __future__ import annotations

import json

from .._common.contracts import load_contract
from .._common.paths import DatasetPaths
from .._common.runtime import asset
from .processing import (
    compute_stats as _stats,
    fit_hvg as _fit_hvg,
    materialize as _materialize,
)
from .selection import DataSelection


def _as_paths(value: DatasetPaths | DataSelection) -> DatasetPaths:
    return value.paths if isinstance(value, DataSelection) else value


def _populations(paths: DatasetPaths) -> tuple[str, ...]:
    shapes = paths.prepared / "materialized_shapes.json"
    if shapes.is_file():
        return tuple(json.loads(shapes.read_text(encoding="utf-8")))
    payload = load_contract(paths.workspace)
    populations = tuple(str(value) for value in payload.get("populations", ()))
    if not populations:
        raise ValueError("The project data contract contains no populations")
    return populations


def _ensure_stats(
    paths: DatasetPaths | DataSelection, *, workers: int = 8, populations=None
):
    if populations is None and isinstance(paths, DataSelection):
        populations = paths.populations
    paths = _as_paths(paths)
    populations = tuple(populations or _populations(paths))
    manifest_path = paths.prepared / "stats_manifest.json"
    required = (
        manifest_path,
        paths.prepared / "global_count_stats.npz",
        paths.prepared / "source_gene_to_state.npy",
        paths.prepared / "state_gene_symbols.json",
    )
    if all(path.is_file() for path in required):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        existing = tuple(str(value) for value in manifest.get("populations", ()))
        if existing != populations:
            raise RuntimeError(
                "Preprocess statistics belong to different populations; use a new project"
            )
        return None
    return _stats(
        paths,
        asset(
            paths,
            "state/Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt",
            "state/gene_embeddings_esm2.pt",
        ),
        workers=workers,
        populations=populations,
    )


def select_hvg(
    paths: DatasetPaths | DataSelection,
    *,
    n_top_genes: int = 2_000,
    workers: int = 8,
):
    selected = paths
    paths = _as_paths(selected)
    _ensure_stats(selected, workers=workers)
    return _fit_hvg(paths, workers=workers, n_top_genes=n_top_genes)


def materialize_selection(
    selection: DataSelection,
    *,
    project_name: str,
    workers: int = 8,
    target_sum: float = 10_000,
    num_gene_tokens: int = 2_048,
    overwrite: bool = False,
):
    if not (selection.prepared / "hvg.json").is_file():
        raise FileNotFoundError("Run preprocess.<dataset>.select_hvg(project_name) first")
    paths = selection.create_project(project_name)
    return _materialize(
        paths,
        workers=workers,
        target_sum=target_sum,
        num_gene_tokens=num_gene_tokens,
        overwrite=overwrite,
    )


__all__ = ["materialize_selection", "select_hvg"]
