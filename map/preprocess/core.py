from __future__ import annotations

import json

from .._common.contracts import load_contract, run_handler_stage
from .._common.feedback import Feedback
from .._common.paths import DatasetPaths
from .._common.runtime import asset
from .processing import (
    compute_stats as _stats,
    fit_hvg as _fit_hvg,
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
    filter_manifest = paths.prepared / "condition_filter.json"
    if not filter_manifest.is_file():
        raise FileNotFoundError(
            "Condition filtering is required before HVG selection; run "
            "preprocess.<dataset>.filter_conditions(project_name) first"
        )
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
        current_filter = json.loads(filter_manifest.read_text(encoding="utf-8"))
        manifest_filter = manifest.get("condition_filter", {})
        current_filter_id = current_filter.get("filter_id")
        manifest_filter_id = manifest_filter.get("filter_id") or manifest.get("filter_id")
        if not current_filter_id or manifest_filter_id != current_filter_id:
            raise RuntimeError(
                "Preprocess statistics belong to a different condition filter; "
                "use a new project or regenerate all preparation outputs"
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


def filter_conditions(
    paths: DatasetPaths | DataSelection,
    *,
    min_cells: int = 500,
    max_cells: int = 5_000,
    seed: int = 42,
    workers: int = 8,
    overwrite: bool = False,
):
    """Keep eligible drug-dose groups independently within each population.

    The source backend writes a deterministic sampling plan.  All later scans
    (statistics, HVG fitting and materialization) consume that same plan.
    """
    selected = paths
    paths = _as_paths(selected)
    if min_cells <= 0 or max_cells <= 0 or min_cells > max_cells:
        raise ValueError("Require 0 < min_cells <= max_cells")
    downstream = (
        paths.prepared / "stats_manifest.json",
        paths.prepared / "hvg.json",
        paths.prepared / "materialized_shapes.json",
    )
    if any(path.is_file() for path in downstream) and not (
        paths.prepared / "condition_filter.json"
    ).is_file():
        raise FileExistsError(
            "This project already has unfiltered preparation outputs; create a new project "
            "to apply condition filtering"
        )
    paths.prepared.mkdir(parents=True, exist_ok=True)
    # Keep an existing filter immutable once downstream caches exist.  The
    # backend reuses an identical plan and rejects a changed one.
    effective_overwrite = bool(overwrite) and not any(
        path.is_file() for path in downstream
    )
    run_handler_stage(
        paths,
        "filter",
        workers=int(workers),
        min_cells=int(min_cells),
        max_cells=int(max_cells),
        seed=int(seed),
        overwrite=effective_overwrite,
    )
    manifest = json.loads(
        (paths.prepared / "condition_filter.json").read_text(encoding="utf-8")
    )
    summary = {
        "filter_id": manifest["filter_id"],
        "min_cells": int(min_cells),
        "max_cells": int(max_cells),
        "seed": int(seed),
        "populations": list(manifest.get("populations", ())),
        "source_conditions": int(manifest["source_conditions"]),
        "retained_conditions": int(manifest["retained_conditions"]),
        "dropped_conditions": int(manifest["dropped_conditions"]),
        "capped_conditions": int(manifest["capped_conditions"]),
        "source_condition_cells": int(manifest["source_condition_cells"]),
        "retained_condition_cells": int(manifest["retained_condition_cells"]),
    }
    outputs = [
        paths.prepared / "condition_filter.json",
        paths.prepared / "condition_filter.parquet",
        paths.prepared / "condition_filter_file_offsets.npy",
        paths.prepared / "condition_filter_condition_ids.npy",
        paths.prepared / "condition_filter_file_counts.npy",
        paths.prepared / "condition_filter_file_quotas.npy",
    ]
    report = Feedback(paths.prepared, "filter_conditions")
    report.emit(
        "summary",
        filter_id=summary["filter_id"],
        retained_conditions=summary["retained_conditions"],
        retained_condition_cells=summary["retained_condition_cells"],
    )
    return report.finish(summary, outputs)


def select_hvg(
    paths: DatasetPaths | DataSelection,
    *,
    n_top_genes: int = 2_000,
    workers: int = 8,
    batch_key: str | None = None,
):
    selected = paths
    paths = _as_paths(selected)
    _ensure_stats(selected, workers=workers)
    return _fit_hvg(
        paths, workers=workers, n_top_genes=n_top_genes, batch_key=batch_key
    )


__all__ = ["filter_conditions", "select_hvg"]
