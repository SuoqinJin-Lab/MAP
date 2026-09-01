from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .._common.contracts import run_handler_stage
from .._common.feedback import Feedback, StageResult
from .._common.paths import DatasetPaths


def _result(paths: DatasetPaths, stage: str, summary: dict, outputs: list[Path]) -> StageResult:
    report = Feedback(paths.prepared, stage)
    printable = {
        key: value for key, value in summary.items()
        if isinstance(value, (str, int, float, bool))
    }
    report.emit("summary", **printable)
    return report.finish(summary, outputs)


def compute_stats(
    paths: DatasetPaths,
    esm_embeddings: Path,
    workers: int = 8,
    populations: tuple[str, ...] | list[str] = (),
) -> StageResult:
    paths.ensure_outputs()
    run_handler_stage(
        paths, "stats", workers=workers, esm_embeddings=esm_embeddings,
    )
    outputs = [paths.prepared / name for name in (
        "stats_manifest.json", "global_count_stats.npz", "batch_count_stats.npz",
        "condition_group_census.json", "eligible_file_cell_counts.npy",
        "preparation_state.pkl", "hvg_sample_plan.pkl", "source_gene_to_state.npy",
        "state_gene_symbols.json", "unmapped_state_gene_symbols.json",
    )]
    manifest = json.loads((paths.prepared / "stats_manifest.json").read_text(encoding="utf-8"))
    eligible = manifest.get("eligible_cells_by_population", {})
    mapping = np.load(paths.prepared / "source_gene_to_state.npy", mmap_mode="r")
    return _result(paths, "stats", {
        "workers": workers,
        "populations": list(populations),
        "source_parts": len(manifest.get("source_parts", [])),
        "eligible_cells": int(sum(eligible.values())),
        "excluded_missing_smiles_or_dose": int(manifest.get("excluded_perturbation_cells_without_smiles_or_dose", 0)),
        "mapped_state_genes": int(np.unique(mapping[mapping >= 0]).size),
        "normalization": "deferred_to_materialize",
    }, outputs)


def fit_hvg(
    paths: DatasetPaths,
    workers: int = 8,
    n_top_genes: int = 2000,
    batch_key: str | None = "population",
) -> StageResult:
    run_handler_stage(
        paths, "hvg", workers=workers, n_top_genes=n_top_genes,
        seurat_span=0.3, hvg_batch_key=batch_key,
    )
    hvg = json.loads((paths.prepared / "hvg.json").read_text(encoding="utf-8"))
    model_name = (
        "seurat_v3_batch_model.npz" if hvg.get("batch_key") == "population"
        else "seurat_v3_model.npz"
    )
    outputs = [paths.prepared / name for name in (
        "hvg.json", "hvg_state_ids.npy", model_name, "seurat_clip_values.npy",
    )]
    return _result(paths, "hvg", {
        "method": hvg.get("flavor", "seurat_v3"),
        "n_hvg": int(hvg.get("n_top_genes", len(hvg.get("state_ids", [])))),
        "batch_key": hvg.get("batch_key"),
        "protocol": hvg.get("protocol"),
        "fingerprint": hvg.get("fingerprint"),
    }, outputs)


__all__ = ["compute_stats", "fit_hvg"]
