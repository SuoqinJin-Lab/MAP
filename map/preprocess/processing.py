from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .._common.contracts import run_source_stage
from .._common.feedback import Feedback, StageResult
from .._common.identifiers import preparation_identifier
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
    run_source_stage(
        paths, "stats", workers=workers, esm_embeddings=esm_embeddings,
    )
    outputs = [paths.prepared / name for name in (
        "stats_manifest.json", "global_count_stats.npz", "condition_group_census.json",
        "source_gene_to_state.npy", "state_gene_symbols.json", "unmapped_state_gene_symbols.json",
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
) -> StageResult:
    run_source_stage(
        paths, "hvg", workers=workers, n_top_genes=n_top_genes,
        seurat_span=0.3,
    )
    outputs = [paths.prepared / name for name in (
        "hvg.json", "hvg_state_ids.npy", "seurat_v3_model.npz", "seurat_clip_values.npy",
    )]
    hvg = json.loads((paths.prepared / "hvg.json").read_text(encoding="utf-8"))
    return _result(paths, "hvg", {
        "method": hvg.get("flavor", "seurat_v3"),
        "n_hvg": int(hvg.get("n_top_genes", len(hvg.get("state_ids", [])))),
    }, outputs)


def materialize(
    paths: DatasetPaths,
    workers: int = 8,
    target_sum: float = 10_000.0,
    log1p: bool = True,
    num_gene_tokens: int = 2048,
    dtype: str = "float16",
    overwrite: bool = False,
) -> StageResult:
    if not log1p:
        raise ValueError("MAP STATE preprocessing requires log1p=True")
    if dtype != "float16":
        raise ValueError("The current memmap format supports dtype='float16'")
    shapes_path = paths.prepared / "materialized_shapes.json"
    if shapes_path.is_file() and not overwrite:
        existing = json.loads(shapes_path.read_text(encoding="utf-8"))
        first = next(iter(existing.values())) if existing else {}
        expected = {
            "target_sum": float(target_sum),
            "token_length": int(num_gene_tokens) + 1,
        }
        if (
            float(first.get("target_sum", target_sum)) == expected["target_sum"]
            and int(first.get("token_length", num_gene_tokens + 1)) == expected["token_length"]
        ):
            hvg_dim = int(first.get("hvg_dim", 0))
            preparation_id = preparation_identifier(
                populations=list(existing),
                n_top_genes=hvg_dim,
                num_gene_tokens=num_gene_tokens,
                target_sum=target_sum,
                log1p=log1p,
            )
            preparation_manifest = paths.prepared / "preparation_config.json"
            if not preparation_manifest.is_file():
                preparation_manifest.write_text(json.dumps({
                    "preparation_id": preparation_id,
                    "populations": list(existing),
                    "n_top_genes": hvg_dim,
                    "num_gene_tokens": num_gene_tokens,
                    "target_sum": target_sum,
                    "log1p": log1p,
                    "dtype": dtype,
                }, indent=2), encoding="utf-8")
            return _result(paths, "materialize", {
                "preparation_id": preparation_id,
                "reused": True,
                "populations": len(existing),
                "cells": int(sum(int(item["n_cells"]) for item in existing.values())),
                "state_tokens": num_gene_tokens,
                "hvg_genes": hvg_dim,
                "target_sum": target_sum,
                "log1p": log1p,
                "dtype": dtype,
            }, [shapes_path, preparation_manifest])
        raise FileExistsError(
            "Materialized data exists with another configuration; pass overwrite=True only after preserving it"
        )
    run_source_stage(
        paths, "materialize", workers=workers,
        target_sum=target_sum, num_gene_tokens=num_gene_tokens,
    )
    shapes = json.loads((paths.prepared / "materialized_shapes.json").read_text(encoding="utf-8"))
    stats_manifest = json.loads(
        (paths.prepared / "stats_manifest.json").read_text(encoding="utf-8")
    )
    first_shape = next(iter(shapes.values()))
    preparation_config = {
        "populations": list(shapes),
        "hvg_flavor": "seurat_v3",
        "n_top_genes": int(first_shape["hvg_dim"]),
        "num_gene_tokens": int(first_shape["token_length"]) - 1,
        "target_sum": float(target_sum),
        "log1p": bool(log1p),
        "dtype": dtype,
        "gene_vocabulary": "STATE_ESM2",
        "quality_population": stats_manifest.get("quality_filter"),
    }
    preparation_id = preparation_identifier(
        populations=preparation_config["populations"],
        n_top_genes=preparation_config["n_top_genes"],
        num_gene_tokens=preparation_config["num_gene_tokens"],
        target_sum=preparation_config["target_sum"],
        log1p=preparation_config["log1p"],
    )
    preparation_manifest = paths.prepared / "preparation_config.json"
    preparation_manifest.write_text(
        json.dumps({"preparation_id": preparation_id, **preparation_config}, indent=2),
        encoding="utf-8",
    )
    return _result(paths, "materialize", {
        "preparation_id": preparation_id,
        "populations": len(shapes),
        "cells": int(sum(int(item["n_cells"]) for item in shapes.values())),
        "state_tokens": int(next(iter(shapes.values()))["token_length"]) - 1,
        "hvg_genes": int(next(iter(shapes.values()))["hvg_dim"]),
        "target_sum": target_sum,
        "log1p": log1p,
        "dtype": dtype,
        "reused": False,
    }, [paths.prepared / "materialized_shapes.json", preparation_manifest])


__all__ = ["compute_stats", "fit_hvg", "materialize"]
