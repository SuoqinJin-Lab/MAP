from __future__ import annotations

import json
import sys
from pathlib import Path

from .._common.feedback import Feedback, StageResult
from .._common.identifiers import evaluation_identifier
from .._common.paths import DatasetPaths
from .._common.runner import run_command
from ..train.engine import _resolve_split


def evaluate_model(
    paths: DatasetPaths,
    model: str,
    regime: str,
    checkpoint: Path,
    se_checkpoint: Path,
    esm_embeddings: Path,
    mapkg_checkpoint: Path,
    mapkg_vocab: Path,
    split_file: str | Path | None = None,
    test_split: str = "test",
    seeds: tuple[int, ...] = (42, 43, 44, 45, 46),
    set_size: int = 24,
    deg_top_k: int = 50,
    deg_fdr: float = 0.05,
    deg_max_cells: int | None = None,
    evaluation_name: str | None = None,
    run_name: str | None = None,
    populations: tuple[str, ...] | list[str] | None = None,
    dry_run: bool = False,
) -> StageResult:
    model = str(model).casefold()
    from ..train.baselines import MODEL_REGISTRY

    if model != "map" and model not in MODEL_REGISTRY:
        choices = ", ".join(("map", *MODEL_REGISTRY))
        raise ValueError(f"Unknown evaluation model {model!r}; choose from {choices}")
    if not seeds:
        raise ValueError("At least one evaluation seed is required")
    if set_size <= 0 or deg_top_k <= 0 or not 0 < deg_fdr <= 1:
        raise ValueError("set_size/deg_top_k must be positive and deg_fdr in (0, 1]")
    split_path, split_payload = _resolve_split(paths, regime, split_file)
    if test_split not in split_payload:
        raise KeyError(f"Split set {test_split!r} is absent from {split_path}")
    split_id = split_payload.get("split_id", split_path.stem)
    shapes = json.loads(
        (paths.prepared / "materialized_shapes.json").read_text(encoding="utf-8")
    )
    first_shape = next(iter(shapes.values()))
    num_gene_tokens = int(first_shape.get("token_length", 2049)) - 1
    hvg_dim = int(first_shape.get("hvg_dim", 2000))
    preparation_config_path = paths.prepared / "preparation_config.json"
    preparation_config = (
        json.loads(preparation_config_path.read_text(encoding="utf-8"))
        if preparation_config_path.is_file()
        else {}
    )
    evaluation_id = evaluation_name or run_name or evaluation_identifier(
        regime,
        split_id,
        checkpoint,
        seeds=seeds,
        set_size=set_size,
        deg_top_k=deg_top_k,
        deg_fdr=deg_fdr,
        extra={
            "model": model,
            "deg_max_cells": deg_max_cells,
            "test_split": test_split,
        },
    )
    if Path(evaluation_id).name != evaluation_id:
        raise ValueError("evaluation_name must be one directory name")
    output = paths.evaluations / evaluation_id
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"Evaluation directory already contains results; change evaluation_name: {output}"
        )
    report = Feedback(paths.evaluations, f"evaluate_{evaluation_id}")
    command = [
        sys.executable,
        "-m",
        "map.eval.program",
        "--checkpoint",
        str(checkpoint),
        "--model",
        model,
        "--data-dir",
        str(paths.prepared),
        "--output-dir",
        str(output),
        "--regime",
        regime,
        "--split-file",
        str(split_path),
        "--test-split",
        test_split,
        "--set-size",
        str(int(set_size)),
        "--num-gene-tokens",
        str(num_gene_tokens),
        "--hvg-dim",
        str(hvg_dim),
        "--deg-top-k",
        str(int(deg_top_k)),
        "--deg-fdr",
        str(float(deg_fdr)),
        "--seeds",
        *[str(int(seed)) for seed in seeds],
    ]
    if populations:
        command.extend(["--populations", *[str(value) for value in populations]])
    if model == "map":
        command.extend([
            "--se-ckpt", str(se_checkpoint),
            "--esm-embeddings", str(esm_embeddings),
            "--mapkg-ckpt", str(mapkg_checkpoint),
            "--mapkg-vocab", str(mapkg_vocab),
            "--static-token-cache", str(paths.prepared / "map_static_tokens.pt"),
        ])
    if preparation_config_path.is_file():
        command.extend(["--preparation-config", str(preparation_config_path)])
    if deg_max_cells is not None:
        command.extend(["--deg-max-cells", str(int(deg_max_cells))])
    run_command(command, cwd=Path(__file__).resolve().parents[2], dry_run=dry_run)
    report.emit(
        "evaluation command ready" if dry_run else "evaluation started",
        evaluation_id=evaluation_id,
        split_id=split_id,
        seeds=list(seeds),
        output=output,
    )
    prediction_files = [output / f"predictions_seed{seed}.parquet" for seed in seeds]
    summary = {
        "model": model,
        "run_name": run_name,
        "evaluation_id": evaluation_id,
        "evaluation_dir": str(output.resolve()),
        "regime": regime,
        "checkpoint": str(checkpoint),
        "split_file": str(split_path),
        "split_id": split_id,
        "split_rule": split_payload.get("rule", regime),
        "split_seed": split_payload.get("seed"),
        "test_split": test_split,
        "seeds": list(seeds),
        "set_size": int(set_size),
        "num_gene_tokens": num_gene_tokens,
        "hvg_dim": hvg_dim,
        "populations": list(populations) if populations else list(shapes),
        "preparation_id": preparation_config.get("preparation_id"),
        "deg_top_k": int(deg_top_k),
        "deg_fdr": float(deg_fdr),
        "deg_max_cells": deg_max_cells,
        "evaluation_file": str((output / "evaluation.json").resolve()),
        "prediction_files": [str(path.resolve()) for path in prediction_files],
        "dry_run": dry_run,
    }
    return report.finish(
        summary,
        [output / "evaluation.json", output / "runs.csv", *prediction_files],
    )


def evaluate_map(paths: DatasetPaths, regime: str, checkpoint: Path, *args, **kwargs):
    """Compatibility wrapper for callers that used the MAP-only evaluator."""
    return evaluate_model(paths, "map", regime, checkpoint, *args, **kwargs)


__all__ = ["evaluate_map", "evaluate_model"]
