from __future__ import annotations

import json
import sys
from pathlib import Path

from .._common.feedback import Feedback, StageResult
from .._common.identifiers import evaluation_identifier
from .._common.paths import DatasetPaths
from .._common.runner import run_command
from ..train.splits import resolve_split


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
    evaluation_splits: tuple[str, ...] | list[str] = (
        "internal_test",
        "external_test",
    ),
    seeds: tuple[int, ...] = (42, 43, 44, 45, 46),
    set_size: int = 24,
    deg_top_k: int = 50,
    deg_fdr: float = 0.05,
    deg_max_cells: int | None = None,
    evaluation_name: str | None = None,
    run_name: str | None = None,
    populations: tuple[str, ...] | list[str] | None = None,
    dry_run: bool = False,
    evaluation_unit: str = "cell_line_drug",
    material_dir: str | Path | None = None,
    drug_representation: str | None = None,
) -> StageResult:
    model = str(model).casefold()
    from ..train.methods.registry import METHOD_REGISTRY

    if model not in METHOD_REGISTRY:
        choices = ", ".join(METHOD_REGISTRY)
        raise ValueError(f"Unknown evaluation model {model!r}; choose from {choices}")
    if not seeds:
        raise ValueError("At least one evaluation seed is required")
    if set_size <= 0 or deg_top_k <= 0 or not 0 < deg_fdr <= 1:
        raise ValueError("set_size/deg_top_k must be positive and deg_fdr in (0, 1]")
    evaluation_unit = str(evaluation_unit).casefold()
    if evaluation_unit not in {"dose_level_condition", "cell_line_drug"}:
        raise ValueError(
            "evaluation_unit must be dose_level_condition or cell_line_drug"
        )
    split_path, split_payload = resolve_split(paths, regime, split_file)
    evaluation_splits = tuple(str(value) for value in evaluation_splits)
    if not evaluation_splits:
        raise ValueError("At least one evaluation split is required")
    unknown_splits = [
        name for name in evaluation_splits
        if name not in {"internal_test", "external_test"}
    ]
    if unknown_splits:
        raise ValueError(
            "evaluation_splits only accepts internal_test/external_test: "
            + ", ".join(unknown_splits)
        )
    missing_splits = [
        name for name in evaluation_splits
        if name not in split_payload
    ]
    if missing_splits:
        raise KeyError(f"Split sets absent from {split_path}: {', '.join(missing_splits)}")
    split_id = split_payload.get("split_id", split_path.stem)
    if not run_name:
        raise ValueError("run_name is required; evaluations belong to a method run")
    shapes = json.loads(
        (paths.prepared / "materialized_shapes.json").read_text(encoding="utf-8")
    )
    first_shape = next(iter(shapes.values()))
    num_gene_tokens = int(first_shape.get("token_length", 2048)) - 1
    hvg_dim = int(first_shape.get("hvg_dim", 2000))
    preparation_config_path = paths.prepared / "preparation_config.json"
    preparation_config = (
        json.loads(preparation_config_path.read_text(encoding="utf-8"))
        if preparation_config_path.is_file()
        else {}
    )
    evaluation_id = evaluation_name or evaluation_identifier(
        regime,
        split_id,
        checkpoint,
        seeds=seeds,
        set_size=set_size,
        deg_top_k=deg_top_k,
        deg_fdr=deg_fdr,
        extra={
            "model": model,
            "run_name": run_name,
            "deg_max_cells": deg_max_cells,
            "batch_size": 1,
            "evaluation_splits": list(evaluation_splits),
            "evaluation_unit": evaluation_unit,
        },
    )
    if Path(evaluation_id).name != evaluation_id:
        raise ValueError("evaluation_name must be one directory name")
    output = paths.evaluation_dir(split_id, model, run_name, evaluation_id)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"Evaluation directory already contains results; change evaluation_name: {output}"
        )
    report = Feedback(output, f"evaluate_{evaluation_id}")
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
        "--evaluation-splits",
        *evaluation_splits,
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
        "--evaluation-unit",
        evaluation_unit,
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
            "--static-token-cache", str(paths.prepared / "knowledge_tokens.pt"),
        ])
    else:
        selected_representation = str(drug_representation or (
            "moa" if model == "cmonge" and regime == "unseen_combination" else ""
        )).casefold()
        selected_material_dir = (
            paths.split_material_dir(split_id, "drug_moa")
            if model == "cmonge" and selected_representation == "moa"
            else Path(material_dir) if material_dir is not None
            else paths.prepared
        )
        command.extend(["--material-dir", str(selected_material_dir)])
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
    prediction_files = [
        output / split_name / f"predictions_seed{seed}.parquet"
        for split_name in evaluation_splits
        for seed in seeds
    ]
    prediction_files_by_split = {
        split_name: [
            str((output / split_name / f"predictions_seed{seed}.parquet").resolve())
            for seed in seeds
        ]
        for split_name in evaluation_splits
    }
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
        "evaluation_splits": list(evaluation_splits),
        "seeds": list(seeds),
        "set_size": int(set_size),
        "batch_size": 1,
        "num_gene_tokens": num_gene_tokens,
        "hvg_dim": hvg_dim,
        "populations": list(populations) if populations else list(shapes),
        "preparation_id": preparation_config.get("preparation_id"),
        "deg_top_k": int(deg_top_k),
        "deg_fdr": float(deg_fdr),
        "deg_max_cells": deg_max_cells,
        "evaluation_unit": evaluation_unit,
        "evaluation_file": str((output / "evaluation.json").resolve()),
        "split_evaluation_files": {
            split_name: str((output / split_name / "evaluation.json").resolve())
            for split_name in evaluation_splits
        },
        "report_files_by_split": {
            split_name: {
                "dose_level_runs": str(
                    (output / split_name / "dose_level_runs.csv").resolve()
                ),
                "dose_level_condition_metrics": str(
                    (output / split_name / "dose_level_condition_metrics.csv").resolve()
                ),
                "cell_line_drug_runs": str(
                    (output / split_name / "cell_line_drug_runs.csv").resolve()
                ),
                "cell_line_drug_metrics": str(
                    (output / split_name / "cell_line_drug_metrics.csv").resolve()
                ),
                "dose_level_per_drug_metrics": str(
                    (output / split_name / "dose_level_per_drug_metrics.csv").resolve()
                ),
                "cell_line_drug_per_drug_metrics": str(
                    (output / split_name / "cell_line_drug_per_drug_metrics.csv").resolve()
                ),
            }
            for split_name in evaluation_splits
        },
        "prediction_files": [str(path.resolve()) for path in prediction_files],
        "prediction_files_by_split": prediction_files_by_split,
        "dry_run": dry_run,
    }
    report_outputs = [
        output / "evaluation.json",
        *[
            output / split_name / "evaluation.json"
            for split_name in evaluation_splits
        ],
        *prediction_files,
    ]
    # The evaluator writes both granular dose-level and merged cell-line/drug
    # reports for every split.  Expose those paths through StageResult so
    # callers do not need to infer the filenames from implementation details.
    for split_name in evaluation_splits:
        split_root = output / split_name
        report_outputs.extend([
            split_root / "dose_level_runs.csv",
            split_root / "cell_line_drug_runs.csv",
            split_root / "dose_level_condition_metrics.csv",
            split_root / "cell_line_drug_metrics.csv",
            split_root / "dose_level_per_drug_metrics.csv",
            split_root / "cell_line_drug_per_drug_metrics.csv",
        ])
    return report.finish(
        summary,
        report_outputs,
    )


__all__ = ["evaluate_model"]
