from __future__ import annotations

import json
from pathlib import Path

from .._common.paths import DatasetPaths
from .._common.runtime import asset
from .._common.identifiers import slug
from .analysis import (
    analyze_degs,
    analyze_embeddings,
    analyze_generalization,
    analyze_predictions,
    build_report,
    summarize_evaluations,
)
from .runner import evaluate_model


def run(
    paths: DatasetPaths,
    *,
    run_name: str | None = None,
    regime=None,
    split_file=None,
    checkpoint=None,
    model: str | None = None,
    evaluation_name: str | None = None,
    **kwargs,
):
    config = {}
    if run_name is not None:
        if not run_name or Path(run_name).name != run_name:
            raise ValueError("run_name must be one directory name")
        run_dir = paths.runs / run_name
        config_file = run_dir / "run_config.json"
        if not config_file.is_file():
            raise FileNotFoundError(f"Run configuration is missing: {config_file}")
        config = json.loads(config_file.read_text(encoding="utf-8"))
        checkpoint = Path(checkpoint) if checkpoint is not None else run_dir / "best.pt"
    elif checkpoint is None:
        raise ValueError("run_name is required unless checkpoint is provided explicitly")

    checkpoint = Path(checkpoint)
    if not checkpoint.is_file() and not kwargs.get("dry_run", False):
        raise FileNotFoundError(checkpoint)
    inferred_model = config.get("model")
    if inferred_model is None:
        inferred_model = config.get("model_configuration", {}).get("model")
    inferred_model = str(inferred_model or "map").casefold()
    if config and model is not None and str(model).casefold() != inferred_model:
        raise ValueError(
            f"Run {run_name!r} contains model {inferred_model!r}, not {model!r}"
        )
    model = inferred_model if model is None else str(model).casefold()
    regime = regime or config.get("regime")
    split_file = split_file or config.get("split_file")
    if regime is None or split_file is None:
        raise ValueError("regime and split_file are required when they are absent from run_config.json")
    return evaluate_model(
        paths,
        model,
        regime,
        Path(checkpoint),
        asset(paths, "state/se600m.safetensors"),
        asset(paths, "state/Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt", "state/gene_embeddings_esm2.pt"),
        asset(paths, "mapkg/mapkg_encoder_v3.pt", "mapkg/mapkg_encoder.pt", "mapkg/mapkg_model.pt"),
        asset(paths, "mapkg/bart_vocab.txt"),
        split_file=split_file,
        evaluation_name=evaluation_name,
        run_name=run_name,
        populations=config.get("populations") or config.get("args", {}).get("populations"),
        **kwargs,
    )


def analyze(
    paths: DatasetPaths,
    *,
    prediction_file=None,
    evaluation_files=None,
    output_name: str | None = None,
):
    summary_name = (
        f"evaluation_summary__{slug(output_name, 80)}.json"
        if output_name else "evaluation_summary.json"
    )
    summary = summarize_evaluations(
        paths, evaluation_files=evaluation_files, output_name=summary_name
    )
    outputs = [summary]
    prediction = Path(prediction_file) if prediction_file else None
    if prediction is not None:
        outputs.append(analyze_predictions(paths, prediction))
        outputs.append(analyze_degs(paths, prediction))
    outputs.append(build_report(
        paths,
        prediction,
        evaluation_summary_file=Path(summary.outputs[0]),
        output_name=output_name,
    ))
    return outputs
