from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch

from ...._common.feedback import Feedback, StageResult
from ...._common.identifiers import training_identifier
from ...._common.paths import DatasetPaths
from ...._common.runner import run_command
from ....preparation.validate import validate_method
from ...splits import resolve_split


# Method 4.4 is the reproducible starting configuration, not an API lock.
METHOD44 = {
    "set_size": 24,
    "batch_size": 1,
    "gradient_accumulation_steps": 8,
    "lr": 1e-5,
    "hvg_loss_weight": 0.1,
    "epochs": 100,
    "max_steps": 100_000,
    "warmup_steps": 10_000,
    "checkpoint_every_epochs": 5,
    "samples_per_epoch": None,
    "amp_dtype": "bf16",
    "compile_mode": "default",
    "seed": 42,
    "combination_fusion": "avg_emb",
    "max_components": 2,
}


def train_map(
    paths: DatasetPaths,
    regime: str,
    se_checkpoint: Path,
    esm_embeddings: Path,
    mapkg_checkpoint: Path,
    mapkg_vocab: Path,
    split_file: str | Path | None = None,
    train_split: str = "train",
    gpus: int = 4,
    num_workers: int = 6,
    run_name: str | None = None,
    resume: str | Path | None = None,
    populations: tuple[str, ...] | list[str] | None = None,
    allow_resource_change_on_resume: bool = True,
    dry_run: bool = False,
    **overrides: Any,
) -> StageResult:
    if regime not in {"unprofiled_drug", "unseen_combination", "combosciplex"}:
        raise ValueError(regime)
    unknown = sorted(set(overrides) - set(METHOD44))
    if unknown:
        raise TypeError(f"Unknown training parameters: {', '.join(unknown)}")
    params = {**METHOD44, **overrides}
    if int(gpus) <= 0 or int(num_workers) < 0:
        raise ValueError("gpus must be positive and num_workers must be non-negative")
    for key in (
        "set_size", "batch_size", "gradient_accumulation_steps", "epochs",
        "max_steps", "checkpoint_every_epochs",
    ):
        if int(params[key]) <= 0:
            raise ValueError(f"{key} must be positive")
    if float(params["lr"]) <= 0 or float(params["hvg_loss_weight"]) < 0:
        raise ValueError("lr must be positive and hvg_loss_weight non-negative")
    if params["compile_mode"] not in {"none", "default"}:
        raise ValueError("compile_mode must be 'none' or 'default'")
    if int(params["warmup_steps"]) < 0 or int(params["warmup_steps"]) >= int(params["max_steps"]):
        raise ValueError("warmup_steps must be non-negative and smaller than max_steps")
    split_path, split_payload = resolve_split(paths, regime, split_file)
    if train_split not in split_payload:
        raise KeyError(f"Split set {train_split!r} is absent from {split_path}")

    material_validation = validate_method(
        paths,
        method="map",
        regime=regime,
        split_file=split_path,
        options=params,
        frozen_assets=(se_checkpoint, esm_embeddings, mapkg_checkpoint, mapkg_vocab),
        strict=not dry_run,
    )

    split_id = split_payload.get("split_id", split_path.stem)
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
    run_id = run_name or training_identifier(
        regime,
        split_id,
        set_size=params["set_size"],
        batch_size=params["batch_size"],
        gradient_accumulation_steps=params["gradient_accumulation_steps"],
        lr=params["lr"],
        seed=params["seed"],
        extra={
            "hvg_loss_weight": params["hvg_loss_weight"],
            "max_steps": params["max_steps"],
            "epochs": params["epochs"],
            "warmup_steps": params["warmup_steps"],
            "checkpoint_every_epochs": params["checkpoint_every_epochs"],
            "samples_per_epoch": params["samples_per_epoch"],
            "amp_dtype": params["amp_dtype"],
            "train_split": train_split,
            "populations": list(populations) if populations else list(shapes),
            "combination_fusion": params["combination_fusion"],
            "max_components": params["max_components"],
        },
    )
    if Path(run_id).name != run_id:
        raise ValueError("run_name must be one directory name")
    output = paths.run_dir(split_id, "map", run_id)
    if output.exists() and any(output.iterdir()) and resume is None:
        raise FileExistsError(
            f"Run directory already contains an experiment; change run_name or pass resume: {output}"
        )
    if resume is not None:
        resume_path = Path(resume)
        if not resume_path.is_file():
            raise FileNotFoundError(resume_path)
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        previous = checkpoint.get("args", {})
        allowed_changes = {"resume", "output_dir"}
        if allow_resource_change_on_resume:
            allowed_changes.update({"num_workers", "compile_mode"})
        current = {
            **params,
            "data_dir": str(paths.prepared),
            "regime": regime,
            "split_file": str(split_path),
            "train_split": train_split,
            "populations": list(populations) if populations else previous.get("populations", list(shapes)),
        }
        incompatible = {
            key: {"checkpoint": previous.get(key), "requested": value}
            for key, value in current.items()
            if key in previous and key not in allowed_changes and previous.get(key) != value
        }
        if incompatible:
            raise ValueError(
                "Resume configuration changes experiment semantics: "
                + json.dumps(incompatible, sort_keys=True, default=str)
            )
    report = Feedback(paths.method_dir(split_id, "map"), f"train_{run_id}")
    module_command = [
        sys.executable,
        "-m",
        "map.train.methods.map.program",
        "--data-dir",
        str(paths.prepared),
        "--output-dir",
        str(output),
        "--regime",
        regime,
        "--split-file",
        str(split_path),
        "--train-split",
        train_split,
        "--se-ckpt",
        str(se_checkpoint),
        "--esm-embeddings",
        str(esm_embeddings),
        "--mapkg-ckpt",
        str(mapkg_checkpoint),
        "--mapkg-vocab",
        str(mapkg_vocab),
        "--static-token-cache",
        str(paths.prepared / "knowledge_tokens.pt"),
        "--num-workers",
        str(int(num_workers)),
        "--num-gene-tokens",
        str(num_gene_tokens),
        "--hvg-dim",
        str(hvg_dim),
    ]
    if preparation_config_path.is_file():
        module_command.extend(["--preparation-config", str(preparation_config_path)])
    if resume is not None:
        module_command.extend(["--resume", str(resume)])
    module_command.extend(["--populations", *[str(value) for value in (populations or shapes)]])
    for key, value in params.items():
        if value is not None:
            module_command.extend([f"--{key.replace('_', '-')}", str(value)])
    if int(gpus) > 1:
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={int(gpus)}",
            *module_command[1:],
        ]
    else:
        command = module_command
    # ``engine.py`` lives below ``map/train/methods/map``.  Launch from the
    # repository root so the module path is importable both from a checkout
    # and from an editable install (the shared method engine already follows this
    # convention).
    run_command(command, cwd=Path(__file__).resolve().parents[4], dry_run=dry_run)
    summary = {
        "model": "map",
        "run_id": run_id,
        "run_dir": str(output),
        "regime": regime,
        "split_file": str(split_path),
        "split_id": split_id,
        "split_rule": split_payload.get("rule", regime),
        "split_seed": split_payload.get("seed"),
        "train_split": train_split,
        "params": params,
        "gpus": int(gpus),
        "num_workers": int(num_workers),
        "num_gene_tokens": num_gene_tokens,
        "hvg_dim": hvg_dim,
        "preparation_id": preparation_config.get("preparation_id"),
        "preparation_config": preparation_config,
        "training_material_validation": material_validation.outputs[0],
        "populations": list(populations or shapes),
        "dry_run": dry_run,
        "checkpoint": str(output / "last.pt"),
        "resume": str(Path(resume).resolve()) if resume is not None else None,
    }
    report.emit(
        "training command ready" if dry_run else "training started",
        run_id=run_id,
        split_id=split_id,
        gpus=gpus,
        output=output,
    )
    return report.finish(
        summary,
        [output / "run_config.json", output / "last.pt", output / "checkpoints"],
    )
