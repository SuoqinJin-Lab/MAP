from __future__ import annotations

import json
import sys
from pathlib import Path

from ..._common.feedback import Feedback
from ..._common.identifiers import config_digest, training_identifier
from ..._common.paths import DatasetPaths
from ..._common.runner import run_command
from ..engine import _resolve_split
from .registry import MODEL_REGISTRY


_COMMON = {
    "set_size": 24,
    "batch_size": 1,
    "epochs": 100,
    "max_steps": 100_000,
    "checkpoint_every_epochs": 5,
    "samples_per_epoch": None,
    "lr": 1e-3,
    "seed": 42,
}

DEFAULTS = {
    "prnet": {
        **_COMMON,
        "weight_decay": 1e-8,
        "hidden_sizes": (128,),
        "latent_dim": 64,
        "adaptor_sizes": (128,),
        "condition_dim": 64,
        "noise_dim": 10,
        "dropout": 0.05,
        "loss_type": "gaussian",
        "prediction_samples": 3,
    },
    "chemcpa": {
        **_COMMON,
        "weight_decay": 1e-6,
        "dim": 256,
        "autoencoder_width": 512,
        "autoencoder_depth": 4,
        "adversary_width": 128,
        "adversary_depth": 3,
        "adversary_steps": 3,
        "reg_adversary": 5.0,
        "reg_adversary_cov": 1.0,
        "penalty_adversary": 3.0,
        "dosers_width": 64,
        "dosers_depth": 2,
        "doser_type": "logsigm",
        "embedding_encoder_width": 512,
        "embedding_encoder_depth": 0,
        "adversary_lr": 3e-4,
        "dosers_lr": 1e-3,
        "adversary_wd": 1e-4,
        "dosers_wd": 1e-7,
        "step_size_lr": 45,
    },
    "trainmean": {
        **_COMMON,
        "epochs": 1,
        "max_steps": 1,
        "checkpoint_every_epochs": 1,
    },
    "crisp": {
        **_COMMON,
        "amp_dtype": "bf16",
        "set_size": 24,
        "batch_size": 2,
        "epochs": 50,
        "checkpoint_every_epochs": 5,
        "seed": 0,
        "weight_decay": 1e-7,
        "cell_weight_decay": 1e-3,
        "latent_dim": 128,
        "encoder_width": 256,
        "encoder_depth": 4,
        "decoder_width": 1028,
        "decoder_depth": 4,
        "embedding_encoder_width": 128,
        "embedding_encoder_depth": 4,
        "doser_width": 64,
        "doser_depth": 3,
        "cell_predictor_width": 128,
        "dropout": 0.2,
        "deg_top_k": 50,
        "mse_weight": 0.75,
        "autofocus_weight": 0.25,
        "celltype_weight": 1.0,
        "contrastive_weight": 0.1,
        "mmd_weight": 0.1,
        "kld_normalizer": 500.0,
        "step_size_lr": 50,
        "compile_mode": "reduce-overhead",
    },
    "xpert": {
        **_COMMON,
        "amp_dtype": "bf16",
        "batch_size": 16,
        "epochs": 2_500,
        "lr": 0.004,
        "weight_decay": 1e-5,
        "hidden_size": 256,
        "attention_heads": 8,
        "attention_dropout": 0.1,
        "hidden_dropout": 0.1,
        "cell_input_dropout": 0.1,
        "drug_input_dropout": 0.1,
        "treated_structure": "CA+SA+SA+CA",
        "control_structure": "SA+SA+SA+SA",
        "expression_bins": 10,
        "expression_min": 0.0,
        "expression_max": 10.0,
        "treated_weight": 0.2,
        "control_weight": 0.003,
        "delta_weight": 0.2,
        "correlation_weight": 1.0,
        "initial_epochs": 70,
        "scheduler_milestone": 40,
        "scheduler_factor": 0.5,
    },
    "cmonge": {
        **_COMMON,
        "set_size": 24,
        "batch_size": 1,
        "epochs": 10_000,
        "max_steps": 10_000,
        "checkpoint_every_epochs": 5,
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "ae_lr": 1e-4,
        "ae_weight_decay": 1e-5,
        "ae_epochs": 50,
        "ae_batch_size": 256,
        "ae_width": 512,
        "latent_dim": 50,
        "context_dim": 50,
        "hidden_sizes": (64, 64, 64, 64),
        "fitting_epsilon": 1.0,
        "regularizer_epsilon": 1e-2,
        "monge_gap_weight": 1e-2,
    },
}


def _append_parameter(command: list[str], name: str, value) -> None:
    if value is None:
        return
    command.append(f"--{name.replace('_', '-')}")
    if isinstance(value, (tuple, list)):
        command.extend(str(item) for item in value)
    else:
        command.append(str(value))


def train_baseline(
    paths: DatasetPaths,
    model: str,
    regime: str,
    *,
    split_file=None,
    train_split="train",
    gpus=1,
    num_workers=6,
    run_name=None,
    resume=None,
    populations=None,
    dry_run=False,
    **overrides,
):
    model = str(model).casefold()
    if model not in MODEL_REGISTRY:
        raise ValueError(f"Unknown training model: {model}")
    unknown = sorted(set(overrides) - set(DEFAULTS[model]))
    if unknown:
        raise TypeError(
            f"Unknown {model} training parameters: {', '.join(unknown)}"
        )
    params = {**DEFAULTS[model], **overrides}
    if int(gpus) <= 0 or int(num_workers) < 0:
        raise ValueError("gpus must be positive and num_workers non-negative")
    if model != "trainmean":
        baseline_manifest = paths.prepared / "baselines" / "manifest.json"
        if not baseline_manifest.is_file():
            raise FileNotFoundError(
                "Baseline inputs are absent; run preparation.prepare_baseline_inputs()"
            )
        prepared = json.loads(
            baseline_manifest.read_text(encoding="utf-8")
        ).get("models", {})
        if model not in prepared:
            raise FileNotFoundError(
                f"Inputs for {model} are absent; rerun prepare_baseline_inputs()"
            )
    split_path, split = _resolve_split(paths, regime, split_file)
    if train_split not in split:
        raise KeyError(f"Split set {train_split!r} is absent from {split_path}")
    shapes = json.loads((paths.prepared / "materialized_shapes.json").read_text())
    selected_populations = list(populations or shapes)
    missing = sorted(set(selected_populations) - set(shapes))
    if missing:
        raise ValueError(f"Unknown populations: {', '.join(missing)}")
    split_id = split.get("split_id", split_path.stem)
    base_id = training_identifier(
        regime,
        split_id,
        set_size=params["set_size"],
        batch_size=params["batch_size"],
        gradient_accumulation_steps=1,
        lr=params["lr"],
        seed=params["seed"],
        extra={"model": model, **params},
    )
    run_id = run_name or f"{model}__{base_id}"
    if Path(run_id).name != run_id:
        raise ValueError("run_name must be one directory name")
    output = paths.runs / run_id
    if output.exists() and any(output.iterdir()) and resume is None:
        raise FileExistsError(f"Run directory is not empty: {output}")
    if resume is not None and not Path(resume).is_file():
        raise FileNotFoundError(resume)

    module_command = [
        sys.executable,
        "-m",
        "map.train.baselines.program",
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
        "--train-split",
        train_split,
        "--num-workers",
        str(int(num_workers)),
        "--populations",
        *selected_populations,
    ]
    if resume:
        module_command.extend(["--resume", str(resume)])
    for key, value in params.items():
        _append_parameter(module_command, key, value)
    if model != "trainmean" and int(gpus) > 1:
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
    run_command(command, cwd=Path(__file__).resolve().parents[3], dry_run=dry_run)
    preparation_file = paths.prepared / "preparation_config.json"
    preparation = (
        json.loads(preparation_file.read_text(encoding="utf-8"))
        if preparation_file.is_file()
        else {}
    )
    summary = {
        "model": model,
        "run_id": run_id,
        "run_dir": str(output),
        "regime": regime,
        "split_file": str(split_path),
        "split_id": split_id,
        "train_split": train_split,
        "params": params,
        "gpus": 0 if model == "trainmean" else int(gpus),
        "num_workers": int(num_workers),
        "populations": selected_populations,
        "preparation_id": preparation.get("preparation_id"),
        "dry_run": dry_run,
        "checkpoint": str(output / "last.pt"),
        "resume": str(Path(resume).resolve()) if resume is not None else None,
        "config_id": config_digest({"model": model, **params}),
    }
    return Feedback(paths.runs, f"train_{run_id}").finish(
        summary, [output / "run_config.json", output / "last.pt", output / "checkpoints"]
    )


__all__ = ["DEFAULTS", "train_baseline"]
