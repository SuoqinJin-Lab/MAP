from __future__ import annotations

import json
import sys
from pathlib import Path

from ..._common.feedback import Feedback
from ..._common.identifiers import config_digest, training_identifier
from ..._common.paths import DatasetPaths
from ..._common.runner import run_command
from ...preparation.validate import validate_method
from ..splits import resolve_split
from .registry import METHOD_REGISTRY


_COMMON = {
    "set_size": 24,
    "batch_size": 1,
    "epochs": 100,
    "max_steps": 100_000,
    "checkpoint_every_epochs": 5,
    "samples_per_epoch": None,
    "lr": 1e-3,
    # Keep all CUDA method trainers on the same mixed-precision contract.
    # Individual models may still promote numerically sensitive reductions to
    # float32 inside their implementation.
    "amp_dtype": "bf16",
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
        "lr": 1e-3,
        "set_size": 24,
        "batch_size": 8,
        "epochs": 60,
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
        "use_deg_mask": False,
        "deg_mask_mode": "official",
        "drug_representation": "official",
        "control_representation": "validation",
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
        "input_mode": "official",
        "use_gene_position_embedding": False,
        "context_tokens": "dose",
        "attention_padding_mode": "masked",
        "include_cell_context": False,
        "loss_reduction_scale": "sample_count",
        "treated_structure": "CA+SA+SA+CA",
        "control_structure": "SA+SA+SA+SA",
        "expression_bins": 128,
        "expression_min": None,
        "expression_max": None,
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
        "drug_representation": "rdkit",
        "set_size": 24,
        "batch_size": 32,
        "epochs": 120,
        "max_steps": 1_000_000,
        "checkpoint_every_epochs": 10,
        "lr": 4e-4,
        "weight_decay": 1e-5,
        "ae_lr": 1e-4,
        "ae_weight_decay": 1e-6,
        "ae_epochs": 50,
        "ae_batch_size": 256,
        "ae_width": 512,
        "ae_depth": 2,
        "latent_dim": 64,
        "context_dim": 64,
        "hidden_sizes": (256, 256, 256, 256),
        "fitting_epsilon": 2.0,
        "regularizer_epsilon": 5e-2,
        "monge_gap_weight": 5e-3,
        "transport_lr_min_ratio": 1e-2,
        "ae_warmup_steps": 100,
        "ae_end_lr": 1e-5,
    },
}


def _append_parameter(command: list[str], name: str, value) -> None:
    if value is None:
        return
    command.append(f"--{name.replace('_', '-')}")
    if isinstance(value, bool):
        if not value:
            command.pop()
        return
    if isinstance(value, (tuple, list)):
        command.extend(str(item) for item in value)
    else:
        command.append(str(value))


def train_method(
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
    if model not in METHOD_REGISTRY or model == "map":
        raise ValueError(f"Unknown training model: {model}")
    unknown = sorted(set(overrides) - set(DEFAULTS[model]))
    if unknown:
        raise TypeError(
            f"Unknown {model} training parameters: {', '.join(unknown)}"
        )
    params = {**DEFAULTS[model], **overrides}
    if model == "cmonge" and regime == "unseen_combination" and "drug_representation" not in overrides:
        params["drug_representation"] = "moa"
    if int(gpus) <= 0 or int(num_workers) < 0:
        raise ValueError("gpus must be positive and num_workers non-negative")
    split_path, split = resolve_split(paths, regime, split_file)
    if train_split not in split:
        raise KeyError(f"Split set {train_split!r} is absent from {split_path}")
    material_validation = validate_method(
        paths,
        method=model,
        regime=regime,
        split_file=split_path,
        options=params,
        strict=not dry_run,
    )
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
    output = paths.run_dir(split_id, model, run_id)
    if output.exists() and any(output.iterdir()) and resume is None:
        raise FileExistsError(f"Run directory is not empty: {output}")
    if resume is not None and not Path(resume).is_file():
        raise FileNotFoundError(resume)

    module_command = [
        sys.executable,
        "-m",
        "map.train.methods.program",
        "--model",
        model,
        "--data-dir",
        str(paths.prepared),
        "--material-dir",
        str(
            paths.split_material_dir(split_id, "drug_moa")
            if model == "cmonge" and params.get("drug_representation") == "moa"
            else paths.prepared
        ),
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
        "training_material_validation": material_validation.outputs[0],
        "populations": selected_populations,
        "preparation_id": preparation.get("preparation_id"),
        "dry_run": dry_run,
        "checkpoint": str(output / "last.pt"),
        "resume": str(Path(resume).resolve()) if resume is not None else None,
        "config_id": config_digest({"model": model, **params}),
    }
    return Feedback(paths.method_dir(split_id, model), f"train_{run_id}").finish(
        summary, [output / "run_config.json", output / "last.pt", output / "checkpoints"]
    )


__all__ = ["DEFAULTS", "train_method"]
