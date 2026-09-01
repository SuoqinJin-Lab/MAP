"""Shared, dataset-independent preparation artifacts.

Each public function writes one neutral artifact.  Directory names describe
the data on disk; consumers are metadata and never control the layout.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .._common.contracts import run_handler_stage
from .._common.feedback import Feedback
from .._common.hvg import load_hvg_contract
from .._common.identifiers import preparation_identifier
from .._common.paths import DatasetPaths
from ..preprocess.selection import DataSelection


SHARED_ARTIFACTS = {
    "cell_metadata": {
        "backend_artifact": "groups",
        "files": (
            "conditions.parquet",
            "<population>/row_condition.int32.dat",
            "<population>/row_group.uint16.dat",
        ),
        "consumers": (
            "MAP", "PRnet", "chemCPA", "TrainMean", "CRISP", "XPert",
            "CMonge", "evaluation",
        ),
    },
    "state_inputs": {
        "backend_artifact": "state",
        "files": (
            "<population>/se_gene_ids.uint16.dat",
            "<population>/se_expr.float16.dat",
        ),
        "consumers": (
            "MAP", "XPert-official", "SE-600M condition embedding cache",
        ),
    },
    "hvg_expression": {
        "backend_artifact": "hvg",
        "files": ("<population>/hvg.float16.dat",),
        "consumers": (
            "MAP", "PRnet", "chemCPA", "TrainMean", "CRISP", "XPert",
            "CMonge", "evaluation",
        ),
    },
}


def create_project(selection: DataSelection, *, project_name: str) -> DatasetPaths:
    """Bind a selected native dataset to a reusable MAP project contract."""
    return selection.create_project(project_name)


def _manifest_path(paths: DatasetPaths) -> Path:
    return paths.prepared / "materialization_manifest.json"


def _load_manifest(paths: DatasetPaths) -> dict[str, Any]:
    path = _manifest_path(paths)
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"format": "map_materialization_v1", "artifacts": {}}


def artifact_files(paths: DatasetPaths, name: str) -> list[Path]:
    """Resolve the concrete files represented by one shared artifact."""
    if name not in SHARED_ARTIFACTS:
        raise ValueError(f"Unknown shared artifact: {name}")
    shapes_path = paths.prepared / "materialized_shapes.json"
    if not shapes_path.is_file():
        return []
    populations = json.loads(shapes_path.read_text(encoding="utf-8"))
    output: list[Path] = []
    for template in SHARED_ARTIFACTS[name]["files"]:
        if "<population>" in template:
            output.extend(
                paths.prepared / template.replace("<population>", population)
                for population in populations
            )
        else:
            output.append(paths.prepared / template)
    return output


def validate_shared_artifacts(paths: DatasetPaths) -> dict[str, Any]:
    """Require all declared shared artifacts and their concrete files."""
    manifest_path = _manifest_path(paths)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Shared-artifact manifest is missing: {manifest_path}")
    manifest = _load_manifest(paths)
    hvg = load_hvg_contract(paths.prepared)
    pending = sorted(set(SHARED_ARTIFACTS) - set(manifest["artifacts"]))
    if not manifest.get("complete") or pending:
        raise RuntimeError(
            "Shared artifacts are incomplete; pending artifacts: "
            + ", ".join(pending)
        )
    missing = [
        str(path)
        for name in SHARED_ARTIFACTS
        for path in artifact_files(paths, name)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Shared artifact files are missing: " + ", ".join(missing)
        )
    recorded = manifest.get("hvg_fingerprint")
    hvg_expression = manifest["artifacts"].get("hvg_expression", {})
    if (
        recorded != hvg["fingerprint"]
        or hvg_expression.get("hvg_fingerprint") != hvg["fingerprint"]
    ):
        raise RuntimeError(
            "Materialized HVG expression belongs to another HVG contract; "
            "regenerate HVG-dependent material for this project."
        )
    return manifest


def _update_preparation_config(
    paths: DatasetPaths,
    manifest: dict[str, Any],
    *,
    target_sum: float,
    pad_length: int,
) -> Path:
    shapes_path = paths.prepared / "materialized_shapes.json"
    shapes = json.loads(shapes_path.read_text(encoding="utf-8"))
    filter_payload = json.loads(
        (paths.prepared / "condition_filter.json").read_text(encoding="utf-8")
    )
    hvg = load_hvg_contract(paths.prepared)
    hvg_dim = int(next(iter(shapes.values()))["hvg_dim"])
    complete = all(name in manifest["artifacts"] for name in SHARED_ARTIFACTS)
    payload = {
        "preparation_id": preparation_identifier(
            populations=list(shapes),
            n_top_genes=hvg_dim,
            pad_length=pad_length,
            target_sum=target_sum,
            log1p=True,
            condition_filter_id=filter_payload["filter_id"],
            hvg_fingerprint=hvg["fingerprint"],
        ),
        "populations": list(shapes),
        "n_top_genes": hvg_dim,
        "pad_length": int(pad_length),
        "target_sum": float(target_sum),
        "log1p": True,
        "dtype": "float16",
        "gene_vocabulary": "STATE_ESM2",
        "hvg_protocol": hvg["protocol"],
        "hvg_fingerprint": hvg["fingerprint"],
        "condition_filter": {
            key: filter_payload[key]
            for key in (
                "filter_id", "min_cells", "max_cells", "seed",
                "retained_conditions", "retained_condition_cells",
            )
        },
        "materialization_manifest": str(_manifest_path(paths).resolve()),
        "materialization_complete": complete,
    }
    target = paths.prepared / "preparation_config.json"
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target


def _prepare_artifact(
    paths: DatasetPaths,
    name: str,
    *,
    workers: int = 8,
    target_sum: float = 10_000,
    pad_length: int = 2_048,
    overwrite: bool = False,
):
    if name not in SHARED_ARTIFACTS:
        raise ValueError(f"Unknown shared artifact: {name}")
    if workers <= 0 or target_sum <= 0 or pad_length < 2:
        raise ValueError("workers/target_sum must be positive and pad_length >= 2")
    for required in ("condition_filter.json", "hvg.json", "stats_manifest.json"):
        if not (paths.prepared / required).is_file():
            raise FileNotFoundError(
                f"{required} is missing; complete common preprocessing first"
            )
    hvg = load_hvg_contract(paths.prepared)
    manifest = _load_manifest(paths)
    previous = manifest["artifacts"].get(name)
    configuration = {
        "target_sum": float(target_sum),
        "pad_length": int(pad_length),
    }
    existing_configuration = manifest.get("configuration")
    if existing_configuration is not None and existing_configuration != configuration:
        raise FileExistsError(
            "Shared artifacts use another target_sum/pad_length; use a new project"
        )
    if previous and not overwrite:
        if previous.get("configuration") != configuration:
            raise FileExistsError(
                f"{name} exists with another configuration; use a new project"
            )
        if name == "hvg_expression" and previous.get("hvg_fingerprint") != hvg[
            "fingerprint"
        ]:
            raise RuntimeError(
                "hvg_expression belongs to another HVG contract; regenerate it "
                "with overwrite=True or use a new project"
            )
        return Feedback(paths.prepared, f"prepare_{name}").finish(
            {**previous, "reused": True}, [_manifest_path(paths)]
        )
    specification = SHARED_ARTIFACTS[name]
    run_handler_stage(
        paths,
        "materialize",
        workers=int(workers),
        target_sum=float(target_sum),
        num_gene_tokens=int(pad_length) - 1,
        artifact=specification["backend_artifact"],
    )
    shapes = json.loads(
        (paths.prepared / "materialized_shapes.json").read_text(encoding="utf-8")
    )
    artifact = {
        "name": name,
        "configuration": configuration,
        "files": list(specification["files"]),
        "consumers": list(specification["consumers"]),
        "populations": list(shapes),
        "cells": sum(int(value["n_cells"]) for value in shapes.values()),
    }
    if name == "hvg_expression":
        artifact["hvg_fingerprint"] = hvg["fingerprint"]
    manifest["artifacts"][name] = artifact
    if name == "hvg_expression":
        manifest["hvg_fingerprint"] = hvg["fingerprint"]
    manifest["configuration"] = configuration
    manifest["complete"] = all(
        value in manifest["artifacts"] for value in SHARED_ARTIFACTS
    )
    manifest["available_artifacts"] = list(SHARED_ARTIFACTS)
    manifest_path = _manifest_path(paths)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    config_path = _update_preparation_config(
        paths, manifest, target_sum=target_sum, pad_length=pad_length
    )
    return Feedback(paths.prepared, f"prepare_{name}").finish(
        {**artifact, "reused": False},
        [paths.prepared / "materialized_shapes.json", manifest_path, config_path],
    )


def prepare_cell_metadata(
    paths: DatasetPaths,
    *,
    workers: int = 8,
    target_sum: float = 10_000,
    pad_length: int = 2_048,
    overwrite: bool = False,
):
    """Write the shared cell/condition metadata artifact."""
    return _prepare_artifact(
        paths, "cell_metadata", workers=workers, target_sum=target_sum,
        pad_length=pad_length, overwrite=overwrite,
    )


def prepare_state_inputs(
    paths: DatasetPaths,
    *,
    workers: int = 8,
    target_sum: float = 10_000,
    pad_length: int = 2_048,
    overwrite: bool = False,
):
    """Write the STATE token-id and expression artifact."""
    return _prepare_artifact(
        paths, "state_inputs", workers=workers, target_sum=target_sum,
        pad_length=pad_length, overwrite=overwrite,
    )


def prepare_hvg_expression(
    paths: DatasetPaths,
    *,
    workers: int = 8,
    target_sum: float = 10_000,
    pad_length: int = 2_048,
    overwrite: bool = False,
):
    """Write the shared HVG expression artifact."""
    return _prepare_artifact(
        paths, "hvg_expression", workers=workers, target_sum=target_sum,
        pad_length=pad_length, overwrite=overwrite,
    )


__all__ = [
    "SHARED_ARTIFACTS",
    "create_project",
    "artifact_files",
    "prepare_cell_metadata",
    "prepare_hvg_expression",
    "prepare_state_inputs",
    "validate_shared_artifacts",
]
