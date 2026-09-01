from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any


CONTRACT_VERSION = 1
REQUIRED_ENTITIES = ("condition", "control_cell", "condition_cell")


def contract_path(workspace: str | Path) -> Path:
    return Path(workspace) / "contract.json"


def write_contract(workspace: str | Path, payload: dict[str, Any]) -> Path:
    """Persist the dataset-independent input contract for preparation."""
    path = contract_path(workspace)
    document = dict(payload)
    document["format_version"] = CONTRACT_VERSION
    missing = [name for name in REQUIRED_ENTITIES if name not in document.get("entities", {})]
    if missing:
        raise ValueError(f"MAP contract is missing entities: {', '.join(missing)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(document, indent=2), encoding="utf-8")
    temporary.replace(path)
    return path


def load_contract(workspace: str | Path) -> dict[str, Any]:
    path = contract_path(workspace)
    if not path.is_file():
        raise FileNotFoundError(
            f"Project contract is missing: {path}; bind a selected dataset with "
            "preparation.create_project(selection, project_name=...) first"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format_version") != CONTRACT_VERSION:
        raise ValueError(
            f"Unsupported MAP contract version {payload.get('format_version')!r}; "
            f"expected {CONTRACT_VERSION}"
        )
    missing = [name for name in REQUIRED_ENTITIES if name not in payload.get("entities", {})]
    if missing:
        raise ValueError(f"MAP contract is missing entities: {', '.join(missing)}")
    return payload


def run_handler_stage(paths, stage: str, **kwargs):
    """Execute the dataset handler declared by the project contract."""
    try:
        contract = load_contract(paths.workspace)
    except FileNotFoundError:
        preprocess_record = Path(paths.workspace) / "preprocess.json"
        if not preprocess_record.is_file():
            raise
        document = json.loads(preprocess_record.read_text(encoding="utf-8"))
        contract = dict(document["contract"])
    entrypoint = str(contract.get("handler_backend", ""))
    if not entrypoint or "." not in entrypoint:
        raise ValueError(
            f"Invalid handler backend in contract: {entrypoint!r}"
        )
    module = importlib.import_module(entrypoint)
    run_stage = getattr(module, "run_stage", None)
    if not callable(run_stage):
        raise TypeError(
            f"Dataset handler {entrypoint!r} must export run_stage()"
        )
    # A materialization handler can run for hours and may be invoked directly
    # by a dataset adapter.  Seed the canonical project manifest before the
    # handler starts so partial/interrupted runs are still discoverable.
    if stage == "materialize":
        output_dir = Path(paths.prepared)
        manifest = output_dir / "manifest.json"
        if not manifest.is_file():
            output_dir.mkdir(parents=True, exist_ok=True)
            manifest.write_text(
                json.dumps(
                    {
                        "format": "map_materialization_v1",
                        "artifacts": {},
                        "available_artifacts": [
                            "cell_metadata", "state_inputs", "hvg_expression"
                        ],
                        "complete": False,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
    return run_stage(
        stage,
        contract=contract,
        output_dir=paths.prepared,
        **kwargs,
    )
