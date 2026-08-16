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
            f"Materialized project contract is missing: {path}; run the dataset "
            "preprocessor's materialize(project_name=...) step first"
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


def run_source_stage(paths, stage: str, **kwargs):
    """Execute the dataset reader declared by a preparation contract."""
    try:
        contract = load_contract(paths.workspace)
    except FileNotFoundError:
        preprocess_record = Path(paths.workspace) / "preprocess.json"
        if not preprocess_record.is_file():
            raise
        document = json.loads(preprocess_record.read_text(encoding="utf-8"))
        contract = dict(document["contract"])
    entrypoint = str(contract.get("preparation_backend", ""))
    if not entrypoint.startswith("map.preprocess.sources."):
        raise ValueError(f"Invalid preparation backend in contract: {entrypoint!r}")
    module = importlib.import_module(entrypoint)
    return module.run_stage(
        stage,
        contract=contract,
        output_dir=paths.prepared,
        **kwargs,
    )
