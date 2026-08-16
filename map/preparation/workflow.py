from __future__ import annotations

import json
from pathlib import Path

from .._common.contracts import load_contract
from .._common.feedback import Feedback
from .._common.paths import DatasetPaths, experiment_paths


def create_workflow(
    project_name: str,
    *,
    storage: str | Path = "storage",
) -> DatasetPaths:
    project_name = str(project_name)
    if not project_name or Path(project_name).name != project_name:
        raise ValueError("project_name must be one directory name")
    root = Path(storage)
    workspace = root / "projects" / project_name
    contract = load_contract(workspace)
    materialized = workspace / "materialized"
    required = (
        materialized / "condition_filter.json",
        materialized / "materialized_shapes.json",
        materialized / "conditions.parquet",
        materialized / "preparation_config.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Materialized project is incomplete: " + ", ".join(missing)
        )
    paths = experiment_paths(workspace, root / "frozen_models")
    workflow_file = workspace / "workflow.json"
    payload = {
        "format": "map_workflow_v1",
        "project_name": project_name,
        "dataset": contract["dataset"],
        "contract": "contract.json",
        "materialized": "materialized",
        "runs": "runs",
        "evaluations": "evaluations",
    }
    workflow_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    Feedback(workspace, "create_workflow").finish(
        {
            "project": project_name,
            "dataset": contract["dataset"],
            "materialized": str(materialized),
        },
        [workflow_file],
    )
    return paths


__all__ = ["create_workflow"]
