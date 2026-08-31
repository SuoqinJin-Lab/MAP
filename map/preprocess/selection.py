from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .._common.contracts import load_contract, write_contract
from .._common.feedback import Feedback
from .._common.identifiers import config_digest, slug
from .._common.paths import DatasetPaths, experiment_paths


def selection_identifier(
    dataset: str,
    source: str | Path,
    populations: tuple[str, ...] | list[str],
) -> str:
    payload = {
        "dataset": str(dataset),
        "source": str(Path(source).resolve()),
        "populations": list(populations),
    }
    return f"selection-{config_digest(payload, 12)}"


@dataclass(frozen=True)
class DataSelection:
    """A population selection and its project-local preprocessing state."""

    dataset: str
    source: Path
    projects: Path
    frozen_models: Path
    populations: tuple[str, ...]
    cache_workspace: Path
    contract_payload: dict[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", Path(self.source))
        object.__setattr__(self, "projects", Path(self.projects))
        object.__setattr__(self, "frozen_models", Path(self.frozen_models))
        object.__setattr__(self, "cache_workspace", Path(self.cache_workspace))
        object.__setattr__(
            self, "populations", tuple(str(value) for value in self.populations)
        )
        if not self.populations:
            raise ValueError("A data selection must contain at least one population")

    @classmethod
    def open(
        cls,
        cache_workspace: str | Path,
        frozen_models: str | Path,
        *,
        projects: str | Path | None = None,
    ) -> "DataSelection":
        cache_workspace = Path(cache_workspace)
        projects = Path(projects) if projects is not None else cache_workspace.parent
        record = cache_workspace / "preprocess.json"
        if record.is_file():
            document = json.loads(record.read_text(encoding="utf-8"))
            contract = dict(document["contract"])
        else:
            contract = load_contract(cache_workspace)
        return cls(
            dataset=str(contract["dataset"]),
            source=Path(contract["native_source"]),
            projects=Path(projects),
            frozen_models=Path(frozen_models),
            populations=tuple(contract["populations"]),
            cache_workspace=cache_workspace,
            contract_payload=contract,
        )

    @property
    def paths(self) -> DatasetPaths:
        return DatasetPaths(
            dataset=self.dataset,
            source=self.source,
            workspace=self.cache_workspace,
            frozen_models=self.frozen_models,
        )

    @property
    def prepared(self) -> Path:
        return self.cache_workspace / "materialize"

    @staticmethod
    def _project_name(project_name: str) -> str:
        project_name = str(project_name)
        if not project_name or Path(project_name).name != project_name:
            raise ValueError("project_name must be one directory name")
        if slug(project_name, 96) != project_name:
            raise ValueError(
                "project_name may contain lowercase letters, numbers, '.', '_' and '-'"
            )
        return project_name

    def reserve_project(self, project_name: str) -> Path:
        project_name = self._project_name(project_name)
        workspace = self.projects / project_name
        workspace.mkdir(parents=True, exist_ok=True)
        record = workspace / "preprocess.json"
        payload = {
            "format": "map_preprocess_project_v1",
            "project_name": project_name,
            "dataset": self.dataset,
            "selection_cache": str(self.cache_workspace.resolve()),
            "populations": list(self.populations),
            "contract": dict(self.contract_payload),
        }
        if record.is_file():
            existing = json.loads(record.read_text(encoding="utf-8"))
            if existing != payload:
                raise FileExistsError(
                    f"Project name is already bound to another selection: {workspace}"
                )
            return workspace
        if workspace != self.cache_workspace and any(workspace.iterdir()):
            raise FileExistsError(f"Project directory is not empty: {workspace}")
        record.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        Feedback(workspace, "reserve_project").finish(
            {
                "project": project_name,
                "dataset": self.dataset,
                "populations": list(self.populations),
            },
            [record],
        )
        return workspace

    def create_project(self, project_name: str) -> DatasetPaths:
        project_name = self._project_name(project_name)
        workspace = self.reserve_project(project_name)
        destination = workspace / "materialize"
        contract_file = workspace / "contract.json"
        same_materialize_directory = destination.resolve() == self.prepared.resolve()
        if contract_file.is_file():
            existing = load_contract(workspace)
            expected_cache = str(self.cache_workspace.resolve())
            if existing.get("selection_cache") != expected_cache:
                raise FileExistsError(
                f"Project belongs to another selection: {workspace}"
                )
            return experiment_paths(workspace, self.frozen_models)
        if destination.exists() and not same_materialize_directory:
            raise FileExistsError(
                f"Incomplete materialize directory has no contract: {destination}"
            )
        destination.mkdir(parents=True, exist_ok=True)
        if self.prepared.is_dir() and not same_materialize_directory:
            for source_path in self.prepared.iterdir():
                if source_path.name == "_source":
                    continue
                target = destination / source_path.name
                if source_path.is_dir():
                    shutil.copytree(source_path, target, dirs_exist_ok=True)
                else:
                    shutil.copy2(source_path, target)

        contract = dict(self.contract_payload)
        contract["selection_cache"] = str(self.cache_workspace.resolve())
        contract["populations"] = list(self.populations)
        filter_file = destination / "condition_filter.json"
        if filter_file.is_file():
            condition_filter = json.loads(filter_file.read_text(encoding="utf-8"))
            contract["condition_filter"] = {
                key: condition_filter[key] for key in (
                    "filter_id", "min_cells", "max_cells", "seed",
                    "retained_conditions", "retained_condition_cells",
                )
            }
        write_contract(workspace, contract)
        paths = experiment_paths(workspace, self.frozen_models)
        Feedback(workspace, "create_project").finish(
            {
                "project": project_name,
                "dataset": self.dataset,
                "populations": list(self.populations),
                "materialize_cache": str(self.prepared),
            },
            [contract_file, destination],
        )
        return paths


__all__ = ["DataSelection", "selection_identifier"]
