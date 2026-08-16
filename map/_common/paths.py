from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .contracts import contract_path, load_contract


@dataclass(frozen=True)
class DatasetPaths:
    """Generic data contract, frozen models and experiment outputs."""

    dataset: str
    source: Path
    workspace: Path
    frozen_models: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", Path(self.source))
        object.__setattr__(self, "workspace", Path(self.workspace))
        object.__setattr__(self, "frozen_models", Path(self.frozen_models))

    @property
    def raw(self) -> Path:
        """Native source root used by the dataset reader."""
        return self.source

    @property
    def prepared(self) -> Path:
        """Dataset-independent materialized data and preparation artifacts."""
        return self.workspace / "materialized"

    @property
    def materialized(self) -> Path:
        return self.prepared

    @property
    def contract(self) -> Path:
        return contract_path(self.workspace)

    @property
    def frozen(self) -> Path:
        """Compatibility alias for read-only pretrained assets."""
        return self.frozen_models

    @property
    def runs(self) -> Path:
        return self.workspace / "runs"

    @property
    def evaluations(self) -> Path:
        return self.workspace / "evaluations"

    @property
    def analysis(self) -> Path:
        return self.evaluations

    def ensure_outputs(self) -> None:
        self.workspace.mkdir(parents=True, exist_ok=True)

    def require_inputs(self) -> None:
        if not self.source.is_dir():
            raise FileNotFoundError(f"Dataset source is missing: {self.source}")


def experiment_paths(
    workspace: str | Path,
    frozen_models: str | Path,
    *,
    source: str | Path | None = None,
    dataset: str = "MAP",
) -> DatasetPaths:
    """Open a MAP project for preparation, training or evaluation."""
    workspace = Path(workspace)
    if contract_path(workspace).is_file():
        contract = load_contract(workspace)
        source = Path(contract["native_source"])
        dataset = str(contract["dataset"])
    elif source is None:
        raise FileNotFoundError(
            f"No data contract under {workspace}; create the project with "
            "preprocess.<dataset>.materialize(project_name=...) first"
        )
    return DatasetPaths(
        dataset=dataset,
        source=Path(source),
        workspace=workspace,
        frozen_models=Path(frozen_models),
    )
