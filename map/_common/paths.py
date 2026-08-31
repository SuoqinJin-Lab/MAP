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
        """Project-level shared material produced before split creation."""
        return self.workspace / "materialize"

    @property
    def splits(self) -> Path:
        """Project-level immutable split directories."""
        return self.workspace / "splits"

    @property
    def contract(self) -> Path:
        return contract_path(self.workspace)

    @property
    def frozen(self) -> Path:
        """Compatibility alias for read-only pretrained assets."""
        return self.frozen_models

    def _name(self, value: str, label: str) -> str:
        value = str(value).casefold() if label == "method" else str(value)
        if not value or value in {".", ".."} or Path(value).name != value:
            raise ValueError(f"{label} must be one directory name")
        return value

    def split_dir(self, split_id: str) -> Path:
        return self.splits / self._name(split_id, "split_id")

    def split_file(self, split_id: str) -> Path:
        return self.split_dir(split_id) / "split.json"

    def method_dir(self, split_id: str, method: str) -> Path:
        """Method state is owned by exactly one split, never by a project root."""
        return self.split_dir(split_id) / "methods" / self._name(method, "method")

    def material_dir(self, artifact: str | None = None) -> Path:
        """Return the directory for one preparation artifact.

        Artifacts are deliberately named by what they contain (for example
        ``drug_rdkit2d`` or ``graph_assets``), never by the model that happens
        to consume them.
        """
        if artifact is None:
            return self.prepared
        return self.prepared / self._name(artifact, "artifact")

    def split_material_dir(self, split_id: str, artifact: str) -> Path:
        """Return one split-dependent artifact directory."""
        return self.split_dir(split_id) / "materialize" / self._name(artifact, "artifact")

    def validation_path(self, split_id: str, method: str) -> Path:
        return self.method_dir(split_id, method) / "validation.json"

    def run_dir(self, split_id: str, method: str, run_name: str) -> Path:
        return self.method_dir(split_id, method) / "runs" / self._name(run_name, "run_name")

    def evaluation_dir(
        self, split_id: str, method: str, run_name: str, evaluation_name: str
    ) -> Path:
        return self.run_dir(split_id, method, run_name) / "evaluations" / self._name(
            evaluation_name, "evaluation_name"
        )

    def find_run_dir(self, split_id: str, method: str, run_name: str) -> Path:
        candidate = self.run_dir(split_id, method, run_name)
        if candidate.is_dir():
            return candidate
        raise FileNotFoundError(f"Run directory is missing: {candidate}")

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
            f"No data contract under {workspace}; create it with "
            "preparation.create_project(selection, project_name=...) first"
        )
    return DatasetPaths(
        dataset=dataset,
        source=Path(source),
        workspace=workspace,
        frozen_models=Path(frozen_models),
    )
