from __future__ import annotations

from pathlib import Path
from typing import Sequence

from ...selection import DataSelection, selection_identifier
from .data import fetch_cell_lines, tahoe_contract, watch_data
from .paths import TahoePaths


class TahoeHandler:
    """Packaged Tahoe handler for the common preprocessing pipeline."""

    def __init__(
        self,
        source: str | Path,
        projects: str | Path,
        *,
        frozen_models: str | Path,
        dataset: str = "Tahoe-100M",
    ) -> None:
        self.source = Path(source)
        self.projects = Path(projects)
        self.frozen_models = Path(frozen_models)
        self.dataset = dataset

    def _selection_paths(
        self, cell_lines: Sequence[str], project_name: str | None = None
    ) -> TahoePaths:
        selection_id = (
            DataSelection._project_name(project_name)
            if project_name is not None
            else selection_identifier(
                self.dataset, self.source, tuple(str(value) for value in cell_lines)
            )
        )
        return TahoePaths(
            dataset=self.dataset,
            source=self.source,
            workspace=self.projects / selection_id,
            frozen_models=self.frozen_models,
        )

    def watch_data(self, *, batch_size: int = 8192, refresh: bool = False):
        return watch_data(
            self.source,
            self.projects,
            batch_size=batch_size,
            refresh=refresh,
            dataset=self.dataset,
        )

    def fetch_cell_line(
        self,
        cell_lines: Sequence[str],
        *,
        project_name: str | None = None,
        batch_size: int = 8192,
    ) -> DataSelection:
        requested = tuple(dict.fromkeys(str(value) for value in cell_lines))
        paths = self._selection_paths(requested, project_name=project_name)
        fetch_cell_lines(paths, cell_lines=requested, batch_size=batch_size)
        contract = tahoe_contract(paths)
        selection = DataSelection(
            dataset=self.dataset,
            source=self.source,
            projects=self.projects,
            frozen_models=self.frozen_models,
            populations=requested,
            cache_workspace=paths.staging,
            contract_payload=contract,
        )
        selection.reserve_project(paths.staging.name)
        return selection

    def fetch_populations(
        self,
        populations: Sequence[str],
        *,
        project_name: str,
        batch_size: int = 8192,
    ) -> DataSelection:
        return self.fetch_cell_line(
            populations,
            project_name=project_name,
            batch_size=batch_size,
        )

    def prepare(self, *, batch_size: int = 8192) -> DataSelection:
        summary = self.watch_data(batch_size=batch_size)
        return self.fetch_cell_line(
            tuple(summary["cells_by_cell_line"]),
            batch_size=batch_size,
        )


def tahoe(source, projects, *, frozen_models, dataset="Tahoe-100M"):
    return TahoeHandler(
        source, projects, frozen_models=frozen_models, dataset=dataset
    )


__all__ = ["TahoeHandler", "tahoe"]
