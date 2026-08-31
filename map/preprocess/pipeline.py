"""Dataset handler protocol and the dataset-independent preprocessing facade.

Dataset packages only need to expose a handler with ``watch_data`` and a
population selection method.  All downstream work is implemented once in
``PreprocessPipeline`` and consumes the MAP contract emitted by that handler.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, Sequence, runtime_checkable

from .core import filter_conditions, select_hvg
from .selection import DataSelection


@runtime_checkable
class DatasetHandler(Protocol):
    """Minimal interface required from a dataset-specific reader."""

    dataset: str
    source: Path

    def watch_data(self, **kwargs: Any) -> dict: ...

    def fetch_populations(
        self, populations: Sequence[str], *, project_name: str, **kwargs: Any
    ) -> DataSelection: ...


class PreprocessPipeline:
    """Run the common preprocessing stages through one dataset handler."""

    def __init__(self, handler: DatasetHandler):
        if not hasattr(handler, "watch_data") or not (
            hasattr(handler, "fetch_populations") or hasattr(handler, "fetch_cell_line")
        ):
            missing = []
            if not hasattr(handler, "watch_data"):
                missing.append("watch_data")
            if not (
                hasattr(handler, "fetch_populations")
                or hasattr(handler, "fetch_cell_line")
            ):
                missing.append("fetch_populations/fetch_cell_line")
            raise TypeError(
                "A dataset handler must provide watch_data() and "
                f"fetch_populations(); missing: {', '.join(missing)}"
            )
        self.handler = handler

    @property
    def dataset(self) -> str:
        return str(self.handler.dataset)

    def watch_data(self, **kwargs: Any) -> dict:
        return self.handler.watch_data(**kwargs)

    def fetch_populations(
        self, populations: Sequence[str], *, project_name: str, **kwargs: Any
    ) -> DataSelection:
        method = getattr(self.handler, "fetch_populations", None)
        if method is None:
            method = getattr(self.handler, "fetch_cell_line")
        return method(populations, project_name=project_name, **kwargs)

    def fetch_cell_line(
        self, populations: Sequence[str], *, project_name: str, **kwargs: Any
    ) -> DataSelection:
        """Population-neutral alias used by cell-line based datasets."""
        return self.fetch_populations(
            populations, project_name=project_name, **kwargs
        )

    def open_project(self, project_name: str) -> DataSelection:
        method = getattr(self.handler, "open_project", None)
        if method is not None:
            return method(project_name)
        try:
            projects = Path(self.handler.projects)
            frozen_models = Path(self.handler.frozen_models)
        except AttributeError as error:
            raise TypeError(
                "A reloadable handler must expose projects and frozen_models "
                "or implement open_project()"
            ) from error
        return DataSelection.open(
            projects / DataSelection._project_name(project_name),
            frozen_models,
            projects=projects,
        )

    def filter_conditions(self, selection: DataSelection, **kwargs: Any):
        return filter_conditions(selection, **kwargs)

    def select_hvg(self, selection: DataSelection, **kwargs: Any):
        return select_hvg(selection, **kwargs)


def pipeline(handler: DatasetHandler) -> PreprocessPipeline:
    """Create the common pipeline for a packaged dataset handler."""
    return PreprocessPipeline(handler)


__all__ = ["DatasetHandler", "PreprocessPipeline", "pipeline"]
