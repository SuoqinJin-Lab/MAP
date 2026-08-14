from __future__ import annotations

from pathlib import Path

from ...._common.paths import DatasetPaths


class TahoePaths(DatasetPaths):
    """Tahoe-100M native layout, private to its preparation reader."""

    @property
    def raw_data(self) -> Path:
        return self.source / "data"

    @property
    def raw_metadata(self) -> Path:
        return self.source / "metadata"

    @property
    def staging(self) -> Path:
        return self.workspace

    @property
    def dataset_cache(self) -> Path:
        return self.source

    @property
    def data_summary(self) -> Path:
        return self.dataset_cache / "data_summary.json"

    @property
    def cell_line_summary(self) -> Path:
        return self.dataset_cache / "cell_lines.parquet"

    @property
    def all_condition_summary(self) -> Path:
        return self.dataset_cache / "conditions.parquet"

    @property
    def selection(self) -> Path:
        return self.staging / "selection.json"

    @property
    def condition_summary(self) -> Path:
        return self.staging / "conditions.parquet"


__all__ = ["TahoePaths"]
