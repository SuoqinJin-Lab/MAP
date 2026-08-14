from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..._common.feedback import Feedback
from ..selection import DataSelection, selection_identifier


def _require_anndata():
    try:
        import anndata as ad
    except ImportError as exc:
        raise RuntimeError(
            "AnnData sources require the optional 'anndata' package"
        ) from exc
    return ad


def _source_file(source: Path) -> Path:
    if source.is_file():
        return source
    files = sorted(source.glob("*.h5ad")) if source.is_dir() else []
    if len(files) != 1:
        raise FileNotFoundError(
            f"Expected one .h5ad file at {source}; found {len(files)}"
        )
    return files[0]


def _resolve_column(columns: Iterable[str], requested: str | None, aliases: Sequence[str], label: str) -> str:
    available = set(str(value) for value in columns)
    candidates = (requested,) if requested else aliases
    for candidate in candidates:
        if candidate and candidate in available:
            return candidate
    raise ValueError(
        f"Cannot resolve {label}; expected one of {', '.join(str(value) for value in candidates if value)}"
    )


def _optional_column(columns: Iterable[str], requested: str | None, aliases: Sequence[str]) -> str | None:
    available = set(str(value) for value in columns)
    candidates = (requested,) if requested else aliases
    return next((value for value in candidates if value and value in available), None)


def _number(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
        return numeric if math.isfinite(numeric) else default
    except (TypeError, ValueError):
        return default


class AnnDataSource:
    """Map perturbational AnnData datasets into MAP's three-entity contract."""

    POPULATION_ALIASES = ("cell_line", "cell_type", "cell_line_id", "celltype")
    PERTURBATION_ALIASES = ("product_name", "drug", "perturbation", "condition")
    DOSE_ALIASES = ("dose", "dose_value", "concentration")
    SMILES_ALIASES = ("canonical_smiles", "SMILES", "smiles")
    GROUP_ALIASES = ("plate", "batch", "batch_id", "replicate")
    GENE_ALIASES = ("gene_symbol", "gene_name", "symbol")

    def __init__(
        self,
        source: str | Path,
        projects: str | Path,
        *,
        frozen_models: str | Path,
        dataset: str,
        population_key: str | None = None,
        perturbation_key: str | None = None,
        dose_key: str | None = None,
        smiles_key: str | None = None,
        group_key: str | None = None,
        gene_symbol_key: str | None = None,
        control_values: Sequence[str] = ("vehicle", "dmso", "control", "untreated"),
        dose_unit: str = "uM",
        smiles_map: Mapping[str, str] | None = None,
    ) -> None:
        self.source = Path(source)
        self.projects = Path(projects)
        self.frozen_models = Path(frozen_models)
        self.dataset = dataset
        self.requested_columns = {
            "population": population_key,
            "perturbation": perturbation_key,
            "dose": dose_key,
            "smiles": smiles_key,
            "group": group_key,
            "gene_symbol": gene_symbol_key,
        }
        self.control_values = tuple(str(value).casefold() for value in control_values)
        self.dose_unit = str(dose_unit)
        self.smiles_map = {str(key): str(value) for key, value in (smiles_map or {}).items()}

    @property
    def dataset_cache(self) -> Path:
        return self.source if self.source.is_dir() else self.source.parent

    def _inspect(self):
        ad = _require_anndata()
        source_file = _source_file(self.source)
        data = ad.read_h5ad(source_file, backed="r")
        columns = data.obs.columns
        resolved = {
            "population": _optional_column(columns, self.requested_columns["population"], self.POPULATION_ALIASES),
            "perturbation": _resolve_column(columns, self.requested_columns["perturbation"], self.PERTURBATION_ALIASES, "perturbation column"),
            "dose": _optional_column(columns, self.requested_columns["dose"], self.DOSE_ALIASES),
            "smiles": _optional_column(columns, self.requested_columns["smiles"], self.SMILES_ALIASES),
            "group": _optional_column(columns, self.requested_columns["group"], self.GROUP_ALIASES),
            "gene_symbol": _optional_column(data.var.columns, self.requested_columns["gene_symbol"], self.GENE_ALIASES),
        }
        return source_file, data, resolved

    @staticmethod
    def _populations(obs, column: str | None, size: int) -> list[str]:
        if column is None:
            return ["__ALL__"] * int(size)
        return [str(value) for value in obs[column]]

    def watch_data(self, *, refresh: bool = False) -> dict:
        summary_file = self.dataset_cache / "data_summary.json"
        if summary_file.is_file() and not refresh:
            return json.loads(summary_file.read_text(encoding="utf-8"))
        source_file, data, columns = self._inspect()
        obs = data.obs
        population_values = self._populations(obs, columns["population"], data.n_obs)
        populations = Counter(population_values)
        perturbations = [str(value) for value in obs[columns["perturbation"]]]
        control_mask = np.asarray([value.casefold() in self.control_values for value in perturbations])
        doses = (
            np.asarray([_number(value) for value in obs[columns["dose"]]], dtype=np.float64)
            if columns["dose"] else np.zeros(data.n_obs, dtype=np.float64)
        )
        noncontrol = np.flatnonzero(~control_mask)
        conditions = {
            (
                population_values[index],
                perturbations[index],
                float(doses[index]),
            )
            for index in noncontrol
        }
        summary = {
            "dataset": self.dataset,
            "source_file": str(source_file.resolve()),
            "cells": int(data.n_obs),
            "genes": int(data.n_vars),
            "populations": len(populations),
            "control_cells": int(control_mask.sum()),
            "condition_cells": int((~control_mask).sum()),
            "conditions": len(conditions),
            "drugs": len({perturbations[index] for index in noncontrol}),
            "dose_min": float(doses[noncontrol].min()) if noncontrol.size else None,
            "dose_median": float(np.median(doses[noncontrol])) if noncontrol.size else None,
            "dose_max": float(doses[noncontrol].max()) if noncontrol.size else None,
            "cells_by_population": dict(sorted(populations.items())),
            "columns": columns,
        }
        self.dataset_cache.mkdir(parents=True, exist_ok=True)
        summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        Feedback(self.dataset_cache, "watch_data").finish(summary, [summary_file])
        if getattr(data, "file", None) is not None:
            data.file.close()
        return summary

    def _selection_workspace(
        self, populations: Sequence[str], project_name: str | None = None
    ) -> Path:
        selection_id = (
            DataSelection._project_name(project_name)
            if project_name is not None
            else selection_identifier(
                self.dataset, self.source, tuple(str(value) for value in populations)
            )
        )
        return self.projects / selection_id

    def fetch_cell_line(
        self, cell_lines: Sequence[str], *, project_name: str | None = None
    ) -> DataSelection:
        summary = self.watch_data()
        requested = tuple(dict.fromkeys(str(value) for value in cell_lines))
        if not requested:
            raise ValueError("At least one population is required")
        available = summary["cells_by_population"]
        missing = sorted(set(requested) - set(available))
        if missing:
            raise ValueError(f"Populations are absent from {self.dataset}: {', '.join(missing)}")
        payload = {
            "populations": list(requested),
            "cells_by_population": {value: int(available[value]) for value in requested},
            "cells": int(sum(available[value] for value in requested)),
        }
        workspace = self._selection_workspace(requested, project_name=project_name)
        path = workspace / "selection.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        Feedback(path.parent, "fetch_cell_line").finish(payload, [path])
        return self._selection(payload["populations"], path, workspace)

    def prepare(self) -> DataSelection:
        """Select every population in the source."""
        summary = self.watch_data()
        return self.fetch_cell_line(list(summary["cells_by_population"]))

    def _selection(
        self,
        populations: Sequence[str],
        selection_file: Path,
        cache_workspace: Path,
    ) -> DataSelection:
        summary = self.watch_data()
        populations = list(populations)
        entities = {
            "condition": {"representation": "logical_view", "fields": ["condition_id", "population", "drug", "dose", "dose_unit", "canonical_smiles"]},
            "control_cell": {"representation": "logical_view", "fields": ["gene_ids", "expression", "population", "matching_group"]},
            "condition_cell": {"representation": "logical_view", "fields": ["gene_ids", "expression", "condition_id", "population", "matching_group"]},
        }
        contract = {
            "dataset": self.dataset,
            "source_format": "anndata",
            "preparation_backend": "map.preprocess.sources.anndata_backend",
            "native_source": str(_source_file(self.source).resolve()),
            "populations": populations,
            "entities": entities,
            "source_schema": {
                "columns": summary["columns"],
                "control_values": list(self.control_values),
                "dose_unit": self.dose_unit,
                "smiles_map": self.smiles_map,
            },
            "selection": str(selection_file.resolve()),
            "canonical_source": str((cache_workspace / "materialized" / "_source").resolve()),
            "counts": {"selected_cells": int(sum(summary["cells_by_population"][value] for value in populations))},
        }
        return DataSelection(
            dataset=self.dataset,
            source=self.source,
            projects=self.projects,
            frozen_models=self.frozen_models,
            populations=tuple(populations),
            cache_workspace=cache_workspace,
            contract_payload=contract,
        )


def anndata(source, projects, *, frozen_models, dataset="AnnData", **kwargs):
    return AnnDataSource(
        source, projects, frozen_models=frozen_models, dataset=dataset, **kwargs
    )


def sciplex(source, projects, *, frozen_models, dataset="SciPlex3", **kwargs):
    kwargs.setdefault("control_values", ("vehicle", "dmso", "control"))
    return anndata(
        source,
        projects,
        frozen_models=frozen_models,
        dataset=dataset,
        **kwargs,
    )


def combosciplex(source, projects, *, frozen_models, dataset="ComboSciPlex", **kwargs):
    kwargs.setdefault("control_values", ("vehicle", "dmso", "control"))
    return anndata(
        source,
        projects,
        frozen_models=frozen_models,
        dataset=dataset,
        **kwargs,
    )


def nips(source, projects, *, frozen_models, dataset="NIPS", **kwargs):
    return anndata(
        source, projects, frozen_models=frozen_models, dataset=dataset, **kwargs
    )


def op3(source, projects, *, frozen_models, dataset="OP3", **kwargs):
    return anndata(
        source, projects, frozen_models=frozen_models, dataset=dataset, **kwargs
    )


__all__ = [
    "AnnDataSource", "anndata", "combosciplex", "nips", "op3", "sciplex",
]
