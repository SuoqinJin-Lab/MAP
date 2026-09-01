from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ...._common.components import combination_key
from ...._common.feedback import Feedback
from ...selection import DataSelection
from ..anndata import (
    AnnDataHandler,
    _number,
    _optional_column,
    _resolve_column,
    _source_file,
)


def _text(value: Any, fallback: str = "") -> str:
    if value is None:
        return fallback
    text = " ".join(str(value).split())
    return fallback if text.casefold() in {"", "nan", "none", "null"} else text


class ComboSciPlexHandler(AnnDataHandler):
    """ComboSciPlex-specific AnnData adapter.

    ComboSciPlex stores two component names in ``Drug1``/``Drug2``.  A
    DMSO+drug row is a singleton (the single-drug training case), while
    DMSO+DMSO is the control.  The generic AnnData handler remains scalar.
    """

    def __init__(
        self,
        source: str | Path,
        projects: str | Path,
        *,
        frozen_models: str | Path,
        dataset: str = "ComboSciPlex",
        component_keys: Sequence[str] = ("Drug1", "Drug2"),
        component_dose_keys: Sequence[str] = (),
        component_smiles_keys: Sequence[str] = (),
        default_dose: float = 1.0,
        **kwargs: Any,
    ) -> None:
        self.component_keys = tuple(component_keys)
        self.component_dose_keys = tuple(component_dose_keys)
        self.component_smiles_keys = tuple(component_smiles_keys)
        self.default_dose = float(default_dose)
        super().__init__(
            source,
            projects,
            frozen_models=frozen_models,
            dataset=dataset,
            perturbation_key=None,
            dose_key=None,
            smiles_key=None,
            **kwargs,
        )

    def _inspect(self):
        ad = self._require()
        source_file = _source_file(self.source)
        data = ad.read_h5ad(source_file, backed="r")
        obs_columns = data.obs.columns
        resolved = {
            "perturbation": None,
            "dose": None,
            "smiles": None,
            "population": _optional_column(obs_columns, self.requested_columns["population"], self.POPULATION_ALIASES),
            "group": _optional_column(obs_columns, self.requested_columns["group"], self.GROUP_ALIASES),
            "gene_symbol": _optional_column(data.var.columns, self.requested_columns["gene_symbol"], self.GENE_ALIASES),
            "component_keys": tuple(_resolve_column(obs_columns, key, (), "component") for key in self.component_keys),
            "component_dose_keys": tuple(_resolve_column(obs_columns, key, (), "component dose") for key in self.component_dose_keys),
            "component_smiles_keys": tuple(_resolve_column(obs_columns, key, (), "component SMILES") for key in self.component_smiles_keys),
        }
        return source_file, data, resolved

    @staticmethod
    def _require():
        from ..anndata import _require_anndata
        return _require_anndata()

    def _row_components(self, row: Mapping[str, Any], columns: Mapping[str, Any]):
        names: list[str] = []
        doses: list[float] = []
        smiles: list[str] = []
        for index, key in enumerate(columns["component_keys"]):
            name = _text(row.get(key))
            if not name or name.casefold() in self.control_values:
                continue
            names.append(name)
            dose_key = columns["component_dose_keys"][index] if index < len(columns["component_dose_keys"]) else None
            smiles_key = columns["component_smiles_keys"][index] if index < len(columns["component_smiles_keys"]) else None
            doses.append(_number(row.get(dose_key), self.default_dose) if dose_key else self.default_dose)
            smiles.append(_text(row.get(smiles_key), self.smiles_map.get(name, "")) if smiles_key else self.smiles_map.get(name, ""))
        return names, doses, smiles

    def watch_data(self, *, refresh: bool = False) -> dict:
        summary_file = self.dataset_cache / "data_summary.json"
        if summary_file.is_file() and not refresh:
            cached = json.loads(summary_file.read_text(encoding="utf-8"))
            cached_columns = cached.get("columns", {})
            if tuple(cached_columns.get("component_keys", ())) == self.component_keys:
                return cached
        source_file, data, columns = self._inspect()
        obs = data.obs
        populations = self._populations(obs, columns["population"], data.n_obs)
        population_counts = Counter(populations)
        condition_keys = set()
        control = 0
        for population, (_, row) in zip(populations, obs.iterrows()):
            names, doses, smiles = self._row_components(row, columns)
            if not names:
                control += 1
                continue
            if any(not value for value in smiles):
                raise ValueError(f"Missing SMILES for ComboSciPlex components {names!r}")
            condition_keys.add((population, combination_key(smiles), tuple(doses)))
        summary = {
            "dataset": self.dataset,
            "source_file": str(source_file.resolve()),
            "cells": int(data.n_obs),
            "genes": int(data.n_vars),
            "populations": len(population_counts),
            "control_cells": int(control),
            "condition_cells": int(data.n_obs - control),
            "conditions": len(condition_keys),
            "drugs": len({key[1] for key in condition_keys}),
            "dose_min": float(min((dose for _, _, doses in condition_keys for dose in doses), default=self.default_dose)),
            "dose_median": float(np.median([dose for _, _, doses in condition_keys for dose in doses])) if condition_keys else None,
            "dose_max": float(max((dose for _, _, doses in condition_keys for dose in doses), default=self.default_dose)),
            "cells_by_population": dict(sorted(population_counts.items())),
            "columns": columns,
            "dose_unit": self.dose_unit,
            "default_dose": self.default_dose,
        }
        self.dataset_cache.mkdir(parents=True, exist_ok=True)
        summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        Feedback(self.dataset_cache, "watch_data").finish(summary, [summary_file])
        if getattr(data, "file", None) is not None:
            data.file.close()
        return summary

    def _selection(self, populations: Sequence[str], selection_file: Path, cache_workspace: Path) -> DataSelection:
        summary = self.watch_data()
        populations = list(populations)
        contract = {
            "dataset": self.dataset,
            "source_format": "anndata",
            "handler_backend": "map.preprocess.handlers.combosciplex.backend",
            "native_source": str(_source_file(self.source).resolve()),
            "populations": populations,
            "entities": {
                "condition": {"representation": "logical_view", "fields": ["condition_id", "population", "drug", "dose", "dose_unit", "canonical_smiles", "component_names", "component_smiles", "component_doses_uM", "combination_key"]},
                "control_cell": {"representation": "logical_view", "fields": ["gene_ids", "expression", "population", "matching_group"]},
                "condition_cell": {"representation": "logical_view", "fields": ["gene_ids", "expression", "condition_id", "population", "matching_group"]},
            },
            "source_schema": {
                "columns": summary["columns"],
                "control_values": list(self.control_values),
                "dose_unit": self.dose_unit,
                "default_dose": self.default_dose,
                "smiles_map": self.smiles_map,
            },
            "selection": str(selection_file.resolve()),
            "canonical_source": str((cache_workspace / "materialize" / "_source").resolve()),
            "counts": {"selected_cells": int(sum(summary["cells_by_population"][value] for value in populations))},
        }
        return DataSelection(dataset=self.dataset, source=self.source, projects=self.projects, frozen_models=self.frozen_models, populations=tuple(populations), cache_workspace=cache_workspace, contract_payload=contract)


def combosciplex(source, projects, *, frozen_models, dataset="ComboSciPlex", **kwargs):
    kwargs.setdefault("control_values", ("vehicle", "dmso", "control", "untreated"))
    return ComboSciPlexHandler(source, projects, frozen_models=frozen_models, dataset=dataset, **kwargs)


__all__ = ["ComboSciPlexHandler", "combosciplex"]
