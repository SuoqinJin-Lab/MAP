from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np

from ...._common.contracts import write_contract
from ...._common.feedback import Feedback
from .constants import CONTROL_NAMES
from .paths import TahoePaths
from .reader import TahoeReader, parse_drug_dose, read_sample_metadata, require_pyarrow


def _require_tahoe_raw(paths: TahoePaths) -> None:
    paths.require_inputs()
    if not paths.raw_data.is_dir():
        raise FileNotFoundError(f"Tahoe expression directory is missing: {paths.raw_data}")
    if not list(paths.raw_data.glob("*.parquet")):
        raise FileNotFoundError(f"No Tahoe expression shards under {paths.raw_data}")


def _paths(
    source: str | Path, workspace: str | Path, dataset: str = "Tahoe-100M"
) -> TahoePaths:
    return TahoePaths(
        dataset=dataset, source=Path(source), workspace=Path(workspace),
        frozen_models=Path(workspace) / ".unused-model-root",
    )


def _condition(row: dict, sample_map: dict) -> tuple:
    drug, dose, unit = parse_drug_dose(
        sample_map.get(str(row.get("sample")), {}).get("drugname_drugconc", "")
    )
    drug = drug or str(row.get("drug", "")).strip()
    smiles = str(row.get("canonical_smiles") or "").strip()
    return drug, dose, unit, smiles


def watch_data(
    source: str | Path,
    projects: str | Path,
    batch_size: int = 8192,
    *,
    refresh: bool = False,
    dataset: str = "Tahoe-100M",
) -> dict:
    """Scan Tahoe metadata once and persist a reusable dataset census."""
    paths = _paths(source, Path(projects) / "_dataset-placeholder", dataset)
    root = paths.dataset_cache
    _require_tahoe_raw(paths)
    summary_file = root / "data_summary.json"
    cell_line_file = root / "cell_lines.parquet"
    condition_file = root / "conditions.parquet"
    if not refresh and summary_file.is_file() and cell_line_file.is_file() and condition_file.is_file():
        summary = json.loads(summary_file.read_text(encoding="utf-8"))
        if "cell_line_ids" not in summary:
            summary["cell_line_ids"] = sorted(summary.get("cells_by_cell_line", {}))
            summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    report = Feedback(root, "watch_data")
    reader = TahoeReader(paths.raw)
    sample_map = read_sample_metadata(paths.raw)
    cell_lines = Counter()
    conditions = Counter()
    drugs = set()
    doses = []
    excluded = 0
    columns = ["cell_line_id", "sample", "drug", "canonical_smiles", "plate"]
    total_shards = len(reader.files)
    progress_every = max(1, total_shards // 100)
    report.emit("scan start", shards=total_shards, batch_size=batch_size)
    for file_index, path in enumerate(reader.files):
        for _, _, batch in reader.iter_file_batches(path, columns, batch_size):
            for line, sample, raw_drug, smiles, plate in zip(
                *(batch[name] for name in columns)
            ):
                line = str(line)
                cell_lines[line] += 1
                drug, dose, unit, canonical = _condition(
                    {"sample": sample, "drug": raw_drug, "canonical_smiles": smiles},
                    sample_map,
                )
                raw_name = str(raw_drug or "").strip()
                control = (
                    drug.casefold() in CONTROL_NAMES
                    or raw_name.casefold() in CONTROL_NAMES
                )
                if control:
                    conditions[
                        (line, "control", "__CONTROL__", 0.0, "uM", "", str(plate))
                    ] += 1
                elif canonical and math.isfinite(dose):
                    conditions[
                        (line, "condition", drug, dose, unit, canonical, str(plate))
                    ] += 1
                    drugs.add(drug)
                    doses.append(float(dose))
                else:
                    excluded += 1
        completed = file_index + 1
        if completed == 1 or completed % progress_every == 0 or completed == total_shards:
            report.emit(
                "scan progress",
                shards=f"{completed}/{total_shards}",
                percent=f"{completed / total_shards:.1%}",
                cells=sum(cell_lines.values()),
            )

    condition_rows = [
        {
            "cell_line": line,
            "kind": kind,
            "drug": drug,
            "dose": dose,
            "unit": unit,
            "canonical_smiles": smiles,
            "plate": plate,
            "cells": int(value),
        }
        for (line, kind, drug, dose, unit, smiles, plate), value in sorted(conditions.items())
    ]
    cell_line_rows = [
        {"cell_line": line, "cells": int(value)}
        for line, value in sorted(cell_lines.items())
    ]
    summary = {
        "cells": int(sum(cell_lines.values())),
        "genes": int(reader.gene_count),
        "shards": len(reader.files),
        "cell_lines": len(cell_lines),
        "condition_cells": int(sum(row["cells"] for row in condition_rows if row["kind"] == "condition")),
        "control_cells": int(sum(row["cells"] for row in condition_rows if row["kind"] == "control")),
        "conditions": len({
            (row["cell_line"], row["drug"], row["dose"], row["unit"], row["canonical_smiles"])
            for row in condition_rows if row["kind"] == "condition"
        }),
        "drugs": len(drugs),
        "excluded_perturbation_cells": int(excluded),
        "dose_min": min(doses) if doses else None,
        "dose_median": float(np.median(doses)) if doses else None,
        "dose_max": max(doses) if doses else None,
        "cell_line_ids": sorted(cell_lines),
        "cells_by_cell_line": dict(sorted(cell_lines.items())),
    }
    pa, pq = require_pyarrow()
    root.mkdir(parents=True, exist_ok=True)
    summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pq.write_table(pa.Table.from_pylist(cell_line_rows), cell_line_file)
    pq.write_table(pa.Table.from_pylist(condition_rows), condition_file)
    report.emit(
        "dataset",
        cells=summary["cells"], genes=summary["genes"], cell_lines=summary["cell_lines"],
        conditions=summary["conditions"], drugs=summary["drugs"],
    )
    report.finish(summary, [summary_file, cell_line_file, condition_file])
    return summary


def fetch_cell_lines(paths: TahoePaths, cell_lines: Iterable[str], batch_size: int = 8192) -> dict:
    _require_tahoe_raw(paths)
    if not paths.data_summary.is_file() or not paths.cell_line_summary.is_file():
        watch_data(
            paths.source,
            paths.workspace.parent,
            batch_size=batch_size,
            dataset=paths.dataset,
        )
    requested = tuple(dict.fromkeys(str(value) for value in cell_lines))
    if not requested:
        raise ValueError("At least one cell line is required")
    report = Feedback(paths.staging, "fetch_cell_line")
    _, pq = require_pyarrow()
    counts = {
        str(row["cell_line"]): int(row["cells"])
        for row in pq.read_table(paths.cell_line_summary).to_pylist()
    }
    missing = sorted(set(requested) - set(counts))
    if missing:
        raise ValueError(f"Cell lines are absent from Tahoe: {', '.join(missing)}")
    payload = {
        "cell_lines": list(requested),
        "cells_by_cell_line": {line: int(counts[line]) for line in requested},
        "cells": int(sum(counts[line] for line in requested)),
        "genes": int(json.loads(paths.data_summary.read_text(encoding="utf-8"))["genes"]),
    }
    paths.selection.parent.mkdir(parents=True, exist_ok=True)
    paths.selection.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report.emit("selected", cell_lines=len(requested), cells=payload["cells"])
    report.finish(payload, [paths.selection])
    return payload


def _selected_cell_lines(paths: TahoePaths) -> tuple[str, ...]:
    if not paths.selection.is_file():
        raise FileNotFoundError("Run fetch_cell_line() first")
    return tuple(json.loads(paths.selection.read_text(encoding="utf-8"))["cell_lines"])


def _select_conditions(paths: TahoePaths, batch_size: int = 8192) -> dict:
    if not paths.all_condition_summary.is_file():
        watch_data(
            paths.source,
            paths.workspace.parent,
            batch_size=batch_size,
            dataset=paths.dataset,
        )
    selected = set(_selected_cell_lines(paths))
    report = Feedback(paths.staging, "selected_conditions")
    pa, pq = require_pyarrow()
    rows = [
        row for row in pq.read_table(paths.all_condition_summary).to_pylist()
        if str(row["cell_line"]) in selected
    ]
    paths.condition_summary.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), paths.condition_summary)
    drugs = {str(row["drug"]) for row in rows if row["kind"] == "condition"}
    doses = [float(row["dose"]) for row in rows if row["kind"] == "condition"]
    summary = {
        "condition_cells": int(sum(row["cells"] for row in rows if row["kind"] == "condition")),
        "control_cells": int(sum(row["cells"] for row in rows if row["kind"] == "control")),
        "conditions": len({
            (row["cell_line"], row["drug"], row["dose"], row["unit"], row["canonical_smiles"])
            for row in rows if row["kind"] == "condition"
        }),
        "drugs": len(drugs), "dose_min": min(doses) if doses else None,
        "dose_median": float(np.median(doses)) if doses else None,
        "dose_max": max(doses) if doses else None,
    }
    report.emit("conditions", conditions=summary["conditions"], drugs=summary["drugs"], condition_cells=summary["condition_cells"], control_cells=summary["control_cells"])
    report.finish(summary, [paths.condition_summary])
    return summary


def tahoe_contract(paths: TahoePaths) -> dict:
    selected = json.loads(paths.selection.read_text(encoding="utf-8"))
    _select_conditions(paths)
    entities = {
        "condition": {"representation": "logical_view", "primary_key": "condition_id", "fields": ["condition_id", "population", "drug", "dose", "dose_unit", "canonical_smiles"]},
        "control_cell": {"representation": "logical_view", "fields": ["gene_ids", "expression", "population", "matching_group"], "matching_keys": ["population", "plate"]},
        "condition_cell": {"representation": "logical_view", "fields": ["gene_ids", "expression", "condition_id", "population", "matching_group"]},
    }
    return {
        "dataset": paths.dataset,
        "source_format": "tahoe",
        "handler_backend": "map.preprocess.handlers.tahoe.backend",
        "native_source": str(paths.source.resolve()),
        "populations": selected["cell_lines"], "entities": entities,
        "selection": str(paths.selection.resolve()),
        "condition_summary": str(paths.condition_summary.resolve()),
        "counts": {
            "selected_cells": selected["cells"],
        },
    }


def create_tahoe_project(paths: TahoePaths) -> dict:
    contract = tahoe_contract(paths)
    report = Feedback(paths.workspace, "create_project")
    contract_file = write_contract(paths.workspace, contract)
    contract["contract_file"] = str(contract_file)
    report.emit(
        "MAP entities",
        project=paths.workspace.name,
        populations=len(contract["populations"]),
        cells=contract["counts"]["selected_cells"],
    )
    report.finish(contract, [contract_file])
    return contract


__all__ = [
    "_condition", "_require_tahoe_raw", "watch_data", "fetch_cell_lines",
    "_select_conditions", "tahoe_contract", "create_tahoe_project",
]
