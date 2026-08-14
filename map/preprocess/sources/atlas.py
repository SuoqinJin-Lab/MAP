from __future__ import annotations

import json
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from rdkit import Chem

from ..._common.feedback import Feedback
from ..selection import DataSelection


CONTROL_NAMES = {
    "control",
    "ctrl",
    "dmso",
    "dmso_tf",
    "negative control",
    "untreated",
    "vehicle",
    "vehicle control",
}


def _first_existing(root: Path, names: Sequence[str]) -> Path:
    for name in names:
        path = root / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"None of {list(names)} found under {root}")


def _read_table(path: Path) -> pd.DataFrame:
    name = path.name.casefold()
    if name.endswith(".parquet"):
        return pd.read_parquet(path)
    if name.endswith((".csv", ".csv.gz", ".csv.zip")):
        return pd.read_csv(path, low_memory=False)
    if name.endswith((".tsv", ".tsv.gz", ".txt", ".txt.gz")):
        return pd.read_csv(path, sep="\t", low_memory=False)
    raise ValueError(f"Unsupported metadata table: {path}")


def _choose_column(
    columns,
    explicit: str | None,
    candidates: Sequence[str],
    *,
    label: str,
    required: bool = False,
) -> str | None:
    available = {str(value).casefold(): str(value) for value in columns}
    if explicit is not None:
        if explicit not in columns:
            raise KeyError(
                f"Requested {label} column {explicit!r} is absent; "
                f"available: {list(columns)}"
            )
        return explicit
    for candidate in candidates:
        if candidate.casefold() in available:
            return available[candidate.casefold()]
    if required:
        raise KeyError(
            f"Cannot infer {label}; expected one of {list(candidates)}, "
            f"available: {list(columns)}"
        )
    return None


def _canonical_smiles(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    if text.casefold() in {"", "nan", "none", "null", "xxx"}:
        return ""
    molecule = Chem.MolFromSmiles(text)
    return Chem.MolToSmiles(molecule, canonical=True) if molecule is not None else ""


def _load_smiles_map(path: str | Path | None) -> dict[str, str]:
    if path is None:
        return {}
    table = _read_table(Path(path))
    name = _choose_column(
        table.columns,
        None,
        ("drug_name", "name", "sm_name", "product_name", "condition"),
        label="drug name",
        required=True,
    )
    smiles = _choose_column(
        table.columns,
        None,
        ("canonical_smiles", "smiles", "SMILES"),
        label="SMILES",
        required=True,
    )
    output = {}
    for drug, value in zip(table[name], table[smiles]):
        canonical = _canonical_smiles(value)
        if canonical:
            output[str(drug).strip().casefold()] = canonical
    return output


def _normalise_metadata(
    frame: pd.DataFrame,
    *,
    schema: Mapping[str, str | None],
    smiles_map: Mapping[str, str],
    default_dose: float | None,
    dose_unit: str,
) -> tuple[pd.DataFrame, dict[str, str | None]]:
    columns = frame.columns
    resolved = {
        "cell_id": _choose_column(
            columns,
            schema.get("cell_id"),
            ("obs_id", "cell_id", "cell", "barcode", "cell_barcode"),
            label="cell ID",
        ),
        "population": _choose_column(
            columns,
            schema.get("population"),
            ("cell_type", "cell_line", "cell_line_id", "celltype", "cell.line"),
            label="cell type/line",
            required=True,
        ),
        "drug": _choose_column(
            columns,
            schema.get("drug"),
            ("sm_name", "product_name", "drug_name", "drug", "condition", "perturbation", "treatment"),
            label="drug",
            required=True,
        ),
        "smiles": _choose_column(
            columns,
            schema.get("smiles"),
            ("SMILES", "smiles", "canonical_smiles", "canonical_isomeric_smiles"),
            label="SMILES",
        ),
        "dose": _choose_column(
            columns,
            schema.get("dose"),
            ("dose_uM", "dose_um", "dose", "dose_val", "dose_value", "concentration"),
            label="dose",
        ),
        "control": _choose_column(
            columns,
            schema.get("control"),
            ("control", "is_control", "vehicle"),
            label="control",
        ),
        "group": _choose_column(
            columns,
            schema.get("group"),
            ("plate", "plate_name", "batch", "batch_id", "replicate", "sample"),
            label="matching group",
        ),
    }
    count = len(frame)
    drug_names = frame[resolved["drug"]].fillna("").astype(str).str.strip()
    control = drug_names.str.casefold().isin(CONTROL_NAMES)
    if resolved["control"] is not None:
        values = frame[resolved["control"]]
        if pd.api.types.is_bool_dtype(values):
            control |= values.fillna(False).astype(bool)
        elif pd.api.types.is_numeric_dtype(values):
            control |= values.fillna(0).astype(float).ne(0)
        else:
            control |= values.fillna("").astype(str).str.casefold().isin(
                {"1", "true", "yes", "control", "vehicle", "dmso"}
            )

    if resolved["dose"] is not None:
        dose = pd.to_numeric(frame[resolved["dose"]], errors="coerce")
    else:
        dose = pd.Series(default_dose, index=frame.index, dtype=np.float64)
    dose = dose.astype(np.float64)
    dose.loc[control] = 0.0

    direct_smiles = (
        frame[resolved["smiles"]].tolist()
        if resolved["smiles"] is not None
        else [""] * count
    )
    smiles = []
    for drug, direct, is_control in zip(drug_names, direct_smiles, control):
        if is_control:
            smiles.append("")
            continue
        canonical = _canonical_smiles(direct)
        if not canonical:
            canonical = smiles_map.get(str(drug).casefold(), "")
        smiles.append(canonical)

    output = pd.DataFrame({
        "source_row": np.arange(count, dtype=np.int64),
        "cell_id": (
            frame[resolved["cell_id"]].astype(str).to_numpy()
            if resolved["cell_id"] is not None
            else np.asarray([f"cell-{index}" for index in range(count)])
        ),
        "population": frame[resolved["population"]].fillna("unknown").astype(str).to_numpy(),
        "drug": drug_names.to_numpy(),
        "dose": dose.to_numpy(),
        "dose_unit": str(dose_unit),
        "canonical_smiles": smiles,
        "control": control.to_numpy(dtype=bool),
        "matching_group": (
            frame[resolved["group"]].fillna("default").astype(str).to_numpy()
            if resolved["group"] is not None
            else "default"
        ),
        "included": True,
    })
    invalid_dose = (~output["control"]) & ~np.isfinite(output["dose"])
    invalid_smiles = (~output["control"]) & output["canonical_smiles"].eq("")
    if invalid_dose.any():
        names = sorted(output.loc[invalid_dose, "drug"].unique())[:8]
        raise ValueError(f"Missing finite dose values for treated drugs: {names}")
    if invalid_smiles.any():
        names = sorted(output.loc[invalid_smiles, "drug"].unique())[:8]
        raise ValueError(
            f"Missing canonical SMILES for treated drugs: {names}; provide smiles_map"
        )
    return output, resolved


def _extract_single_parquet(archive: Path) -> Path:
    if archive.suffix.casefold() != ".zip":
        return archive
    target = archive.with_suffix("")
    with zipfile.ZipFile(archive) as handle:
        members = [name for name in handle.namelist() if name.casefold().endswith(".parquet")]
        if len(members) != 1:
            raise ValueError(f"Expected one parquet file in {archive}; found {members}")
        info = handle.getinfo(members[0])
        if target.is_file() and target.stat().st_size == info.file_size:
            return target
        temporary = target.with_suffix(target.suffix + ".tmp")
        with handle.open(members[0]) as source, temporary.open("wb") as destination:
            while chunk := source.read(16 * 1024 * 1024):
                destination.write(chunk)
        temporary.replace(target)
    return target


def _inspect_nips(source: Path, options: Mapping[str, Any]):
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("NIPS/OP3 preprocessing requires pyarrow") from exc
    metadata_file = _first_existing(
        source,
        ("adata_obs_meta.csv.zip", "adata_obs_meta.csv.gz", "adata_obs_meta.csv"),
    )
    expression_archive = _first_existing(
        source, ("adata_train.parquet", "adata_train.parquet.zip")
    )
    expression_file = _extract_single_parquet(expression_archive)
    metadata = _read_table(metadata_file)
    parquet = pq.ParquetFile(expression_file)
    parquet_columns = list(parquet.schema_arrow.names)
    long_columns = {"obs_id", "gene", "count"}
    if long_columns.issubset(parquet_columns):
        row_id = str(options.get("expression_cell_id") or "obs_id")
        gene_column = str(options.get("expression_gene") or "gene")
        value_column = str(options.get("expression_value") or "count")
        for column in (row_id, gene_column, value_column):
            if column not in parquet_columns:
                raise KeyError(
                    f"Requested NIPS long-table column {column!r} is absent; "
                    f"available: {parquet_columns}"
                )
        schema = {name: options.get(name) for name in (
            "cell_id", "population", "drug", "smiles", "dose", "control", "group"
        )}
        cells, resolved = _normalise_metadata(
            metadata,
            schema=schema,
            smiles_map=_load_smiles_map(options.get("smiles_map")),
            default_dose=1.0,
            dose_unit=str(options.get("dose_unit", "uM")),
        )
        if resolved["cell_id"] is None:
            raise KeyError("NIPS long-table input requires obs_id or cell_id in metadata")

        import pyarrow.compute as pc

        observed_ids: set[str] = set()
        gene_symbols: set[str] = set()
        for batch in parquet.iter_batches(
            batch_size=1_000_000, columns=[row_id, gene_column]
        ):
            observed_ids.update(
                str(value) for value in pc.unique(batch.column(0)).to_pylist()
            )
            gene_symbols.update(
                str(value) for value in pc.unique(batch.column(1)).to_pylist()
            )
        metadata_ids = set(cells["cell_id"].astype(str))
        missing_metadata = sorted(observed_ids - metadata_ids)
        if missing_metadata:
            raise ValueError(
                f"{len(missing_metadata)} NIPS expression cell IDs are absent from "
                f"metadata; examples: {missing_metadata[:5]}"
            )
        cells["included"] = cells["cell_id"].astype(str).isin(observed_ids)
        excluded_file = next(
            (
                path for path in (
                    source / "adata_excluded_ids.csv.zip",
                    source / "adata_excluded_ids.csv.gz",
                    source / "adata_excluded_ids.csv",
                )
                if path.is_file()
            ),
            None,
        )
        return cells, sorted(gene_symbols), {
            "kind": "nips_long_parquet",
            "expression_file": str(expression_file.resolve()),
            "expression_rows": int(parquet.metadata.num_rows),
            "row_id_column": row_id,
            "gene_column": gene_column,
            "value_column": value_column,
            "metadata_file": str(metadata_file.resolve()),
            "excluded_file": str(excluded_file.resolve()) if excluded_file else None,
            "resolved_columns": resolved,
        }

    row_id = next(
        (name for name in ("obs_id", "cell_id", "__index_level_0__") if name in parquet_columns),
        None,
    )
    metadata_id = _choose_column(
        metadata.columns,
        options.get("cell_id"),
        ("obs_id", "cell_id"),
        label="cell ID",
    )
    if row_id is not None and metadata_id is not None:
        ordered_ids = pq.read_table(expression_file, columns=[row_id]).column(0).to_pylist()
        lookup = metadata.assign(_map_id=metadata[metadata_id].astype(str)).set_index("_map_id", drop=False)
        missing = [str(value) for value in ordered_ids if str(value) not in lookup.index]
        if missing:
            raise ValueError(
                f"{len(missing)} expression row IDs are absent from metadata; examples: {missing[:5]}"
            )
        metadata = lookup.loc[[str(value) for value in ordered_ids]].reset_index(drop=True)
    elif len(metadata) != parquet.metadata.num_rows:
        raise ValueError(
            "NIPS expression and metadata rows cannot be aligned without a shared cell ID"
        )
    metadata_columns = {str(value) for value in metadata.columns}
    expression_columns = [
        value for value in parquet_columns
        if value != row_id and value not in metadata_columns
    ]
    if not expression_columns:
        raise ValueError(f"No expression columns found in {expression_file}")
    schema = {name: options.get(name) for name in (
        "cell_id", "population", "drug", "smiles", "dose", "control", "group"
    )}
    cells, resolved = _normalise_metadata(
        metadata,
        schema=schema,
        smiles_map=_load_smiles_map(options.get("smiles_map")),
        default_dose=1.0,
        dose_unit=str(options.get("dose_unit", "uM")),
    )
    excluded_file = next(
        (
            path for path in (
                source / "adata_excluded_ids.csv.zip",
                source / "adata_excluded_ids.csv.gz",
                source / "adata_excluded_ids.csv",
            )
            if path.is_file()
        ),
        None,
    )
    return cells, expression_columns, {
        "kind": "nips_parquet",
        "expression_file": str(expression_file.resolve()),
        "row_id_column": row_id,
        "metadata_file": str(metadata_file.resolve()),
        "excluded_file": str(excluded_file.resolve()) if excluded_file else None,
        "resolved_columns": resolved,
    }


def _sciplex_export_root(source: Path) -> Path:
    candidates = (source, source / "exported")
    for candidate in candidates:
        if any((candidate / name).is_file() for name in ("matrix.mtx", "matrix.mtx.gz", "counts.mtx", "counts.mtx.gz")):
            return candidate
    rds = sorted(source.glob("*.RDS")) + sorted(source.glob("*.rds"))
    if rds:
        raise FileNotFoundError(
            f"SciPlex3 RDS is present but not exported: {rds[0]}; "
            "run preprocess.sciplex.export_rds() first"
        )
    raise FileNotFoundError(f"SciPlex3 Matrix Market export is missing under {source}")


def _inspect_sciplex(source: Path, options: Mapping[str, Any]):
    from scipy.io import mminfo

    root = _sciplex_export_root(source)
    matrix_file = _first_existing(
        root, ("matrix.mtx.gz", "matrix.mtx", "counts.mtx.gz", "counts.mtx")
    )
    cell_file = _first_existing(
        root,
        ("cell_metadata.tsv.gz", "cell_metadata.tsv", "cells.tsv.gz", "cells.tsv"),
    )
    gene_file = _first_existing(
        root,
        ("gene_metadata.tsv.gz", "gene_metadata.tsv", "genes.tsv.gz", "genes.tsv"),
    )
    metadata = _read_table(cell_file)
    genes = _read_table(gene_file)
    gene_column = _choose_column(
        genes.columns,
        options.get("gene_symbol"),
        ("gene_short_name", "gene_name", "gene_symbol", "symbol", "feature_name", "feature_id"),
        label="gene symbol",
        required=True,
    )
    rows, columns = mminfo(matrix_file)[:2]
    if rows == len(genes) and columns == len(metadata):
        orientation = "gene_by_cell"
    elif rows == len(metadata) and columns == len(genes):
        orientation = "cell_by_gene"
    else:
        raise ValueError(
            f"SciPlex matrix shape {(rows, columns)} is incompatible with "
            f"{len(metadata)} cells and {len(genes)} genes"
        )
    schema = {name: options.get(name) for name in (
        "cell_id", "population", "drug", "smiles", "dose", "control", "group"
    )}
    cells, resolved = _normalise_metadata(
        metadata,
        schema=schema,
        smiles_map=_load_smiles_map(options.get("smiles_map")),
        default_dose=None,
        dose_unit=str(options.get("dose_unit", "uM")),
    )
    gene_symbols = genes[gene_column].fillna("").astype(str).tolist()
    return cells, gene_symbols, {
        "kind": "sciplex_mtx",
        "expression_file": str(matrix_file.resolve()),
        "orientation": orientation,
        "cell_metadata_file": str(cell_file.resolve()),
        "gene_metadata_file": str(gene_file.resolve()),
        "gene_symbol_column": gene_column,
        "resolved_columns": resolved,
    }


def _summary(cells: pd.DataFrame, genes: int, dataset: str, population_label: str) -> dict:
    source_cells = cells
    cells = cells.loc[cells["included"]]
    populations = Counter(cells["population"].astype(str))
    treated = cells.loc[~cells["control"]]
    condition_columns = ["population", "drug", "dose", "dose_unit", "canonical_smiles"]
    values = {
        "dataset": dataset,
        "cells": int(len(cells)),
        "source_cells": int(len(source_cells)),
        "excluded_cells": int((~source_cells["included"]).sum()),
        "genes": int(genes),
        "populations": len(populations),
        "population_ids": sorted(populations),
        "cells_by_population": dict(sorted(populations.items())),
        "control_cells": int(cells["control"].sum()),
        "condition_cells": int((~cells["control"]).sum()),
        "conditions": int(len(treated[condition_columns].drop_duplicates())),
        "drugs": int(treated["drug"].nunique()),
        "dose_min": float(treated["dose"].min()) if len(treated) else None,
        "dose_median": float(treated["dose"].median()) if len(treated) else None,
        "dose_max": float(treated["dose"].max()) if len(treated) else None,
    }
    values[population_label] = list(values["population_ids"])
    return values


class NativeAtlasSource:
    def __init__(
        self,
        source: str | Path,
        projects: str | Path,
        *,
        frozen_models: str | Path,
        dataset: str,
        kind: str,
        population_label: str,
        **options: Any,
    ) -> None:
        self.source = Path(source)
        self.projects = Path(projects)
        self.frozen_models = Path(frozen_models)
        self.dataset = str(dataset)
        self.kind = str(kind)
        self.population_label = str(population_label)
        self.options = {key: value for key, value in options.items() if value is not None}

    @property
    def summary_file(self) -> Path:
        return self.source / "data_summary.json"

    @property
    def cells_file(self) -> Path:
        return self.source / "cells.parquet"

    @property
    def genes_file(self) -> Path:
        return self.source / "genes.parquet"

    @property
    def conditions_file(self) -> Path:
        return self.source / "conditions.parquet"

    @property
    def manifest_file(self) -> Path:
        return self.source / "native_manifest.json"

    def watch_data(self, *, refresh: bool = False) -> dict:
        required = (
            self.summary_file,
            self.cells_file,
            self.genes_file,
            self.conditions_file,
            self.manifest_file,
        )
        if not refresh and all(path.is_file() for path in required):
            return json.loads(self.summary_file.read_text(encoding="utf-8"))
        if not self.source.is_dir():
            raise FileNotFoundError(f"Raw dataset directory is missing: {self.source}")
        if self.kind == "nips":
            cells, gene_symbols, manifest = _inspect_nips(self.source, self.options)
        elif self.kind == "sciplex":
            cells, gene_symbols, manifest = _inspect_sciplex(self.source, self.options)
        else:
            raise ValueError(f"Unsupported native atlas kind: {self.kind}")
        genes = pd.DataFrame({
            "source_gene_id": np.arange(len(gene_symbols), dtype=np.int64),
            "gene_symbol": [str(value) for value in gene_symbols],
        })
        condition_columns = [
            "population", "drug", "dose", "dose_unit", "canonical_smiles", "control"
        ]
        conditions = (
            cells.loc[cells["included"]].groupby(condition_columns, dropna=False)
            .size()
            .rename("cells")
            .reset_index()
        )
        summary = _summary(
            cells, len(gene_symbols), self.dataset, self.population_label
        )
        serializable_options = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in self.options.items()
        }
        manifest.update({
            "format": "map_native_atlas_v1",
            "dataset": self.dataset,
            "source_kind": self.kind,
            "cells_file": str(self.cells_file.resolve()),
            "genes_file": str(self.genes_file.resolve()),
            "options": serializable_options,
        })
        self.source.mkdir(parents=True, exist_ok=True)
        cells.to_parquet(self.cells_file, index=False)
        genes.to_parquet(self.genes_file, index=False)
        conditions.to_parquet(self.conditions_file, index=False)
        self.summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        self.manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        Feedback(self.source, "watch_data").finish(
            summary,
            [
                self.summary_file,
                self.cells_file,
                self.genes_file,
                self.conditions_file,
                self.manifest_file,
            ],
        )
        return summary

    def fetch_populations(
        self,
        populations: Sequence[str],
        *,
        project_name: str,
    ) -> DataSelection:
        requested = tuple(dict.fromkeys(str(value) for value in populations))
        if not requested:
            raise ValueError("At least one population is required")
        project_name = DataSelection._project_name(project_name)
        workspace = self.projects / project_name
        record = workspace / "preprocess.json"
        if record.is_file():
            selection = DataSelection.open(
                workspace, self.frozen_models, projects=self.projects
            )
            if selection.populations != requested:
                raise FileExistsError(
                    f"Project {project_name!r} is already bound to different populations"
                )
            return selection
        summary = self.watch_data()
        available = set(summary["population_ids"])
        missing = sorted(set(requested) - available)
        if missing:
            raise ValueError(
                f"Populations are absent from {self.dataset}: {', '.join(missing)}"
            )
        workspace.mkdir(parents=True, exist_ok=True)
        cells = pd.read_parquet(
            self.cells_file, columns=["population", "included"]
        )
        counts = cells.loc[cells["included"], "population"].value_counts().to_dict()
        selection_file = workspace / "selection.json"
        selection_payload = {
            "populations": list(requested),
            "cells_by_population": {value: int(counts[value]) for value in requested},
            "cells": int(sum(counts[value] for value in requested)),
            "genes": int(summary["genes"]),
        }
        selection_file.write_text(
            json.dumps(selection_payload, indent=2), encoding="utf-8"
        )
        selected_conditions = pd.read_parquet(self.conditions_file)
        selected_conditions = selected_conditions.loc[
            selected_conditions["population"].isin(requested)
        ]
        condition_file = workspace / "conditions.parquet"
        selected_conditions.to_parquet(condition_file, index=False)
        entities = {
            "condition": {
                "representation": "logical_view",
                "primary_key": "condition_id",
                "fields": [
                    "condition_id", "population", "drug", "dose", "dose_unit", "canonical_smiles"
                ],
            },
            "control_cell": {
                "representation": "logical_view",
                "fields": ["gene_ids", "expression", "population", "matching_group"],
                "matching_keys": ["population", "matching_group"],
            },
            "condition_cell": {
                "representation": "logical_view",
                "fields": [
                    "gene_ids", "expression", "condition_id", "population", "matching_group"
                ],
            },
        }
        contract = {
            "dataset": self.dataset,
            "source_format": self.kind,
            "preparation_backend": "map.preprocess.sources.atlas_backend",
            "native_source": str(self.source.resolve()),
            "populations": list(requested),
            "entities": entities,
            "selection": str(selection_file.resolve()),
            "condition_summary": str(condition_file.resolve()),
            "native_manifest": str(self.manifest_file.resolve()),
            "canonical_source": str((workspace / "materialized" / "_source").resolve()),
            "counts": {"selected_cells": selection_payload["cells"]},
        }
        selection = DataSelection(
            dataset=self.dataset,
            source=self.source,
            projects=self.projects,
            frozen_models=self.frozen_models,
            populations=requested,
            cache_workspace=workspace,
            contract_payload=contract,
        )
        Feedback(workspace, "fetch_population").finish(
            selection_payload, [selection_file, condition_file]
        )
        selection.reserve_project(project_name)
        return selection


__all__ = ["NativeAtlasSource"]
