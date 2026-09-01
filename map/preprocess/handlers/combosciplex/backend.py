from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..anndata import _require_anndata
from ...._common.components import combination_key
from ..tahoe import pipeline
from ..tahoe.reader import require_pyarrow


def _value(row, column: str | None, default=""):
    return row[column] if column else default


def _text(value, fallback=""):
    if value is None:
        return fallback
    text = " ".join(str(value).split())
    return fallback if text.casefold() in {"", "nan", "none", "null"} else text


def _components(row, columns, perturbation, dose, smiles, dose_unit, smiles_map, controls=()):
    # Keep the component/dose association stable when the native table swaps
    # Drug1 and Drug2.  ``combination_key`` is order independent, so the dose
    # tuple must be canonicalized with the same component ordering as well.
    components = []
    component_keys = columns.get("component_keys", ())
    component_dose_keys = columns.get("component_dose_keys", ())
    component_smiles_keys = columns.get("component_smiles_keys", ())
    if component_keys:
        for index, key in enumerate(component_keys):
            name = _text(row.get(key), "")
            # ComboSciPlex encodes single-drug wells as DMSO + drug.  Vehicle
            # entries are null components, while DMSO + DMSO is the control.
            if name.casefold() in {str(value).casefold() for value in controls}:
                continue
            if not name:
                continue
            dose_key = component_dose_keys[index] if index < len(component_dose_keys) else None
            smiles_key = component_smiles_keys[index] if index < len(component_smiles_keys) else None
            component_dose = float(_value(row, dose_key, dose))
            component_smiles = _text(
                _value(row, smiles_key, ""),
                smiles_map.get(name, smiles_map.get(name.casefold(), "")),
            )
            components.append((component_smiles, component_dose, name))
    if not components and component_keys:
        return [], [], [], ""
    if not components:
        # Single-component schemas encode controls in the perturbation column.
        # Treat those values as null components just like explicit component
        # columns, so controls do not require a drug SMILES mapping.
        if _text(perturbation, "").casefold() in {
            str(value).casefold() for value in controls
        }:
            return [], [], [], ""
        name = _text(perturbation)
        components = [(
            _text(smiles, smiles_map.get(name, smiles_map.get(name.casefold(), ""))),
            float(dose),
            name,
        )]
    components.sort(key=lambda value: (value[0], value[1], value[2]))
    smiles_values = [value[0] for value in components]
    doses = [value[1] for value in components]
    names = [value[2] for value in components]
    if any(not value for value in smiles_values):
        raise ValueError(f"Missing SMILES for drug components {names!r}")
    return names, doses, smiles_values, combination_key(smiles_values)


def _canonical_source(contract: dict[str, Any], output_dir: Path, shard_size: int = 20_000) -> Path:
    configured = contract.get("canonical_source")
    root = Path(configured) if configured else output_dir / "_source"
    marker = root / "_SUCCESS"
    if marker.is_file():
        return root
    ad = _require_anndata()
    pa, pq = require_pyarrow()
    data = ad.read_h5ad(contract["native_source"], backed="r")
    source_schema = contract["source_schema"]
    columns = source_schema["columns"]
    populations = set(str(value) for value in contract["populations"])
    controls = set(str(value).casefold() for value in source_schema["control_values"])
    default_dose = float(source_schema.get("default_dose", 1.0))
    smiles_map = {str(key): str(value) for key, value in source_schema.get("smiles_map", {}).items()}
    smiles_map.update({key.casefold(): value for key, value in smiles_map.items()})
    data_dir = root / "data"
    metadata_dir = root / "metadata"
    data_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    symbols = (
        [str(value) for value in data.var[columns["gene_symbol"]]]
        if columns.get("gene_symbol") else [str(value) for value in data.var_names]
    )
    pq.write_table(
        pa.Table.from_pylist([
            {"token_id": index, "gene_symbol": symbol}
            for index, symbol in enumerate(symbols)
        ]),
        metadata_dir / "gene_metadata.parquet",
    )
    sample_rows = []
    output_rows = []
    component_map = {}
    shard_index = 0
    for start in range(0, data.n_obs, shard_size):
        stop = min(data.n_obs, start + shard_size)
        obs = data.obs.iloc[start:stop]
        matrix = data[start:stop].X
        try:
            from scipy import sparse
            sparse_matrix = sparse.issparse(matrix)
            if sparse_matrix:
                matrix = matrix.tocsr()
        except ImportError:
            sparse_matrix = False
        for offset, (_, row) in enumerate(obs.iterrows()):
            population = _text(_value(row, columns["population"], "__ALL__"), "__ALL__")
            if population not in populations:
                continue
            perturbation = _text(_value(row, columns["perturbation"]))
            dose = float(_value(row, columns.get("dose"), default_dose))
            smiles = _text(
                _value(row, columns.get("smiles"), ""),
                smiles_map.get(perturbation, ""),
            )
            component_names, component_doses, component_smiles, combination = _components(
                row, columns, perturbation, dose,
                smiles, source_schema.get("dose_unit", "uM"), smiles_map, controls,
            )
            control = not component_names
            if control:
                drug = "DMSO_TF"
                dose = 0.0
                smiles = ""
            else:
                # Preserve the full dose tuple in the condition identity.
                # Tahoe's legacy schema has one dose column, so encoding pair
                # doses in the drug key prevents accidental merges.
                drug = combination
                if len(component_names) > 1:
                    suffix = ",".join(f"{value:g}" for value in component_doses)
                    drug = f"{combination}__dose_{suffix}"
                if not component_smiles:
                    raise ValueError(
                        f"Missing SMILES for drug components {component_names!r}"
                    )
                # ``dose`` is the scalar compatibility field consumed by the
                # Tahoe preparation pipeline.  Set it before constructing the
                # component-map key so explicit component-dose columns use the
                # same key during materialization enrichment.
                dose = component_doses[0]
                component_map[f"{drug}||{dose:g}"] = {
                    "component_names": component_names,
                    "component_smiles": component_smiles,
                    "component_doses_uM": component_doses,
                    "component_dose_units": [source_schema.get("dose_unit", "uM")] * len(component_doses),
                    "combination_key": combination,
                }
                smiles = component_smiles[0]
            group = _text(_value(row, columns.get("group"), "default"), "default")
            vector = matrix[offset]
            if sparse_matrix:
                gene_ids = vector.indices.astype(np.int32).tolist()
                expressions = vector.data.astype(np.float32).tolist()
            else:
                values = np.asarray(vector).reshape(-1)
                gene_ids = np.flatnonzero(values).astype(np.int32).tolist()
                expressions = values[gene_ids].astype(np.float32).tolist()
            sample = f"cell-{start + offset}"
            output_rows.append({
                "genes": gene_ids,
                "expressions": expressions,
                "cell_line_id": population,
                "sample": sample,
                "drug": drug,
                "canonical_smiles": smiles,
                "plate": group,
            })
            sample_rows.append({
                "sample": sample,
                "drug": drug,
                "drugname_drugconc": repr([(drug, dose, source_schema.get("dose_unit", "uM"))]),
            })
            if len(output_rows) >= shard_size:
                pq.write_table(
                    pa.Table.from_pylist(output_rows),
                    data_dir / f"part-{shard_index:05d}.parquet",
                )
                output_rows.clear()
                shard_index += 1
    if output_rows:
        pq.write_table(
            pa.Table.from_pylist(output_rows),
            data_dir / f"part-{shard_index:05d}.parquet",
        )
    pq.write_table(pa.Table.from_pylist(sample_rows), metadata_dir / "sample_metadata.parquet")
    (metadata_dir / "component_map.json").write_text(json.dumps(component_map, indent=2), encoding="utf-8")
    marker.write_text("ok\n", encoding="utf-8")
    if getattr(data, "file", None) is not None:
        data.file.close()
    return root


def run_stage(
    stage: str,
    *,
    contract: dict[str, Any],
    output_dir: str | Path,
    workers: int,
    esm_embeddings: str | Path | None = None,
    seed: int = 42,
    **kwargs: Any,
) -> None:
    if stage not in {"filter", "stats", "hvg", "materialize"}:
        raise ValueError(f"AnnData handler does not support preparation stage {stage!r}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = _canonical_source(contract, output_dir)
    values = {
        "raw_dir": str(raw_dir),
        "output_dir": str(output_dir),
        "workers": int(workers),
        "seed": int(seed),
        **kwargs,
    }
    if stage in {"filter", "stats"}:
        values["cell_lines"] = [str(value) for value in contract["populations"]]
        if stage == "stats" and esm_embeddings is None:
            raise ValueError("stats requires the frozen STATE ESM2 gene table")
    if esm_embeddings is not None:
        values["esm_embeddings"] = str(esm_embeddings)
    getattr(pipeline, f"run_{stage}")(Namespace(**values))
    if stage == "materialize":
        conditions_path = output_dir / "conditions.parquet"
        component_path = raw_dir / "metadata" / "component_map.json"
        if conditions_path.is_file() and component_path.is_file():
            import pandas as pd
            conditions = pd.read_parquet(conditions_path)
            mapping = json.loads(component_path.read_text(encoding="utf-8"))
            for field in ("component_names", "component_smiles", "component_doses_uM", "component_dose_units", "combination_key"):
                values = []
                for drug, smiles, dose, unit in zip(conditions["drug"], conditions["canonical_smiles"], conditions["dose"], conditions["dose_unit"]):
                    fallback = str(drug) if field == "combination_key" else [smiles] if field == "component_smiles" else [dose] if field == "component_doses_uM" else [str(drug)] if field == "component_names" else [unit]
                    values.append(mapping.get(f"{drug}||{float(dose):g}", mapping.get(str(drug), {})).get(field, fallback))
                conditions[field] = values
            # Include the full component-dose tuple in the logical condition
            # identity.  The legacy scalar ``dose`` field stores only the
            # first component dose for compatibility with Tahoe consumers;
            # using it alone would collide for (A@1,B@2) and (A@1,B@3).
            conditions["condition_key"] = [
                f"{population}|{key}|{','.join(f'{float(value):g}' for value in doses)}"
                for population, key, doses in zip(
                    conditions["population"],
                    conditions["combination_key"],
                    conditions["component_doses_uM"],
                )
            ]
            conditions.to_parquet(conditions_path, index=False)


__all__ = ["run_stage"]
