from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import Any

import numpy as np

from .anndata import _require_anndata
from .tahoe import pipeline
from .tahoe.reader import require_pyarrow


def _value(row, column: str | None, default=""):
    return row[column] if column else default


def _text(value, fallback=""):
    if value is None:
        return fallback
    text = str(value).strip()
    return fallback if text.casefold() in {"", "nan", "none", "null"} else text


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
    smiles_map = {str(key): str(value) for key, value in source_schema.get("smiles_map", {}).items()}
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
            control = perturbation.casefold() in controls
            drug = "DMSO_TF" if control else perturbation
            dose = 0.0 if control else float(_value(row, columns.get("dose"), 0.0))
            smiles = "" if control else _text(
                _value(row, columns.get("smiles"), ""),
                smiles_map.get(perturbation, ""),
            )
            if not control and not smiles:
                raise ValueError(
                    f"Missing SMILES for perturbation {perturbation!r}; set smiles_key or smiles_map"
                )
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
        raise ValueError(f"AnnData source reader does not support preparation stage {stage!r}")
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


__all__ = ["run_stage"]
