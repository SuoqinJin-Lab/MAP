from __future__ import annotations

import csv
import gzip
import io
import json
import zipfile
from contextlib import contextmanager
from argparse import Namespace
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .tahoe import pipeline
from .tahoe.reader import require_pyarrow


class _ShardWriter:
    def __init__(self, output: Path, shard_size: int) -> None:
        self.output = output
        self.shard_size = int(shard_size)
        self.rows: list[dict[str, Any]] = []
        self.index = 0
        self._pa, self._pq = require_pyarrow()

    def append(self, row: dict[str, Any]) -> None:
        self.rows.append(row)
        if len(self.rows) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self.rows:
            return
        path = self.output / f"part-{self.index:05d}.parquet"
        self._pq.write_table(
            self._pa.Table.from_pylist(self.rows), path, compression="zstd"
        )
        self.rows.clear()
        self.index += 1


def _sample_id(row) -> str:
    if bool(row.control):
        return f"control::{row.population}::{row.matching_group}"
    return "condition::" + json.dumps(
        [row.population, row.drug, float(row.dose), row.dose_unit, row.canonical_smiles],
        separators=(",", ":"),
    )


def _cell_record(row, gene_ids: Iterable[int], expressions: Iterable[float]) -> dict:
    control = bool(row.control)
    return {
        "genes": [int(value) for value in gene_ids],
        "expressions": [float(value) for value in expressions],
        "cell_line_id": str(row.population),
        "sample": _sample_id(row),
        "drug": "DMSO_TF" if control else str(row.drug),
        "canonical_smiles": "" if control else str(row.canonical_smiles),
        "plate": str(row.matching_group),
    }


def _sample_rows(cells) -> list[dict[str, Any]]:
    output = {}
    for row in cells.itertuples(index=False):
        sample = _sample_id(row)
        control = bool(row.control)
        drug = "DMSO_TF" if control else str(row.drug)
        dose = 0.0 if control else float(row.dose)
        unit = str(row.dose_unit)
        output[sample] = {
            "sample": sample,
            "drug": drug,
            "drugname_drugconc": repr([(drug, dose, unit)]),
        }
    return list(output.values())


def _write_nips(manifest: dict, cells, writer: _ShardWriter, batch_size: int) -> None:
    _, pq = require_pyarrow()
    genes = pq.read_table(manifest["genes_file"]).to_pandas()
    columns = genes["gene_symbol"].astype(str).tolist()
    parquet = pq.ParquetFile(manifest["expression_file"])
    start = 0
    for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
        matrix = batch.to_pandas().to_numpy(dtype=np.float32, copy=False)
        stop = start + len(matrix)
        metadata = cells.iloc[start:stop]
        for offset, row in enumerate(metadata.itertuples(index=False)):
            if not bool(row.selected):
                continue
            values = matrix[offset]
            gene_ids = np.flatnonzero(np.isfinite(values) & (values > 0))
            writer.append(_cell_record(row, gene_ids, values[gene_ids]))
        start = stop
    if start != len(cells):
        raise RuntimeError(
            f"NIPS expression rows ({start}) do not match metadata rows ({len(cells)})"
        )


@contextmanager
def _text_table(path: Path):
    if path.suffix.casefold() == ".zip":
        with zipfile.ZipFile(path) as archive:
            members = [name for name in archive.namelist() if not name.endswith("/")]
            if len(members) != 1:
                raise ValueError(f"Expected one CSV in {path}; found {members}")
            with archive.open(members[0]) as binary:
                with io.TextIOWrapper(binary, encoding="utf-8", newline="") as text:
                    yield text
        return
    if path.suffix.casefold() == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", newline="") as text:
            yield text
        return
    with path.open("r", encoding="utf-8", newline="") as text:
        yield text


def _excluded_groups(path: str | None, gene_to_id: dict[str, int]):
    if path is None:
        return iter(())

    def groups():
        with _text_table(Path(path)) as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                return
            id_column = next(
                (name for name in ("obs_id", "cell_id", "id") if name in reader.fieldnames),
                None,
            )
            gene_column = next(
                (name for name in ("gene", "gene_symbol") if name in reader.fieldnames),
                None,
            )
            if id_column is None or gene_column is None:
                raise KeyError(
                    f"NIPS excluded-pair table requires obs_id/cell_id and gene; "
                    f"available: {reader.fieldnames}"
                )
            current = None
            excluded: set[int] = set()
            for row in reader:
                obs_id = str(row[id_column])
                if current is not None and obs_id != current:
                    yield current, excluded
                    excluded = set()
                current = obs_id
                gene_id = gene_to_id.get(str(row[gene_column]))
                if gene_id is not None:
                    excluded.add(gene_id)
            if current is not None:
                yield current, excluded

    return groups()


def _write_nips_long(
    manifest: dict, cells, writer: _ShardWriter, batch_size: int
) -> None:
    import pyarrow.compute as pc

    _, pq = require_pyarrow()
    genes = pq.read_table(manifest["genes_file"]).to_pandas()
    gene_to_id = {
        str(row.gene_symbol): int(row.source_gene_id)
        for row in genes.itertuples(index=False)
    }
    selected_cells = {
        str(row.cell_id): row
        for row in cells.itertuples(index=False)
        if bool(row.selected)
    }
    exclusions = _excluded_groups(manifest.get("excluded_file"), gene_to_id)
    excluded_id, excluded_gene_ids = next(exclusions, (None, set()))

    def exclusions_for(obs_id: str) -> set[int]:
        nonlocal excluded_id, excluded_gene_ids
        while excluded_id is not None and excluded_id < obs_id:
            excluded_id, excluded_gene_ids = next(exclusions, (None, set()))
        if excluded_id == obs_id:
            result = excluded_gene_ids
            excluded_id, excluded_gene_ids = next(exclusions, (None, set()))
            return result
        return set()

    parquet = pq.ParquetFile(manifest["expression_file"])
    columns = [
        manifest["row_id_column"],
        manifest["gene_column"],
        manifest["value_column"],
    ]
    current_id: str | None = None
    current_gene_ids: list[np.ndarray] = []
    current_values: list[np.ndarray] = []
    matched = 0

    def flush() -> None:
        nonlocal current_id, current_gene_ids, current_values, matched
        if current_id is None:
            return
        row = selected_cells.get(current_id)
        if row is not None:
            gene_ids = np.concatenate(current_gene_ids)
            values = np.concatenate(current_values).astype(np.float32, copy=False)
            excluded = exclusions_for(current_id)
            valid = np.isfinite(values) & (values > 0)
            if excluded:
                valid &= ~np.isin(gene_ids, np.fromiter(excluded, dtype=np.int64))
            writer.append(_cell_record(row, gene_ids[valid], values[valid]))
            matched += 1
        current_id = None
        current_gene_ids = []
        current_values = []

    previous_id: str | None = None
    for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
        ids = np.asarray(batch.column(0).to_pylist(), dtype=object)
        if not len(ids):
            continue
        if previous_id is not None and str(ids[0]) < previous_id:
            raise ValueError("NIPS long-table parquet must be ordered by obs_id")
        if np.any(ids[1:] < ids[:-1]):
            raise ValueError("NIPS long-table parquet must be ordered by obs_id")
        encoded = pc.dictionary_encode(batch.column(1))
        dictionary_ids = np.asarray(
            [gene_to_id[str(value)] for value in encoded.dictionary.to_pylist()],
            dtype=np.int64,
        )
        gene_ids = dictionary_ids[np.asarray(encoded.indices)]
        values = np.asarray(batch.column(2), dtype=np.float32)
        boundaries = np.flatnonzero(ids[1:] != ids[:-1]) + 1
        starts = np.concatenate(([0], boundaries))
        stops = np.concatenate((boundaries, [len(ids)]))
        for start, stop in zip(starts, stops):
            obs_id = str(ids[start])
            if current_id is not None and obs_id != current_id:
                flush()
            current_id = obs_id
            current_gene_ids.append(gene_ids[start:stop])
            current_values.append(values[start:stop])
        previous_id = str(ids[-1])
    flush()
    if matched != len(selected_cells):
        raise RuntimeError(
            f"NIPS long-table produced {matched} selected cells; "
            f"expected {len(selected_cells)}"
        )


def _write_sciplex(manifest: dict, cells, writer: _ShardWriter) -> None:
    from scipy import sparse
    from scipy.io import mmread

    try:
        matrix = mmread(manifest["expression_file"], spmatrix=True)
    except TypeError:
        matrix = mmread(manifest["expression_file"])
    if manifest["orientation"] == "gene_by_cell":
        matrix = matrix.T
    matrix = sparse.csr_matrix(matrix, dtype=np.float32)
    if matrix.shape[0] != len(cells):
        raise RuntimeError(
            f"SciPlex expression rows ({matrix.shape[0]}) do not match metadata rows ({len(cells)})"
        )
    for index, row in enumerate(cells.itertuples(index=False)):
        if not bool(row.selected):
            continue
        vector = matrix.getrow(index)
        valid = np.isfinite(vector.data) & (vector.data > 0)
        writer.append(_cell_record(row, vector.indices[valid], vector.data[valid]))


def _canonical_source(
    contract: dict[str, Any], output_dir: Path, *, batch_size: int = 256,
    shard_size: int = 20_000,
) -> Path:
    root = Path(contract.get("canonical_source") or output_dir / "_source")
    marker = root / "_SUCCESS"
    if marker.is_file():
        return root
    _, pq = require_pyarrow()
    manifest = json.loads(
        Path(contract["native_manifest"]).read_text(encoding="utf-8")
    )
    cells = pq.read_table(manifest["cells_file"]).to_pandas()
    selected = set(str(value) for value in contract["populations"])
    cells["selected"] = (
        cells["population"].astype(str).isin(selected) & cells["included"].astype(bool)
    )
    if not cells["selected"].any():
        raise ValueError("The selected project populations contain no source cells")
    data_dir = root / "data"
    metadata_dir = root / "metadata"
    data_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    writer = _ShardWriter(data_dir, shard_size)
    if manifest["kind"] == "nips_long_parquet":
        _write_nips_long(manifest, cells, writer, max(batch_size, 262_144))
    elif manifest["kind"] == "nips_parquet":
        _write_nips(manifest, cells, writer, batch_size)
    elif manifest["kind"] == "sciplex_mtx":
        _write_sciplex(manifest, cells, writer)
    else:
        raise ValueError(f"Unsupported native atlas manifest kind: {manifest['kind']}")
    writer.flush()
    genes = pq.read_table(manifest["genes_file"]).to_pylist()
    pa, pq = require_pyarrow()
    pq.write_table(
        pa.Table.from_pylist([
            {"token_id": int(row["source_gene_id"]), "gene_symbol": str(row["gene_symbol"])}
            for row in genes
        ]),
        metadata_dir / "gene_metadata.parquet",
        compression="zstd",
    )
    selected_cells = cells.loc[cells["selected"]]
    pq.write_table(
        pa.Table.from_pylist(_sample_rows(selected_cells)),
        metadata_dir / "sample_metadata.parquet",
        compression="zstd",
    )
    (root / "canonical_manifest.json").write_text(
        json.dumps({
            "format": "map_canonical_cells_v1",
            "dataset": contract["dataset"],
            "populations": sorted(selected),
            "cells": int(cells["selected"].sum()),
            "genes": len(genes),
            "shards": writer.index,
        }, indent=2),
        encoding="utf-8",
    )
    marker.write_text("ok\n", encoding="utf-8")
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
        raise ValueError(f"Native atlas reader does not support stage {stage!r}")
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
