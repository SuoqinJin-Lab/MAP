from __future__ import annotations

import ast
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def require_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        return pa, pq
    except ImportError as exc:
        raise RuntimeError("Tahoe streaming requires pyarrow") from exc


def raw_files(raw_dir: Path) -> list[Path]:
    files = sorted((Path(raw_dir) / "data").glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet shards under {Path(raw_dir) / 'data'}")
    return files


def read_sample_metadata(raw_dir: Path) -> dict[str, dict[str, Any]]:
    _, pq = require_pyarrow()
    path = Path(raw_dir) / "metadata" / "sample_metadata.parquet"
    if not path.is_file():
        raise FileNotFoundError(path)
    records = pq.read_table(path).to_pylist()
    return {str(row["sample"]): row for row in records if row.get("sample") is not None}


def parse_drug_dose(value: Any) -> tuple[str, float, str]:
    try:
        parsed = ast.literal_eval(str(value))
        drug, dose, unit = parsed[0]
        return " ".join(str(drug).split()), float(dose), str(unit).strip()
    except (ValueError, SyntaxError, IndexError, TypeError):
        return "", float("nan"), ""


class TahoeReader:
    """Stream Tahoe rows while keeping the raw nested expression shape."""

    def __init__(self, raw_dir: Path):
        self.raw_dir = Path(raw_dir)
        self.files = raw_files(self.raw_dir)
        self._pa, self._pq = require_pyarrow()

    @property
    def total_rows(self) -> int:
        return sum(self._pq.ParquetFile(path).metadata.num_rows for path in self.files)

    @property
    def gene_count(self) -> int:
        path = self.raw_dir / "metadata" / "gene_metadata.parquet"
        if not path.is_file():
            return -1
        return self._pq.ParquetFile(path).metadata.num_rows

    def iter_batches(self, columns: list[str], batch_size: int = 4096) -> Iterator[tuple[Path, int, dict[str, list[Any]]]]:
        for path in self.files:
            yield from self.iter_file_batches(path, columns, batch_size)

    def iter_file_batches(self, path: Path, columns: list[str], batch_size: int = 4096) -> Iterator[tuple[Path, int, dict[str, list[Any]]]]:
        parquet = self._pq.ParquetFile(path)
        for offset, batch in enumerate(parquet.iter_batches(batch_size=batch_size, columns=columns)):
            yield path, offset * batch_size, self._pa.Table.from_batches([batch]).to_pydict()

    def gene_symbols(self) -> dict[int, str]:
        _, pq = require_pyarrow()
        path = self.raw_dir / "metadata" / "gene_metadata.parquet"
        if not path.is_file():
            return {}
        rows = pq.read_table(path).to_pylist()
        token_col = "token_id" if rows and "token_id" in rows[0] else "gene_id"
        symbol_col = "gene_symbol" if rows and "gene_symbol" in rows[0] else "gene_name"
        return {int(row[token_col]): str(row[symbol_col]) for row in rows if row.get(token_col) is not None}

    def schema_report(self) -> dict[str, Any]:
        # Parquet ``schema.names`` returns repeated leaf names (for Tahoe this
        # includes two ``element`` fields); schema_arrow exposes the real
        # top-level columns such as genes and expressions.
        schema = self._pq.ParquetFile(self.files[0]).schema_arrow.names
        return {
            "shards": len(self.files),
            "rows": self.total_rows,
            "gene_count": self.gene_count,
            "first_shard_columns": schema,
        }
