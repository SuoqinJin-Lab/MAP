"""Self-contained Tahoe reader for the generic MAP preparation contract."""

from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import json
import math
import os
import pickle
import random
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch

from ...._common.hvg import hvg_fingerprint


CELL_LINES = (
    "CVCL_0131",  # A-172
    "CVCL_1056",  # A-498
    "CVCL_0023",  # A549
    "CVCL_1098",  # HepG2/C3A
    "CVCL_0480",  # PANC-1
    "CVCL_0069",  # SK-MEL-2
)
CELL_LINE_NAMES = {
    "CVCL_0131": "A-172",
    "CVCL_1056": "A-498",
    "CVCL_0023": "A549",
    "CVCL_1098": "HepG2/C3A",
    "CVCL_0480": "PANC-1",
    "CVCL_0069": "SK-MEL-2",
}
CONTROL_DRUG = "DMSO_TF"
N_GENE_TOKENS = 2047
N_HVG = 2000
TARGET_SUM = 10_000.0
SPECIAL_TOKEN = 3
SEURAT_SPAN = 0.3

_WORKER = {}


def _dump_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _canonical_drug(value: str) -> str:
    return " ".join(str(value).strip().split())


def _parse_drug_dose(value: str, fallback_drug: str) -> tuple[str, float, str]:
    try:
        parsed = ast.literal_eval(value)
        drug, dose, unit = parsed[0]
        return _canonical_drug(drug), float(dose), str(unit).strip()
    except Exception:
        return _canonical_drug(fallback_drug), float("nan"), "uM"


def _condition_key(cell_line, drug, dose, unit, smiles):
    return (cell_line, drug, float(dose), unit, smiles)


def _condition_key_text(key) -> str:
    cell_line, drug, dose, unit, smiles = key
    return json.dumps(
        [cell_line, drug, float(dose), unit, smiles],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _condition_key_from_text(value: str):
    cell_line, drug, dose, unit, smiles = json.loads(value)
    return _condition_key(cell_line, drug, dose, unit, smiles)


def _control_key(cell_line: str) -> tuple:
    """Synthetic key used only for deterministic control-cell sampling."""
    return _condition_key(cell_line, "__CONTROL__", 0.0, "uM", "")


def _raw_files(raw_dir: Path) -> list[str]:
    files = sorted(str(path) for path in (raw_dir / "data").glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Tahoe parquet shards under {raw_dir / 'data'}")
    return files


def _filter_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "manifest": output_dir / "condition_filter.json",
        "table": output_dir / "condition_filter.parquet",
        "offsets": output_dir / "condition_filter_file_offsets.npy",
        "ids": output_dir / "condition_filter_condition_ids.npy",
        "counts": output_dir / "condition_filter_file_counts.npy",
        "quotas": output_dir / "condition_filter_file_quotas.npy",
    }


def _stable_seed(seed: int, *values: int) -> np.random.SeedSequence:
    return np.random.SeedSequence([int(seed), *(int(value) for value in values)])


def _condition_seed_words(key: tuple) -> tuple[int, ...]:
    digest = hashlib.sha256(_condition_key_text(key).encode("utf-8")).digest()
    return tuple(int(value) for value in np.frombuffer(digest[:16], dtype="<u4"))


def _allocate_condition_quota(
    counts: np.ndarray, retained: int, seed: int, key: tuple
) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.int64)
    total = int(counts.sum())
    retained = int(retained)
    if retained < 0 or retained > total:
        raise ValueError(f"Invalid retained cell count {retained} for source count {total}")
    if retained == total:
        return counts.copy()
    rng = np.random.default_rng(_stable_seed(seed, *_condition_seed_words(key)))
    if hasattr(rng, "multivariate_hypergeometric"):
        return np.asarray(
            rng.multivariate_hypergeometric(counts, retained), dtype=np.int64
        )
    output = np.zeros_like(counts)
    remaining_cells = total
    remaining_draws = retained
    for index, available in enumerate(counts[:-1]):
        output[index] = rng.hypergeometric(
            int(available), int(remaining_cells - available), int(remaining_draws)
        )
        remaining_draws -= int(output[index])
        remaining_cells -= int(available)
    if len(output):
        output[-1] = remaining_draws
    return output


def _load_condition_filter(output_dir: Path) -> dict:
    paths = _filter_paths(output_dir)
    if not paths["manifest"].is_file():
        raise FileNotFoundError(
            "Condition filter is missing; run preprocess.<dataset>.filter_conditions() "
            "after selecting populations"
        )
    manifest = _load_json(paths["manifest"])
    state = {
        "seed": int(manifest["seed"]),
        "source_to_index": {
            str(path): index for index, path in enumerate(manifest["source_parts"])
        },
        "key_to_id": {
            _condition_key_from_text(value): index
            for index, value in enumerate(manifest["condition_keys"])
        },
        "offsets": np.load(paths["offsets"], mmap_mode="r"),
        "ids": np.load(paths["ids"], mmap_mode="r"),
        "counts": np.load(paths["counts"], mmap_mode="r"),
        "quotas": np.load(paths["quotas"], mmap_mode="r"),
    }
    return state


def _file_condition_selector(file_path: str) -> dict | None:
    state = _WORKER.get("condition_filter")
    if state is None:
        return None
    file_index = state["source_to_index"].get(str(file_path))
    if file_index is None:
        raise KeyError(f"Source shard is absent from condition filter: {file_path}")
    start = int(state["offsets"][file_index])
    stop = int(state["offsets"][file_index + 1])
    selected: dict[int, frozenset[int] | None] = {}
    for condition_id, count, quota in zip(
        state["ids"][start:stop],
        state["counts"][start:stop],
        state["quotas"][start:stop],
    ):
        condition_id = int(condition_id)
        count = int(count)
        quota = int(quota)
        if quota == count:
            selected[condition_id] = None
        elif quota == 0:
            selected[condition_id] = frozenset()
        else:
            rng = np.random.default_rng(
                _stable_seed(state["seed"], file_index, condition_id)
            )
            selected[condition_id] = frozenset(
                int(value)
                for value in rng.choice(count, size=quota, replace=False)
            )
    return {
        "key_to_id": state["key_to_id"],
        "selected": selected,
        "seen": Counter(),
    }


def _condition_is_selected(selector: dict | None, key: tuple) -> bool:
    if selector is None:
        return True
    condition_id = selector["key_to_id"].get(key)
    if condition_id is None:
        return False
    occurrence = int(selector["seen"][condition_id])
    selector["seen"][condition_id] += 1
    positions = selector["selected"].get(condition_id, frozenset())
    return positions is None or occurrence in positions


def _control_is_selected(selector: dict | None, cell_line: str) -> bool:
    """Apply the control-cell sampling cap for a selected cell line."""
    if selector is None:
        return True
    key = _control_key(cell_line)
    if key not in selector["key_to_id"]:
        return False
    return _condition_is_selected(selector, key)


def _iter_selected_batches(file_path, columns, batch_size=4096, cell_lines=None):
    """Yield only paper cell lines without materializing a whole nested shard."""
    parquet = pq.ParquetFile(file_path)
    selected = pa.array(tuple(cell_lines or CELL_LINES))
    for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
        table = pa.Table.from_batches([batch])
        mask = pc.is_in(table["cell_line_id"], value_set=selected)
        if pc.any(mask).as_py():
            yield table.filter(mask).to_pydict()


def _build_gene_mapping(raw_dir: Path, esm_path: Path, output_dir: Path):
    gene_meta = pd.read_parquet(raw_dir / "metadata" / "gene_metadata.parquet")
    esm = torch.load(esm_path, map_location="cpu", weights_only=False)
    if not isinstance(esm, dict):
        raise TypeError("Expected the released ESM2 embedding file to be a gene-symbol dict")
    state_symbols = list(esm)
    symbol_to_state = {symbol: index for index, symbol in enumerate(state_symbols)}
    max_token = int(gene_meta["token_id"].max())
    token_to_state = np.full(max_token + 1, -1, dtype=np.int32)
    for row in gene_meta.itertuples(index=False):
        state_id = symbol_to_state.get(row.gene_symbol)
        if state_id is not None:
            token_to_state[int(row.token_id)] = state_id
    np.save(output_dir / "source_gene_to_state.npy", token_to_state)
    _dump_json(output_dir / "state_gene_symbols.json", state_symbols)
    mapped_state_ids = set(int(value) for value in token_to_state[token_to_state >= 0])
    missing_symbols = [
        symbol for index, symbol in enumerate(state_symbols) if index not in mapped_state_ids
    ]
    _dump_json(output_dir / "unmapped_state_gene_symbols.json", missing_symbols)
    if missing_symbols:
        print(
            f"warning: {len(missing_symbols)} STATE genes have no Tahoe token and will be masked",
            flush=True,
        )
    return token_to_state, state_symbols


def _load_sample_map(raw_dir: Path):
    metadata = pd.read_parquet(raw_dir / "metadata" / "sample_metadata.parquet")
    result = {}
    for row in metadata.itertuples(index=False):
        result[str(row.sample)] = _parse_drug_dose(row.drugname_drugconc, row.drug)
    return result


def _init_filter_worker(sample_map_path: str, cell_lines):
    with open(sample_map_path, "rb") as handle:
        _WORKER["sample_map"] = pickle.load(handle)
    _WORKER["cell_lines"] = tuple(cell_lines)


def _scan_filter_chunk(args):
    chunk_id, files = args
    sample_map = _WORKER["sample_map"]
    cell_lines = _WORKER["cell_lines"]
    output = []
    columns = ["drug", "sample", "cell_line_id", "canonical_smiles"]
    for file_path in files:
        counts = Counter()
        for table in _iter_selected_batches(file_path, columns, cell_lines=cell_lines):
            for raw_drug, sample, cell_line, smiles in zip(
                table["drug"], table["sample"], table["cell_line_id"],
                table["canonical_smiles"],
            ):
                fallback = _canonical_drug(raw_drug)
                drug, dose, unit = sample_map.get(
                    str(sample), (fallback, float("nan"), "uM")
                )
                is_control = drug == CONTROL_DRUG or fallback == CONTROL_DRUG
                valid_smiles = smiles is not None and str(smiles).strip() not in {
                    "", "nan", "None"
                }
                if is_control:
                    counts[_control_key(cell_line)] += 1
                    continue
                if not valid_smiles or not math.isfinite(dose):
                    continue
                counts[
                    _condition_key(cell_line, drug, dose, unit, str(smiles))
                ] += 1
        output.append((file_path, counts))
        gc.collect()
        pa.default_memory_pool().release_unused()
    return chunk_id, output


def run_filter(args):
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = _raw_files(raw_dir)
    cell_lines = tuple(getattr(args, "cell_lines", None) or CELL_LINES)
    min_cells = int(getattr(args, "min_cells", 500))
    max_cells = int(getattr(args, "max_cells", 5000))
    seed = int(getattr(args, "seed", 42))
    overwrite = bool(getattr(args, "overwrite", False))
    workers = int(args.workers)
    if min_cells <= 0 or max_cells <= 0:
        raise ValueError("min_cells and max_cells must be positive")
    if min_cells > max_cells:
        raise ValueError("min_cells cannot exceed max_cells")
    if workers <= 0:
        raise ValueError("workers must be positive")

    print(
        f"[filter] start: {len(files):,} shards, "
        f"{len(cell_lines)} cell lines, workers={workers}",
        flush=True,
    )

    paths = _filter_paths(output_dir)
    if paths["manifest"].is_file():
        existing = _load_json(paths["manifest"])
        requested = {
            "format": "map_population_condition_filter_v2",
            "populations": list(cell_lines),
            "min_cells": min_cells,
            "max_cells": max_cells,
            "seed": seed,
            "source_parts": files,
        }
        if all(existing.get(key) == value for key, value in requested.items()):
            print(f"condition filter reused: {existing['filter_id']}")
            return
        downstream = (
            output_dir / "stats_manifest.json",
            output_dir / "hvg.json",
            output_dir / "materialized_shapes.json",
        )
        if any(path.is_file() for path in downstream):
            raise FileExistsError(
                "A different condition filter cannot replace preparation outputs; "
                "use a new project"
            )
        if not overwrite:
            raise FileExistsError(
                "A different condition filter already exists; use a new project or pass overwrite=True"
            )

    sample_map_path = output_dir / "condition_filter_sample_map.pkl"
    try:
        with sample_map_path.open("wb") as handle:
            pickle.dump(_load_sample_map(raw_dir), handle)
        results = _run_parallel(
            files,
            workers,
            _scan_filter_chunk,
            _init_filter_worker,
            (str(sample_map_path), cell_lines),
        )
    finally:
        # This pickle is only an IPC payload for the filter scan.
        sample_map_path.unlink(missing_ok=True)
    by_file = {}
    for _, chunk in results:
        by_file.update(dict(chunk))
    file_counts = [by_file[path] for path in files]
    total_counts = Counter()
    for counts in file_counts:
        total_counts.update(counts)

    # The 500-cell eligibility rule applies to perturbation drug-dose groups.
    # Controls are always retained, with the same deterministic 5,000-cell cap
    # applied independently for each population.
    perturbation_counts = Counter(
        {key: count for key, count in total_counts.items() if key[1] != "__CONTROL__"}
    )
    control_keys = sorted(
        (key for key in total_counts if key[1] == "__CONTROL__"),
        key=_condition_key_text,
    )
    retained_keys = sorted(
        (key for key, count in perturbation_counts.items() if int(count) >= min_cells),
        key=_condition_key_text,
    )
    if not retained_keys:
        raise ValueError(
            f"No drug-dose condition has at least {min_cells} cells in the selected populations"
        )
    # Keep condition ids stable and separate from the control-only sampling
    # entries.  The latter are not written to conditions.parquet.
    all_filter_keys = retained_keys + control_keys
    key_to_id = {key: index for index, key in enumerate(all_filter_keys)}
    quotas_by_file = [dict() for _ in files]
    counts_by_file = [dict() for _ in files]
    table_rows = []
    retained_cells = 0
    retained_control_cells = 0
    retained_controls_by_population = Counter()
    retained_by_file_population = [Counter() for _ in files]
    for key in all_filter_keys:
        counts = np.asarray([item.get(key, 0) for item in file_counts], dtype=np.int64)
        source_count = int(counts.sum())
        keep = min(source_count, max_cells)
        quotas = _allocate_condition_quota(counts, keep, seed, key)
        condition_id = key_to_id[key]
        for file_index in np.flatnonzero(counts):
            counts_by_file[int(file_index)][condition_id] = int(counts[file_index])
            quotas_by_file[int(file_index)][condition_id] = int(quotas[file_index])
        population, drug, dose, unit, smiles = key
        if drug == "__CONTROL__":
            retained_control_cells += keep
            retained_controls_by_population[str(population)] += keep
        else:
            table_rows.append({
                "condition_id": int(condition_id),
                "population": str(population),
                "drug": str(drug),
                "dose": float(dose),
                "dose_unit": str(unit),
                "canonical_smiles": str(smiles),
                "source_cells": source_count,
                "retained_cells": keep,
                "capped": bool(source_count > max_cells),
            })
            retained_cells += keep
        for file_index, quota in enumerate(quotas):
            retained_by_file_population[file_index][str(population)] += int(quota)

    offsets = [0]
    flat_ids = []
    flat_counts = []
    flat_quotas = []
    for counts, quotas in zip(counts_by_file, quotas_by_file):
        for condition_id in sorted(counts):
            flat_ids.append(condition_id)
            flat_counts.append(counts[condition_id])
            flat_quotas.append(quotas[condition_id])
        offsets.append(len(flat_ids))
    np.save(paths["offsets"], np.asarray(offsets, dtype=np.int64))
    np.save(paths["ids"], np.asarray(flat_ids, dtype=np.int32))
    np.save(paths["counts"], np.asarray(flat_counts, dtype=np.int32))
    np.save(paths["quotas"], np.asarray(flat_quotas, dtype=np.int32))
    pq.write_table(pa.Table.from_pylist(table_rows), paths["table"], compression="zstd")

    configuration = {
        "format": "map_population_condition_filter_v2",
        "populations": list(cell_lines),
        "min_cells": min_cells,
        "max_cells": max_cells,
        "seed": seed,
        "source_parts": files,
    }
    digest = hashlib.sha256(
        json.dumps(configuration, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    source_cells = int(sum(perturbation_counts.values()))
    source_control_cells = int(sum(total_counts[key] for key in control_keys))
    retained_by_population = Counter()
    for row in table_rows:
        retained_by_population[row["population"]] += int(row["retained_cells"])
    manifest = {
        "filter_id": (
            f"population-condition__min-{min_cells}__max-{max_cells}__seed-{seed}__cfg-{digest}"
        ),
        **configuration,
        "grouping_fields": [
            "population", "drug", "dose", "dose_unit", "canonical_smiles"
        ],
        "source_conditions": len(perturbation_counts),
        "retained_conditions": len(retained_keys),
        "dropped_conditions": len(perturbation_counts) - len(retained_keys),
        "capped_conditions": int(sum(row["capped"] for row in table_rows)),
        "source_condition_cells": source_cells,
        "retained_condition_cells": int(retained_cells),
        "dropped_condition_cells": source_cells - int(retained_cells),
        "retained_condition_cells_by_population": dict(sorted(retained_by_population.items())),
        "source_control_cells": source_control_cells,
        "retained_control_cells": int(retained_control_cells),
        "retained_control_cells_by_population": dict(
            sorted(
                (population, int(retained_controls_by_population[population]))
                for population in cell_lines
            )
        ),
        "retained_cells_by_source_part_population": [
            dict(sorted(counter.items())) for counter in retained_by_file_population
        ],
        # condition_keys is the complete sampling-key registry; the parquet
        # table contains perturbation conditions only.
        "condition_keys": [_condition_key_text(key) for key in all_filter_keys],
        "perturbation_keys": [_condition_key_text(key) for key in retained_keys],
        "condition_table": paths["table"].name,
    }
    _dump_json(paths["manifest"], manifest)
    print(
        "condition filter complete: "
        f"{len(retained_keys):,}/{len(perturbation_counts):,} conditions, "
        f"{retained_cells:,}/{source_cells:,} condition cells"
    )


def _init_worker(
    mapping_path: str,
    sample_map_path: str,
    cell_lines,
    output_dir,
    hvg_sample_plan_path: str | None = None,
):
    _WORKER["mapping"] = np.load(mapping_path, mmap_mode="r")
    _WORKER["n_genes"] = len(_load_json(Path(mapping_path).parent / "state_gene_symbols.json"))
    with open(sample_map_path, "rb") as handle:
        _WORKER["sample_map"] = pickle.load(handle)
    _WORKER["cell_lines"] = tuple(cell_lines)
    _WORKER["cell_to_index"] = {value: index for index, value in enumerate(cell_lines)}
    _WORKER["condition_filter"] = _load_condition_filter(Path(output_dir))
    if hvg_sample_plan_path is not None:
        with open(hvg_sample_plan_path, "rb") as handle:
            _WORKER["hvg_sample_plan"] = pickle.load(handle)
    else:
        _WORKER.pop("hvg_sample_plan", None)


def _valid_gene_values(gene_ids, expressions, mapping):
    raw_ids = np.asarray(gene_ids, dtype=np.int64)
    raw_values = np.asarray(expressions, dtype=np.float64)
    in_range = (raw_ids >= 0) & (raw_ids < mapping.shape[0])
    raw_ids = raw_ids[in_range]
    raw_values = raw_values[in_range]
    state_ids = np.asarray(mapping[raw_ids], dtype=np.int32)
    valid = (state_ids >= 0) & np.isfinite(raw_values) & (raw_values > 0)
    return state_ids[valid], raw_values[valid]


def _scan_stats_chunk(args):
    chunk_id, files = args
    mapping = _WORKER["mapping"]
    sample_map = _WORKER["sample_map"]
    cell_lines = _WORKER["cell_lines"]
    cell_to_index = _WORKER["cell_to_index"]
    n_genes = _WORKER["n_genes"]
    sample_plan = _WORKER["hvg_sample_plan"]
    n_batches = len(cell_lines)

    sums = np.zeros(n_genes, dtype=np.float64)
    sumsq = np.zeros(n_genes, dtype=np.float64)
    batch_sums = np.zeros((n_batches, n_genes), dtype=np.float64)
    batch_sumsq = np.zeros((n_batches, n_genes), dtype=np.float64)
    batch_counts = np.zeros(n_batches, dtype=np.int64)
    maxima = np.zeros(n_genes, dtype=np.float64)
    n_cells = 0
    file_counts = []
    condition_counts = Counter()
    excluded_missing_smiles = 0

    columns = [
        "genes",
        "expressions",
        "drug",
        "sample",
        "cell_line_id",
        "canonical_smiles",
        "plate",
    ]
    for file_path in files:
        counts = np.zeros(len(cell_lines), dtype=np.int64)
        population_seen = Counter()
        file_plan = sample_plan.get(file_path, {})
        selector = _file_condition_selector(file_path)
        for table in _iter_selected_batches(file_path, columns, cell_lines=cell_lines):
            for genes, expressions, raw_drug, sample, cell_line, smiles, plate in zip(
                table["genes"],
                table["expressions"],
                table["drug"],
                table["sample"],
                table["cell_line_id"],
                table["canonical_smiles"],
                table["plate"],
            ):
                fallback = _canonical_drug(raw_drug)
                drug, dose, unit = sample_map.get(str(sample), (fallback, float("nan"), "uM"))
                is_control = drug == CONTROL_DRUG or fallback == CONTROL_DRUG
                valid_smiles = smiles is not None and str(smiles).strip() not in {"", "nan", "None"}
                if not is_control and (not valid_smiles or not math.isfinite(dose)):
                    excluded_missing_smiles += 1
                    continue
                if is_control:
                    if not _control_is_selected(selector, cell_line):
                        continue
                elif not _condition_is_selected(
                    selector, _condition_key(cell_line, drug, dose, unit, str(smiles))
                ):
                    continue
                population_index = cell_to_index[cell_line]
                occurrence = int(population_seen[cell_line])
                population_seen[cell_line] += 1
                selected_for_hvg = occurrence in file_plan.get(cell_line, frozenset())
                if not selected_for_hvg:
                    counts[population_index] += 1
                    if is_control:
                        condition_counts[(cell_line, "__CONTROL__", 0.0, "uM", "", str(plate))] += 1
                    else:
                        condition_counts[(cell_line, drug, dose, unit, str(smiles), str(plate))] += 1
                    continue
                n_cells += 1
                state_ids, values = _valid_gene_values(genes, expressions, mapping)
                if state_ids.size:
                    sums += np.bincount(state_ids, weights=values, minlength=n_genes)
                    sumsq += np.bincount(state_ids, weights=values * values, minlength=n_genes)
                    np.maximum.at(maxima, state_ids, values)
                batch_sums[population_index] += np.bincount(
                    state_ids, weights=values, minlength=n_genes
                )
                batch_sumsq[population_index] += np.bincount(
                    state_ids, weights=values * values, minlength=n_genes
                )
                batch_counts[population_index] += 1
                counts[population_index] += 1
                if is_control:
                    condition_counts[(cell_line, "__CONTROL__", 0.0, "uM", "", str(plate))] += 1
                else:
                    condition_counts[(cell_line, drug, dose, unit, str(smiles), str(plate))] += 1
        file_counts.append((file_path, counts.tolist()))
        gc.collect()
        pa.default_memory_pool().release_unused()

    return {
        "chunk_id": chunk_id,
        "sums": sums,
        "sumsq": sumsq,
        "batch_sums": batch_sums,
        "batch_sumsq": batch_sumsq,
        "batch_counts": batch_counts,
        "maxima": maxima,
        "n_cells": n_cells,
        "file_counts": file_counts,
        "condition_counts": condition_counts,
        "excluded_missing_smiles": excluded_missing_smiles,
    }


def _chunks(items, n_chunks):
    return [items[index::n_chunks] for index in range(n_chunks) if items[index::n_chunks]]


def _run_parallel(files, workers, function, initializer, initargs, extra=None):
    chunks = _chunks(files, min(workers, len(files)))
    tasks = [(index, chunk) if extra is None else (index, chunk, extra) for index, chunk in enumerate(chunks)]
    results = []
    label = function.__name__.strip("_")
    print(
        f"[{label}] scanning {len(files):,} parquet shards in "
        f"{len(chunks)} worker chunks",
        flush=True,
    )
    with ProcessPoolExecutor(
        max_workers=len(chunks),
        initializer=initializer,
        initargs=initargs,
    ) as pool:
        futures = [pool.submit(function, task) for task in tasks]
        completed = 0
        for future in as_completed(futures):
            results.append(future.result())
            completed += 1
            print(
                f"[{label}] chunks {completed}/{len(futures)} complete "
                f"({completed / len(futures):.0%})",
                flush=True,
            )
    return results


def _build_hvg_sample_plan(
    output_dir: Path,
    files: list[str],
    cell_lines: tuple[str, ...],
    *,
    max_cells_per_population: int = 10_000,
    seed: int = 42,
) -> dict[str, dict[str, frozenset[int]]]:
    """Allocate a deterministic, balanced HVG sample across source shards."""
    filter_manifest = _load_json(output_dir / "condition_filter.json")
    by_part = filter_manifest.get("retained_cells_by_source_part_population")
    if not by_part or len(by_part) != len(files):
        raise RuntimeError(
            "Condition filter is missing per-shard population quotas; rerun "
            "fetch_cell_line before computing statistics."
        )
    counts_by_population = {
        population: np.asarray(
            [int(part.get(population, 0)) for part in by_part], dtype=np.int64
        )
        for population in cell_lines
    }
    plan: dict[str, dict[str, frozenset[int]]] = {str(path): {} for path in files}
    for population, counts in counts_by_population.items():
        total = int(counts.sum())
        keep = min(total, int(max_cells_per_population))
        if keep <= 0:
            continue
        quotas = _allocate_condition_quota(
            counts,
            keep,
            seed,
            _condition_key(population, "__HVG_SAMPLE__", 0.0, "uM", ""),
        )
        for file_index, (count, quota) in enumerate(zip(counts, quotas)):
            if int(quota) == 0:
                continue
            rng = np.random.default_rng(
                _stable_seed(seed, file_index, *_condition_seed_words(
                    _condition_key(population, "__HVG_FILE__", file_index, "uM", "")
                ))
            )
            positions = frozenset(
                int(value) for value in rng.choice(int(count), size=int(quota), replace=False)
            )
            plan[str(files[file_index])][population] = positions
    return plan


def run_stats(args):
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _load_condition_filter(output_dir)
    files = _raw_files(raw_dir)
    selected_cell_lines = tuple(getattr(args, "cell_lines", None) or CELL_LINES)
    print(
        f"[stats] start: {len(files):,} shards, "
        f"{len(selected_cell_lines)} cell lines, workers={args.workers}",
        flush=True,
    )
    mapping_path = output_dir / "source_gene_to_state.npy"
    if not mapping_path.exists():
        _build_gene_mapping(raw_dir, Path(args.esm_embeddings), output_dir)
    sample_map = _load_sample_map(raw_dir)
    sample_map_path = output_dir / "preparation_state.pkl"
    with sample_map_path.open("wb") as handle:
        pickle.dump(sample_map, handle)

    sample_plan = _build_hvg_sample_plan(
        output_dir,
        files,
        selected_cell_lines,
        max_cells_per_population=int(getattr(args, "hvg_cells_per_population", 10_000)),
        seed=int(getattr(args, "seed", 42)),
    )
    sample_plan_path = output_dir / "hvg_sample_plan.pkl"
    with sample_plan_path.open("wb") as handle:
        pickle.dump(sample_plan, handle)

    results = _run_parallel(
        files,
        args.workers,
        _scan_stats_chunk,
        _init_worker,
        (
            str(mapping_path), str(sample_map_path), selected_cell_lines,
            str(output_dir),
            str(sample_plan_path),
        ),
    )
    n_genes = len(_load_json(output_dir / "state_gene_symbols.json"))
    sums = np.zeros(n_genes, dtype=np.float64)
    sumsq = np.zeros(n_genes, dtype=np.float64)
    batch_sums = np.zeros((len(selected_cell_lines), n_genes), dtype=np.float64)
    batch_sumsq = np.zeros((len(selected_cell_lines), n_genes), dtype=np.float64)
    batch_counts = np.zeros(len(selected_cell_lines), dtype=np.int64)
    maxima = np.zeros(n_genes, dtype=np.float64)
    n_cells = 0
    condition_counts = Counter()
    file_count_map = {}
    excluded = 0
    for result in results:
        sums += result["sums"]
        sumsq += result["sumsq"]
        batch_sums += result["batch_sums"]
        batch_sumsq += result["batch_sumsq"]
        batch_counts += result["batch_counts"]
        maxima = np.maximum(maxima, result["maxima"])
        n_cells += result["n_cells"]
        condition_counts.update(result["condition_counts"])
        file_count_map.update(dict(result["file_counts"]))
        excluded += result["excluded_missing_smiles"]

    np.savez(
        output_dir / "global_count_stats.npz",
        sums=sums,
        sumsq=sumsq,
        maxima=maxima,
        n_cells=np.asarray(n_cells, dtype=np.int64),
    )
    np.savez(
        output_dir / "batch_count_stats.npz",
        sums=batch_sums,
        sumsq=batch_sumsq,
        n_cells=batch_counts,
        populations=np.asarray(selected_cell_lines),
    )
    file_counts = np.asarray([file_count_map[path] for path in files], dtype=np.int64)
    np.save(output_dir / "eligible_file_cell_counts.npy", file_counts)
    conditions = [
        {"key": json.dumps(list(key), ensure_ascii=False), "count": int(count)}
        for key, count in sorted(condition_counts.items())
    ]
    _dump_json(output_dir / "condition_group_census.json", conditions)
    _dump_json(
        output_dir / "stats_manifest.json",
        {
            "filter_id": _load_json(output_dir / "condition_filter.json")["filter_id"],
            "populations": list(selected_cell_lines),
            "source_parts": files,
            "n_hvg_model_cells": int(n_cells),
            "hvg_cells_per_population": int(getattr(args, "hvg_cells_per_population", 10_000)),
            "hvg_sampled_cells_by_population": {
                str(population): int(batch_counts[index])
                for index, population in enumerate(selected_cell_lines)
            },
            "eligible_cells_by_population": {
                cell_line: int(file_counts[:, index].sum())
                for index, cell_line in enumerate(selected_cell_lines)
            },
            "excluded_perturbation_cells_without_smiles_or_dose": int(excluded),
            "target_sum": TARGET_SUM,
            "pad_length": N_GENE_TOKENS + 1,
            "num_hvg": N_HVG,
            "hvg_flavor": "seurat_v3",
            # Statistics are collected per population so the default HVG
            # protocol can reproduce the cell-line-aware shared gene space.
            # An explicit global selection updates this field in run_hvg().
            "hvg_batch_key": "population",
            "condition_filter": _load_json(output_dir / "condition_filter.json"),
        },
    )
    print(f"stats complete: {n_cells:,} cells across {len(selected_cell_lines)} cell lines")


def _fit_seurat_clip(output_dir: Path, span: float = SEURAT_SPAN):
    stats = np.load(output_dir / "global_count_stats.npz")
    n_cells = int(stats["n_cells"])
    means = stats["sums"] / n_cells
    variances = (stats["sumsq"] - n_cells * means * means) / max(n_cells - 1, 1)
    fitted = np.zeros_like(means)
    nonconstant = variances > 0
    if int(nonconstant.sum()) < 20:
        raise ValueError(
            "Seurat-v3 HVG selection needs at least 20 non-constant mapped genes; "
            f"found {int(nonconstant.sum())}"
        )
    x = np.log10(np.maximum(means[nonconstant], 1e-12))
    y = np.log10(np.maximum(variances[nonconstant], 1e-12))
    try:
        from skmisc.loess import loess

        model = loess(x, y, span=float(span), degree=2)
        model.fit()
        fitted[nonconstant] = model.outputs.fitted_values
    except ImportError:
        # scikit-misc is optional in the minimal environment.  The quadratic
        # trend is the same regression family and keeps the cache reproducible.
        degree = min(2, max(1, len(x) - 1))
        fitted[nonconstant] = np.polyval(np.polyfit(x, y, degree), x)
    reg_std = np.sqrt(10**fitted)
    clip_values = reg_std * np.sqrt(n_cells) + means
    np.savez(
        output_dir / "seurat_v3_model.npz",
        means=means,
        variances=variances,
        reg_std=reg_std,
        clip_values=clip_values,
        n_cells=np.asarray(n_cells, dtype=np.int64),
    )
    return clip_values


def _fit_batch_seurat_clips(output_dir: Path, span: float = SEURAT_SPAN):
    """Fit one Seurat-v3 variance trend per population batch."""
    stats = np.load(output_dir / "batch_count_stats.npz")
    sums = np.asarray(stats["sums"], dtype=np.float64)
    sumsq = np.asarray(stats["sumsq"], dtype=np.float64)
    counts = np.asarray(stats["n_cells"], dtype=np.int64)
    n_batches, n_genes = sums.shape
    clips = np.zeros((n_batches, n_genes), dtype=np.float64)
    means_out = np.zeros_like(clips)
    reg_std_out = np.ones_like(clips)
    for batch in range(n_batches):
        n_cells = int(counts[batch])
        if n_cells < 2:
            raise ValueError(
                f"Batch-aware HVG selection needs at least 2 cells in batch {batch}"
            )
        means = sums[batch] / n_cells
        variances = (sumsq[batch] - n_cells * means * means) / max(n_cells - 1, 1)
        nonconstant = (variances > 0) & np.isfinite(means) & (means > 0)
        if int(nonconstant.sum()) < 20:
            raise ValueError(
                "Seurat-v3 HVG selection needs at least 20 non-constant mapped genes; "
                f"batch {batch} has {int(nonconstant.sum())}"
            )
        x = np.log10(np.maximum(means[nonconstant], 1e-12))
        y = np.log10(np.maximum(variances[nonconstant], 1e-12))
        fitted = np.zeros(n_genes, dtype=np.float64)
        try:
            from skmisc.loess import loess

            model = loess(x, y, span=float(span), degree=2)
            model.fit()
            fitted[nonconstant] = model.outputs.fitted_values
        except ImportError:
            degree = min(2, max(1, len(x) - 1))
            fitted[nonconstant] = np.polyval(np.polyfit(x, y, degree), x)
        reg_std = np.sqrt(np.maximum(10.0 ** fitted, 1e-12))
        clip = reg_std * np.sqrt(n_cells) + means
        means_out[batch] = means
        reg_std_out[batch] = reg_std
        clips[batch] = clip
    np.savez(
        output_dir / "seurat_v3_batch_model.npz",
        means=means_out,
        reg_std=reg_std_out,
        clip_values=clips,
        n_cells=counts,
    )
    return clips


def _init_hvg_worker(
    mapping_path,
    clip_path,
    sample_map_path,
    cell_lines,
    output_dir,
    sample_plan_path,
):
    _WORKER["mapping"] = np.load(mapping_path, mmap_mode="r")
    _WORKER["clip"] = np.load(clip_path, mmap_mode="r")
    _WORKER["cell_lines"] = tuple(cell_lines)
    with open(sample_map_path, "rb") as handle:
        _WORKER["sample_map"] = pickle.load(handle)
    _WORKER["condition_filter"] = _load_condition_filter(Path(output_dir))
    with open(sample_plan_path, "rb") as handle:
        _WORKER["hvg_sample_plan"] = pickle.load(handle)


def _scan_clipped_chunk(task):
    chunk_id, files = task
    mapping = _WORKER["mapping"]
    clip = _WORKER["clip"]
    cell_lines = _WORKER["cell_lines"]
    cell_to_index = {value: index for index, value in enumerate(cell_lines)}
    sample_map = _WORKER["sample_map"]
    n_genes = clip.shape[-1]
    batch_mode = clip.ndim == 2
    n_batches = len(cell_lines)
    output_shape = (n_batches, n_genes) if batch_mode else (n_genes,)
    clipped_sum = np.zeros(output_shape, dtype=np.float64)
    clipped_sumsq = np.zeros(output_shape, dtype=np.float64)
    clipped_counts = np.zeros(n_batches, dtype=np.int64) if batch_mode else np.asarray(0, dtype=np.int64)
    sample_plan = _WORKER["hvg_sample_plan"]
    for file_path in files:
        selector = _file_condition_selector(file_path)
        population_seen = Counter()
        file_plan = sample_plan.get(file_path, {})
        columns = [
            "genes", "expressions", "cell_line_id", "drug", "sample", "canonical_smiles"
        ]
        for data in _iter_selected_batches(file_path, columns, cell_lines=cell_lines):
            for genes, expressions, cell_line, raw_drug, sample, smiles in zip(
                data["genes"], data["expressions"], data["cell_line_id"],
                data["drug"], data["sample"], data["canonical_smiles"],
            ):
                fallback = _canonical_drug(raw_drug)
                drug, dose, unit = sample_map.get(
                    str(sample), (fallback, float("nan"), "uM")
                )
                is_control = drug == CONTROL_DRUG or fallback == CONTROL_DRUG
                valid_smiles = smiles is not None and str(smiles).strip() not in {
                    "", "nan", "None"
                }
                if not is_control and (not valid_smiles or not math.isfinite(dose)):
                    continue
                if is_control:
                    if not _control_is_selected(selector, cell_line):
                        continue
                elif not _condition_is_selected(
                    selector, _condition_key(cell_line, drug, dose, unit, str(smiles))
                ):
                    continue
                occurrence = int(population_seen[cell_line])
                population_seen[cell_line] += 1
                if occurrence not in file_plan.get(cell_line, frozenset()):
                    continue
                state_ids, values = _valid_gene_values(genes, expressions, mapping)
                batch = cell_to_index[cell_line]
                if batch_mode:
                    clipped_counts[batch] += 1
                else:
                    clipped_counts += 1
                if state_ids.size:
                    clipped = np.minimum(
                        values, clip[batch, state_ids] if batch_mode else clip[state_ids]
                    )
                    partial_sum = np.bincount(
                        state_ids, weights=clipped, minlength=n_genes
                    )
                    partial_sumsq = np.bincount(
                        state_ids, weights=clipped * clipped, minlength=n_genes
                    )
                    if batch_mode:
                        clipped_sum[batch] += partial_sum
                        clipped_sumsq[batch] += partial_sumsq
                    else:
                        clipped_sum += partial_sum
                        clipped_sumsq += partial_sumsq
        gc.collect()
        pa.default_memory_pool().release_unused()
    return chunk_id, clipped_sum, clipped_sumsq, clipped_counts


def run_hvg(args):
    output_dir = Path(args.output_dir)
    raw_dir = Path(args.raw_dir)
    _load_condition_filter(output_dir)
    n_hvg = int(getattr(args, "n_top_genes", N_HVG))
    span = float(getattr(args, "seurat_span", SEURAT_SPAN))
    if n_hvg <= 0:
        raise ValueError("n_top_genes must be positive")
    requested_batch_key = getattr(args, "hvg_batch_key", "population")
    if requested_batch_key in (None, "", "none", "None"):
        requested_batch_key = None
    elif requested_batch_key != "population":
        raise ValueError("hvg_batch_key must be None or 'population'")
    batch_stats_path = output_dir / "batch_count_stats.npz"
    if requested_batch_key is None:
        clip = _fit_seurat_clip(output_dir, span)
        batch_mode = False
    else:
        if not batch_stats_path.is_file():
            raise RuntimeError(
                "Cell-line-aware HVG selection requires batch_count_stats.npz. "
                "Cell-line-aware HVG selection requires batch_count_stats.npz. "
                "Run data statistics before selecting HVGs."
            )
        clip = _fit_batch_seurat_clips(output_dir, span)
        batch_mode = True
    clip_path = output_dir / "seurat_clip_values.npy"
    np.save(clip_path, clip)
    files = _load_json(output_dir / "stats_manifest.json")["source_parts"]
    selected_cell_lines = tuple(
        _load_json(output_dir / "stats_manifest.json").get("populations", CELL_LINES)
    )
    print(
        f"[hvg] start: {len(files):,} shards, "
        f"{len(selected_cell_lines)} cell lines, workers={args.workers}",
        flush=True,
    )
    results = _run_parallel(
        files,
        args.workers,
        _scan_clipped_chunk,
        _init_hvg_worker,
        (
            str(output_dir / "source_gene_to_state.npy"),
            str(clip_path),
            str(output_dir / "preparation_state.pkl"),
            selected_cell_lines,
            str(output_dir),
            str(output_dir / "hvg_sample_plan.pkl"),
        ),
    )
    if batch_mode:
        stats = np.load(output_dir / "batch_count_stats.npz")
        means = np.asarray(stats["sums"], dtype=np.float64) / np.maximum(
            np.asarray(stats["n_cells"], dtype=np.int64)[:, None], 1
        )
        reg_std = np.asarray(np.load(output_dir / "seurat_v3_batch_model.npz")["reg_std"])
        clipped_sum = np.zeros_like(means)
        clipped_sumsq = np.zeros_like(means)
        clipped_counts = np.zeros(means.shape[0], dtype=np.int64)
        for _, partial_sum, partial_sumsq, partial_counts in results:
            clipped_sum += partial_sum
            clipped_sumsq += partial_sumsq
            clipped_counts += partial_counts
        per_batch_variance = (
            clipped_counts[:, None] * means * means
            + clipped_sumsq
            - 2.0 * clipped_sum * means
        ) / np.maximum(
            (clipped_counts[:, None] - 1) * np.square(reg_std), 1e-12
        )
        per_batch_variance[~np.isfinite(per_batch_variance)] = -np.inf
        ranks = np.empty_like(per_batch_variance, dtype=np.float64)
        selected = np.zeros_like(per_batch_variance, dtype=bool)
        state_ids = np.arange(per_batch_variance.shape[1])
        for batch in range(per_batch_variance.shape[0]):
            order = np.lexsort((state_ids, -per_batch_variance[batch]))
            ranks[batch, order] = np.arange(1, len(order) + 1)
            selected[batch, order[: min(n_hvg, len(order))]] = True
        n_batches = selected.sum(axis=0).astype(np.int32)
        median_rank = np.full(per_batch_variance.shape[1], np.inf, dtype=np.float64)
        for gene_id in np.flatnonzero(n_batches):
            median_rank[gene_id] = float(np.median(ranks[selected[:, gene_id], gene_id]))
        # scanpy flavor="seurat_v3" sorts by median within-batch rank first,
        # then by the number of populations in which the gene is top-ranked.
        ranked = np.lexsort((state_ids, -n_batches, median_rank))
        hvg_state_ids = ranked[:n_hvg].astype(np.int32)
        normalized_variance = np.mean(per_batch_variance, axis=0)
        hvg_nbatches = n_batches
    else:
        model = np.load(output_dir / "seurat_v3_model.npz")
        n_cells = int(model["n_cells"])
        clipped_sum = np.zeros_like(model["means"])
        clipped_sumsq = np.zeros_like(model["means"])
        for _, partial_sum, partial_sumsq, _ in results:
            clipped_sum += partial_sum
            clipped_sumsq += partial_sumsq
        means = model["means"]
        reg_std = model["reg_std"]
        normalized_variance = (
            n_cells * means * means
            + clipped_sumsq
            - 2.0 * clipped_sum * means
        ) / ((n_cells - 1) * np.square(reg_std))
        normalized_variance[~np.isfinite(normalized_variance)] = -np.inf
        state_ids = np.arange(normalized_variance.size)
        ranked = np.lexsort((state_ids, -normalized_variance))
        hvg_state_ids = ranked[:n_hvg].astype(np.int32)
        hvg_nbatches = None
    symbols = _load_json(output_dir / "state_gene_symbols.json")
    np.save(output_dir / "hvg_state_ids.npy", hvg_state_ids)
    np.save(output_dir / "seurat_v3_normalized_variance.npy", normalized_variance)
    # Record the exact HVG contract used for this project. ``population``
    # fits within-cell-line statistics and then merges one shared gene space.
    stats_manifest_path = output_dir / "stats_manifest.json"
    stats_manifest = _load_json(stats_manifest_path)
    stats_manifest["hvg_batch_key"] = requested_batch_key
    stats_manifest["hvg_protocol"] = (
        "cell_line_aware_shared_v1" if batch_mode else "global_seurat_v3_v1"
    )
    hvg = {
        "protocol": stats_manifest["hvg_protocol"],
        "flavor": "seurat_v3",
        "batch_key": requested_batch_key if batch_mode else None,
        "cells_per_batch": int(stats_manifest.get("hvg_cells_per_population", 10_000))
        if batch_mode else None,
        "n_top_genes": n_hvg,
        "seurat_span": span,
        "merge_rule": "median_rank_then_nbatches" if batch_mode else "global_normalized_variance",
        "populations": list(selected_cell_lines),
        "sampled_cells_by_population": dict(
            stats_manifest.get("hvg_sampled_cells_by_population", {})
        ) if batch_mode else {},
        "highly_variable_nbatches": hvg_nbatches[hvg_state_ids].tolist()
        if hvg_nbatches is not None else None,
        "state_ids": hvg_state_ids.tolist(),
        "gene_symbols": [symbols[index] for index in hvg_state_ids],
    }
    hvg["fingerprint"] = hvg_fingerprint(hvg)
    stats_manifest["hvg_fingerprint"] = hvg["fingerprint"]
    _dump_json(stats_manifest_path, stats_manifest)
    _dump_json(output_dir / "hvg.json", hvg)
    mode = "batch-aware" if batch_mode else "global"
    print(f"hvg complete: {mode} Seurat-v3 top {n_hvg:,}")


def _build_condition_tables(output_dir: Path, cell_lines):
    census = _load_json(output_dir / "condition_group_census.json")
    perturb_counts = Counter()
    plate_values = {cell_line: set() for cell_line in cell_lines}
    for record in census:
        cell_line, drug, dose, unit, smiles, plate = json.loads(record["key"])
        plate_values[cell_line].add(plate)
        if drug != "__CONTROL__":
            key = _condition_key(cell_line, drug, dose, unit, smiles)
            perturb_counts[key] += int(record["count"])
    keys = sorted(perturb_counts, key=_condition_key_text)
    key_to_id = {key: index for index, key in enumerate(keys)}
    rows = []
    for key, condition_id in key_to_id.items():
        cell_line, drug, dose, unit, smiles = key
        rows.append(
            {
                "condition_id": condition_id,
                "population": cell_line,
                "population_name": CELL_LINE_NAMES.get(cell_line, cell_line),
                "drug": drug,
                "dose": float(dose),
                "dose_unit": unit,
                "canonical_smiles": smiles,
                "n_cells": int(perturb_counts[key]),
            }
        )
    conditions = pd.DataFrame(rows)
    conditions.to_parquet(output_dir / "conditions.parquet", index=False)
    plate_maps = {
        cell_line: {plate: index for index, plate in enumerate(sorted(values))}
        for cell_line, values in plate_values.items()
    }
    with (output_dir / "condition_key_to_id.pkl").open("wb") as handle:
        pickle.dump(key_to_id, handle)
    _dump_json(output_dir / "group_maps.json", plate_maps)
    return key_to_id, plate_maps


def _init_materialize_worker(
    mapping_path,
    sample_map_path,
    hvg_path,
    condition_map_path,
    plate_maps_path,
    output_dir,
    file_offsets_path,
    cell_counts,
    n_gene_tokens,
    target_sum,
    cell_lines,
    artifact,
):
    _WORKER["mapping"] = np.load(mapping_path, mmap_mode="r")
    with open(sample_map_path, "rb") as handle:
        _WORKER["sample_map"] = pickle.load(handle)
    _WORKER["hvg"] = np.load(hvg_path)
    with open(condition_map_path, "rb") as handle:
        _WORKER["condition_map"] = pickle.load(handle)
    _WORKER["plate_maps"] = _load_json(Path(plate_maps_path))
    _WORKER["output_dir"] = Path(output_dir)
    _WORKER["condition_filter"] = _load_condition_filter(Path(output_dir))
    _WORKER["file_offsets"] = np.load(file_offsets_path, mmap_mode="r")
    _WORKER["cell_counts"] = tuple(int(value) for value in cell_counts)
    _WORKER["cell_lines"] = tuple(cell_lines)
    _WORKER["cell_to_index"] = {value: index for index, value in enumerate(cell_lines)}
    _WORKER["n_gene_tokens"] = int(n_gene_tokens)
    _WORKER["target_sum"] = float(target_sum)
    _WORKER["artifact"] = str(artifact)
    mapping = _WORKER["mapping"]
    _WORKER["all_state_ids"] = np.unique(mapping[mapping >= 0]).astype(np.int32)
    state_to_hvg = np.full(
        len(_load_json(Path(output_dir) / "state_gene_symbols.json")),
        -1,
        dtype=np.int32,
    )
    state_to_hvg[_WORKER["hvg"]] = np.arange(len(_WORKER["hvg"]), dtype=np.int32)
    _WORKER["state_to_hvg"] = state_to_hvg


def _open_materialized_arrays(cell_line, n_cells):
    base = _WORKER["output_dir"] / cell_line
    n_gene_tokens = _WORKER["n_gene_tokens"]
    n_hvg = len(_WORKER["hvg"])
    artifact = _WORKER.get("artifact", "all")
    arrays = {}
    if artifact in {"all", "state"}:
        arrays["genes"] = np.memmap(
            base / "se_gene_ids.uint16.dat", dtype=np.uint16, mode="r+",
            shape=(n_cells, n_gene_tokens + 1)
        )
        arrays["expr"] = np.memmap(
            base / "se_expr.float16.dat", dtype=np.float16, mode="r+",
            shape=(n_cells, n_gene_tokens + 1)
        )
    if artifact in {"all", "hvg"}:
        arrays["hvg"] = np.memmap(
            base / "hvg.float16.dat", dtype=np.float16, mode="r+",
            shape=(n_cells, n_hvg)
        )
    if artifact in {"all", "groups"}:
        arrays["condition"] = np.memmap(
            base / "row_condition.int32.dat", dtype=np.int32, mode="r+",
            shape=(n_cells,)
        )
        arrays["plate"] = np.memmap(
            base / "row_group.uint16.dat", dtype=np.uint16, mode="r+",
            shape=(n_cells,)
        )
    return arrays


def _paper_cell_arrays(
    genes,
    expressions,
    mapping,
    hvg_ids,
    all_state_ids,
    state_to_hvg=None,
    n_gene_tokens=N_GENE_TOKENS,
    target_sum=TARGET_SUM,
):
    state_ids, counts = _valid_gene_values(genes, expressions, mapping)
    total = float(counts.sum())
    if total > 0:
        log_normalized = np.log1p(counts * (float(target_sum) / total))
    else:
        log_normalized = np.zeros_like(counts)
    total_log_expression = float(log_normalized.sum())

    # Stable descending expression order, with STATE id as the tie breaker.
    order = np.lexsort((state_ids, -log_normalized))
    state_ids = state_ids[order]
    log_normalized = log_normalized[order]
    if state_ids.size >= n_gene_tokens:
        selected_ids = state_ids[:n_gene_tokens]
        selected_values = log_normalized[:n_gene_tokens]
    else:
        missing = n_gene_tokens - state_ids.size
        present = np.zeros(int(all_state_ids.max()) + 1, dtype=np.bool_)
        present[state_ids] = True
        fill_ids = all_state_ids[~present[all_state_ids]][:missing]
        selected_ids = np.concatenate([state_ids, fill_ids])
        selected_values = np.concatenate([log_normalized, np.zeros(missing, dtype=np.float64)])

    weights = np.zeros(n_gene_tokens + 1, dtype=np.float32)
    denom = total_log_expression
    if denom > 0:
        weights[1:] = (100.0 * selected_values / denom).astype(np.float32)
    gene_tokens = np.empty(n_gene_tokens + 1, dtype=np.uint16)
    gene_tokens[0] = SPECIAL_TOKEN
    gene_tokens[1:] = selected_ids.astype(np.uint16)

    if state_to_hvg is None:
        state_to_hvg = np.full(
            int(max(all_state_ids.max(), hvg_ids.max())) + 1,
            -1,
            dtype=np.int32,
        )
        state_to_hvg[hvg_ids] = np.arange(len(hvg_ids), dtype=np.int32)
    hvg = np.zeros(len(hvg_ids), dtype=np.float32)
    hvg_positions = state_to_hvg[state_ids]
    in_hvg = hvg_positions >= 0
    hvg[hvg_positions[in_hvg]] = log_normalized[in_hvg].astype(np.float32)
    return gene_tokens, weights, hvg


def _materialize_chunk(task):
    chunk_id, indexed_files, _ = task
    mapping = _WORKER["mapping"]
    sample_map = _WORKER["sample_map"]
    hvg_ids = _WORKER["hvg"]
    condition_map = _WORKER["condition_map"]
    plate_maps = _WORKER["plate_maps"]
    file_offsets = _WORKER["file_offsets"]
    cell_counts = _WORKER["cell_counts"]
    cell_to_index = _WORKER["cell_to_index"]
    all_state_ids = _WORKER["all_state_ids"]
    state_to_hvg = _WORKER["state_to_hvg"]
    n_gene_tokens = _WORKER["n_gene_tokens"]
    target_sum = _WORKER["target_sum"]
    cell_lines = _WORKER["cell_lines"]
    arrays = {
        cell_line: _open_materialized_arrays(cell_line, cell_counts[index])
        for index, cell_line in enumerate(cell_lines)
    }
    written = np.zeros(len(cell_lines), dtype=np.int64)
    columns = [
        "genes",
        "expressions",
        "drug",
        "sample",
        "cell_line_id",
        "canonical_smiles",
        "plate",
    ]
    for file_index, file_path in indexed_files:
        local = np.zeros(len(cell_lines), dtype=np.int64)
        selector = _file_condition_selector(file_path)
        for data in _iter_selected_batches(file_path, columns, cell_lines=cell_lines):
            for genes, expressions, raw_drug, sample, cell_line, smiles, plate in zip(
                data["genes"],
                data["expressions"],
                data["drug"],
                data["sample"],
                data["cell_line_id"],
                data["canonical_smiles"],
                data["plate"],
            ):
                cell_index = cell_to_index[cell_line]
                fallback = _canonical_drug(raw_drug)
                drug, dose, unit = sample_map.get(str(sample), (fallback, float("nan"), "uM"))
                is_control = drug == CONTROL_DRUG or fallback == CONTROL_DRUG
                valid_smiles = smiles is not None and str(smiles).strip() not in {"", "nan", "None"}
                if not is_control and (not valid_smiles or not math.isfinite(dose)):
                    continue
                if is_control:
                    if not _control_is_selected(selector, cell_line):
                        continue
                elif not _condition_is_selected(
                    selector, _condition_key(cell_line, drug, dose, unit, str(smiles))
                ):
                    continue
                row = int(file_offsets[file_index, cell_index] + local[cell_index])
                local[cell_index] += 1
                gene_tokens, soft_expression, hvg = _paper_cell_arrays(
                    genes,
                    expressions,
                    mapping,
                    hvg_ids,
                    all_state_ids,
                    state_to_hvg,
                    n_gene_tokens,
                    target_sum,
                )
                target = arrays[cell_line]
                if "genes" in target:
                    target["genes"][row] = gene_tokens
                    target["expr"][row] = soft_expression.astype(np.float16)
                if "hvg" in target:
                    target["hvg"][row] = hvg.astype(np.float16)
                if "plate" in target:
                    target["plate"][row] = int(plate_maps[cell_line][str(plate)])
                if "condition" in target:
                    if is_control:
                        target["condition"][row] = -1
                    else:
                        key = _condition_key(cell_line, drug, dose, unit, str(smiles))
                        target["condition"][row] = int(condition_map[key])
                written[cell_index] += 1
        gc.collect()
        pa.default_memory_pool().release_unused()
    for cell_arrays in arrays.values():
        for value in cell_arrays.values():
            value.flush()
    return chunk_id, written


def run_materialize(args):
    output_dir = Path(args.output_dir)
    manifest = _load_json(output_dir / "stats_manifest.json")
    cell_lines = tuple(manifest.get("populations", CELL_LINES))
    key_to_id, plate_maps = _build_condition_tables(output_dir, cell_lines)
    files = manifest["source_parts"]
    file_counts = np.load(output_dir / "eligible_file_cell_counts.npy")
    file_offsets = np.zeros_like(file_counts)
    if len(files) > 1:
        file_offsets[1:] = np.cumsum(file_counts[:-1], axis=0)
    np.save(output_dir / "eligible_file_cell_offsets.npy", file_offsets)
    cell_counts = file_counts.sum(axis=0).astype(np.int64)
    print(
        f"[materialize] start: {len(files):,} shards, "
        f"{len(cell_lines)} cell lines, "
        f"{int(cell_counts.sum()):,} cells, workers={args.workers}",
        flush=True,
    )
    hvg_ids = np.load(output_dir / "hvg_state_ids.npy")
    n_hvg = int(len(hvg_ids))
    n_gene_tokens = int(getattr(args, "num_gene_tokens", N_GENE_TOKENS))
    target_sum = float(getattr(args, "target_sum", TARGET_SUM))
    if n_gene_tokens <= 0 or target_sum <= 0:
        raise ValueError("num_gene_tokens and target_sum must be positive")

    artifact = str(getattr(args, "artifact", "all")).casefold()
    if artifact not in {"all", "state", "hvg", "groups"}:
        raise ValueError("materialize artifact must be all, state, hvg, or groups")
    for cell_index, cell_line in enumerate(cell_lines):
        base = output_dir / cell_line
        base.mkdir(parents=True, exist_ok=True)
        n_cells = int(cell_counts[cell_index])
        specifications = []
        if artifact in {"all", "state"}:
            specifications.extend([
                ("se_gene_ids.uint16.dat", np.uint16, (n_cells, n_gene_tokens + 1)),
                ("se_expr.float16.dat", np.float16, (n_cells, n_gene_tokens + 1)),
            ])
        if artifact in {"all", "hvg"}:
            specifications.append(("hvg.float16.dat", np.float16, (n_cells, n_hvg)))
        if artifact in {"all", "groups"}:
            specifications.extend([
                ("row_condition.int32.dat", np.int32, (n_cells,)),
                ("row_group.uint16.dat", np.uint16, (n_cells,)),
            ])
        for filename, dtype, shape in specifications:
            array = np.memmap(base / filename, dtype=dtype, mode="w+", shape=shape)
            array.flush()
            del array

    indexed_files = list(enumerate(files))
    chunks = _chunks(indexed_files, min(args.workers, len(indexed_files)))
    tasks = [(index, chunk, None) for index, chunk in enumerate(chunks)]
    with ProcessPoolExecutor(
        max_workers=len(chunks),
        initializer=_init_materialize_worker,
        initargs=(
            str(output_dir / "source_gene_to_state.npy"),
            str(output_dir / "preparation_state.pkl"),
            str(output_dir / "hvg_state_ids.npy"),
            str(output_dir / "condition_key_to_id.pkl"),
            str(output_dir / "group_maps.json"),
            str(output_dir),
            str(output_dir / "eligible_file_cell_offsets.npy"),
            cell_counts.tolist(),
            n_gene_tokens,
            target_sum,
            cell_lines,
            artifact,
        ),
    ) as pool:
        written = np.zeros(len(cell_lines), dtype=np.int64)
        futures = [pool.submit(_materialize_chunk, task) for task in tasks]
        for future in as_completed(futures):
            _, partial = future.result()
            written += partial
    if not np.array_equal(written, cell_counts):
        raise RuntimeError(f"Materialization row mismatch: expected={cell_counts}, wrote={written}")
    shapes_path = output_dir / "materialized_shapes.json"
    existing_shapes = _load_json(shapes_path) if shapes_path.is_file() else {}
    for index, cell_line in enumerate(cell_lines):
        existing_shapes[str(cell_line)] = {
            **existing_shapes.get(str(cell_line), {}),
            "n_cells": int(cell_counts[index]),
            "token_length": n_gene_tokens + 1,
            "hvg_dim": n_hvg,
            "target_sum": target_sum,
            "log1p": True,
        }
    _dump_json(shapes_path, existing_shapes)
    print(f"materialize complete: artifact={artifact}")
