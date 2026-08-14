"""Self-contained Tahoe reader for the generic MAP preparation contract."""

from __future__ import annotations

import argparse
import ast
import gc
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
N_GENE_TOKENS = 2048
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


def _number_token(value) -> str:
    numeric = float(value)
    if numeric.is_integer():
        return str(int(numeric))
    return format(numeric, ".8g").replace("-", "m").replace(".", "p")


def _size_token(value) -> str:
    numeric = float(value)
    return ("n" if numeric >= 1 else "p") + _number_token(value)


def _default_split_id(rule, test_size, val_size, seed, disjoint=True):
    parts = [
        rule,
        f"test-{_size_token(test_size)}",
        f"val-{_size_token(val_size)}",
        f"seed-{int(seed)}",
    ]
    if rule == "unseen_combination":
        parts.append("testdisjoint" if disjoint else "testindependent")
    return "__".join(parts)


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


def _raw_files(raw_dir: Path) -> list[str]:
    files = sorted(str(path) for path in (raw_dir / "data").glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Tahoe parquet shards under {raw_dir / 'data'}")
    return files


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


def _init_worker(mapping_path: str, sample_map_path: str, cell_lines):
    _WORKER["mapping"] = np.load(mapping_path, mmap_mode="r")
    _WORKER["n_genes"] = len(_load_json(Path(mapping_path).parent / "state_gene_symbols.json"))
    with open(sample_map_path, "rb") as handle:
        _WORKER["sample_map"] = pickle.load(handle)
    _WORKER["cell_lines"] = tuple(cell_lines)
    _WORKER["cell_to_index"] = {value: index for index, value in enumerate(cell_lines)}


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

    sums = np.zeros(n_genes, dtype=np.float64)
    sumsq = np.zeros(n_genes, dtype=np.float64)
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
                n_cells += 1
                state_ids, values = _valid_gene_values(genes, expressions, mapping)
                if state_ids.size:
                    sums += np.bincount(state_ids, weights=values, minlength=n_genes)
                    sumsq += np.bincount(state_ids, weights=values * values, minlength=n_genes)
                    np.maximum.at(maxima, state_ids, values)
                counts[cell_to_index[cell_line]] += 1
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
    with ProcessPoolExecutor(
        max_workers=len(chunks),
        initializer=initializer,
        initargs=initargs,
    ) as pool:
        futures = [pool.submit(function, task) for task in tasks]
        for future in as_completed(futures):
            results.append(future.result())
    return results


def run_stats(args):
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = _raw_files(raw_dir)
    selected_cell_lines = tuple(getattr(args, "cell_lines", None) or CELL_LINES)
    mapping_path = output_dir / "source_gene_to_state.npy"
    if not mapping_path.exists():
        _build_gene_mapping(raw_dir, Path(args.esm_embeddings), output_dir)
    sample_map = _load_sample_map(raw_dir)
    sample_map_path = output_dir / "preparation_state.pkl"
    with sample_map_path.open("wb") as handle:
        pickle.dump(sample_map, handle)

    results = _run_parallel(
        files,
        args.workers,
        _scan_stats_chunk,
        _init_worker,
        (str(mapping_path), str(sample_map_path), selected_cell_lines),
    )
    n_genes = len(_load_json(output_dir / "state_gene_symbols.json"))
    sums = np.zeros(n_genes, dtype=np.float64)
    sumsq = np.zeros(n_genes, dtype=np.float64)
    maxima = np.zeros(n_genes, dtype=np.float64)
    n_cells = 0
    condition_counts = Counter()
    file_count_map = {}
    excluded = 0
    for result in results:
        sums += result["sums"]
        sumsq += result["sumsq"]
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
            "populations": list(selected_cell_lines),
            "source_parts": files,
            "n_hvg_model_cells": int(n_cells),
            "eligible_cells_by_population": {
                cell_line: int(file_counts[:, index].sum())
                for index, cell_line in enumerate(selected_cell_lines)
            },
            "excluded_perturbation_cells_without_smiles_or_dose": int(excluded),
            "target_sum": TARGET_SUM,
            "num_gene_tokens": N_GENE_TOKENS,
            "num_hvg": N_HVG,
            "hvg_flavor": "seurat_v3",
            "hvg_batch_key": None,
        },
    )
    print(f"stats complete: {n_cells:,} cells across {len(selected_cell_lines)} cell lines")


def _fit_seurat_clip(output_dir: Path, span: float = SEURAT_SPAN):
    from skmisc.loess import loess

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
    model = loess(
        np.log10(means[nonconstant]),
        np.log10(variances[nonconstant]),
        span=float(span),
        degree=2,
    )
    model.fit()
    fitted[nonconstant] = model.outputs.fitted_values
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


def _init_hvg_worker(
    mapping_path,
    clip_path,
    sample_map_path,
    cell_lines,
):
    _WORKER["mapping"] = np.load(mapping_path, mmap_mode="r")
    _WORKER["clip"] = np.load(clip_path, mmap_mode="r")
    _WORKER["cell_lines"] = set(cell_lines)
    with open(sample_map_path, "rb") as handle:
        _WORKER["sample_map"] = pickle.load(handle)


def _scan_clipped_chunk(task):
    chunk_id, files = task
    mapping = _WORKER["mapping"]
    clip = _WORKER["clip"]
    cell_lines = _WORKER["cell_lines"]
    sample_map = _WORKER["sample_map"]
    n_genes = clip.shape[0]
    clipped_sum = np.zeros(n_genes, dtype=np.float64)
    clipped_sumsq = np.zeros(n_genes, dtype=np.float64)
    for file_path in files:
        columns = [
            "genes", "expressions", "cell_line_id", "drug", "sample", "canonical_smiles"
        ]
        for data in _iter_selected_batches(file_path, columns, cell_lines=cell_lines):
            for genes, expressions, cell_line, raw_drug, sample, smiles in zip(
                data["genes"], data["expressions"], data["cell_line_id"],
                data["drug"], data["sample"], data["canonical_smiles"],
            ):
                fallback = _canonical_drug(raw_drug)
                drug, dose, _ = sample_map.get(
                    str(sample), (fallback, float("nan"), "uM")
                )
                is_control = drug == CONTROL_DRUG or fallback == CONTROL_DRUG
                valid_smiles = smiles is not None and str(smiles).strip() not in {
                    "", "nan", "None"
                }
                if not is_control and (not valid_smiles or not math.isfinite(dose)):
                    continue
                state_ids, values = _valid_gene_values(genes, expressions, mapping)
                if state_ids.size:
                    clipped = np.minimum(values, clip[state_ids])
                    clipped_sum += np.bincount(state_ids, weights=clipped, minlength=n_genes)
                    clipped_sumsq += np.bincount(
                        state_ids,
                        weights=clipped * clipped,
                        minlength=n_genes,
                    )
        gc.collect()
        pa.default_memory_pool().release_unused()
    return chunk_id, clipped_sum, clipped_sumsq


def run_hvg(args):
    output_dir = Path(args.output_dir)
    raw_dir = Path(args.raw_dir)
    n_hvg = int(getattr(args, "n_top_genes", N_HVG))
    span = float(getattr(args, "seurat_span", SEURAT_SPAN))
    if n_hvg <= 0:
        raise ValueError("n_top_genes must be positive")
    clip = _fit_seurat_clip(output_dir, span)
    clip_path = output_dir / "seurat_clip_values.npy"
    np.save(clip_path, clip)
    files = _load_json(output_dir / "stats_manifest.json")["source_parts"]
    selected_cell_lines = tuple(
        _load_json(output_dir / "stats_manifest.json").get("populations", CELL_LINES)
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
        ),
    )
    model = np.load(output_dir / "seurat_v3_model.npz")
    n_cells = int(model["n_cells"])
    clipped_sum = np.zeros_like(model["means"])
    clipped_sumsq = np.zeros_like(model["means"])
    for _, partial_sum, partial_sumsq in results:
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
    symbols = _load_json(output_dir / "state_gene_symbols.json")
    np.save(output_dir / "hvg_state_ids.npy", hvg_state_ids)
    np.save(output_dir / "seurat_v3_normalized_variance.npy", normalized_variance)
    _dump_json(
        output_dir / "hvg.json",
        {
            "flavor": "seurat_v3",
            "batch_key": None,
            "n_top_genes": n_hvg,
            "seurat_span": span,
            "state_ids": hvg_state_ids.tolist(),
            "gene_symbols": [symbols[index] for index in hvg_state_ids],
        },
    )
    print(f"hvg complete: exact global Seurat-v3 top {n_hvg:,}")


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
):
    _WORKER["mapping"] = np.load(mapping_path, mmap_mode="r")
    with open(sample_map_path, "rb") as handle:
        _WORKER["sample_map"] = pickle.load(handle)
    _WORKER["hvg"] = np.load(hvg_path)
    with open(condition_map_path, "rb") as handle:
        _WORKER["condition_map"] = pickle.load(handle)
    _WORKER["plate_maps"] = _load_json(Path(plate_maps_path))
    _WORKER["output_dir"] = Path(output_dir)
    _WORKER["file_offsets"] = np.load(file_offsets_path, mmap_mode="r")
    _WORKER["cell_counts"] = tuple(int(value) for value in cell_counts)
    _WORKER["cell_lines"] = tuple(cell_lines)
    _WORKER["cell_to_index"] = {value: index for index, value in enumerate(cell_lines)}
    _WORKER["n_gene_tokens"] = int(n_gene_tokens)
    _WORKER["target_sum"] = float(target_sum)
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
    return {
        "genes": np.memmap(
            base / "se_gene_ids.uint16.dat",
            dtype=np.uint16,
            mode="r+",
            shape=(n_cells, n_gene_tokens + 1),
        ),
        "expr": np.memmap(
            base / "se_expr.float16.dat",
            dtype=np.float16,
            mode="r+",
            shape=(n_cells, n_gene_tokens + 1),
        ),
        "hvg": np.memmap(
            base / "hvg.float16.dat",
            dtype=np.float16,
            mode="r+",
            shape=(n_cells, n_hvg),
        ),
        "condition": np.memmap(
            base / "row_condition.int32.dat",
            dtype=np.int32,
            mode="r+",
            shape=(n_cells,),
        ),
        "plate": np.memmap(
            base / "row_group.uint16.dat",
            dtype=np.uint16,
            mode="r+",
            shape=(n_cells,),
        ),
    }


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
                target["genes"][row] = gene_tokens
                target["expr"][row] = soft_expression.astype(np.float16)
                target["hvg"][row] = hvg.astype(np.float16)
                target["plate"][row] = int(plate_maps[cell_line][str(plate)])
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
    hvg_ids = np.load(output_dir / "hvg_state_ids.npy")
    n_hvg = int(len(hvg_ids))
    n_gene_tokens = int(getattr(args, "num_gene_tokens", N_GENE_TOKENS))
    target_sum = float(getattr(args, "target_sum", TARGET_SUM))
    if n_gene_tokens <= 0 or target_sum <= 0:
        raise ValueError("num_gene_tokens and target_sum must be positive")

    for cell_index, cell_line in enumerate(cell_lines):
        base = output_dir / cell_line
        base.mkdir(parents=True, exist_ok=True)
        n_cells = int(cell_counts[cell_index])
        specifications = [
            ("se_gene_ids.uint16.dat", np.uint16, (n_cells, n_gene_tokens + 1)),
            ("se_expr.float16.dat", np.float16, (n_cells, n_gene_tokens + 1)),
            ("hvg.float16.dat", np.float16, (n_cells, n_hvg)),
            ("row_condition.int32.dat", np.int32, (n_cells,)),
            ("row_group.uint16.dat", np.uint16, (n_cells,)),
        ]
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
        ),
    ) as pool:
        written = np.zeros(len(cell_lines), dtype=np.int64)
        futures = [pool.submit(_materialize_chunk, task) for task in tasks]
        for future in as_completed(futures):
            _, partial = future.result()
            written += partial
    if not np.array_equal(written, cell_counts):
        raise RuntimeError(f"Materialization row mismatch: expected={cell_counts}, wrote={written}")
    _dump_json(
        output_dir / "materialized_shapes.json",
        {
            cell_line: {
                "n_cells": int(cell_counts[index]),
                "token_length": n_gene_tokens + 1,
                "hvg_dim": n_hvg,
                "num_gene_tokens": n_gene_tokens,
                "target_sum": target_sum,
                "log1p": True,
            }
            for index, cell_line in enumerate(cell_lines)
        },
    )
    print("materialize complete")


def _write_csr(base: Path, prefix: str, group_ids: np.ndarray, rows: np.ndarray):
    order = np.lexsort((rows, group_ids))
    sorted_groups = group_ids[order]
    sorted_rows = rows[order].astype(np.int64, copy=False)
    unique, starts = np.unique(sorted_groups, return_index=True)
    offsets = np.concatenate([starts, [len(sorted_rows)]]).astype(np.int64)
    np.save(base / f"{prefix}_ids.npy", unique)
    np.save(base / f"{prefix}_offsets.npy", offsets)
    np.save(base / f"{prefix}_rows.npy", sorted_rows)


def _resolve_holdout_size(value, total, label):
    value = float(value)
    if value <= 0:
        raise ValueError(f"{label} must be positive")
    count = int(round(value * total)) if value < 1 else int(round(value))
    return max(1, count)


def _make_unprofiled_split(
    conditions: pd.DataFrame,
    seed: int,
    test_size=16,
    val_size=16,
):
    rng = np.random.default_rng(seed)
    all_drugs = np.asarray(sorted(conditions["drug"].unique()))
    n_unprofiled_test = _resolve_holdout_size(
        test_size, len(all_drugs), "unprofiled_test_size"
    )
    n_unprofiled_val = _resolve_holdout_size(
        val_size, len(all_drugs), "unprofiled_val_size"
    )
    if all_drugs.size <= n_unprofiled_test + n_unprofiled_val:
        raise RuntimeError(
            "Unprofiled holdouts leave no training drugs: "
            f"total={all_drugs.size}, test={n_unprofiled_test}, val={n_unprofiled_val}"
        )
    shuffled = rng.permutation(all_drugs)
    unprofiled_test = sorted(shuffled[:n_unprofiled_test].tolist())
    unprofiled_val = sorted(
        shuffled[n_unprofiled_test:n_unprofiled_test + n_unprofiled_val].tolist()
    )

    unprofiled_split = {"train": [], "val": [], "test": []}
    for row in conditions.itertuples(index=False):
        split = "test" if row.drug in unprofiled_test else "val" if row.drug in unprofiled_val else "train"
        unprofiled_split[split].append(int(row.condition_id))
    unprofiled_split.update(
        {
            "regime": "unprofiled_drug",
            "seed": seed,
            "rule": "unprofiled_drug",
            "test_size_requested": test_size,
            "validation_size_requested": val_size,
            "test_drug_count": n_unprofiled_test,
            "validation_drug_count": n_unprofiled_val,
            "test_drugs": unprofiled_test,
            "validation_drugs": unprofiled_val,
        }
    )
    return unprofiled_split


def _make_combination_split(
    conditions: pd.DataFrame,
    seed: int,
    test_size=0.05,
    val_size=0.05,
    disjoint_test_drugs=True,
):
    rng = np.random.default_rng(seed)
    cell_lines = tuple(sorted(conditions["cell_line"].unique()))

    drugs_by_cell = {
        cell_line: set(conditions.loc[conditions.cell_line == cell_line, "drug"])
        for cell_line in cell_lines
    }
    test_by_cell = {}
    used_test = set()
    for cell_line in cell_lines:
        candidates = sorted(
            drug
            for drug in drugs_by_cell[cell_line]
            if (not disjoint_test_drugs or drug not in used_test) and any(
                drug in drugs_by_cell[other] for other in cell_lines if other != cell_line
            )
        )
        n_holdout = _resolve_holdout_size(
            test_size,
            len(drugs_by_cell[cell_line]),
            f"combination_test_size[{cell_line}]",
        )
        if len(candidates) < n_holdout:
            raise RuntimeError(f"Cannot construct disjoint unseen-combination drugs for {cell_line}")
        selected = sorted(rng.choice(candidates, size=n_holdout, replace=False).tolist())
        test_by_cell[cell_line] = selected
        if disjoint_test_drugs:
            used_test.update(selected)

    val_by_cell = {}
    for cell_line in cell_lines:
        candidates = sorted(drugs_by_cell[cell_line] - set(test_by_cell[cell_line]))
        n_holdout = _resolve_holdout_size(
            val_size,
            len(drugs_by_cell[cell_line]),
            f"combination_val_size[{cell_line}]",
        )
        val_by_cell[cell_line] = sorted(
            rng.choice(candidates, size=n_holdout, replace=False).tolist()
        )
    combination_split = {"train": [], "val": [], "test": []}
    for row in conditions.itertuples(index=False):
        if row.drug in test_by_cell[row.cell_line]:
            split = "test"
        elif row.drug in val_by_cell[row.cell_line]:
            split = "val"
        else:
            split = "train"
        combination_split[split].append(int(row.condition_id))
    combination_split.update(
        {
            "regime": "unseen_combination",
            "seed": seed,
            "rule": "unseen_combination",
            "test_size_requested": test_size,
            "validation_size_requested": val_size,
            "test_drugs_by_cell_line": test_by_cell,
            "validation_drugs_by_cell_line": val_by_cell,
            "test_drugs_are_disjoint_across_cell_lines": bool(
                disjoint_test_drugs
            ),
        }
    )
    return combination_split


def _make_splits(
    conditions: pd.DataFrame,
    seed: int,
    unprofiled_test_size=16,
    unprofiled_val_size=16,
    combination_test_size=0.05,
    combination_val_size=0.05,
    combination_disjoint_test_drugs=True,
):
    """Backward-compatible helper returning both published regimes."""
    return (
        _make_unprofiled_split(
            conditions, seed, unprofiled_test_size, unprofiled_val_size
        ),
        _make_combination_split(
            conditions,
            seed,
            combination_test_size,
            combination_val_size,
            combination_disjoint_test_drugs,
        ),
    )


def run_index(args):
    output_dir = Path(args.output_dir)
    shapes = _load_json(output_dir / "materialized_shapes.json")
    for cell_line in shapes:
        base = output_dir / cell_line
        n_cells = int(shapes[cell_line]["n_cells"])
        conditions = np.memmap(
            base / "row_condition.int32.dat", dtype=np.int32, mode="r", shape=(n_cells,)
        )
        plates = np.memmap(
            base / "row_group.uint16.dat", dtype=np.uint16, mode="r", shape=(n_cells,)
        )
        rows = np.arange(n_cells, dtype=np.int64)
        perturb = conditions >= 0
        control = ~perturb
        control_plate_ids = set(int(value) for value in np.unique(plates[control]))
        unmatched_plates = sorted(
            set(int(value) for value in np.unique(plates[perturb])) - control_plate_ids
        )
        if unmatched_plates:
            raise RuntimeError(
                f"{cell_line} has perturbation cells on plates without DMSO controls: "
                f"{unmatched_plates}"
            )
        _write_csr(base, "condition", np.asarray(conditions[perturb]), rows[perturb])
        _write_csr(base, "control_plate", np.asarray(plates[control]), rows[control])

    if getattr(args, "skip_splits", False):
        print("condition/control index complete")
        return

    run_splits(args)
    print("index complete")


def run_splits(args):
    """Generate experiment splits after condition/control indexes exist."""
    output_dir = Path(args.output_dir)
    shapes = _load_json(output_dir / "materialized_shapes.json")

    conditions = pd.read_parquet(output_dir / "conditions.parquet")
    rule = getattr(args, "rule", "all")
    splits = output_dir / "splits"
    requested = []
    if rule in {"all", "unprofiled_drug"}:
        test_size = (
            getattr(args, "test_size", None)
            if rule == "unprofiled_drug"
            else None
        )
        val_size = (
            getattr(args, "val_size", None)
            if rule == "unprofiled_drug"
            else None
        )
        test_size = 16 if test_size is None else test_size
        val_size = 16 if val_size is None else val_size
        if rule == "all":
            test_size = getattr(args, "unprofiled_test_size", test_size)
            val_size = getattr(args, "unprofiled_val_size", val_size)
        payload = _make_unprofiled_split(conditions, args.seed, test_size, val_size)
        requested.append(("unprofiled_drug", payload))
    if rule in {"all", "unseen_combination"}:
        test_size = (
            getattr(args, "test_size", None)
            if rule == "unseen_combination"
            else None
        )
        val_size = (
            getattr(args, "val_size", None)
            if rule == "unseen_combination"
            else None
        )
        test_size = 0.05 if test_size is None else test_size
        val_size = 0.05 if val_size is None else val_size
        if rule == "all":
            test_size = getattr(args, "combination_test_size", test_size)
            val_size = getattr(args, "combination_val_size", val_size)
        disjoint = getattr(args, "disjoint_test_drugs", None)
        if disjoint is None:
            disjoint = getattr(args, "combination_disjoint_test_drugs", True)
        payload = _make_combination_split(
            conditions, args.seed, test_size, val_size, disjoint
        )
        requested.append(("unseen_combination", payload))

    manifest = _load_json(output_dir / "stats_manifest.json")
    existing_manifest = output_dir / "manifest.json"
    if existing_manifest.is_file():
        manifest.update(_load_json(existing_manifest))
    preparation_config = output_dir / "preparation_config.json"
    if preparation_config.is_file():
        manifest["preparation"] = _load_json(preparation_config)
    manifest.setdefault("splits", {})
    manifest.setdefault("split_registry", {})
    written = []
    for regime, payload in requested:
        if len(requested) == 1:
            filename = getattr(args, "output_filename", None)
            split_id = getattr(args, "split_id", None)
        else:
            filename = getattr(
                args,
                "unprofiled_filename" if regime == "unprofiled_drug" else "combination_filename",
                None,
            )
            split_id = None
        if filename is None:
            split_id = split_id or _default_split_id(
                regime,
                payload["test_size_requested"],
                payload["validation_size_requested"],
                payload["seed"],
                payload.get("test_drugs_are_disjoint_across_cell_lines", True),
            )
            filename = f"{split_id}.json"
        split_id = split_id or Path(filename).stem
        path = splits / filename
        relative_path = str(path.relative_to(output_dir))
        payload.update(
            {
                "split_id": split_id,
                "split_file": relative_path,
                "counts": {
                    name: len(payload[name]) for name in ("train", "val", "test")
                },
            }
        )
        _dump_json(path, payload)
        manifest["splits"][regime] = relative_path
        manifest["split_registry"][split_id] = {
            "path": relative_path,
            "rule": payload["rule"],
            "seed": int(payload["seed"]),
            "test_size_requested": payload["test_size_requested"],
            "validation_size_requested": payload["validation_size_requested"],
            "counts": payload["counts"],
        }
        written.append(relative_path)
    manifest.update(
        {
            "format_version": 1,
            "materialized_shapes": shapes,
            "conditions": "conditions.parquet",
            "normalization": "library_size_10000_log1p",
            "state_expression_encoding": "100 * selected_log_expression / selected_log_expression_sum",
            "control_matching": "cell_line_and_plate",
        }
    )
    _dump_json(output_dir / "manifest.json", manifest)
    print(f"split generation complete: {', '.join(written)}")
