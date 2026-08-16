from __future__ import annotations

import hashlib
import json
import os
import shutil
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .._common.feedback import Feedback, StageResult
from .._common.paths import DatasetPaths
from ..model import MAPKGEncoder, StateEncoder
from ..model.state import load_gene_embeddings


def _inference_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _precision(device: torch.device):
    return (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )


def _fingerprint(path: Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        return {"path": str(resolved), "exists": False}
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return {
        "path": str(resolved),
        "exists": True,
        "size_bytes": resolved.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _validate_partition(partition_index: int, num_partitions: int) -> None:
    if partition_index < 0 or num_partitions <= 0 or partition_index >= num_partitions:
        raise ValueError("partition_index must be in [0, num_partitions)")


def _part_stem(partition_index: int, num_partitions: int) -> str:
    return f"part-{partition_index:05d}-of-{num_partitions:05d}"


def _state_part_paths(
    root: Path,
    population: str,
    partition_index: int,
    num_partitions: int,
) -> tuple[Path, Path]:
    if num_partitions == 1:
        base = root / population
        return (
            base / "state_embeddings.float16.dat",
            base / "embedding_complete.json",
        )
    base = root / population / "state_embedding_parts"
    stem = _part_stem(partition_index, num_partitions)
    return base / f"{stem}.float16.dat", base / f"{stem}.json"


class _StateInputs(torch.utils.data.Dataset):
    def __init__(self, root: Path, total: int, token_length: int, start: int, end: int):
        self.start = int(start)
        self.end = int(end)
        self.genes = np.memmap(
            root / "se_gene_ids.uint16.dat", dtype=np.uint16, mode="r",
            shape=(int(total), int(token_length)),
        )
        self.expression = np.memmap(
            root / "se_expr.float16.dat", dtype=np.float16, mode="r",
            shape=(int(total), int(token_length)),
        )

    def __len__(self) -> int:
        return self.end - self.start

    def __getitem__(self, index: int):
        row = self.start + int(index)
        return (
            torch.from_numpy(self.genes[row].astype(np.int64)),
            torch.from_numpy(self.expression[row].astype(np.float32)),
        )


def _embed_population(
    paths: DatasetPaths,
    population: str,
    total: int,
    row_start: int,
    row_end: int,
    se_checkpoint: Path,
    esm_embeddings: Path,
    *,
    batch_size: int,
    workers: int,
    partition_index: int,
    num_partitions: int,
    max_batches: int | None,
    probe_output_dir: str | Path | None,
) -> None:
    shape = json.loads(
        (paths.prepared / "materialized_shapes.json").read_text(encoding="utf-8")
    )[population]
    token_length = int(shape.get("token_length", 2048))
    dataset = _StateInputs(
        paths.prepared / population, total, token_length, row_start, row_end
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
        pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0,
    )
    device = _inference_device()
    model = StateEncoder.from_pretrained(se_checkpoint, esm_embeddings).to(device).eval()
    output_root = (
        Path(probe_output_dir)
        if max_batches is not None and probe_output_dir
        else paths.prepared
    )
    output_path, completion_path = _state_part_paths(
        output_root, population, partition_index, num_partitions
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partition_cells = row_end - row_start
    output = (
        np.memmap(output_path, dtype=np.float16, mode="w+", shape=(partition_cells, 2048))
        if partition_cells else None
    )
    if output is None:
        output_path.touch()
    cursor = 0
    with torch.inference_mode():
        for batch_index, (gene_ids, expression) in enumerate(loader):
            with _precision(device):
                values = model.encode(
                    gene_ids.to(device, non_blocking=True),
                    expression.to(device, non_blocking=True),
                )
            batch = values.float().cpu().numpy().astype(np.float16)
            if output is not None:
                output[cursor:cursor + len(batch)] = batch
            cursor += len(batch)
            if max_batches is not None and batch_index + 1 >= max_batches:
                break
    if output is not None:
        output.flush()
    if max_batches is None:
        completion_path.write_text(json.dumps({
            "n_cells": partition_cells,
            "total_cells": total,
            "row_start": row_start,
            "row_end": row_end,
            "partition_index": partition_index,
            "num_partitions": num_partitions,
            "embedding_dim": 2048,
            "dtype": "float16",
        }, indent=2), encoding="utf-8")


def _static_part_path(
    paths: DatasetPaths,
    kind: str,
    partition_index: int,
    num_partitions: int,
) -> Path:
    return (
        paths.prepared
        / "static_token_parts"
        / kind
        / f"{_part_stem(partition_index, num_partitions)}.pt"
    )


def embed_state(
    paths: DatasetPaths,
    se_checkpoint: Path,
    esm_embeddings: Path,
    populations: tuple[str, ...] = (),
    batch_size: int = 32,
    workers: int = 8,
    dtype: str = "float16",
    partition_index: int = 0,
    num_partitions: int = 1,
    resume: bool = True,
    max_batches: int | None = None,
    probe_output_dir: str | Path | None = None,
    dry_run: bool = False,
) -> StageResult:
    """Precompute treated-cell SE embeddings used as the embedding target."""
    if dtype != "float16":
        raise ValueError("The released STATE cache format supports dtype='float16'")
    if batch_size <= 0 or workers < 0:
        raise ValueError("batch_size must be positive and workers non-negative")
    if max_batches is not None and max_batches <= 0:
        raise ValueError("max_batches must be positive")
    _validate_partition(partition_index, num_partitions)

    report = Feedback(paths.prepared, "embed_perturbed_cells")
    shapes_path = paths.prepared / "materialized_shapes.json"
    shapes = (
        json.loads(shapes_path.read_text(encoding="utf-8"))
        if shapes_path.is_file()
        else {}
    )
    source = {
        "se_checkpoint": _fingerprint(se_checkpoint),
        "esm_embeddings": _fingerprint(esm_embeddings),
    }
    output_root = (
        Path(probe_output_dir)
        if max_batches is not None and probe_output_dir
        else paths.prepared
    )
    completed: list[str] = []
    skipped: list[str] = []
    outputs: list[Path] = []
    for population in populations:
        n_cells = int(shapes.get(population, {}).get("n_cells", 0))
        row_start = n_cells * partition_index // num_partitions
        row_end = n_cells * (partition_index + 1) // num_partitions
        partition_cells = row_end - row_start
        embedding, completion = _state_part_paths(
            output_root, population, partition_index, num_partitions
        )
        expected_bytes = partition_cells * 2048 * 2
        outputs.append(completion)
        if resume and completion.is_file() and embedding.is_file():
            payload = json.loads(completion.read_text(encoding="utf-8"))
            matches = (
                embedding.stat().st_size == expected_bytes
                and int(payload.get("row_start", row_start)) == row_start
                and int(payload.get("row_end", row_end)) == row_end
                and int(payload.get("num_partitions", num_partitions)) == num_partitions
                and payload.get("source") in (None, source)
            )
            if matches:
                skipped.append(population)
                report.emit(
                    "partition already complete",
                    population=population,
                    partition=f"{partition_index + 1}/{num_partitions}",
                    cells=partition_cells,
                )
                continue
            raise FileExistsError(
                f"Embedding partition for {population} exists with different metadata"
            )
        report.emit(
            "embedding treated cells",
            population=population,
            partition=f"{partition_index + 1}/{num_partitions}",
            rows=f"{row_start}:{row_end}",
        )
        if dry_run:
            completed.append(population)
            continue
        _embed_population(
            paths, population, n_cells, row_start, row_end,
            se_checkpoint, esm_embeddings,
            batch_size=batch_size, workers=workers,
            partition_index=partition_index, num_partitions=num_partitions,
            max_batches=max_batches, probe_output_dir=probe_output_dir,
        )
        if max_batches is None:
            payload = json.loads(completion.read_text(encoding="utf-8"))
            payload.update(
                {
                    "source": source,
                    "batch_size": int(batch_size),
                    "workers": int(workers),
                    "dtype": dtype,
                    "role": "treated_cell_embedding_supervision",
                }
            )
            completion.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        completed.append(population)
    return report.finish(
        {
            "populations": list(populations),
            "completed": completed,
            "skipped": skipped,
            "embedding_dim": 2048,
            "dtype": dtype,
            "batch_size": batch_size,
            "workers": workers,
            "partition_index": partition_index,
            "num_partitions": num_partitions,
            "resume": resume,
            "max_batches": max_batches,
            "probe_output_dir": str(probe_output_dir) if probe_output_dir else None,
            "source": source,
            "role": "embedding_loss_target",
            "dry_run": dry_run,
        },
        outputs,
    )


def merge_perturbed_cell_embeddings(
    paths: DatasetPaths,
    populations: tuple[str, ...] = (),
    num_partitions: int = 1,
    overwrite: bool = False,
) -> StageResult:
    """Validate and concatenate treated-cell embedding partitions."""
    _validate_partition(0, num_partitions)
    report = Feedback(paths.prepared, "merge_perturbed_cell_embeddings")
    shapes = json.loads(
        (paths.prepared / "materialized_shapes.json").read_text(encoding="utf-8")
    )
    outputs: list[Path] = []
    merged: list[str] = []
    for population in populations:
        n_cells = int(shapes[population]["n_cells"])
        final_data, final_manifest = _state_part_paths(
            paths.prepared, population, 0, 1
        )
        expected_final_bytes = n_cells * 2048 * 2
        if final_data.is_file() and final_manifest.is_file() and not overwrite:
            if final_data.stat().st_size == expected_final_bytes:
                report.emit("merged embedding already exists", population=population)
                outputs.append(final_manifest)
                continue
            raise FileExistsError(f"Invalid existing embedding cache: {final_data}")

        manifests = []
        source = None
        for index in range(num_partitions):
            data_path, manifest_path = _state_part_paths(
                paths.prepared, population, index, num_partitions
            )
            if not data_path.is_file() or not manifest_path.is_file():
                raise FileNotFoundError(
                    f"Missing embedding partition {index + 1}/{num_partitions} "
                    f"for {population}"
                )
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            row_start = n_cells * index // num_partitions
            row_end = n_cells * (index + 1) // num_partitions
            expected = {
                "partition_index": index,
                "num_partitions": num_partitions,
                "row_start": row_start,
                "row_end": row_end,
                "total_cells": n_cells,
                "embedding_dim": 2048,
                "dtype": "float16",
            }
            for key, value in expected.items():
                if payload.get(key) != value:
                    raise ValueError(
                        f"Invalid {key} in {manifest_path}: "
                        f"expected {value!r}, got {payload.get(key)!r}"
                    )
            expected_bytes = (row_end - row_start) * 2048 * 2
            if data_path.stat().st_size != expected_bytes:
                raise ValueError(f"Invalid partition size: {data_path}")
            if source is None:
                source = payload.get("source")
            elif payload.get("source") != source:
                raise ValueError(f"Frozen asset mismatch in {manifest_path}")
            manifests.append((data_path, payload))

        final_data.parent.mkdir(parents=True, exist_ok=True)
        temporary = final_data.with_suffix(final_data.suffix + ".tmp")
        with temporary.open("wb") as output_handle:
            for data_path, _ in manifests:
                with data_path.open("rb") as input_handle:
                    shutil.copyfileobj(input_handle, output_handle, 16 * 1024 * 1024)
        if temporary.stat().st_size != expected_final_bytes:
            temporary.unlink(missing_ok=True)
            raise ValueError(f"Merged embedding has the wrong size for {population}")
        os.replace(temporary, final_data)
        final_payload = {
            "n_cells": n_cells,
            "total_cells": n_cells,
            "row_start": 0,
            "row_end": n_cells,
            "partition_index": 0,
            "num_partitions": 1,
            "merged_from_partitions": num_partitions,
            "embedding_dim": 2048,
            "dtype": "float16",
            "source": source,
            "role": "treated_cell_embedding_supervision",
        }
        temporary_manifest = final_manifest.with_suffix(".json.tmp")
        temporary_manifest.write_text(
            json.dumps(final_payload, indent=2), encoding="utf-8"
        )
        os.replace(temporary_manifest, final_manifest)
        outputs.append(final_manifest)
        merged.append(population)
        report.emit("embedding partitions merged", population=population, cells=n_cells)
    return report.finish(
        {
            "populations": list(populations),
            "num_partitions": num_partitions,
            "merged": merged,
            "role": "embedding_loss_target",
        },
        outputs,
    )


def _embed_static_kind(
    paths: DatasetPaths,
    kind: str,
    esm_embeddings: Path,
    mapkg_checkpoint: Path,
    mapkg_vocab: Path,
    batch_size: int,
    dtype: str,
    partition_index: int,
    num_partitions: int,
    resume: bool,
    overwrite: bool,
    dry_run: bool,
) -> StageResult:
    if kind not in {"genes", "drugs"}:
        raise ValueError(kind)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if dtype not in {"bfloat16", "float16", "float32"}:
        raise ValueError("dtype must be one of: bfloat16, float16, float32")
    _validate_partition(partition_index, num_partitions)
    stage = f"embed_{kind}"
    report = Feedback(paths.prepared, stage)
    output = _static_part_path(paths, kind, partition_index, num_partitions)
    expected_source = {
        "conditions": str((paths.prepared / "conditions.parquet").resolve()),
        "esm_embeddings": str(Path(esm_embeddings).resolve()),
        "mapkg_ckpt": str(Path(mapkg_checkpoint).resolve()),
        "mapkg_vocab": str(Path(mapkg_vocab).resolve()),
    }
    if output.is_file() and resume and not overwrite:
        cache = torch.load(output, map_location="cpu", weights_only=False)
        if (
            cache.get("kind") == kind
            and cache.get("source") == expected_source
            and cache.get("dtype") == dtype
            and cache.get("partition_index") == partition_index
            and cache.get("num_partitions") == num_partitions
        ):
            report.emit(
                "partition already complete",
                partition=f"{partition_index + 1}/{num_partitions}",
                output=output,
            )
            return report.finish(
                {
                    "kind": kind,
                    "partition_index": partition_index,
                    "num_partitions": num_partitions,
                    "reused": True,
                    "output": str(output),
                },
                [output],
            )
        raise FileExistsError(f"Static token partition has different metadata: {output}")
    if output.is_file():
        if not overwrite:
            raise FileExistsError(output)
        output.unlink()
    output.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        return report.finish(
            {
                "kind": kind,
                "partition_index": partition_index,
                "num_partitions": num_partitions,
                "output": str(output),
                "dry_run": True,
            },
            [output],
        )

    import pandas as pd

    target_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype]
    conditions = pd.read_parquet(
        paths.prepared / "conditions.parquet", columns=["canonical_smiles"]
    )
    drug_smiles = sorted(conditions["canonical_smiles"].astype(str).unique().tolist())
    raw_gene_embeddings = torch.load(
        esm_embeddings, map_location="cpu", weights_only=False
    )
    gene_symbols = list(raw_gene_embeddings) if isinstance(raw_gene_embeddings, dict) else None
    gene_embeddings = load_gene_embeddings(esm_embeddings)
    device = _inference_device()
    model = MAPKGEncoder(vocab_path=mapkg_vocab).load_from_full_model(mapkg_checkpoint)
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    payload: dict[str, Any] = {
        "format": "map_static_token_cache_v1",
        "kind": kind,
        "embedding_dim": 1024,
        "dtype": dtype,
        "source": expected_source,
        "partition_index": partition_index,
        "num_partitions": num_partitions,
    }
    if kind == "genes":
        indices = list(range(partition_index, len(gene_embeddings), num_partitions))
        parts = []
        with torch.inference_mode(), _precision(device):
            for start in range(0, len(indices), batch_size):
                selected = indices[start:start + batch_size]
                parts.append(
                    model.encode_genes(gene_embeddings[selected].to(device)).cpu()
                )
        tokens = (
            torch.cat(parts).to(target_dtype).contiguous()
            if parts else torch.empty((0, 1024), dtype=target_dtype)
        )
        payload.update({
            "gene_tokens": tokens,
            "gene_symbols": [gene_symbols[index] for index in indices] if gene_symbols else None,
            "gene_indices": indices,
            "gene_total": len(gene_embeddings),
        })
    else:
        indices = list(range(partition_index, len(drug_smiles), num_partitions))
        tokens = torch.empty((len(indices), 1024), dtype=target_dtype)
        order = sorted(range(len(indices)), key=lambda index: len(drug_smiles[indices[index]]))
        with torch.inference_mode(), _precision(device):
            for start in range(0, len(order), batch_size):
                local = order[start:start + batch_size]
                smiles = [drug_smiles[indices[index]] for index in local]
                tokens[torch.tensor(local)] = model(smiles).cpu().to(target_dtype)
        payload.update({
            "drug_tokens": tokens.contiguous(),
            "drug_smiles": [drug_smiles[index] for index in indices],
            "drug_indices": indices,
            "drug_total": len(drug_smiles),
        })
    torch.save(payload, output)
    report.emit(
        "static token partition ready",
        kind=kind,
        partition=f"{partition_index + 1}/{num_partitions}",
        output=output,
    )
    return report.finish(
        {
            "kind": kind,
            "embedding_dim": 1024,
            "dtype": dtype,
            "batch_size": batch_size,
            "partition_index": partition_index,
            "num_partitions": num_partitions,
            "resume": resume,
            "overwrite": overwrite,
            "reused": False,
            "output": str(output),
            "dry_run": dry_run,
        },
        [output],
    )


def embed_genes(
    paths: DatasetPaths,
    esm_embeddings: Path,
    mapkg_checkpoint: Path,
    mapkg_vocab: Path,
    batch_size: int = 512,
    dtype: str = "bfloat16",
    partition_index: int = 0,
    num_partitions: int = 1,
    resume: bool = True,
    overwrite: bool = False,
    dry_run: bool = False,
) -> StageResult:
    return _embed_static_kind(
        paths,
        "genes",
        esm_embeddings,
        mapkg_checkpoint,
        mapkg_vocab,
        batch_size,
        dtype,
        partition_index,
        num_partitions,
        resume,
        overwrite,
        dry_run,
    )


def embed_drugs(
    paths: DatasetPaths,
    esm_embeddings: Path,
    mapkg_checkpoint: Path,
    mapkg_vocab: Path,
    batch_size: int = 32,
    dtype: str = "bfloat16",
    partition_index: int = 0,
    num_partitions: int = 1,
    resume: bool = True,
    overwrite: bool = False,
    dry_run: bool = False,
) -> StageResult:
    return _embed_static_kind(
        paths,
        "drugs",
        esm_embeddings,
        mapkg_checkpoint,
        mapkg_vocab,
        batch_size,
        dtype,
        partition_index,
        num_partitions,
        resume,
        overwrite,
        dry_run,
    )


def _load_static_part(
    path: Path,
    kind: str,
    index: int,
    num_partitions: int,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    expected = {
        "format": "map_static_token_cache_v1",
        "kind": kind,
        "embedding_dim": 1024,
        "partition_index": index,
        "num_partitions": num_partitions,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(
                f"Invalid {key} in {path}: expected {value!r}, "
                f"got {payload.get(key)!r}"
            )
    return payload


def _assemble_static_kind(
    paths: DatasetPaths,
    kind: str,
    num_partitions: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    index_key = "gene_indices" if kind == "genes" else "drug_indices"
    total_key = "gene_total" if kind == "genes" else "drug_total"
    token_key = "gene_tokens" if kind == "genes" else "drug_tokens"
    label_key = "gene_symbols" if kind == "genes" else "drug_smiles"
    parts = [
        _load_static_part(
            _static_part_path(paths, kind, index, num_partitions),
            kind,
            index,
            num_partitions,
        )
        for index in range(num_partitions)
    ]
    reference = parts[0]
    total = int(reference[total_key])
    tokens = torch.empty((total, 1024), dtype=reference[token_key].dtype)
    labels: list[str | None] = [None] * total
    seen: set[int] = set()
    for part in parts:
        for key in ("source", "dtype", total_key):
            if part.get(key) != reference.get(key):
                raise ValueError(f"Inconsistent {key} across {kind} partitions")
        indices = [int(value) for value in part[index_key]]
        part_tokens = part[token_key]
        part_labels = part[label_key]
        if part_tokens.shape != (len(indices), 1024):
            raise ValueError(f"Token shape does not match indices in {kind} partition")
        if part_labels is not None and len(part_labels) != len(indices):
            raise ValueError(f"Labels do not match indices in {kind} partition")
        for local_index, global_index in enumerate(indices):
            if global_index < 0 or global_index >= total or global_index in seen:
                raise ValueError(f"Invalid or duplicate {kind} index: {global_index}")
            seen.add(global_index)
            tokens[global_index] = part_tokens[local_index]
            if part_labels is not None:
                labels[global_index] = str(part_labels[local_index])
    if seen != set(range(total)):
        missing = sorted(set(range(total)) - seen)
        raise ValueError(f"Missing {kind} indices: {missing[:8]}")
    if kind == "drugs" and any(value is None for value in labels):
        raise ValueError("Drug SMILES labels are required")
    return {
        token_key: tokens.contiguous(),
        label_key: None if all(value is None for value in labels) else labels,
    }, reference


def merge_static_tokens(
    paths: DatasetPaths,
    gene_partitions: int = 1,
    drug_partitions: int = 1,
    overwrite: bool = False,
) -> StageResult:
    """Assemble gene and drug partitions into the cache consumed by training."""
    _validate_partition(0, gene_partitions)
    _validate_partition(0, drug_partitions)
    report = Feedback(paths.prepared, "merge_static_tokens")
    output = paths.prepared / "map_static_tokens.pt"
    if output.is_file() and not overwrite:
        cache = torch.load(output, map_location="cpu", weights_only=False)
        if cache.get("format") == "map_static_token_cache_v1":
            report.emit("static token cache already exists", output=output)
            return report.finish(
                {"output": str(output), "reused": True}, [output]
            )
        raise FileExistsError(f"Invalid existing static token cache: {output}")
    genes, gene_reference = _assemble_static_kind(
        paths, "genes", gene_partitions
    )
    drugs, drug_reference = _assemble_static_kind(
        paths, "drugs", drug_partitions
    )
    if gene_reference.get("source") != drug_reference.get("source"):
        raise ValueError("Gene and drug partitions were produced from different assets")
    if gene_reference.get("dtype") != drug_reference.get("dtype"):
        raise ValueError("Gene and drug partitions use different dtypes")
    payload = {
        "format": "map_static_token_cache_v1",
        "embedding_dim": 1024,
        "dtype": gene_reference["dtype"],
        "source": gene_reference["source"],
        "gene_partitions": gene_partitions,
        "drug_partitions": drug_partitions,
        **genes,
        **drugs,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, output)
    report.emit(
        "static token cache assembled",
        genes=len(payload["gene_tokens"]),
        drugs=len(payload["drug_tokens"]),
        output=output,
    )
    return report.finish(
        {
            "output": str(output),
            "genes": len(payload["gene_tokens"]),
            "drugs": len(payload["drug_tokens"]),
            "dtype": payload["dtype"],
            "gene_partitions": gene_partitions,
            "drug_partitions": drug_partitions,
            "reused": False,
        },
        [output],
    )


def embed_mapkg(
    paths: DatasetPaths,
    esm_embeddings: Path,
    mapkg_checkpoint: Path,
    mapkg_vocab: Path,
    batch_size_genes: int = 512,
    batch_size_drugs: int = 32,
    dtype: str = "bfloat16",
    overwrite: bool = False,
    dry_run: bool = False,
) -> StageResult:
    """Compatibility wrapper for the original single-job static cache API."""
    embed_genes(
        paths,
        esm_embeddings,
        mapkg_checkpoint,
        mapkg_vocab,
        batch_size=batch_size_genes,
        dtype=dtype,
        overwrite=overwrite,
        dry_run=dry_run,
    )
    embed_drugs(
        paths,
        esm_embeddings,
        mapkg_checkpoint,
        mapkg_vocab,
        batch_size=batch_size_drugs,
        dtype=dtype,
        overwrite=overwrite,
        dry_run=dry_run,
    )
    if dry_run:
        return StageResult(
            stage="embed_mapkg",
            status="ok",
            summary={"dry_run": True, "gene_partitions": 1, "drug_partitions": 1},
            outputs=[str(paths.prepared / "map_static_tokens.pt")],
        )
    return merge_static_tokens(paths, overwrite=overwrite)
