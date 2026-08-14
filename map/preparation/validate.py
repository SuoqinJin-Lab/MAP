from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .._common.feedback import Feedback, StageResult
from .._common.paths import DatasetPaths


def validate_preparation(
    paths: DatasetPaths,
    require_embeddings: bool = True,
    sample_rows: int = 256,
    split_files: tuple[str | Path, ...] | list[str | Path] | None = None,
) -> StageResult:
    report = Feedback(paths.prepared, "validate_preparation")
    manifest_path = paths.prepared / "manifest.json"
    shapes_path = paths.prepared / "materialized_shapes.json"
    preparation_config_path = paths.prepared / "preparation_config.json"
    required = [
        manifest_path,
        shapes_path,
        preparation_config_path,
        paths.prepared / "conditions.parquet",
    ]
    if require_embeddings:
        required.append(paths.prepared / "map_static_tokens.pt")
    manifest = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if split_files is None:
        split_files = list(manifest.get("splits", {}).values())
    resolved_splits = []
    for split_file in split_files:
        split_path = Path(split_file)
        if not split_path.is_absolute():
            split_path = paths.prepared / split_path
        resolved_splits.append(split_path)
    if not resolved_splits:
        required.append(paths.prepared / "manifest.json#splits")
    else:
        required.extend(resolved_splits)
    missing = [str(path) for path in required if not path.is_file()]
    checks = []
    core_ready = manifest_path.is_file() and shapes_path.is_file()
    if core_ready:
        shapes = json.loads(shapes_path.read_text(encoding="utf-8"))
        first_shape = next(iter(shapes.values())) if shapes else {}
        token_length = int(first_shape.get("token_length", 2049))
        hvg_dim = int(first_shape.get("hvg_dim", 2000))
        for population in shapes:
            n_cells = int(shapes[population]["n_cells"])
            root = paths.prepared / population
            files = {
                "genes": (root / "se_gene_ids.uint16.dat", np.uint16, (n_cells, token_length)),
                "expression": (root / "se_expr.float16.dat", np.float16, (n_cells, token_length)),
                "hvg": (root / "hvg.float16.dat", np.float16, (n_cells, hvg_dim)),
                "condition": (root / "row_condition.int32.dat", np.int32, (n_cells,)),
                "group": (root / "row_group.uint16.dat", np.uint16, (n_cells,)),
            }
            if require_embeddings:
                files["embedding"] = (root / "state_embeddings.float16.dat", np.float16, (n_cells, 2048))
            status = {"population": population, "cells": n_cells, "files": {}}
            for name, (path, dtype, shape) in files.items():
                expected = int(np.prod(shape)) * np.dtype(dtype).itemsize
                exists = path.is_file()
                size_ok = exists and path.stat().st_size == expected
                finite = None
                if size_ok and name in {"expression", "hvg", "embedding"}:
                    array = np.memmap(path, dtype=dtype, mode="r", shape=shape)
                    finite = bool(np.isfinite(np.asarray(array[:min(sample_rows, n_cells)])).all())
                status["files"][name] = {"exists": exists, "size_ok": size_ok, "sample_finite": finite}
                if not exists or not size_ok or finite is False:
                    missing.append(str(path))
            checks.append(status)
            report.emit("population checked", population=population, cells=n_cells, status="pass" if not any(not x["exists"] or not x["size_ok"] or x["sample_finite"] is False for x in status["files"].values()) else "fail")
    split_checks = []
    for split_path in resolved_splits:
        if not split_path.is_file():
            continue
        split_payload = json.loads(split_path.read_text(encoding="utf-8"))
        sets = {
            name: {int(value) for value in split_payload.get(name, [])}
            for name in ("train", "val", "test")
        }
        disjoint = not (
            sets["train"] & sets["val"]
            or sets["train"] & sets["test"]
            or sets["val"] & sets["test"]
        )
        if not disjoint or any(not values for values in sets.values()):
            missing.append(str(split_path))
        split_checks.append({
            "split_file": str(split_path),
            "split_id": split_payload.get("split_id", split_path.stem),
            "rule": split_payload.get("rule"),
            "seed": split_payload.get("seed"),
            "counts": {name: len(values) for name, values in sets.items()},
            "condition_sets_disjoint": disjoint,
        })
    missing = list(dict.fromkeys(missing))
    payload = {
        "status": "pass" if not missing else "fail",
        "missing_or_invalid": missing,
        "populations": checks,
        "splits": split_checks,
        "preparation": (
            json.loads(preparation_config_path.read_text(encoding="utf-8"))
            if preparation_config_path.is_file()
            else None
        ),
    }
    output = paths.prepared / "preparation_validation.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    result = report.finish(payload, [output], status="ok" if not missing else "failed")
    if missing:
        raise ValueError(f"Preparation validation failed; see {output}")
    (paths.prepared / "_SUCCESS").write_text(
        json.dumps(
            {
                "status": "ok",
                "default_splits": manifest.get("splits", {}),
                "validated_split_ids": [item["split_id"] for item in split_checks],
                "preparation_id": payload.get("preparation", {}).get("preparation_id")
                if payload.get("preparation")
                else None,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    return result
