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
        paths.prepared / "condition_filter.json",
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
        token_length = int(first_shape.get("token_length", 2048))
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
    condition_filter_check = None
    filter_path = paths.prepared / "condition_filter.json"
    if filter_path.is_file() and (paths.prepared / "conditions.parquet").is_file():
        import pyarrow.parquet as pq

        condition_filter = json.loads(filter_path.read_text(encoding="utf-8"))
        condition_cells = np.asarray(
            pq.read_table(
                paths.prepared / "conditions.parquet", columns=["n_cells"]
            ).column(0)
        )
        minimum = int(condition_filter["min_cells"])
        maximum = int(condition_filter["max_cells"])
        within_bounds = bool(
            condition_cells.size
            and np.all(condition_cells >= minimum)
            and np.all(condition_cells <= maximum)
        )
        retained_matches = (
            int(condition_cells.sum())
            == int(condition_filter["retained_condition_cells"])
        )
        control_counts = {}
        if shapes_path.is_file():
            for population, shape in json.loads(
                shapes_path.read_text(encoding="utf-8")
            ).items():
                n_cells = int(shape["n_cells"])
                condition_path = paths.prepared / population / "row_condition.int32.dat"
                if not condition_path.is_file():
                    control_counts[population] = -1
                    continue
                condition_array = np.memmap(
                    condition_path,
                    dtype=np.int32,
                    mode="r",
                    shape=(n_cells,),
                )
                control_counts[population] = int(np.count_nonzero(condition_array < 0))
        controls_within_cap = all(
            0 <= value <= maximum for value in control_counts.values()
        )
        retained_controls_match = (
            not control_counts
            or int(sum(control_counts.values()))
            == int(condition_filter.get("retained_control_cells", sum(control_counts.values())))
        )
        preparation = (
            json.loads(preparation_config_path.read_text(encoding="utf-8"))
            if preparation_config_path.is_file()
            else {}
        )
        id_matches = (
            preparation.get("condition_filter", {}).get("filter_id")
            == condition_filter.get("filter_id")
        )
        condition_filter_check = {
            "filter_id": condition_filter.get("filter_id"),
            "min_cells": minimum,
            "max_cells": maximum,
            "conditions_within_bounds": within_bounds,
            "retained_cells_match": retained_matches,
            "control_cells_by_population": control_counts,
            "controls_within_cap": controls_within_cap,
            "retained_controls_match": retained_controls_match,
            "preparation_id_matches": id_matches,
        }
        if (
            not within_bounds
            or not retained_matches
            or not controls_within_cap
            or not retained_controls_match
            or not id_matches
        ):
            missing.append(str(filter_path))
    split_checks = []
    for split_path in resolved_splits:
        if not split_path.is_file():
            continue
        split_payload = json.loads(split_path.read_text(encoding="utf-8"))
        sets = {
            name: {int(value) for value in split_payload.get(name, [])}
            for name in ("train", "internal_test", "external_test")
        }
        external_disjoint = not (
            sets["external_test"] & (sets["train"] | sets["internal_test"])
        )
        row_membership = {}
        for name in ("train", "internal_test", "external_test"):
            membership = {
                str(population): np.asarray(rows, dtype=np.int64)
                for population, rows in split_payload.get(f"{name}_rows", {}).items()
            }
            for population, filename in split_payload.get(
                f"{name}_rows_files", {}
            ).items():
                row_path = Path(filename)
                if not row_path.is_absolute():
                    row_path = paths.prepared / row_path
                if not row_path.is_file():
                    missing.append(str(row_path))
                    continue
                population = str(population)
                if population in membership:
                    missing.append(
                        f"{split_path} defines duplicate row sources for {name}/{population}"
                    )
                    continue
                membership[population] = np.load(
                    row_path, mmap_mode="r"
                )
            row_membership[name] = membership
        has_row_protocol = any(row_membership[name] for name in row_membership)
        row_disjoint = True
        if has_row_protocol:
            populations = set().union(
                *(set(values) for values in row_membership.values())
            )
            for population in populations:
                arrays = [
                    np.asarray(row_membership[name].get(population, ()), dtype=np.int64)
                    for name in ("train", "internal_test", "external_test")
                ]
                if any(len(np.unique(values)) != len(values) for values in arrays):
                    row_disjoint = False
                    break
                if any(
                    np.intersect1d(arrays[left], arrays[right], assume_unique=False).size
                    for left, right in ((0, 1), (0, 2), (1, 2))
                ):
                    row_disjoint = False
                    break
        valid_sets = (
            all(
                sum(len(rows) for rows in row_membership[name].values()) > 0
                for name in row_membership
            )
            if has_row_protocol
            else all(sets[name] for name in sets)
        )
        if (not row_disjoint if has_row_protocol else not external_disjoint) or not valid_sets:
            missing.append(str(split_path))
        split_checks.append({
            "split_file": str(split_path),
            "split_id": split_payload.get("split_id", split_path.stem),
            "rule": split_payload.get("rule"),
            "seed": split_payload.get("seed"),
            "counts": {name: len(values) for name, values in sets.items()},
            "external_condition_set_disjoint": external_disjoint,
            "row_sets_disjoint": row_disjoint if has_row_protocol else None,
            "row_protocol": has_row_protocol,
            "row_counts": {
                name: sum(
                    len(rows) for rows in row_membership[name].values()
                )
                for name in row_membership
            } if has_row_protocol else None,
        })
    missing = list(dict.fromkeys(missing))
    payload = {
        "status": "pass" if not missing else "fail",
        "missing_or_invalid": missing,
        "populations": checks,
        "splits": split_checks,
        "condition_filter": condition_filter_check,
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
