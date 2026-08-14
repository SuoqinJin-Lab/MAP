from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .._common.feedback import Feedback, StageResult
from .._common.identifiers import split_identifier
from .._common.paths import DatasetPaths


def _result(paths: DatasetPaths, stage: str, summary: dict, outputs: list[Path]) -> StageResult:
    report = Feedback(paths.prepared, stage)
    printable = {
        key: value for key, value in summary.items()
        if isinstance(value, (str, int, float, bool))
    }
    report.emit("summary", **printable)
    return report.finish(summary, outputs)


def _write_csr(base: Path, prefix: str, group_ids: np.ndarray, rows: np.ndarray) -> None:
    order = np.lexsort((rows, group_ids))
    sorted_groups = group_ids[order]
    sorted_rows = rows[order].astype(np.int64, copy=False)
    unique, starts = np.unique(sorted_groups, return_index=True)
    np.save(base / f"{prefix}_ids.npy", unique)
    np.save(
        base / f"{prefix}_offsets.npy",
        np.concatenate([starts, [len(sorted_rows)]]).astype(np.int64),
    )
    np.save(base / f"{prefix}_rows.npy", sorted_rows)


def _write_condition_indexes(root: Path, shapes: dict) -> None:
    for population, shape in shapes.items():
        base = root / population
        n_cells = int(shape["n_cells"])
        conditions = np.memmap(
            base / "row_condition.int32.dat", dtype=np.int32, mode="r", shape=(n_cells,)
        )
        groups = np.memmap(
            base / "row_group.uint16.dat", dtype=np.uint16, mode="r", shape=(n_cells,)
        )
        rows = np.arange(n_cells, dtype=np.int64)
        condition_mask = conditions >= 0
        control_mask = ~condition_mask
        missing = sorted(
            set(int(value) for value in np.unique(groups[condition_mask]))
            - set(int(value) for value in np.unique(groups[control_mask]))
        )
        if missing:
            raise RuntimeError(
                f"{population} has condition cells without matched controls in groups: {missing}"
            )
        _write_csr(base, "condition", np.asarray(conditions[condition_mask]), rows[condition_mask])
        _write_csr(base, "control_group", np.asarray(groups[control_mask]), rows[control_mask])


def _holdout_count(value: int | float, total: int, label: str) -> int:
    numeric = float(value)
    if numeric <= 0:
        raise ValueError(f"{label} must be positive")
    return max(1, int(round(numeric * total)) if numeric < 1 else int(round(numeric)))


def _unprofiled_split(conditions, seed: int, test_size, val_size) -> dict:
    rng = np.random.default_rng(seed)
    drugs = np.asarray(sorted(conditions["drug"].unique()))
    n_test = _holdout_count(test_size, len(drugs), "test_size")
    n_val = _holdout_count(val_size, len(drugs), "val_size")
    if len(drugs) <= n_test + n_val:
        raise RuntimeError("Holdouts leave no training drugs")
    shuffled = rng.permutation(drugs)
    test_drugs = set(shuffled[:n_test].tolist())
    val_drugs = set(shuffled[n_test:n_test + n_val].tolist())
    payload = {"train": [], "val": [], "test": []}
    for row in conditions.itertuples(index=False):
        group = "test" if row.drug in test_drugs else "val" if row.drug in val_drugs else "train"
        payload[group].append(int(row.condition_id))
    payload.update({
        "regime": "unprofiled_drug", "rule": "unprofiled_drug", "seed": int(seed),
        "test_size_requested": test_size, "validation_size_requested": val_size,
        "test_drugs": sorted(test_drugs), "validation_drugs": sorted(val_drugs),
    })
    return payload


def _combination_split(conditions, seed: int, test_size, val_size, disjoint: bool) -> dict:
    rng = np.random.default_rng(seed)
    populations = tuple(sorted(conditions["population"].unique()))
    drugs_by_population = {
        value: set(conditions.loc[conditions.population == value, "drug"])
        for value in populations
    }
    test_by_population: dict[str, list[str]] = {}
    used: set[str] = set()
    for population in populations:
        candidates = sorted(
            drug for drug in drugs_by_population[population]
            if (not disjoint or drug not in used)
            and any(drug in drugs_by_population[other] for other in populations if other != population)
        )
        count = _holdout_count(test_size, len(drugs_by_population[population]), f"test_size[{population}]")
        if len(candidates) < count:
            raise RuntimeError(f"Cannot construct unseen combinations for {population}")
        selected = sorted(rng.choice(candidates, size=count, replace=False).tolist())
        test_by_population[population] = selected
        if disjoint:
            used.update(selected)
    val_by_population = {}
    for population in populations:
        candidates = sorted(drugs_by_population[population] - set(test_by_population[population]))
        count = _holdout_count(val_size, len(drugs_by_population[population]), f"val_size[{population}]")
        val_by_population[population] = sorted(
            rng.choice(candidates, size=count, replace=False).tolist()
        )
    payload = {"train": [], "val": [], "test": []}
    for row in conditions.itertuples(index=False):
        group = (
            "test" if row.drug in test_by_population[row.population]
            else "val" if row.drug in val_by_population[row.population]
            else "train"
        )
        payload[group].append(int(row.condition_id))
    payload.update({
        "regime": "unseen_combination", "rule": "unseen_combination", "seed": int(seed),
        "test_size_requested": test_size, "validation_size_requested": val_size,
        "test_drugs_by_population": test_by_population,
        "validation_drugs_by_population": val_by_population,
        "test_drugs_are_disjoint_across_populations": bool(disjoint),
    })
    return payload


def _update_prepared_manifest(
    paths: DatasetPaths,
    payloads: dict[str, dict],
    target_paths: dict[str, Path],
    shapes_path: Path,
) -> None:
    manifest_path = paths.prepared / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file()
        else {}
    )
    stats_path = paths.prepared / "stats_manifest.json"
    if stats_path.is_file():
        manifest.update(json.loads(stats_path.read_text(encoding="utf-8")))
    preparation_path = paths.prepared / "preparation_config.json"
    if preparation_path.is_file():
        manifest["preparation"] = json.loads(preparation_path.read_text(encoding="utf-8"))
    manifest.setdefault("splits", {})
    manifest.setdefault("split_registry", {})
    for regime, payload in payloads.items():
        relative = str(target_paths[regime].relative_to(paths.prepared))
        manifest["splits"][regime] = relative
        manifest["split_registry"][payload["split_id"]] = {
            "path": relative,
            "rule": payload["rule"],
            "seed": payload["seed"],
            "test_size_requested": payload["test_size_requested"],
            "validation_size_requested": payload["validation_size_requested"],
            "counts": payload["counts"],
        }
    manifest.update({
        "format_version": 1,
        "materialized_shapes": json.loads(shapes_path.read_text(encoding="utf-8")),
        "conditions": "conditions.parquet",
        "normalization": "library_size_10000_log1p",
        "state_expression_encoding": "100 * selected_log_expression / selected_log_expression_sum",
        "control_matching": "population_and_group",
    })
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def build_condition_index(
    paths: DatasetPaths,
    workers: int = 8,
    control_match: str = "population_and_group",
    overwrite: bool = False,
) -> StageResult:
    if control_match != "population_and_group":
        raise ValueError("The current sampler supports control_match='population_and_group'")
    shapes = json.loads(
        (paths.prepared / "materialized_shapes.json").read_text(encoding="utf-8")
    )
    complete = all(
        (paths.prepared / population / "condition_rows.npy").is_file()
        and (paths.prepared / population / "control_group_rows.npy").is_file()
        for population in shapes
    )
    if complete and not overwrite:
        return _result(paths, "condition_index", {
            "control_match": control_match,
            "workers": workers,
            "populations": len(shapes),
            "reused": True,
        }, [paths.prepared / "conditions.parquet"])
    _write_condition_indexes(paths.prepared, shapes)
    outputs = []
    for population in shapes:
        population_dir = paths.prepared / population
        outputs.extend(population_dir / name for name in (
            "condition_ids.npy", "condition_offsets.npy", "condition_rows.npy",
            "control_group_ids.npy", "control_group_offsets.npy", "control_group_rows.npy",
        ))

    import pyarrow.parquet as pq
    condition_count = pq.ParquetFile(paths.prepared / "conditions.parquet").metadata.num_rows
    return _result(
        paths,
        "condition_index",
        {
            "control_match": "population_and_group",
            "workers": workers,
            "populations": len(shapes),
            "conditions": int(condition_count),
            "reused": False,
        },
        outputs,
    )


def generate_splits(
    paths: DatasetPaths,
    rule: str = "all",
    test_size: int | float | None = None,
    val_size: int | float | None = None,
    seed: int = 42,
    disjoint_test_drugs: bool = True,
    output_name: str | None = None,
    workers: int = 8,
    overwrite: bool = False,
    *,
    unprofiled_test_size: int | float = 16,
    unprofiled_val_size: int | float = 16,
    combination_test_size: int | float = 0.05,
    combination_val_size: int | float = 0.05,
) -> StageResult:
    """Generate one identified generalization split."""
    if rule not in {"all", "unprofiled_drug", "unseen_combination"}:
        raise ValueError(f"Unknown split rule: {rule}")
    shapes_path = paths.prepared / "materialized_shapes.json"
    if not shapes_path.is_file():
        raise FileNotFoundError("materialized_shapes.json is missing; run materialize() first")
    first_population = next(iter(json.loads(shapes_path.read_text(encoding="utf-8"))))
    if not (paths.prepared / first_population / "condition_rows.npy").is_file():
        raise FileNotFoundError(
            "Condition/control indexes are missing; run build_condition_index() first"
        )

    split_specs: dict[str, dict] = {}
    if rule in {"all", "unprofiled_drug"}:
        effective_test = unprofiled_test_size if test_size is None or rule == "all" else test_size
        effective_val = unprofiled_val_size if val_size is None or rule == "all" else val_size
        split_id = split_identifier("unprofiled_drug", effective_test, effective_val, seed)
        split_specs["unprofiled_drug"] = {
            "split_id": split_id,
            "test_size": effective_test,
            "val_size": effective_val,
            "filename": f"{split_id}.json",
        }
    if rule in {"all", "unseen_combination"}:
        effective_test = combination_test_size if test_size is None or rule == "all" else test_size
        effective_val = combination_val_size if val_size is None or rule == "all" else val_size
        split_id = split_identifier(
            "unseen_combination",
            effective_test,
            effective_val,
            seed,
            disjoint_test_drugs=disjoint_test_drugs,
        )
        split_specs["unseen_combination"] = {
            "split_id": split_id,
            "test_size": effective_test,
            "val_size": effective_val,
            "filename": f"{split_id}.json",
        }
    if output_name is not None:
        if rule == "all":
            raise ValueError("output_name is only valid when generating one split rule")
        if Path(output_name).name != output_name:
            raise ValueError("output_name must be a file name, not a path")
        if not output_name.endswith(".json"):
            output_name += ".json"
        split_specs[rule]["filename"] = output_name
        split_specs[rule]["split_id"] = Path(output_name).stem

    target_paths = {
        regime: paths.prepared / "splits" / spec["filename"]
        for regime, spec in split_specs.items()
    }
    existing = [path for path in target_paths.values() if path.is_file()]
    if existing and not overwrite:
        if len(existing) != len(target_paths):
            raise FileExistsError(
                "Only part of the requested split set exists; pass overwrite=True or use a new seed"
            )
        payloads = {
            regime: json.loads(path.read_text(encoding="utf-8"))
            for regime, path in target_paths.items()
        }
    else:
        import pandas as pd

        conditions = pd.read_parquet(paths.prepared / "conditions.parquet")
        if "population" not in conditions.columns:
            raise ValueError("conditions.parquet must contain the MAP population field")
        payloads = {}
        for regime, spec in split_specs.items():
            payload = (
                _unprofiled_split(conditions, seed, spec["test_size"], spec["val_size"])
                if regime == "unprofiled_drug"
                else _combination_split(
                    conditions, seed, spec["test_size"], spec["val_size"],
                    disjoint_test_drugs,
                )
            )
            payload["split_id"] = spec["split_id"]
            payload["split_file"] = str(target_paths[regime].relative_to(paths.prepared))
            payload["counts"] = {name: len(payload[name]) for name in ("train", "val", "test")}
            target_paths[regime].parent.mkdir(parents=True, exist_ok=True)
            target_paths[regime].write_text(json.dumps(payload, indent=2), encoding="utf-8")
            payloads[regime] = payload
        _update_prepared_manifest(paths, payloads, target_paths, shapes_path)

    split_counts = {
        regime: {name: len(payload.get(name, [])) for name in ("train", "val", "test")}
        for regime, payload in payloads.items()
    }
    split_files = {regime: str(path.resolve()) for regime, path in target_paths.items()}
    outputs = [paths.prepared / "manifest.json", *target_paths.values()]
    stage_id = next(iter(split_specs.values()))["split_id"] if len(split_specs) == 1 else f"defaults_seed-{seed}"
    summary = {
        "rule": rule,
        "seed": seed,
        "split_ids": {regime: payload["split_id"] for regime, payload in payloads.items()},
        "split_files": split_files,
        "counts": split_counts,
        "reused": bool(existing and not overwrite),
    }
    if len(payloads) == 1:
        regime, payload = next(iter(payloads.items()))
        summary.update({
            "split_id": payload["split_id"],
            "split_file": split_files[regime],
            "test_size": payload["test_size_requested"],
            "val_size": payload["validation_size_requested"],
        })
    return _result(
        paths,
        f"splits_{stage_id}",
        summary,
        outputs,
    )


def index_and_split(
    paths: DatasetPaths,
    workers: int = 8,
    seed: int = 42,
    control_match: str = "population_and_group",
    overwrite: bool = False,
    **split_kwargs,
) -> StageResult:
    build_condition_index(paths, workers, control_match, overwrite)
    splits = generate_splits(
        paths, workers=workers, seed=seed, overwrite=overwrite, **split_kwargs
    )
    outputs = [Path(path) for path in splits.outputs]
    return _result(paths, "index_and_split", {"seed": seed, "control_match": control_match}, outputs)
