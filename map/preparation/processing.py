from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

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


def _load_condition_rows(paths: DatasetPaths, conditions):
    """Load the materialized condition CSR index when available."""
    shapes_path = paths.prepared / "materialized_shapes.json"
    if not shapes_path.is_file():
        return None
    try:
        shapes = json.loads(shapes_path.read_text(encoding="utf-8"))
        rows_by_condition = {}
        for population in shapes:
            base = paths.prepared / str(population)
            ids = np.load(base / "condition_ids.npy")
            offsets = np.load(base / "condition_offsets.npy")
            rows = np.load(base / "condition_rows.npy", mmap_mode="r")
            for index, condition_id in enumerate(ids):
                rows_by_condition[int(condition_id)] = (
                    str(population),
                    np.asarray(rows[int(offsets[index]): int(offsets[index + 1])], dtype=np.int64),
                )
        if not rows_by_condition:
            return None
        expected = set(int(value) for value in conditions["condition_id"])
        if not expected.issubset(rows_by_condition):
            return None
        return rows_by_condition
    except (FileNotFoundError, ValueError, OSError):
        # Keep the condition-only format usable for hand-authored fixtures and
        # materialized data produced by older releases.
        return None


def _assign_internal_rows(condition_rows, seed: int, train_fraction: float = 0.8):
    train_rows = {}
    internal_test_rows = {}
    train_ids = []
    internal_test_ids = []
    for condition_id, (population, rows) in condition_rows.items():
        rows = np.asarray(rows, dtype=np.int64)
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(condition_id), 7103]))
        shuffled = rows[rng.permutation(len(rows))]
        n_train = int(len(shuffled) * float(train_fraction))
        train_rows.setdefault(population, []).extend(int(value) for value in shuffled[:n_train])
        internal_test_rows.setdefault(population, []).extend(
            int(value) for value in shuffled[n_train:]
        )
        if n_train:
            train_ids.append(int(condition_id))
        if n_train < len(shuffled):
            internal_test_ids.append(int(condition_id))
    for values in train_rows.values():
        values.sort()
    for values in internal_test_rows.values():
        values.sort()
    return train_rows, internal_test_rows, train_ids, internal_test_ids


def _row_level_unprofiled_split(
    conditions,
    condition_rows,
    seed: int,
    external_test_size,
    internal_test_fraction,
    external_drugs=None,
):
    internal_fraction = float(internal_test_fraction)
    if not 0 < internal_fraction < 1:
        raise ValueError(
            "internal_test_fraction must be between 0 and 1"
        )
    drugs = sorted(str(value) for value in conditions["drug"].unique())
    requested = None if external_drugs is None else sorted(
        {" ".join(str(value).strip().split()) for value in external_drugs}
    )
    available = set(drugs)
    missing = sorted(set(requested or ()) - available)
    if requested is not None:
        external_test_drugs = sorted(set(requested) & available)
        if not external_test_drugs:
            raise ValueError("None of the requested external_drugs occur in conditions.parquet")
    else:
        n_external = _holdout_count(
            external_test_size, len(drugs), "external_test_size"
        )
        if len(drugs) <= n_external:
            raise RuntimeError("Unprofiled holdout leaves no training drugs")
        rng = np.random.default_rng(seed)
        external_test_drugs = sorted(
            rng.choice(np.asarray(drugs), size=n_external, replace=False).tolist()
        )
    external_set = set(external_test_drugs)
    external_test_rows = {}
    internal = {}
    external_test_conditions = []
    for row in conditions.itertuples(index=False):
        condition_id = int(row.condition_id)
        population, rows = condition_rows[condition_id]
        if str(row.drug) in external_set:
            external_test_conditions.append(condition_id)
            external_test_rows.setdefault(population, []).extend(
                int(value) for value in rows
            )
        else:
            internal[condition_id] = (population, rows)
    train_rows, internal_test_rows, train_ids, internal_test_ids = _assign_internal_rows(
        internal, seed, train_fraction=1.0 - internal_fraction
    )
    for values in external_test_rows.values():
        values.sort()
    return {
        "train": sorted(set(train_ids)),
        "internal_test": sorted(set(internal_test_ids)),
        "external_test": sorted(external_test_conditions),
        "regime": "unprofiled_drug",
        "rule": "unprofiled_drug",
        "seed": int(seed),
        "external_test_size_requested": external_test_size,
        "internal_test_fraction": internal_fraction,
        "split_mode": "held_out_conditions_plus_internal_cell_holdout",
        "train_fraction": 1.0 - internal_fraction,
        "external_test_drugs": external_test_drugs,
        "requested_external_drugs": requested,
        "external_test_drug_count": len(external_test_drugs),
        "missing_external_test_drugs": missing,
        "train_rows": train_rows,
        "internal_test_rows": internal_test_rows,
        "external_test_rows": external_test_rows,
        "row_counts": {
            "train": sum(len(values) for values in train_rows.values()),
            "internal_test": sum(len(values) for values in internal_test_rows.values()),
            "external_test": sum(len(values) for values in external_test_rows.values()),
        },
    }


def _row_level_combination_split(
    conditions, condition_rows, seed, external_test_size, internal_test_fraction, disjoint
):
    internal_fraction = float(internal_test_fraction)
    if not 0 < internal_fraction < 1:
        raise ValueError(
            "internal_test_fraction must be between 0 and 1"
        )
    populations = tuple(sorted(conditions["population"].unique()))
    drugs_by_population = {
        value: set(conditions.loc[conditions.population == value, "drug"])
        for value in populations
    }
    external_by_population = {}
    used = set()
    rng = np.random.default_rng(seed)
    for population in populations:
        candidates = sorted(
            drug for drug in drugs_by_population[population]
            if (not disjoint or drug not in used)
            and any(drug in drugs_by_population[other] for other in populations if other != population)
        )
        count = _holdout_count(
            external_test_size,
            len(drugs_by_population[population]),
            f"external_test_size[{population}]",
        )
        if len(candidates) < count:
            raise RuntimeError(f"Cannot construct unseen combinations for {population}")
        selected = sorted(rng.choice(candidates, size=count, replace=False).tolist())
        external_by_population[population] = selected
        if disjoint:
            used.update(selected)
    external_test_rows, internal = {}, {}
    external_test_ids = []
    for row in conditions.itertuples(index=False):
        condition_id = int(row.condition_id)
        population, rows = condition_rows[condition_id]
        if str(row.drug) in set(external_by_population[population]):
            external_test_ids.append(condition_id)
            external_test_rows.setdefault(population, []).extend(int(value) for value in rows)
        else:
            internal[condition_id] = (population, rows)
    train_rows, internal_test_rows, train_ids, internal_test_ids = _assign_internal_rows(
        internal, seed, train_fraction=1.0 - internal_fraction
    )
    for values in external_test_rows.values():
        values.sort()
    return {
        "train": sorted(set(train_ids)),
        "internal_test": sorted(set(internal_test_ids)),
        "external_test": sorted(external_test_ids),
        "regime": "unseen_combination", "rule": "unseen_combination", "seed": int(seed),
        "external_test_size_requested": external_test_size,
        "internal_test_fraction": internal_fraction,
        "split_mode": "held_out_conditions_plus_internal_cell_holdout",
        "train_fraction": 1.0 - internal_fraction,
        "external_test_drugs_by_population": external_by_population,
        "external_test_drugs_are_disjoint_across_populations": bool(disjoint),
        "train_rows": train_rows,
        "internal_test_rows": internal_test_rows,
        "external_test_rows": external_test_rows,
        "row_counts": {
            "train": sum(len(values) for values in train_rows.values()),
            "internal_test": sum(len(values) for values in internal_test_rows.values()),
            "external_test": sum(len(values) for values in external_test_rows.values()),
        },
    }


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
            "external_test_size_requested": payload["external_test_size_requested"],
            "internal_test_fraction": payload["internal_test_fraction"],
            "counts": payload["counts"],
            "row_counts": payload.get("row_counts"),
            "external_test_drugs": payload.get("external_test_drugs"),
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
    external_test_size: int | float | None = None,
    internal_test_fraction: float = 0.2,
    seed: int = 42,
    disjoint_external_drugs: bool = True,
    output_name: str | None = None,
    workers: int = 8,
    overwrite: bool = False,
    *,
    unprofiled_external_test_size: int | float = 16,
    combination_external_test_size: int | float = 0.05,
    external_drugs: list[str] | tuple[str, ...] | None = None,
) -> StageResult:
    """Generate one identified generalization split."""
    if rule not in {"all", "unprofiled_drug", "unseen_combination"}:
        raise ValueError(f"Unknown split rule: {rule}")
    if external_drugs and rule == "unseen_combination":
        raise ValueError("external_drugs is only valid for rule='unprofiled_drug'")
    shapes_path = paths.prepared / "materialized_shapes.json"
    if not shapes_path.is_file():
        raise FileNotFoundError("materialized_shapes.json is missing; run materialize() first")
    first_population = next(iter(json.loads(shapes_path.read_text(encoding="utf-8"))))
    if not (paths.prepared / first_population / "condition_rows.npy").is_file():
        raise FileNotFoundError(
            "Condition/control indexes are missing; run build_condition_index() first"
        )
    conditions = pd.read_parquet(paths.prepared / "conditions.parquet")
    if "population" not in conditions.columns:
        raise ValueError("conditions.parquet must contain the MAP population field")
    condition_rows = _load_condition_rows(paths, conditions)
    if condition_rows is None:
        raise RuntimeError(
            "The split contract requires condition row indexes; rebuild the sampling index"
        )

    split_specs: dict[str, dict] = {}
    if rule in {"all", "unprofiled_drug"}:
        effective_external = (
            unprofiled_external_test_size
            if external_test_size is None or rule == "all"
            else external_test_size
        )
        split_id = split_identifier(
            "unprofiled_drug", effective_external, internal_test_fraction, seed,
            external_drugs=external_drugs,
        )
        split_specs["unprofiled_drug"] = {
            "split_id": split_id,
            "external_test_size": effective_external,
            "filename": f"{split_id}.json",
        }
    if rule in {"all", "unseen_combination"}:
        effective_external = (
            combination_external_test_size
            if external_test_size is None or rule == "all"
            else external_test_size
        )
        split_id = split_identifier(
            "unseen_combination",
            effective_external,
            internal_test_fraction,
            seed,
            disjoint_external_drugs=disjoint_external_drugs,
        )
        split_specs["unseen_combination"] = {
            "split_id": split_id,
            "external_test_size": effective_external,
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
        for regime, payload in payloads.items():
            required = {"train", "internal_test", "external_test"}
            if not required.issubset(payload):
                raise ValueError(
                    f"Existing {regime} split uses the legacy val/test contract; "
                    "choose a new output name or pass overwrite=True"
                )
    else:
        payloads = {}
        for regime, spec in split_specs.items():
            if regime == "unprofiled_drug":
                payload = _row_level_unprofiled_split(
                    conditions, condition_rows, seed, spec["external_test_size"],
                    internal_test_fraction, external_drugs,
                )
            else:
                payload = _row_level_combination_split(
                    conditions, condition_rows, seed, spec["external_test_size"],
                    internal_test_fraction, disjoint_external_drugs,
                )
            payload["split_id"] = spec["split_id"]
            payload["split_file"] = str(target_paths[regime].relative_to(paths.prepared))
            payload["counts"] = {
                name: len(payload[name])
                for name in ("train", "internal_test", "external_test")
            }
            target_paths[regime].parent.mkdir(parents=True, exist_ok=True)
            target_paths[regime].write_text(json.dumps(payload, indent=2), encoding="utf-8")
            payloads[regime] = payload
        _update_prepared_manifest(paths, payloads, target_paths, shapes_path)

    split_counts = {
        regime: {
            name: len(payload.get(name, []))
            for name in ("train", "internal_test", "external_test")
        }
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
            "external_test_size": payload["external_test_size_requested"],
            "internal_test_fraction": payload["internal_test_fraction"],
            "row_counts": payload.get("row_counts"),
            "external_test_drugs": payload.get("external_test_drugs"),
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
