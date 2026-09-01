from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .._common.feedback import Feedback, StageResult
from .._common.identifiers import split_identifier
from .._common.paths import DatasetPaths


# The published validation protocol used this fixed external-drug panel.  It
# remains available as an explicit preset; generic datasets default to seeded
# selection unless callers pass this panel through ``external_drugs``.
PAPER_UNPROFILED_DRUGS = (
    "BI-78D3",
    "Balsalazide (sodium hydrate)",
    "Bergenin",
    "Bortezomib",
    "Brivudine",
    "CP21R7",
    "Carbidopa (monohydrate)",
    "Ciclopirox",
    "Drospirenone",
    "ERK5-IN-2",
    "Estrone sulfate (potassium)",
    "Idarubicin (hydrochloride)",
    "Lidocaine (hydrochloride)",
    "Nafamostat (mesylate)",
    "Ralimetinib dimesylate",
    "Sildenafil",
    "Sivelestat (sodium tetrahydrate)",
    "ULK-101",
)


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


def _legacy_pair_groups(conditions, condition_rows):
    groups = {}
    for row in conditions.itertuples(index=False):
        condition_id = int(row.condition_id)
        population, rows = condition_rows[condition_id]
        # MAP-validation samples each (cell-line, drug, concentration) combo
        # independently (max 5000 cells), then sequential evaluation merges
        # concentrations back to one cell-line–drug entity.
        key = (str(population), str(row.drug), float(row.dose))
        entry = groups.setdefault(key, {
            "population": str(population), "drug": str(row.drug),
            "ids": [], "rows": [],
        })
        entry["ids"].append(condition_id)
        entry["rows"].extend(int(value) for value in rows)
    for entry in groups.values():
        entry["ids"] = sorted(set(entry["ids"]))
        entry["rows"] = np.asarray(sorted(set(entry["rows"])), dtype=np.int64)
    return groups


def _legacy_group_split(groups, external_keys, seed, internal_fraction):
    train_rows, internal_rows, external_rows = {}, {}, {}
    train_ids, internal_ids, external_ids = [], [], []
    rng = np.random.default_rng(int(seed))
    external_keys = set(external_keys)
    for key in sorted(groups):
        entry = groups[key]
        rows = entry["rows"]
        if key in external_keys:
            selected = rng.choice(rows, min(5000, len(rows)), replace=False)
            external_rows.setdefault(entry["population"], []).extend(map(int, selected))
            external_ids.extend(entry["ids"])
            continue
        selected = rng.choice(rows, min(5000, len(rows)), replace=False)
        split_point = int(len(selected) * (1.0 - float(internal_fraction)))
        train = selected[:split_point]
        internal = selected[split_point:]
        train_rows.setdefault(entry["population"], []).extend(map(int, train))
        internal_rows.setdefault(entry["population"], []).extend(map(int, internal))
        if len(train):
            train_ids.extend(entry["ids"])
        if len(internal):
            internal_ids.extend(entry["ids"])
    for values in (*train_rows.values(), *internal_rows.values(), *external_rows.values()):
        values.sort()
    return {
        "train": sorted(set(train_ids)),
        "internal_test": sorted(set(internal_ids)),
        "external_test": sorted(set(external_ids)),
        "train_rows": train_rows,
        "internal_test_rows": internal_rows,
        "external_test_rows": external_rows,
        "row_counts": {
            "train": sum(map(len, train_rows.values())),
            "internal_test": sum(map(len, internal_rows.values())),
            "external_test": sum(map(len, external_rows.values())),
        },
    }


def _row_level_unprofiled_split(
    conditions,
    condition_rows,
    seed: int,
    external_test_size,
    internal_test_fraction,
    external_drugs=None,
):
    internal_fraction = float(internal_test_fraction)
    if not 0 <= internal_fraction < 1:
        raise ValueError(
            "internal_test_fraction must be between 0 (inclusive) and 1"
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
    groups = _legacy_pair_groups(conditions, condition_rows)
    sampled = _legacy_group_split(
        groups,
        {key for key, entry in groups.items() if entry["drug"] in set(external_test_drugs)},
        seed,
        internal_fraction,
    )
    return {
        "train": sampled["train"],
        "internal_test": sampled["internal_test"],
        "external_test": sampled["external_test"],
        "regime": "unprofiled_drug",
        "rule": "unprofiled_drug",
        "seed": int(seed),
        "external_test_size_requested": external_test_size,
        "internal_test_fraction": internal_fraction,
        "split_mode": "legacy_cell_line_drug_sampled_rows",
        "train_fraction": 1.0 - internal_fraction,
        "external_test_drugs": external_test_drugs,
        "requested_external_drugs": requested,
        "external_test_drug_count": len(external_test_drugs),
        "missing_external_test_drugs": missing,
        "train_rows": sampled["train_rows"],
        "internal_test_rows": sampled["internal_test_rows"],
        "external_test_rows": sampled["external_test_rows"],
        "row_counts": sampled["row_counts"],
    }


def _row_level_combination_split(
    conditions, condition_rows, seed, external_test_size, internal_test_fraction, disjoint
):
    internal_fraction = float(internal_test_fraction)
    if not 0 <= internal_fraction < 1:
        raise ValueError(
            "internal_test_fraction must be between 0 (inclusive) and 1"
        )
    populations = tuple(sorted(conditions["population"].unique()))
    drugs_by_population = {
        value: set(conditions.loc[conditions.population == value, "drug"])
        for value in populations
    }
    external_by_population = {}
    used = set()
    rng = np.random.default_rng(seed)
    allocation_order = tuple(sorted(populations, key=lambda value: len(drugs_by_population[value])))
    for population in allocation_order:
        # Legacy allocation walks cell lines from fewest to most drugs and
        # removes already allocated drugs globally.  It does not require the
        # held-out drug to occur in another cell line; that detail matters for
        # reproducing the old unseen-combination panel.
        candidates = sorted(
            drug for drug in drugs_by_population[population]
            if (not disjoint or drug not in used)
        )
        count = (
            max(1, int(float(external_test_size) * len(drugs_by_population[population])))
            if float(external_test_size) < 1
            else _holdout_count(
                external_test_size,
                len(drugs_by_population[population]),
                f"external_test_size[{population}]",
            )
        )
        if len(candidates) < count:
            raise RuntimeError(f"Cannot construct unseen combinations for {population}")
        selected = sorted(rng.choice(candidates, size=count, replace=False).tolist())
        external_by_population[population] = selected
        if disjoint:
            used.update(selected)
    groups = _legacy_pair_groups(conditions, condition_rows)
    external_keys = {
        key
        for key, entry in groups.items()
        if entry["population"] in external_by_population
        and entry["drug"] in set(external_by_population[entry["population"]])
    }
    sampled = _legacy_group_split(groups, external_keys, seed, internal_fraction)
    return {
        "train": sampled["train"],
        "internal_test": sampled["internal_test"],
        "external_test": sampled["external_test"],
        "regime": "unseen_combination", "rule": "unseen_combination", "seed": int(seed),
        "external_test_size_requested": external_test_size,
        "internal_test_fraction": internal_fraction,
        "split_mode": "legacy_cell_line_drug_sampled_rows",
        "train_fraction": 1.0 - internal_fraction,
        "external_test_drugs_by_population": external_by_population,
        "external_test_drugs_are_disjoint_across_populations": bool(disjoint),
        "train_rows": sampled["train_rows"],
        "internal_test_rows": sampled["internal_test_rows"],
        "external_test_rows": sampled["external_test_rows"],
        "row_counts": sampled["row_counts"],
    }


def _combosciplex_split(conditions, condition_rows, seed, internal_test_fraction):
    """ComboSciPlex protocol: train on singles, classify held-out pairs by seen drugs."""
    fraction = float(internal_test_fraction)
    # ``0`` is useful for the paper-style combination benchmark: train on all
    # single-drug rows and evaluate only on held-out two-drug conditions.  The
    # generic split validator treats this as an intentional empty internal set
    # for the ComboSciPlex rule.
    if not 0 <= fraction < 1:
        raise ValueError("internal_test_fraction must be between 0 (inclusive) and 1")
    components = {}
    for row in conditions.itertuples(index=False):
        values = getattr(row, "component_smiles", None)
        if hasattr(values, "tolist"):
            values = values.tolist()
        if isinstance(values, str):
            values = [values]
        if not isinstance(values, (list, tuple)):
            values = [getattr(row, "canonical_smiles")]
        components[int(row.condition_id)] = tuple(str(value) for value in values)
    seen = {value for values in components.values() if len(values) == 1 for value in values}
    train_rows, internal_rows = {}, {}
    train_ids, internal_ids = [], []
    for condition_id, (population, rows) in condition_rows.items():
        if len(components.get(int(condition_id), ())) != 1:
            continue
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(condition_id), 7103]))
        selected = np.asarray(rows, dtype=np.int64)[rng.permutation(len(rows))]
        split_point = int(len(selected) * (1.0 - fraction))
        train_rows.setdefault(population, []).extend(map(int, selected[:split_point]))
        internal_rows.setdefault(population, []).extend(map(int, selected[split_point:]))
        if split_point:
            train_ids.append(int(condition_id))
        if split_point < len(selected):
            internal_ids.append(int(condition_id))
    category_rows = {"both_seen": {}, "one_seen": {}, "both_unseen": {}}
    category_ids = {key: [] for key in category_rows}
    grouped = {}
    for row in conditions.itertuples(index=False):
        cid = int(row.condition_id)
        values = components[cid]
        if len(values) < 2:
            continue
        key = (str(row.population), str(getattr(row, "combination_key", "|".join(sorted(values)))))
        grouped.setdefault(key, []).append(cid)
    for (population, _), ids in sorted(grouped.items()):
        values = components[ids[0]]
        n_seen = sum(value in seen for value in values)
        category = "both_seen" if n_seen == len(values) else "both_unseen" if n_seen == 0 else "one_seen"
        rows = np.concatenate([np.asarray(condition_rows[cid][1], dtype=np.int64) for cid in ids])
        category_rows[category].setdefault(population, []).extend(map(int, rows))
        category_ids[category].extend(ids)
    for values in (*train_rows.values(), *internal_rows.values(), *(rows for category in category_rows.values() for rows in category.values())):
        values.sort()
    external_ids = sorted(set(sum(category_ids.values(), [])))
    external_rows = {pop: sorted(set(sum((category.get(pop, []) for category in category_rows.values()), []))) for pop in set().union(*(set(category) for category in category_rows.values()))}
    payload = {
        "train": sorted(set(train_ids)), "internal_test": sorted(set(internal_ids)),
        "external_test": external_ids,
        "both_seen": sorted(set(category_ids["both_seen"])),
        "one_seen": sorted(set(category_ids["one_seen"])),
        "both_unseen": sorted(set(category_ids["both_unseen"])),
        "train_rows": train_rows, "internal_test_rows": internal_rows,
        "external_test_rows": external_rows,
        "both_seen_rows": category_rows["both_seen"],
        "one_seen_rows": category_rows["one_seen"],
        "both_unseen_rows": category_rows["both_unseen"],
        "row_counts": {"train": sum(map(len, train_rows.values())), "internal_test": sum(map(len, internal_rows.values())), "external_test": sum(map(len, external_rows.values()))},
        "regime": "combosciplex", "rule": "combosciplex", "seed": int(seed),
        "split_mode": "combosciplex_single_drug_train",
    }
    return payload


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
    if rule not in {"all", "unprofiled_drug", "unseen_combination", "combosciplex"}:
        raise ValueError(f"Unknown split rule: {rule}")
    internal_test_fraction = float(internal_test_fraction)
    if not np.isfinite(internal_test_fraction) or not 0 <= internal_test_fraction < 1:
        raise ValueError(
            "internal_test_fraction must be between 0 (inclusive) and 1"
        )
    if external_drugs and rule == "unseen_combination":
        raise ValueError("external_drugs is only valid for rule='unprofiled_drug'")
    shapes_path = paths.prepared / "materialized_shapes.json"
    if not shapes_path.is_file():
        raise FileNotFoundError(
            "materialized_shapes.json is missing; prepare cell metadata, STATE inputs, and HVG expression first"
        )
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
    effective_external_drugs = None if external_drugs is None else list(external_drugs)
    if rule in {"all", "unprofiled_drug"}:
        effective_external = (
            unprofiled_external_test_size
            if external_test_size is None or rule == "all"
            else external_test_size
        )
        split_id = split_identifier(
            "unprofiled_drug", effective_external, internal_test_fraction, seed,
            external_drugs=effective_external_drugs,
        )
        split_specs["unprofiled_drug"] = {
            "split_id": split_id,
            "external_test_size": (
                unprofiled_external_test_size
                if external_test_size is None or rule == "all"
                else external_test_size
            ),
            "external_drugs": effective_external_drugs,
            "filename": "split.json",
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
            "filename": "split.json",
        }
    if rule in {"all", "combosciplex"}:
        split_id = split_identifier("combosciplex", 0.0 + combination_external_test_size, internal_test_fraction, seed)
        split_specs["combosciplex"] = {
            "split_id": split_id, "external_test_size": combination_external_test_size,
            "filename": "split.json",
        }
    if output_name is not None:
        if rule == "all":
            raise ValueError("output_name is only valid when generating one split rule")
        if Path(output_name).name != output_name:
            raise ValueError("output_name must be a split_id, not a path")
        split_specs[rule]["split_id"] = output_name.removesuffix(".json")

    target_paths = {
        regime: paths.split_file(spec["split_id"])
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
                    internal_test_fraction, effective_external_drugs,
                )
            elif regime == "combosciplex":
                payload = _combosciplex_split(
                    conditions, condition_rows, seed, internal_test_fraction,
                )
            else:
                payload = _row_level_combination_split(
                    conditions, condition_rows, seed, spec["external_test_size"],
                    internal_test_fraction, disjoint_external_drugs,
                )
            # Keep one summary schema across split rules.  ComboSciPlex uses
            # all observed pairs as external conditions, so its helper does
            # not need the requested size to construct the split, but the
            # value is still recorded for reproducibility and reporting.
            payload.setdefault(
                "external_test_size_requested", spec["external_test_size"]
            )
            payload.setdefault("internal_test_fraction", float(internal_test_fraction))
            payload["split_id"] = spec["split_id"]
            payload["split_file"] = str(target_paths[regime].relative_to(paths.workspace))
            payload["counts"] = {
                name: len(payload[name])
                for name in ("train", "internal_test", "external_test")
            }
            target_paths[regime].parent.mkdir(parents=True, exist_ok=True)
            target_paths[regime].write_text(json.dumps(payload, indent=2), encoding="utf-8")
            payloads[regime] = payload
    split_counts = {
        regime: {
            name: len(payload.get(name, []))
            for name in ("train", "internal_test", "external_test")
        }
        for regime, payload in payloads.items()
    }
    split_files = {regime: str(path.resolve()) for regime, path in target_paths.items()}
    outputs = [
        *target_paths.values(),
    ]
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
