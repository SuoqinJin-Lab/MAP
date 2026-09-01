from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from .components import combination_key, normalize_components


class MAPDataset(Dataset):
    """Sample condition cells with controls matched by population and group."""

    FIELD_DTYPES: dict[str, object] = {}

    DATA_FIELDS = frozenset({
        "control_gene_ids",
        "control_expressions",
        "condition_gene_ids",
        "condition_expressions",
        "control_embeddings",
        "condition_embeddings",
        "condition_hvg_vectors",
        "control_hvg_vectors",
        "condition_rows",
        "control_rows",
    })

    def __init__(
        self,
        data_dir: str | Path,
        regime: str,
        split: str,
        *,
        set_size: int = 24,
        seed: int = 42,
        training: bool = False,
        samples_per_epoch: int | None = None,
        populations: list[str] | tuple[str, ...] | None = None,
        split_file: str | Path | None = None,
        fields: set[str] | frozenset[str] | tuple[str, ...] | list[str] | None = None,
        sampling_mode: str = "condition_uniform",
    ) -> None:
        self.data_dir = Path(data_dir)
        filter_path = self.data_dir / "condition_filter.json"
        if not filter_path.is_file():
            raise FileNotFoundError(f"Condition filter is missing: {filter_path}")
        self.condition_filter = json.loads(filter_path.read_text(encoding="utf-8"))
        self.regime = regime
        self.split = split
        self.set_size = int(set_size)
        self.seed = int(seed)
        self.training = bool(training)
        self.fields = self.DATA_FIELDS if fields is None else frozenset(fields)
        self.sampling_mode = str(sampling_mode).casefold()
        if self.sampling_mode not in {"condition_uniform", "cell_abundance"}:
            raise ValueError(
                "MAPDataset sampling_mode must be condition_uniform or cell_abundance"
            )
        unknown_fields = sorted(self.fields - self.DATA_FIELDS)
        if unknown_fields:
            raise ValueError(
                "Unknown MAPDataset fields: " + ", ".join(unknown_fields)
            )
        self.epoch = 0
        manifest_path = self.data_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Prepared manifest is missing: {self.data_dir / 'manifest.json'}"
            )
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        shapes_path = self.data_dir / "materialized_shapes.json"
        if not shapes_path.is_file():
            raise FileNotFoundError(f"Materialized shapes are missing: {shapes_path}")
        self.materialized_shapes = json.loads(
            shapes_path.read_text(encoding="utf-8")
        )
        preparation_filter = self.manifest.get("preparation", {}).get(
            "condition_filter", {}
        )
        if preparation_filter.get("filter_id") != self.condition_filter.get("filter_id"):
            raise RuntimeError("Prepared manifest and condition filter do not match")
        if split_file is None:
            split_file = self.manifest.get("splits", {}).get(regime)
        if split_file is None:
            raise ValueError(f"No split registered for {regime}; pass split_file")
        split_path = Path(split_file)
        if not split_path.is_absolute():
            split_path = self.data_dir / split_path
        self.split_file = split_path.resolve()
        self.split_manifest = json.loads(split_path.read_text(encoding="utf-8"))
        if self.split_manifest.get("rule", self.split_manifest.get("regime")) != regime:
            raise ValueError("Split rule does not match the requested regime")
        split_name = str(split)
        if split_name not in self.split_manifest:
            raise KeyError(f"Split set {split!r} is absent from {split_path}")
        self.split_name = split_name
        self.split = split_name
        self.split_id = self.split_manifest.get("split_id", split_path.stem)
        table = pq.read_table(self.data_dir / "conditions.parquet")
        conditions = table.to_pandas().set_index("condition_id")
        # Paper Tahoe calls the grouping column ``cell_line``; the generic
        # contract calls it ``population``.  Normalize this metadata in
        # memory (the parquet artifact remains untouched).
        if "population" not in conditions.columns:
            if "cell_line" not in conditions.columns:
                raise ValueError(
                    "conditions.parquet must contain population or cell_line"
                )
            conditions["population"] = conditions["cell_line"].astype(str)
        if (
            "population_name" not in conditions.columns
            and "cell_line_name" in conditions.columns
        ):
            conditions["population_name"] = conditions["cell_line_name"]
        # Normalize the optional combination schema in memory.  Legacy
        # condition tables remain valid and become singleton components.
        component_smiles = []
        component_doses = []
        component_names = []
        component_keys = []
        for _, row in conditions.iterrows():
            raw_smiles = row.get("component_smiles", row.get("canonical_smiles", ""))
            raw_doses = row.get("component_doses_uM", row.get("dose", 0.0))
            smiles_values, dose_values = normalize_components(raw_smiles, raw_doses)
            names = row.get("component_names", None)
            if isinstance(names, str):
                try:
                    import ast
                    names = ast.literal_eval(names)
                except (ValueError, SyntaxError):
                    names = [names]
            if not isinstance(names, (list, tuple)) or len(names) != len(smiles_values):
                names = list(smiles_values)
            component_smiles.append(smiles_values)
            component_doses.append(dose_values)
            component_names.append([str(value) for value in names])
            component_keys.append(str(row.get("combination_key", "")) or combination_key(smiles_values))
        conditions["component_smiles"] = component_smiles
        conditions["component_doses_uM"] = component_doses
        conditions["component_names"] = component_names
        conditions["combination_key"] = component_keys
        if "condition_key" not in conditions.columns:
            conditions["condition_key"] = [
                f"{population}|{key}|{float(dose):g}"
                for population, key, dose in zip(
                    conditions["population"], conditions["combination_key"], conditions["dose"]
                )
            ]
        self.conditions = conditions
        self._arrays: dict[str, dict] = {}
        selected = [int(value) for value in self.split_manifest[split_name]]
        if populations:
            allowed = set(populations)
            selected = [value for value in selected if self.conditions.loc[value, "population"] in allowed]
        if not selected:
            raise ValueError(f"No conditions for regime={regime}, split={split}")
        self.condition_ids = np.asarray(sorted(selected), dtype=np.int64)
        self.rows_by_condition: dict[int, np.ndarray] = {}
        row_field = f"{split_name}_rows"
        row_files_field = f"{split_name}_rows_files"
        if row_field in self.split_manifest or row_files_field in self.split_manifest:
            population_rows = {
                str(population): np.asarray(rows, dtype=np.int64)
                for population, rows in self.split_manifest.get(row_field, {}).items()
            }
            for population, filename in self.split_manifest.get(
                row_files_field, {}
            ).items():
                row_path = Path(filename)
                if not row_path.is_absolute():
                    row_path = self.data_dir / row_path
                if str(population) in population_rows:
                    raise ValueError(
                        f"Split {split!r} defines both inline and file-backed rows "
                        f"for population {population}"
                    )
                population_rows[str(population)] = np.load(
                    row_path, mmap_mode="r"
                )
            # Row ids are local to each population's materialized mmap.  Build
            # the condition mapping directly from the row labels.
            for population in self.conditions.loc[self.condition_ids, "population"].unique():
                arrays = self._open_population(str(population))
                selected_ids = set(int(value) for value in self.condition_ids)
                grouped: dict[int, list[int]] = {}
                for row in population_rows.get(str(population), ()):
                    condition_id = int(arrays["row_condition"][int(row)])
                    if condition_id in selected_ids:
                        grouped.setdefault(condition_id, []).append(int(row))
                self.rows_by_condition.update({
                    condition_id: np.asarray(rows, dtype=np.int64)
                    for condition_id, rows in grouped.items()
                })
            missing_rows = sorted(
                set(int(value) for value in self.condition_ids)
                - set(self.rows_by_condition)
            )
            if missing_rows:
                raise ValueError(
                    f"Split {split_name!r} has no row membership for condition ids: {missing_rows[:8]}"
                )
        self.condition_sampling_probabilities = None
        if self.training and self.sampling_mode == "cell_abundance":
            counts = []
            for condition_id in self.condition_ids:
                condition_id = int(condition_id)
                if condition_id in self.rows_by_condition:
                    count = len(self.rows_by_condition[condition_id])
                else:
                    population = str(self.conditions.loc[condition_id, "population"])
                    arrays = self._open_population(population)
                    count = len(self._group_rows(arrays, "condition", condition_id))
                counts.append(count)
            counts = np.asarray(counts, dtype=np.float64)
            if np.any(counts <= 0) or not np.isfinite(counts).all():
                raise ValueError("Cell-abundance sampling requires positive condition counts")
            self.condition_sampling_probabilities = counts / counts.sum()
            default_samples = max(2 * int(counts.sum()) // self.set_size, 1000)
        else:
            default_samples = len(selected)
        self.samples_per_epoch = (
            int(samples_per_epoch)
            if samples_per_epoch is not None
            else int(default_samples)
        )
        evaluation_groups: dict[tuple[str, str], list[int]] = {}
        if not self.training:
            for condition_id in self.condition_ids:
                condition = self.conditions.loc[int(condition_id)]
                key = (str(condition["population"]), str(condition["combination_key"]))
                evaluation_groups.setdefault(key, []).append(int(condition_id))
        self.evaluation_groups = tuple(
            (key, tuple(values)) for key, values in sorted(evaluation_groups.items())
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        if not self.training:
            return len(self.evaluation_groups)
        return self.samples_per_epoch

    def _open_population(self, population: str) -> dict:
        if population in self._arrays:
            return self._arrays[population]
        base = self.data_dir / population
        shape = self.materialized_shapes[population]
        n_cells = int(shape["n_cells"])
        token_length = int(shape.get("token_length", 2048))
        hvg_dim = int(shape.get("hvg_dim", 2000))

        def existing(*names: str) -> Path:
            for name in names:
                path = base / name
                if path.is_file():
                    return path
            # Let ``np.memmap``/``np.load`` raise their usual precise error if
            # a required artifact is genuinely absent.  Returning the primary
            # name also keeps lightweight reader probes (which monkeypatch
            # those functions) independent of on-disk fixture files.
            return base / names[0]

        # Current artifacts use generic ``row_group``/``control_group`` names;
        # the paper materializer uses plate-specific names.  They represent the
        # same matching relation, so expose both through one internal schema.
        row_group_path = existing("row_group.uint16.dat", "row_plate.uint16.dat")
        row_condition_path = existing("row_condition.int32.dat")
        arrays = {
            "row_group": np.memmap(row_group_path, dtype=np.uint16, mode="r", shape=(n_cells,)),
            "row_condition": np.memmap(row_condition_path, dtype=np.int32, mode="r", shape=(n_cells,)),
        }
        if self.fields & {"control_gene_ids", "condition_gene_ids"}:
            arrays["genes"] = np.memmap(
                existing("se_gene_ids.uint16.dat"), dtype=np.uint16, mode="r",
                shape=(n_cells, token_length),
            )
        if self.fields & {"control_expressions", "condition_expressions"}:
            arrays["expression"] = np.memmap(
                existing("se_expr.float16.dat"), dtype=np.float16, mode="r",
                shape=(n_cells, token_length),
            )
        if self.fields & {"condition_hvg_vectors", "control_hvg_vectors"}:
            arrays["hvg"] = np.memmap(
                existing("hvg.float16.dat"), dtype=np.float16, mode="r",
                shape=(n_cells, hvg_dim),
            )
        if self.fields & {"control_embeddings", "condition_embeddings"}:
            arrays["embedding"] = np.memmap(
                existing("state_embeddings.float16.dat"), dtype=np.float16,
                mode="r", shape=(n_cells, 2048),
            )
        group_files = {
            "condition": (
                "condition_ids.npy",
                "condition_offsets.npy",
                "condition_rows.npy",
            ),
            "control_group": (
                ("control_group_ids.npy", "control_plate_ids.npy"),
                ("control_group_offsets.npy", "control_plate_offsets.npy"),
                ("control_group_rows.npy", "control_plate_rows.npy"),
            ),
        }
        for prefix, names in group_files.items():
            id_names, offset_names, row_names = names
            if isinstance(id_names, str):
                id_names, offset_names, row_names = (
                    (id_names,),
                    (offset_names,),
                    (row_names,),
                )
            arrays[f"{prefix}_ids"] = np.load(existing(*id_names))
            if prefix == "condition":
                raw_condition_ids = np.asarray(arrays[f"{prefix}_ids"], dtype=np.int64)
                normalized, local_map = self._normalize_population_condition_ids(
                    population, raw_condition_ids
                )
                arrays[f"{prefix}_ids"] = normalized
                if local_map is not None:
                    raw_rows = np.asarray(arrays["row_condition"], dtype=np.int64)
                    translated = raw_rows.copy()
                    valid = (raw_rows >= 0) & (raw_rows < len(local_map))
                    translated[valid] = local_map[raw_rows[valid]]
                    arrays["row_condition"] = translated.astype(np.int32, copy=False)
            arrays[f"{prefix}_offsets"] = np.load(existing(*offset_names))
            arrays[f"{prefix}_rows"] = np.load(existing(*row_names), mmap_mode="r")
            arrays[f"{prefix}_lookup"] = {
                int(value): index for index, value in enumerate(arrays[f"{prefix}_ids"])
            }
        self._arrays[population] = arrays
        return arrays

    def _normalize_population_condition_ids(self, population: str, values):
        """Return global condition ids for either MAP or paper population files.

        The generic materializer stores global ids.  Early paper artifacts used
        population-local ids in some files, so detect that layout from the
        condition metadata and translate only when necessary.  This keeps the
        reader compatible with both layouts without rewriting multi-gigabyte
        row arrays on disk.
        """
        ids = np.asarray(values, dtype=np.int64)
        # A few lightweight consumers construct the dataset object without
        # loading condition metadata (for example resource probes).  In that
        # mode there is no reliable population mapping, so preserve the ids
        # exactly as stored instead of dereferencing an absent attribute.
        if not hasattr(self, "conditions"):
            return ids, None
        expected = np.asarray(
            self.conditions.index[
                self.conditions["population"].astype(str) == str(population)
            ],
            dtype=np.int64,
        )
        if len(expected) == 0 or len(ids) == 0:
            return ids, None
        valid_global = np.isin(ids, expected).mean()
        if valid_global >= 0.95:
            return ids, None
        # Local ids are positional within this population's sorted condition
        # table.  Preserve unknown/sentinel values (e.g. -1) for diagnostics.
        local = ids.copy()
        in_range = (local >= 0) & (local < len(expected))
        local[in_range] = expected[local[in_range]]
        return local, expected

    @staticmethod
    def _group_rows(arrays: dict, prefix: str, group_id: int):
        position = arrays[f"{prefix}_lookup"].get(int(group_id))
        if position is None:
            raise KeyError(f"Missing {prefix} group {group_id}")
        offsets = arrays[f"{prefix}_offsets"]
        rows = arrays[f"{prefix}_rows"]
        return rows[int(offsets[position]):int(offsets[position + 1])]

    def _rng(self, index: int, condition_id: int, salt: int = 0):
        epoch = self.epoch if self.training else 0
        return np.random.default_rng(
            np.random.SeedSequence([self.seed, epoch, index, condition_id, salt])
        )

    def _sample_condition(
        self, index: int, condition_id: int, *, salt: int = 0
    ) -> dict:
        condition = self.conditions.loc[condition_id]
        population = str(condition["population"])
        arrays = self._open_population(population)
        rng = self._rng(int(index), condition_id, salt)
        available = self.rows_by_condition.get(
            int(condition_id), self._group_rows(arrays, "condition", condition_id)
        )
        condition_rows = rng.choice(
            available, size=self.set_size, replace=len(available) < self.set_size
        ).astype(np.int64)
        matching_groups = np.asarray(arrays["row_group"][condition_rows])
        control_rows = np.empty(self.set_size, dtype=np.int64)
        for group_id in np.unique(matching_groups):
            positions = np.flatnonzero(matching_groups == group_id)
            control_rows[positions] = rng.choice(
                self._group_rows(arrays, "control_group", int(group_id)),
                size=len(positions),
                replace=True,
            )
        tensor = lambda values, dtype: torch.from_numpy(np.asarray(values, dtype=dtype).copy())
        sample = {
            "drug_smiles": str(condition["canonical_smiles"]),
            "drug_conc": float(condition["dose"]),
            "component_smiles": list(condition["component_smiles"]),
            "component_doses": list(condition["component_doses_uM"]),
            "component_names": list(condition["component_names"]),
            "combination_key": str(condition["combination_key"]),
            "condition_key": str(condition["condition_key"]),
            "population": population,
            "condition_id": condition_id,
        }
        array_fields = {
            "control_gene_ids": ("genes", control_rows, np.int64),
            "control_expressions": ("expression", control_rows, np.float32),
            "condition_gene_ids": ("genes", condition_rows, np.int64),
            "condition_expressions": ("expression", condition_rows, np.float32),
            "control_embeddings": ("embedding", control_rows, np.float32),
            "condition_embeddings": ("embedding", condition_rows, np.float32),
            "condition_hvg_vectors": ("hvg", condition_rows, np.float32),
            "control_hvg_vectors": ("hvg", control_rows, np.float32),
        }
        for field, (array, rows, dtype) in array_fields.items():
            if field in self.fields:
                sample[field] = tensor(
                    arrays[array][rows], self.FIELD_DTYPES.get(field, dtype)
                )
        if "condition_rows" in self.fields:
            sample["condition_rows"] = tensor(condition_rows, np.int64)
        if "control_rows" in self.fields:
            sample["control_rows"] = tensor(control_rows, np.int64)
        return sample

    def sample_evaluation_group(self, index: int) -> dict:
        """Sample one pseudobulk for a cell-line/drug pair across all doses."""
        if self.training:
            raise RuntimeError("Evaluation groups are unavailable in training mode")
        (_, combination), condition_ids = self.evaluation_groups[int(index)]
        first = self.conditions.loc[int(condition_ids[0])]
        population = str(first["population"])
        arrays = self._open_population(population)
        available = np.concatenate([
            np.asarray(
                self.rows_by_condition.get(
                    int(condition_id),
                    self._group_rows(arrays, "condition", int(condition_id)),
                ),
                dtype=np.int64,
            )
            for condition_id in condition_ids
        ])
        rng = self._rng(int(index), int(condition_ids[0]), salt=811)
        condition_rows = rng.choice(
            available, size=self.set_size, replace=len(available) < self.set_size
        ).astype(np.int64)
        matching_groups = np.asarray(arrays["row_group"][condition_rows])
        control_rows = np.empty(self.set_size, dtype=np.int64)
        for group_id in np.unique(matching_groups):
            positions = np.flatnonzero(matching_groups == group_id)
            control_rows[positions] = rng.choice(
                self._group_rows(arrays, "control_group", int(group_id)),
                size=len(positions),
                replace=True,
            )
        sampled_condition_id = int(arrays["row_condition"][condition_rows[0]])
        sampled_condition = self.conditions.loc[sampled_condition_id]
        tensor = lambda values, dtype: torch.from_numpy(
            np.asarray(values, dtype=dtype).copy()
        )
        sample = {
            "drug_smiles": str(sampled_condition["canonical_smiles"]),
            "drug_conc": float(sampled_condition["dose"]),
            "drug": str(sampled_condition["drug"]),
            "component_smiles": list(sampled_condition["component_smiles"]),
            "component_doses": list(sampled_condition["component_doses_uM"]),
            "component_names": list(sampled_condition["component_names"]),
            "combination_key": str(combination),
            "condition_key": str(sampled_condition["condition_key"]),
            "population": population,
            "condition_id": sampled_condition_id,
            "condition_ids": condition_ids,
        }
        array_fields = {
            "control_gene_ids": ("genes", control_rows, np.int64),
            "control_expressions": ("expression", control_rows, np.float32),
            "condition_gene_ids": ("genes", condition_rows, np.int64),
            "condition_expressions": ("expression", condition_rows, np.float32),
            "control_embeddings": ("embedding", control_rows, np.float32),
            "condition_embeddings": ("embedding", condition_rows, np.float32),
            "condition_hvg_vectors": ("hvg", condition_rows, np.float32),
            "control_hvg_vectors": ("hvg", control_rows, np.float32),
        }
        for field, (array, rows, dtype) in array_fields.items():
            if field in self.fields:
                sample[field] = tensor(
                    arrays[array][rows], self.FIELD_DTYPES.get(field, dtype)
                )
        if "condition_rows" in self.fields:
            sample["condition_rows"] = tensor(condition_rows, np.int64)
        if "control_rows" in self.fields:
            sample["control_rows"] = tensor(control_rows, np.int64)
        return sample

    def __getitem__(self, index: int) -> dict:
        if self.training and self.condition_sampling_probabilities is not None:
            rng = np.random.default_rng(
                np.random.SeedSequence([self.seed, self.epoch, int(index), 638])
            )
            condition_id = int(
                rng.choice(
                    self.condition_ids,
                    p=self.condition_sampling_probabilities,
                )
            )
        else:
            condition_id = int(self.condition_ids[int(index) % len(self.condition_ids)])
        return self._sample_condition(int(index), condition_id)


__all__ = ["MAPDataset"]
