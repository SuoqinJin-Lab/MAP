from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset


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
    ) -> None:
        self.data_dir = Path(data_dir)
        filter_path = self.data_dir / "condition_filter.json"
        if not filter_path.is_file():
            raise FileNotFoundError(
                f"Prepared data predates the population-condition filter contract: {filter_path}"
            )
        self.condition_filter = json.loads(filter_path.read_text(encoding="utf-8"))
        self.regime = regime
        self.split = split
        self.set_size = int(set_size)
        self.seed = int(seed)
        self.training = bool(training)
        self.fields = self.DATA_FIELDS if fields is None else frozenset(fields)
        unknown_fields = sorted(self.fields - self.DATA_FIELDS)
        if unknown_fields:
            raise ValueError(
                "Unknown MAPDataset fields: " + ", ".join(unknown_fields)
            )
        self.epoch = 0
        self.manifest = json.loads(
            (self.data_dir / "manifest.json").read_text(encoding="utf-8")
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
        if split not in self.split_manifest:
            raise KeyError(f"Split set {split!r} is absent from {split_path}")
        self.split_id = self.split_manifest.get("split_id", split_path.stem)
        table = pq.read_table(self.data_dir / "conditions.parquet")
        self.conditions = table.to_pandas().set_index("condition_id")
        self._arrays: dict[str, dict] = {}
        selected = [int(value) for value in self.split_manifest[split]]
        if populations:
            allowed = set(populations)
            selected = [value for value in selected if self.conditions.loc[value, "population"] in allowed]
        if not selected:
            raise ValueError(f"No conditions for regime={regime}, split={split}")
        self.condition_ids = np.asarray(sorted(selected), dtype=np.int64)
        self.rows_by_condition: dict[int, np.ndarray] = {}
        row_field = f"{split}_rows"
        row_files_field = f"{split}_rows_files"
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
                    f"Split {split!r} has no row membership for condition ids: {missing_rows[:8]}"
                )
        self.samples_per_epoch = int(samples_per_epoch) if samples_per_epoch else len(selected)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _open_population(self, population: str) -> dict:
        if population in self._arrays:
            return self._arrays[population]
        base = self.data_dir / population
        shape = self.manifest["materialized_shapes"][population]
        n_cells = int(shape["n_cells"])
        token_length = int(shape.get("token_length", 2048))
        hvg_dim = int(shape.get("hvg_dim", 2000))
        arrays = {
            "row_group": np.memmap(base / "row_group.uint16.dat", dtype=np.uint16, mode="r", shape=(n_cells,)),
            "row_condition": np.memmap(base / "row_condition.int32.dat", dtype=np.int32, mode="r", shape=(n_cells,)),
        }
        if self.fields & {"control_gene_ids", "condition_gene_ids"}:
            arrays["genes"] = np.memmap(
                base / "se_gene_ids.uint16.dat", dtype=np.uint16, mode="r",
                shape=(n_cells, token_length),
            )
        if self.fields & {"control_expressions", "condition_expressions"}:
            arrays["expression"] = np.memmap(
                base / "se_expr.float16.dat", dtype=np.float16, mode="r",
                shape=(n_cells, token_length),
            )
        if self.fields & {"condition_hvg_vectors", "control_hvg_vectors"}:
            arrays["hvg"] = np.memmap(
                base / "hvg.float16.dat", dtype=np.float16, mode="r",
                shape=(n_cells, hvg_dim),
            )
        if self.fields & {"control_embeddings", "condition_embeddings"}:
            arrays["embedding"] = np.memmap(
                base / "state_embeddings.float16.dat", dtype=np.float16,
                mode="r", shape=(n_cells, 2048),
            )
        for prefix in ("condition", "control_group"):
            arrays[f"{prefix}_ids"] = np.load(base / f"{prefix}_ids.npy")
            arrays[f"{prefix}_offsets"] = np.load(base / f"{prefix}_offsets.npy")
            arrays[f"{prefix}_rows"] = np.load(base / f"{prefix}_rows.npy", mmap_mode="r")
            arrays[f"{prefix}_lookup"] = {
                int(value): index for index, value in enumerate(arrays[f"{prefix}_ids"])
            }
        self._arrays[population] = arrays
        return arrays

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

    def __getitem__(self, index: int) -> dict:
        condition_id = int(self.condition_ids[int(index) % len(self.condition_ids)])
        return self._sample_condition(int(index), condition_id)


__all__ = ["MAPDataset"]
