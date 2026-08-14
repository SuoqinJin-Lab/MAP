from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset


class MAPDataset(Dataset):
    """Sample condition cells with controls matched by population and group."""

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
    ) -> None:
        self.data_dir = Path(data_dir)
        self.regime = regime
        self.split = split
        self.set_size = int(set_size)
        self.seed = int(seed)
        self.training = bool(training)
        self.epoch = 0
        self.manifest = json.loads(
            (self.data_dir / "manifest.json").read_text(encoding="utf-8")
        )
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
        selected = [int(value) for value in self.split_manifest[split]]
        if populations:
            allowed = set(populations)
            selected = [value for value in selected if self.conditions.loc[value, "population"] in allowed]
        if not selected:
            raise ValueError(f"No conditions for regime={regime}, split={split}")
        self.condition_ids = np.asarray(sorted(selected), dtype=np.int64)
        self.samples_per_epoch = int(samples_per_epoch) if samples_per_epoch else len(selected)
        self._arrays: dict[str, dict] = {}

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
        token_length = int(shape.get("token_length", 2049))
        hvg_dim = int(shape.get("hvg_dim", 2000))
        arrays = {
            "genes": np.memmap(base / "se_gene_ids.uint16.dat", dtype=np.uint16, mode="r", shape=(n_cells, token_length)),
            "expression": np.memmap(base / "se_expr.float16.dat", dtype=np.float16, mode="r", shape=(n_cells, token_length)),
            "hvg": np.memmap(base / "hvg.float16.dat", dtype=np.float16, mode="r", shape=(n_cells, hvg_dim)),
            "embedding": np.memmap(base / "state_embeddings.float16.dat", dtype=np.float16, mode="r", shape=(n_cells, 2048)),
            "row_group": np.memmap(base / "row_group.uint16.dat", dtype=np.uint16, mode="r", shape=(n_cells,)),
        }
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
        available = self._group_rows(arrays, "condition", condition_id)
        condition_rows = rng.choice(
            available, size=self.set_size, replace=len(available) < self.set_size
        ).astype(np.int64)
        matching_groups = np.asarray(arrays["row_group"][condition_rows])
        control_rows = np.asarray([
            rng.choice(self._group_rows(arrays, "control_group", int(group_id)))
            for group_id in matching_groups
        ], dtype=np.int64)
        tensor = lambda values, dtype: torch.from_numpy(np.asarray(values, dtype=dtype).copy())
        return {
            "control_gene_ids": tensor(arrays["genes"][control_rows], np.int64),
            "control_expressions": tensor(arrays["expression"][control_rows], np.float32),
            "control_embeddings": tensor(arrays["embedding"][control_rows], np.float32),
            "condition_embeddings": tensor(arrays["embedding"][condition_rows], np.float32),
            "condition_hvg_vectors": tensor(arrays["hvg"][condition_rows], np.float32),
            "control_hvg_vectors": tensor(arrays["hvg"][control_rows], np.float32),
            "drug_smiles": str(condition["canonical_smiles"]),
            "drug_conc": float(condition["dose"]),
            "population": population,
            "condition_id": condition_id,
            "condition_rows": tensor(condition_rows, np.int64),
            "control_rows": tensor(control_rows, np.int64),
        }

    def __getitem__(self, index: int) -> dict:
        condition_id = int(self.condition_ids[int(index) % len(self.condition_ids)])
        return self._sample_condition(int(index), condition_id)


__all__ = ["MAPDataset"]
