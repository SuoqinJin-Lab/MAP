from __future__ import annotations

import json
import os
import random
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from ..._common.dataset import MAPDataset


class MethodAssets:
    """Read neutral artifact manifests from the shared preparation root."""

    def __init__(
        self, data_dir: str | Path, model: str, material_dir: str | Path | None
    ) -> None:
        self.data_dir = Path(data_dir)
        if material_dir is None:
            raise ValueError(f"{model} requires the prepared artifact root")
        self.root = Path(material_dir)
        artifact_dirs = {
            "chemcpa": ("drug_ecfp4",),
            "prnet": ("drug_fcfp4",),
            "crisp": ("drug_rdkit2d", "control_means", "deg_masks"),
            "cmonge": ("drug_moa", "drug_rdkit2d"),
            "xpert": ("graph_assets", "drug_unimol", "expression_bins"),
        }.get(str(model).casefold(), ())
        roots = [self.root] if (self.root / "manifest.json").is_file() else [
            self.root / name for name in artifact_dirs
            if (self.root / name / "manifest.json").is_file()
        ]
        if not roots:
            raise FileNotFoundError(
                f"{model} inputs are absent; prepare the required artifacts first"
            )
        self._roots = roots
        self.manifest = {}
        for root in roots:
            manifest_file = root / "manifest.json"
            self.manifest.update(json.loads(manifest_file.read_text(encoding="utf-8")))
        self.model_manifest = self.manifest
        self.smiles = tuple(str(value) for value in self.manifest["smiles"])
        self.smiles_to_index = {value: index for index, value in enumerate(self.smiles)}

    def matrix(self, filename: str) -> torch.Tensor:
        path = self.file(filename)
        if not path.is_file():
            raise FileNotFoundError(path)
        return torch.from_numpy(np.asarray(np.load(path), dtype=np.float32))

    def file(self, filename: str) -> Path:
        """Resolve one file across the artifact directories in this view."""
        return next(
            (root / filename for root in self._roots if (root / filename).exists()),
            self.root / filename,
        )


def setup_distributed() -> tuple[int, int, int, torch.device]:
    torch.set_float32_matmul_precision("high")
    if "RANK" not in os.environ:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return 0, 1, 0, device
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ["LOCAL_RANK"])
    if not torch.cuda.is_available():
            raise RuntimeError("Distributed method training requires CUDA")
    torch.cuda.set_device(local)
    dist.init_process_group("nccl")
    return rank, world, local, torch.device(f"cuda:{local}")


def precision(device: torch.device, amp_dtype: str):
    if amp_dtype == "bf16" and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


@dataclass
class EarlyStopping:
    """Step-based early stopping shared by every iterative method.

    ``patience`` counts optimizer updates, rather than epochs.  A strict new
    low point is required when ``min_delta`` is zero, matching the paper's
    "stop after 1,000 steps without a new loss minimum" rule.  A patience of
    zero explicitly disables the criterion.
    """

    patience: int = 1000
    min_delta: float = 0.0
    best_loss: float | None = None
    bad_steps: int = 0
    stopped: bool = False

    def __post_init__(self) -> None:
        self.patience = int(self.patience)
        self.min_delta = float(self.min_delta)
        if self.patience < 0:
            raise ValueError("early_stopping_patience must be non-negative")
        if self.min_delta < 0:
            raise ValueError("early_stopping_min_delta must be non-negative")

    @property
    def enabled(self) -> bool:
        return self.patience > 0

    def update(self, loss: float) -> bool:
        """Record one optimizer-step loss and return whether training stops."""
        value = float(loss)
        improved = (
            self.best_loss is None
            or (value == value and value < self.best_loss - self.min_delta)
        )
        if improved:
            self.best_loss = value
            self.bad_steps = 0
        else:
            self.bad_steps += 1
        self.stopped = self.enabled and self.bad_steps >= self.patience
        return self.stopped

    def state_dict(self) -> dict[str, object]:
        return {
            "patience": self.patience,
            "min_delta": self.min_delta,
            "best_loss": self.best_loss,
            "bad_steps": self.bad_steps,
            "stopped": self.stopped,
        }

    def load_state_dict(self, state: dict[str, object] | None) -> None:
        if not state:
            return
        self.best_loss = (
            None if state.get("best_loss") is None else float(state["best_loss"])
        )
        self.bad_steps = int(state.get("bad_steps", 0))
        self.stopped = bool(state.get("stopped", False))


def synchronized_loss(loss: torch.Tensor | float, device: torch.device) -> float:
    """Return the mean loss across ranks for a deterministic stop decision."""
    value = torch.as_tensor(float(loss), dtype=torch.float64, device=device)
    if dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        value /= dist.get_world_size()
    return float(value.item())


def seed_everything(seed: int, rank: int = 0) -> None:
    effective = int(seed) + int(rank)
    random.seed(effective)
    np.random.seed(effective)
    torch.manual_seed(effective)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(effective)


def move_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def build_loaders(
    args,
    rank: int,
    world: int,
    dataset_class=MAPDataset,
    *,
    fields=None,
    dataset_kwargs=None,
):
    common = {
        "data_dir": args.data_dir,
        "regime": args.regime,
        "set_size": args.set_size,
        "seed": args.seed,
        "populations": args.populations,
        "split_file": args.split_file,
        "fields": fields,
    }
    if dataset_kwargs:
        common.update(dataset_kwargs)
    training = dataset_class(
        split=args.train_split,
        training=True,
        samples_per_epoch=args.samples_per_epoch,
        **common,
    )
    train_sampler = DistributedSampler(
        training,
        num_replicas=world,
        rank=rank,
        shuffle=True,
        seed=args.seed,
        drop_last=True,
    )
    options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": args.num_workers > 0,
    }
    return (
        training,
        train_sampler,
        DataLoader(training, sampler=train_sampler, **options),
    )


def raw_model(model):
    return model.module if isinstance(model, DDP) else model


def atomic_torch_save(payload: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def write_run_config(output: str | Path, payload: dict) -> None:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "run_config.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )


def finish_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


__all__ = [
    "MethodAssets",
    "atomic_torch_save",
    "build_loaders",
    "EarlyStopping",
    "finish_distributed",
    "move_batch",
    "precision",
    "raw_model",
    "seed_everything",
    "synchronized_loss",
    "setup_distributed",
    "write_run_config",
]
