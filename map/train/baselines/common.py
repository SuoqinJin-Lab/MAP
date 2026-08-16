from __future__ import annotations

import json
import os
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from ..._common.dataset import MAPDataset


class BaselineAssets:
    """Read model-specific frozen inputs prepared for the generic data contract."""

    def __init__(self, data_dir: str | Path, model: str) -> None:
        self.data_dir = Path(data_dir)
        self.root = self.data_dir / "baselines"
        manifest_file = self.root / "manifest.json"
        if not manifest_file.is_file():
            raise FileNotFoundError(
                "Baseline inputs are absent; run preparation.prepare_baseline_inputs()"
            )
        self.manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        try:
            self.model_manifest = self.manifest["models"][model]
        except KeyError as error:
            raise FileNotFoundError(
                f"Prepared inputs for {model} are absent; rerun prepare_baseline_inputs()"
            ) from error
        self.smiles = tuple(str(value) for value in self.manifest["drug_vocabulary"])
        self.smiles_to_index = {value: index for index, value in enumerate(self.smiles)}

    def matrix(self, filename: str) -> torch.Tensor:
        directory = self.model_manifest.get("directory", self.model_manifest["model"])
        path = self.root / directory / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        return torch.from_numpy(np.asarray(np.load(path), dtype=np.float32))


def setup_distributed() -> tuple[int, int, int, torch.device]:
    torch.set_float32_matmul_precision("high")
    if "RANK" not in os.environ:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return 0, 1, 0, device
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ["LOCAL_RANK"])
    if not torch.cuda.is_available():
        raise RuntimeError("Distributed baseline training requires CUDA")
    torch.cuda.set_device(local)
    dist.init_process_group("nccl")
    return rank, world, local, torch.device(f"cuda:{local}")


def precision(device: torch.device, amp_dtype: str):
    if amp_dtype == "bf16" and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


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
    "BaselineAssets",
    "atomic_torch_save",
    "build_loaders",
    "finish_distributed",
    "move_batch",
    "precision",
    "raw_model",
    "seed_everything",
    "setup_distributed",
    "write_run_config",
]
