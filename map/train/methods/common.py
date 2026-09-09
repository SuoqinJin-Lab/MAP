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
from .registry import method_spec


class MethodAssets:
    """Read neutral artifact manifests from the shared preparation root."""

    def __init__(
        self, data_dir: str | Path, model: str, material_dir: str | Path | None
    ) -> None:
        self.data_dir = Path(data_dir)
        if material_dir is None:
            raise ValueError(f"{model} requires the prepared artifact root")
        self.root = Path(material_dir)
        spec = method_spec(model)
        artifact_dirs = spec.artifacts + spec.split_artifacts
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
        for root in self._roots:
            path = root / filename
            if path.exists():
                return path
        path = self.root / filename
        if not path.exists():
            raise FileNotFoundError(
                f"{self.manifest.get('model', 'method')} asset is missing: {path}"
            )
        return path


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


def precision(device: torch.device, amp_dtype):
    """Autocast context for one precision setting.

    ``amp_dtype`` is either a string (``fp32``/``bf16``/``fp16``) or a
    ``torch.dtype``.  Non-CUDA devices and fp32 fall back to ``nullcontext``,
    which matches every method's previous behaviour in one place.
    """
    if device.type != "cuda":
        return nullcontext()
    if isinstance(amp_dtype, torch.dtype):
        if amp_dtype == torch.float32:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=amp_dtype)
    if amp_dtype == "fp32":
        return nullcontext()
    if amp_dtype == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if amp_dtype == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    raise ValueError(f"Unknown amp_dtype: {amp_dtype!r}")


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


RESUME_FORMATS = ("map_method_v2", "map_baseline_v2")


def load_hvg_dim(data_dir: str | Path, *, required: int | None = None) -> int:
    """Read the single HVG dimension from the materialized shapes.

    ``required`` asserts that every population shares that dimension (the
    CRISP and CMonge rules); ``None`` accepts the first population's value
    for the other methods.
    """
    shapes = json.loads(
        (Path(data_dir) / "materialized_shapes.json").read_text(encoding="utf-8")
    )
    dimensions = {int(value["hvg_dim"]) for value in shapes.values()}
    if required is not None:
        if dimensions != {required}:
            raise ValueError(
                f"Method requires exactly {required} HVGs; "
                f"materialized dimensions={sorted(dimensions)}"
            )
        return required
    if not dimensions:
        raise ValueError("materialized_shapes.json contains no populations")
    return sorted(dimensions)[0]


def validate_resume_checkpoint(
    checkpoint: dict,
    model: str,
    *,
    formats: tuple[str, ...] = RESUME_FORMATS,
) -> None:
    """Reject a resume checkpoint that does not belong to this method."""
    name = str(model).casefold()
    if checkpoint.get("format") not in formats or checkpoint.get("model") != name:
        raise ValueError(f"Resume checkpoint is not a {name} checkpoint")


def build_checkpoint(
    *,
    model_name: str,
    model,
    epoch: int,
    step: int,
    args,
    optimizer=None,
    scheduler=None,
    early_stopping=None,
    extra: dict | None = None,
) -> dict:
    """Canonical method checkpoint with one key layout for every baseline.

    The keys here are the resume contract; adding a new method must not
    invent a different spelling for the same state.
    """
    payload = {
        "format": "map_method_v2",
        "model": str(model_name).casefold(),
        "epoch": int(epoch),
        "global_step": int(step),
        "model_state_dict": raw_model(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "args": vars(args),
        "model_configuration": raw_model(model).configuration(),
        "early_stopping": (
            early_stopping.state_dict() if early_stopping is not None else None
        ),
    }
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    if extra:
        payload.update(extra)
    return payload


def wrap_ddp(
    model,
    *,
    local: int,
    world: int,
    static_graph: bool = False,
    gradient_as_bucket_view: bool = False,
    broadcast_buffers: bool = False,
    find_unused_parameters: bool = False,
) -> torch.nn.Module:
    """Wrap a model for distributed training with the shared defaults."""
    if world <= 1:
        return model
    return DDP(
        model,
        device_ids=[local],
        output_device=local,
        broadcast_buffers=broadcast_buffers,
        gradient_as_bucket_view=gradient_as_bucket_view,
        static_graph=static_graph,
        find_unused_parameters=find_unused_parameters,
    )


def write_method_config(
    output: str | Path,
    args,
    *,
    model: str,
    world: int,
    training,
    model_configuration: dict,
) -> None:
    """Write the per-run configuration card shared by every method."""
    write_run_config(
        output,
        {
            **vars(args),
            "model": model,
            "run_name": Path(output).name,
            "world_size": world,
            "split_id": training.split_id,
            "model_configuration": model_configuration,
        },
    )


def apply_resume_state(
    model,
    optimizer,
    scheduler,
    early_stopping,
    checkpoint: dict,
    *,
    load_scheduler: bool = True,
) -> tuple[int, int]:
    """Restore optimizer/scheduler/early-stopping state from a checkpoint.

    Returns the ``(start_epoch, global_step)`` continuation point.  The
    checkpoint itself has already been validated by
    :func:`validate_resume_checkpoint`.  Method-specific extra state (for
    example CMonge's autoencoder early-stopping) is restored by the caller
    from the same ``extra`` keys it wrote via :func:`build_checkpoint`.
    """
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and load_scheduler and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    early_stopping.load_state_dict(checkpoint.get("early_stopping"))
    return int(checkpoint["epoch"]) + 1, int(checkpoint["global_step"])


def save_epoch_checkpoint(
    output: str | Path,
    payload_fn,
    *,
    epoch: int,
    every: int,
    last_every_epoch: bool = False,
) -> None:
    """Save the periodic checkpoint and, optionally, the rolling ``last``."""
    output = Path(output)
    if last_every_epoch:
        atomic_torch_save(payload_fn(), output / "last.pt")
    if (epoch + 1) % every == 0:
        atomic_torch_save(
            payload_fn(), output / "checkpoints" / f"epoch_{epoch + 1:04d}.pt"
        )


__all__ = [
    "MethodAssets",
    "RESUME_FORMATS",
    "apply_resume_state",
    "atomic_torch_save",
    "build_checkpoint",
    "build_loaders",
    "EarlyStopping",
    "finish_distributed",
    "load_hvg_dim",
    "move_batch",
    "precision",
    "raw_model",
    "save_epoch_checkpoint",
    "seed_everything",
    "setup_distributed",
    "synchronized_loss",
    "validate_resume_checkpoint",
    "wrap_ddp",
    "write_method_config",
    "write_run_config",
]

