from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from ...._common.dataset import MAPDataset
from ..common import (
    atomic_torch_save,
    build_loaders,
    distributed_mean,
    finish_distributed,
    move_batch,
    raw_model,
    seed_everything,
    setup_distributed,
    write_run_config,
)
from .model import CRISP


class CRISPDataset(MAPDataset):
    """Add CRISP's same-drug, different-population negative condition."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._control_means: dict[tuple[str, int], tuple[torch.Tensor, torch.Tensor]] = {}
        selected = [int(value) for value in self.condition_ids]
        self.negative_candidates: dict[int, tuple[int, ...]] = {}
        for condition_id in selected:
            condition = self.conditions.loc[condition_id]
            same_drug = [
                candidate
                for candidate in selected
                if candidate != condition_id
                and str(self.conditions.loc[candidate, "canonical_smiles"])
                == str(condition["canonical_smiles"])
                and str(self.conditions.loc[candidate, "population"])
                != str(condition["population"])
            ]
            different_population = [
                candidate
                for candidate in selected
                if candidate != condition_id
                and str(self.conditions.loc[candidate, "population"])
                != str(condition["population"])
            ]
            fallback = [candidate for candidate in selected if candidate != condition_id]
            candidates = same_drug or different_population or fallback or [condition_id]
            self.negative_candidates[condition_id] = tuple(candidates)

    def _use_paired_control_means(self, sample: dict) -> None:
        population = str(sample["population"])
        arrays = self._open_population(population)
        condition_rows = sample["condition_rows"].numpy()
        embeddings = []
        hvg_vectors = []
        for group_id in np.asarray(arrays["row_group"][condition_rows], dtype=np.int64):
            key = (population, int(group_id))
            cached = self._control_means.get(key)
            if cached is None:
                rows = self._group_rows(arrays, "control_group", int(group_id))
                cached = (
                    torch.from_numpy(
                        np.asarray(arrays["embedding"][rows], dtype=np.float32)
                        .mean(axis=0)
                        .copy()
                    ),
                    torch.from_numpy(
                        np.asarray(arrays["hvg"][rows], dtype=np.float32)
                        .mean(axis=0)
                        .copy()
                    ),
                )
                self._control_means[key] = cached
            embedding, hvg = cached
            embeddings.append(embedding)
            hvg_vectors.append(hvg)
        sample["control_embeddings"] = torch.stack(embeddings)
        sample["control_hvg_vectors"] = torch.stack(hvg_vectors)

    def __getitem__(self, index: int) -> dict:
        sample = super().__getitem__(index)
        self._use_paired_control_means(sample)
        condition_id = int(sample["condition_id"])
        rng = self._rng(int(index), condition_id, salt=17)
        negative_id = int(rng.choice(self.negative_candidates[condition_id]))
        negative = self._sample_condition(int(index), negative_id, salt=23)
        self._use_paired_control_means(negative)
        for key in (
            "control_embeddings",
            "condition_hvg_vectors",
            "control_hvg_vectors",
            "drug_smiles",
            "drug_conc",
            "population",
            "condition_id",
        ):
            sample[f"negative_{key}"] = negative[key]
        return sample


def add_arguments(parser) -> None:
    parser.add_argument("--weight-decay", type=float, default=1e-7)
    parser.add_argument("--cell-weight-decay", type=float, default=1e-3)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--encoder-width", type=int, default=256)
    parser.add_argument("--encoder-depth", type=int, default=4)
    parser.add_argument("--decoder-width", type=int, default=1028)
    parser.add_argument("--decoder-depth", type=int, default=4)
    parser.add_argument("--embedding-encoder-width", type=int, default=128)
    parser.add_argument("--embedding-encoder-depth", type=int, default=4)
    parser.add_argument("--doser-width", type=int, default=64)
    parser.add_argument("--doser-depth", type=int, default=3)
    parser.add_argument("--cell-predictor-width", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--deg-top-k", type=int, default=50)
    parser.add_argument("--mse-weight", type=float, default=0.75)
    parser.add_argument("--autofocus-weight", type=float, default=0.25)
    parser.add_argument("--celltype-weight", type=float, default=1.0)
    parser.add_argument("--contrastive-weight", type=float, default=0.1)
    parser.add_argument("--mmd-weight", type=float, default=0.1)
    parser.add_argument("--kld-normalizer", type=float, default=500.0)
    parser.add_argument("--step-size-lr", type=int, default=50)


def _model_options(args) -> dict:
    names = (
        "latent_dim",
        "encoder_width",
        "encoder_depth",
        "decoder_width",
        "decoder_depth",
        "embedding_encoder_width",
        "embedding_encoder_depth",
        "doser_width",
        "doser_depth",
        "cell_predictor_width",
        "dropout",
        "deg_top_k",
        "mse_weight",
        "autofocus_weight",
        "celltype_weight",
        "contrastive_weight",
        "mmd_weight",
        "kld_normalizer",
    )
    return {name: getattr(args, name) for name in names}


def build_model(data_dir, hvg_dim: int, populations, **options) -> CRISP:
    return CRISP(data_dir, hvg_dim, populations, **options)


def load_model(checkpoint, data_dir, hvg_dim: int, populations, device) -> CRISP:
    configuration = dict(checkpoint.get("model_configuration", {}))
    configuration.pop("model", None)
    configuration.pop("hvg_dim", None)
    model = build_model(data_dir, hvg_dim, populations, **configuration)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device)


def _optimizer(model: CRISP, args):
    cell_ids = {id(value) for value in model.pertae.cell_predictor.parameters()}
    main, cell = [], []
    for parameter in model.parameters():
        if parameter.requires_grad:
            (cell if id(parameter) in cell_ids else main).append(parameter)
    return torch.optim.Adam(
        [
            {"params": main, "weight_decay": args.weight_decay},
            {"params": cell, "weight_decay": args.cell_weight_decay},
        ],
        lr=args.lr,
    )


@torch.no_grad()
def _validate(model, loader, device) -> tuple[float, float]:
    model.eval()
    total = torch.zeros(4, device=device, dtype=torch.float64)
    implementation = raw_model(model)
    for batch in loader:
        batch = move_batch(batch, device)
        output = model(batch)
        value, _ = implementation.loss(output, batch)
        predicted = output["prediction"].mean(1)
        observed = batch["condition_hvg_vectors"].float().mean(1)
        control = batch["control_hvg_vectors"].float().mean(1)
        first = predicted - control
        second = observed - control
        first = first - first.mean(1, keepdim=True)
        second = second - second.mean(1, keepdim=True)
        denominator = first.square().sum(1).mul(second.square().sum(1)).sqrt()
        valid = denominator > 1e-8
        correlation = (
            (first[valid] * second[valid]).sum(1) / denominator[valid]
            if valid.any()
            else denominator.new_empty(0)
        )
        total += torch.tensor(
            [value.item(), 1.0, correlation.sum().item(), correlation.numel()],
            device=device,
            dtype=torch.float64,
        )
    loss = distributed_mean(total[:2])
    pearson = distributed_mean(total[2:])
    model.train()
    return loss, pearson


def _checkpoint(model, optimizer, scheduler, args, epoch, step, best, best_step):
    implementation = raw_model(model)
    return {
        "format": "map_baseline_v2",
        "model": "crisp",
        "epoch": int(epoch),
        "global_step": int(step),
        "best_validation_loss": float(best),
        "best_step": int(best_step),
        "model_state_dict": implementation.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "args": vars(args),
        "model_configuration": implementation.configuration(),
    }


def train(args) -> None:
    dimensions = (
        args.latent_dim,
        args.encoder_width,
        args.encoder_depth,
        args.decoder_width,
        args.decoder_depth,
        args.embedding_encoder_width,
        args.embedding_encoder_depth,
        args.doser_width,
        args.doser_depth,
        args.cell_predictor_width,
        args.deg_top_k,
        args.step_size_lr,
    )
    if min(dimensions) <= 0 or not 0 <= args.dropout < 1:
        raise ValueError("CRISP dimensions must be positive and dropout in [0, 1)")
    rank, world, local, device = setup_distributed()
    seed_everything(args.seed, rank)
    training, _, train_sampler, train_loader, validation_loader = build_loaders(
        args, rank, world, dataset_class=CRISPDataset
    )
    shapes = json.loads((Path(args.data_dir) / "materialized_shapes.json").read_text())
    hvg_dim = int(next(iter(shapes.values()))["hvg_dim"])
    model = build_model(args.data_dir, hvg_dim, args.populations, **_model_options(args)).to(device)
    optimizer = _optimizer(model, args)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.step_size_lr, gamma=0.5
    )
    start_epoch = global_step = best_step = 0
    best = float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if checkpoint.get("format") != "map_baseline_v2" or checkpoint.get("model") != "crisp":
            raise ValueError("Resume checkpoint is not a CRISP checkpoint")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["global_step"])
        best = float(checkpoint["best_validation_loss"])
        best_step = int(checkpoint["best_step"])
    if world > 1:
        model = DDP(model, device_ids=[local], output_device=local, broadcast_buffers=False)
    output = Path(args.output_dir)
    if rank == 0:
        write_run_config(output, {
            **vars(args),
            "model": "crisp",
            "run_name": output.name,
            "world_size": world,
            "split_id": training.split_id,
            "model_configuration": raw_model(model).configuration(),
        })
    stop = False
    last_epoch = max(start_epoch - 1, 0)
    for epoch in range(start_epoch, args.epochs):
        last_epoch = epoch
        training.set_epoch(epoch)
        train_sampler.set_epoch(epoch)
        model.train()
        for batch in train_loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            value, _ = raw_model(model).loss(model(batch), batch)
            value.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            global_step += 1
            if global_step % args.eval_every_steps == 0:
                validation_loss, _ = _validate(model, validation_loader, device)
                if validation_loss < best:
                    best, best_step = validation_loss, global_step
                    if rank == 0:
                        atomic_torch_save(
                            _checkpoint(model, optimizer, scheduler, args, epoch, global_step, best, best_step),
                            output / "best.pt",
                        )
                if rank == 0:
                    atomic_torch_save(
                        _checkpoint(model, optimizer, scheduler, args, epoch, global_step, best, best_step),
                        output / "last.pt",
                    )
                if global_step - best_step >= args.early_stopping_patience:
                    stop = True
                    break
            if global_step >= args.max_steps:
                stop = True
                break
        scheduler.step()
        if stop:
            break
    if not (output / "best.pt").is_file():
        best, _ = _validate(model, validation_loader, device)
        best_step = global_step
        if rank == 0:
            atomic_torch_save(
                _checkpoint(model, optimizer, scheduler, args, last_epoch, global_step, best, best_step),
                output / "best.pt",
            )
    if rank == 0:
        atomic_torch_save(
            _checkpoint(model, optimizer, scheduler, args, last_epoch, global_step, best, best_step),
            output / "last.pt",
        )
    finish_distributed()


__all__ = ["CRISPDataset", "add_arguments", "build_model", "load_model", "train"]
