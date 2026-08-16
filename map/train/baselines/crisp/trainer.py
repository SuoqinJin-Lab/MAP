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
    finish_distributed,
    move_batch,
    precision,
    raw_model,
    seed_everything,
    setup_distributed,
    write_run_config,
)
from .model import CRISP
from ..utils import baseline_data_fields


class CRISPDataset(MAPDataset):
    """Add CRISP's same-drug, different-population negative condition."""

    FIELD_DTYPES = {"condition_hvg_vectors": np.float16}

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        baseline_manifest = json.loads(
            (self.data_dir / "baselines" / "manifest.json").read_text(
                encoding="utf-8"
            )
        )
        self._drug_to_index = {
            str(smiles): index
            for index, smiles in enumerate(
                baseline_manifest["drug_vocabulary"]
            )
        }
        try:
            mean_manifest = baseline_manifest["models"]["crisp"][
                "control_group_means"
            ]
        except KeyError as error:
            raise FileNotFoundError(
                "CRISP control-group means are absent; rerun "
                "preparation.prepare_baseline_inputs(models=('crisp',))"
            ) from error
        self._control_group_means: dict[str, dict] = {}
        selected_populations = {
            str(value)
            for value in self.conditions.loc[self.condition_ids, "population"]
        }
        requested_populations = kwargs.get("populations")
        population_order = tuple(
            str(value)
            for value in (
                requested_populations
                if requested_populations
                else sorted(selected_populations)
            )
        )
        self._population_to_index = {
            value: index for index, value in enumerate(population_order)
        }
        crisp_root = self.data_dir / "baselines" / "crisp"
        for population in selected_populations:
            try:
                root = crisp_root / mean_manifest[population]["directory"]
            except KeyError as error:
                raise FileNotFoundError(
                    f"CRISP control-group means are absent for {population}"
                ) from error
            group_ids = np.asarray(
                np.load(root / "group_ids.int64.npy"), dtype=np.int64
            )
            embeddings = np.load(root / "embeddings.float16.npy", mmap_mode="r")
            hvg = np.load(root / "hvg.float16.npy", mmap_mode="r")
            if embeddings.shape != (len(group_ids), 2048):
                raise ValueError(
                    f"Invalid CRISP control embedding means for {population}"
                )
            self._control_group_means[population] = {
                "lookup": {
                    int(group_id): index
                    for index, group_id in enumerate(group_ids)
                },
                "embeddings": embeddings,
                "hvg": hvg,
            }
        selected = tuple(int(value) for value in self.condition_ids)
        condition_table = self.conditions.loc[
            list(selected), ["canonical_smiles", "population"]
        ]
        smiles_by_condition = {
            condition_id: str(condition_table.loc[condition_id, "canonical_smiles"])
            for condition_id in selected
        }
        population_by_condition = {
            condition_id: str(condition_table.loc[condition_id, "population"])
            for condition_id in selected
        }
        by_smiles: dict[str, list[int]] = {}
        for condition_id, smiles in smiles_by_condition.items():
            by_smiles.setdefault(smiles, []).append(condition_id)
        outside_population = {
            population: tuple(
                condition_id
                for condition_id in selected
                if population_by_condition[condition_id] != population
            )
            for population in set(population_by_condition.values())
        }
        self.negative_candidates: dict[int, tuple[int, ...]] = {}
        for condition_id in selected:
            population = population_by_condition[condition_id]
            same_drug = tuple(
                candidate
                for candidate in by_smiles[smiles_by_condition[condition_id]]
                if population_by_condition[candidate] != population
            )
            different_population = outside_population[population]
            candidates = same_drug or different_population
            if not candidates:
                candidates = tuple(
                    candidate for candidate in selected if candidate != condition_id
                ) or (condition_id,)
            self.negative_candidates[condition_id] = tuple(candidates)

    def _use_paired_control_means(self, sample: dict) -> None:
        population = str(sample["population"])
        arrays = self._open_population(population)
        condition_rows = sample["condition_rows"].numpy()
        group_ids = np.asarray(
            arrays["row_group"][condition_rows], dtype=np.int64
        )
        prepared = self._control_group_means[population]
        try:
            positions = np.asarray(
                [prepared["lookup"][int(group_id)] for group_id in group_ids],
                dtype=np.int64,
            )
        except KeyError as error:
            raise KeyError(
                f"Missing prepared CRISP control group {error.args[0]} in {population}"
            ) from error
        sample["control_embeddings"] = torch.from_numpy(
            np.asarray(prepared["embeddings"][positions], dtype=np.float16).copy()
        )
        sample["control_hvg_vectors"] = torch.from_numpy(
            np.asarray(prepared["hvg"][positions], dtype=np.float16).copy()
        )

    def _use_categorical_indices(self, sample: dict) -> None:
        sample["drug_index"] = self._drug_to_index[str(sample["drug_smiles"])]
        sample["population_index"] = self._population_to_index[
            str(sample["population"])
        ]

    def __getitem__(self, index: int) -> dict:
        sample = super().__getitem__(index)
        self._use_paired_control_means(sample)
        self._use_categorical_indices(sample)
        condition_id = int(sample["condition_id"])
        rng = self._rng(int(index), condition_id, salt=17)
        negative_id = int(rng.choice(self.negative_candidates[condition_id]))
        negative = self._sample_condition(int(index), negative_id, salt=23)
        self._use_paired_control_means(negative)
        self._use_categorical_indices(negative)
        for key in (
            "control_embeddings",
            "condition_hvg_vectors",
            "control_hvg_vectors",
            "drug_smiles",
            "drug_index",
            "drug_conc",
            "population",
            "population_index",
            "condition_id",
        ):
            sample[f"negative_{key}"] = negative[key]
        return sample


class CRISPObjective(torch.nn.Module):
    """Compile CRISP's forward and loss as one graph."""

    def __init__(self, model: CRISP) -> None:
        super().__init__()
        self.model = model

    def forward(self, batch: dict) -> torch.Tensor:
        return self.model.loss(self.model(batch), batch)[0]


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
    parser.add_argument(
        "--compile-mode",
        choices=("none", "reduce-overhead"),
        default="reduce-overhead",
    )


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
    configuration.pop("deg_mask_contract", None)
    model = build_model(data_dir, hvg_dim, populations, **configuration)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device)


def _optimizer(model: CRISP, args, device: torch.device):
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
        fused=device.type == "cuda",
    )


def _checkpoint(model, optimizer, scheduler, args, epoch, step):
    implementation = raw_model(model)
    return {
        "format": "map_baseline_v2",
        "model": "crisp",
        "epoch": int(epoch),
        "global_step": int(step),
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
    training, train_sampler, train_loader = build_loaders(
        args, rank, world, dataset_class=CRISPDataset,
        fields=baseline_data_fields("crisp"),
    )
    shapes = json.loads((Path(args.data_dir) / "materialized_shapes.json").read_text())
    hvg_dim = int(next(iter(shapes.values()))["hvg_dim"])
    model = build_model(args.data_dir, hvg_dim, args.populations, **_model_options(args)).to(device)
    optimizer = _optimizer(model, args, device)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.step_size_lr, gamma=0.5
    )
    start_epoch = global_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if checkpoint.get("format") != "map_baseline_v2" or checkpoint.get("model") != "crisp":
            raise ValueError("Resume checkpoint is not a CRISP checkpoint")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["global_step"])
    objective: torch.nn.Module = CRISPObjective(model)
    compile_enabled = args.compile_mode != "none" and device.type == "cuda"
    if compile_enabled:
        objective = torch.compile(
            objective,
            mode=args.compile_mode,
            fullgraph=True,
            dynamic=False,
        )
    if world > 1:
        objective = DDP(
            objective,
            device_ids=[local],
            output_device=local,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
            static_graph=True,
        )
    output = Path(args.output_dir)
    if rank == 0:
        write_run_config(output, {
            **vars(args),
            "model": "crisp",
            "run_name": output.name,
            "world_size": world,
            "split_id": training.split_id,
            "compile_enabled": compile_enabled,
            "model_configuration": model.configuration(),
        })
    last_epoch = max(start_epoch - 1, 0)
    for epoch in range(start_epoch, args.epochs):
        last_epoch = epoch
        training.set_epoch(epoch)
        train_sampler.set_epoch(epoch)
        objective.train()
        for batch in train_loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            if compile_enabled and args.compile_mode == "reduce-overhead":
                torch.compiler.cudagraph_mark_step_begin()
            with precision(device, args.amp_dtype):
                value = objective(batch)
            value.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            global_step += 1
            if global_step >= args.max_steps:
                break
        scheduler.step()
        if rank == 0:
            if (epoch + 1) % args.checkpoint_every_epochs == 0:
                payload = _checkpoint(
                    model, optimizer, scheduler, args, epoch, global_step
                )
                atomic_torch_save(
                    payload, output / "checkpoints" / f"epoch_{epoch + 1:04d}.pt"
                )
        if global_step >= args.max_steps:
            break
    if rank == 0:
        atomic_torch_save(
            _checkpoint(model, optimizer, scheduler, args, last_epoch, global_step),
            output / "last.pt",
        )
    finish_distributed()


__all__ = ["CRISPDataset", "add_arguments", "build_model", "load_model", "train"]
