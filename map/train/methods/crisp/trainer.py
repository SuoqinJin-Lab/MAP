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
    EarlyStopping,
    finish_distributed,
    move_batch,
    precision,
    raw_model,
    seed_everything,
    synchronized_loss,
    setup_distributed,
    write_run_config,
)
from .model import CRISP
from ..utils import method_data_fields
from ..common import MethodAssets


class CRISPDataset(MAPDataset):
    """Add CRISP's same-drug, different-population negative condition."""

    FIELD_DTYPES = {"condition_hvg_vectors": np.float16}

    def __init__(self, *args, **kwargs) -> None:
        material_dir = Path(kwargs.pop("material_dir"))
        legacy_use_deg_mask = bool(kwargs.pop("use_deg_mask", False))
        self.deg_mask_mode = str(
            kwargs.pop("deg_mask_mode", "validation")
        ).casefold()
        if legacy_use_deg_mask:
            self.deg_mask_mode = "official"
        if self.deg_mask_mode not in {"official", "validation"}:
            raise ValueError("CRISP deg_mask_mode must be official or validation")
        self.use_deg_mask = self.deg_mask_mode == "official"
        self.deg_top_k = int(kwargs.pop("deg_top_k", 50))
        if self.deg_top_k <= 0:
            raise ValueError("CRISP deg_top_k must be positive")
        self.drug_representation = str(
            kwargs.pop("drug_representation", "official")
        ).casefold()
        self.control_representation = str(
            kwargs.pop("control_representation", "official")
        ).casefold()
        for name, value in (
            ("drug_representation", self.drug_representation),
            ("control_representation", self.control_representation),
        ):
            if value not in {"official", "validation"}:
                raise ValueError(f"CRISP {name} must be official or validation")
        super().__init__(*args, **kwargs)
        assets = MethodAssets(kwargs.get("data_dir", ""), "crisp", material_dir)
        crisp_manifest = assets.model_manifest
        if self.use_deg_mask and crisp_manifest.get("deg_mask_mode", "official") != "official":
            raise ValueError(
                "CRISP training requested official DEG masks, but preparation "
                "contains the validation all-zero contract"
            )
        self._drug_to_index = {
            str(smiles): index
            for index, smiles in enumerate(
                crisp_manifest["smiles"]
            )
        }
        matrix_file = crisp_manifest.get("matrix", "rdkit2d.float32.npy")
        self._drug_features = np.load(
            assets.file(matrix_file),
            mmap_mode="r",
        )
        if self._drug_features.shape[0] != len(self._drug_to_index):
            raise ValueError("CRISP drug matrix does not match the drug vocabulary")
        try:
            mean_manifest = crisp_manifest["control_group_means"]
        except KeyError as error:
            raise FileNotFoundError(
                "CRISP control-group means are absent; rerun "
                "preparation.prepare_control_means()"
            ) from error
        self._control_group_means: dict[str, dict] = {}
        self._deg_masks: dict[str, dict] = {}
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
        crisp_root = material_dir
        for population in selected_populations:
            try:
                root = assets.file(mean_manifest[population]["directory"])
                root = root.parent if root.is_file() else root
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
        if self.use_deg_mask:
            try:
                deg_manifest = crisp_manifest["deg_masks"]
            except KeyError as error:
                raise FileNotFoundError(
                    "CRISP DEG masks are absent; rerun preparation with "
                    "preparation.prepare_deg_masks()"
                ) from error
            for population in selected_populations:
                try:
                    root = assets.file(deg_manifest[population]["directory"])
                    root = root.parent if root.is_file() else root
                except KeyError as error:
                    raise FileNotFoundError(
                        f"CRISP DEG masks are absent for {population}"
                    ) from error
                ids = np.asarray(
                    np.load(root / "condition_ids.int64.npy"), dtype=np.int64
                )
                masks = np.load(root / "mask.bool.npy", mmap_mode="r")
                expected_hvg_dim = int(
                    self._control_group_means[population]["hvg"].shape[1]
                )
                if masks.shape[0] != len(ids) or masks.shape[1] != expected_hvg_dim:
                    raise ValueError(
                        f"Invalid CRISP DEG masks for {population}: {masks.shape}"
                    )
                self._deg_masks[population] = {
                    "lookup": {int(value): index for index, value in enumerate(ids)},
                    "masks": masks,
                    "top_k": int(deg_manifest[population].get("top_k", 0)),
                }
                prepared_top_k = self._deg_masks[population]["top_k"]
                if prepared_top_k != self.deg_top_k:
                    raise ValueError(
                        "CRISP prepared DEG mask top_k does not match training "
                        f"deg_top_k for {population}: prepared={prepared_top_k}, "
                        f"requested={self.deg_top_k}; rerun preparation with "
                        "crisp_deg_top_k or use the matching training parameter"
                    )
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

    def _use_control_representation(self, sample: dict) -> None:
        if self.control_representation == "validation":
            return
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
        drug_index = self._drug_to_index[str(sample["drug_smiles"])]
        if self.drug_representation == "official":
            sample["drug_index"] = drug_index
        else:
            sample["drug_features"] = torch.from_numpy(
                np.asarray(self._drug_features[drug_index], dtype=np.float32).copy()
            )
        sample["population_index"] = self._population_to_index[
            str(sample["population"])
        ]

    def _use_deg_mask(self, sample: dict) -> None:
        if not self.use_deg_mask:
            return
        population = str(sample["population"])
        prepared = self._deg_masks[population]
        condition_id = int(sample["condition_id"])
        try:
            position = prepared["lookup"][condition_id]
        except KeyError as error:
            raise KeyError(
                f"Missing prepared CRISP DEG mask for condition {condition_id} "
                f"in {population}"
            ) from error
        mask = np.asarray(prepared["masks"][position], dtype=np.bool_)
        set_size = int(sample["condition_hvg_vectors"].shape[0])
        sample["condition_deg_mask"] = torch.from_numpy(
            np.broadcast_to(mask, (set_size, mask.shape[0])).copy()
        )

    def __getitem__(self, index: int) -> dict:
        sample = super().__getitem__(index)
        self._use_control_representation(sample)
        self._use_categorical_indices(sample)
        self._use_deg_mask(sample)
        condition_id = int(sample["condition_id"])
        rng = self._rng(int(index), condition_id, salt=17)
        negative_id = int(rng.choice(self.negative_candidates[condition_id]))
        negative = self._sample_condition(int(index), negative_id, salt=23)
        self._use_control_representation(negative)
        self._use_categorical_indices(negative)
        self._use_deg_mask(negative)
        keys = (
            "control_embeddings",
            "condition_hvg_vectors",
            "control_hvg_vectors",
            "drug_smiles",
            "drug_conc",
            "population",
            "population_index",
            "condition_id",
        )
        keys += (
            ("drug_index",)
            if self.drug_representation == "official"
            else ("drug_features",)
        )
        if self.use_deg_mask:
            keys += ("condition_deg_mask",)
        for key in keys:
            sample[f"negative_{key}"] = negative[key]
        return sample

    def sample_evaluation_group(self, index: int) -> dict:
        sample = super().sample_evaluation_group(index)
        self._use_control_representation(sample)
        self._use_categorical_indices(sample)
        self._use_deg_mask(sample)
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
    parser.add_argument(
        "--deg-mask-mode",
        choices=("official", "validation"),
        default="official",
        help="Use official statistical DEG masks or validation all-zero masks",
    )
    parser.add_argument(
        "--use-deg-mask",
        action="store_true",
        help=(
            "Use the CRISP rank_genes_groups_by_cov-style DEG autofocus mask "
            "prepared per condition"
        ),
    )
    parser.add_argument(
        "--drug-representation",
        choices=("official", "validation"),
        default="official",
        help=(
            "Use official frozen-embedding index lookup or validation's "
            "direct precomputed drug vectors"
        ),
    )
    parser.add_argument(
        "--control-representation",
        choices=("official", "validation"),
        default="validation",
        help=(
            "Use official paired control-group means or validation's sampled "
            "control-cell representations"
        ),
    )
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
        "deg_mask_mode",
        "drug_representation",
        "control_representation",
    )
    return {name: getattr(args, name) for name in names}


def build_model(data_dir, hvg_dim: int, populations, *, material_dir=None, **options) -> CRISP:
    return CRISP(data_dir, hvg_dim, populations, material_dir=material_dir, **options)


def load_model(checkpoint, data_dir, hvg_dim: int, populations, device, *, material_dir=None) -> CRISP:
    configuration = dict(checkpoint.get("model_configuration", {}))
    configuration.pop("model", None)
    configuration.pop("hvg_dim", None)
    configuration.pop("deg_mask_contract", None)
    model = build_model(data_dir, hvg_dim, populations, material_dir=material_dir, **configuration)
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


def _checkpoint(model, optimizer, scheduler, args, epoch, step, early_stopping=None):
    implementation = raw_model(model)
    return {
        "format": "map_method_v2",
        "model": "crisp",
        "epoch": int(epoch),
        "global_step": int(step),
        "model_state_dict": implementation.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "args": vars(args),
        "model_configuration": implementation.configuration(),
        "early_stopping": (
            early_stopping.state_dict() if early_stopping is not None else None
        ),
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
        fields=(
            method_data_fields("crisp")
            if args.control_representation == "validation"
            else frozenset({"condition_hvg_vectors", "condition_rows"})
        ),
        dataset_kwargs={
            "use_deg_mask": args.use_deg_mask,
            "deg_mask_mode": args.deg_mask_mode,
            "deg_top_k": args.deg_top_k,
            "drug_representation": args.drug_representation,
            "control_representation": args.control_representation,
            "material_dir": args.material_dir,
        },
    )
    shapes = json.loads((Path(args.data_dir) / "materialized_shapes.json").read_text())
    hvg_dim = int(next(iter(shapes.values()))["hvg_dim"])
    model = build_model(args.data_dir, hvg_dim, args.populations, material_dir=args.material_dir, **_model_options(args)).to(device)
    optimizer = _optimizer(model, args, device)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.step_size_lr, gamma=0.5
    )
    start_epoch = global_step = 0
    early_stopping = EarlyStopping(
        args.early_stopping_patience, args.early_stopping_min_delta
    )
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if checkpoint.get("format") not in {"map_method_v2", "map_baseline_v2"} or checkpoint.get("model") != "crisp":
            raise ValueError("Resume checkpoint is not a CRISP checkpoint")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["global_step"])
        early_stopping.load_state_dict(checkpoint.get("early_stopping"))
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
    stop_reason = "max_steps_or_epochs"
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
            optimizer.step()
            global_step += 1
            stop_requested = early_stopping.update(
                synchronized_loss(value.detach(), device)
            )
            if stop_requested or global_step >= args.max_steps:
                if stop_requested:
                    stop_reason = "early_stopping"
                break
        scheduler.step()
        if rank == 0:
            if (epoch + 1) % args.checkpoint_every_epochs == 0:
                payload = _checkpoint(
                    model, optimizer, scheduler, args, epoch, global_step,
                    early_stopping,
                )
                atomic_torch_save(
                    payload, output / "checkpoints" / f"epoch_{epoch + 1:04d}.pt"
                )
        if early_stopping.stopped or global_step >= args.max_steps:
            break
    if rank == 0:
        atomic_torch_save(
            _checkpoint(
                model, optimizer, scheduler, args, last_epoch, global_step,
                early_stopping,
            ),
            output / "last.pt",
        )
    finish_distributed()


__all__ = ["CRISPDataset", "add_arguments", "build_model", "load_model", "train"]
