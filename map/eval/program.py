from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from .._common.dataset import MAPDataset
from ..model.map import MAPModel
from ..train.methods.registry import METHOD_REGISTRY, load_model
from ..train.methods.crisp.trainer import CRISPDataset
from ..train.methods.utils import method_data_fields
from .metrics import (
    METRIC_CATALOG,
    biological_metrics,
    condition_metrics,
    covariance_structure_score,
    disentanglement_score,
    perturbation_discrimination,
    significant_deg_mask,
    summarize_runs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MAP or a bundled method")
    parser.add_argument("--model", choices=METHOD_REGISTRY, default="map")
    parser.add_argument("--material-dir", dest="material_dir")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split-file", required=True)
    parser.add_argument(
        "--evaluation-splits",
        nargs="+",
        choices=("internal_test", "external_test", "both_seen", "one_seen", "both_unseen"),
        default=["internal_test", "external_test"],
    )
    parser.add_argument(
        "--regime", choices=("unprofiled_drug", "unseen_combination", "combosciplex"), required=True
    )
    parser.add_argument("--se-ckpt")
    parser.add_argument("--esm-embeddings")
    parser.add_argument("--mapkg-ckpt")
    parser.add_argument("--mapkg-vocab")
    parser.add_argument("--static-token-cache")
    parser.add_argument("--preparation-config")
    parser.add_argument("--populations", nargs="+")
    parser.add_argument("--set-size", type=int, default=24,
                        help="Cells sampled for each cell-line/drug pseudobulk")
    parser.add_argument(
        "--evaluation-unit",
        choices=("dose_level_condition", "cell_line_drug", "cell_line_combination"),
        default="cell_line_drug",
        help=(
            "Primary paper protocol evaluates each cell-line/drug/dose "
            "condition; cell_line_drug is retained for legacy summaries"
        ),
    )
    parser.add_argument("--num-gene-tokens", type=int, default=2048)
    parser.add_argument("--hvg-dim", type=int, default=2000)
    parser.add_argument("--combination-fusion", choices=("avg_emb", "two_tokens"))
    parser.add_argument("--max-components", type=int)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--deg-top-k", type=int, default=50)
    parser.add_argument("--deg-fdr", type=float, default=0.05)
    parser.add_argument("--deg-max-cells", type=int)
    return parser.parse_args()


def _components():
    return {
        "model": MAPModel,
        "deg_mask": significant_deg_mask,
        "catalog": METRIC_CATALOG,
        "condition_metrics": condition_metrics,
        "css": covariance_structure_score,
        "discrimination": perturbation_discrimination,
        "disentanglement": disentanglement_score,
        "biological": biological_metrics,
        "summarize": summarize_runs,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _batched_item(item: dict, device: torch.device) -> dict:
    """Construct the MAP-validation-style batch-size-1 model input."""
    batch = {}
    for key, value in item.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.unsqueeze(0).to(device, non_blocking=True)
        elif key.endswith("drug_index"):
            # CRISP keeps categorical indices as Python ints in Dataset items;
            # prediction expects a batch tensor, including negative samples.
            batch[key] = torch.tensor([value], device=device, dtype=torch.long)
        elif key.endswith("drug_conc"):
            # The same applies to doses, which are scalar floats in samples.
            batch[key] = torch.tensor(
                [value], device=device, dtype=torch.float32
            )
        elif key in {"drug_smiles", "population"}:
            batch[key] = [value]
        else:
            batch[key] = value
    return batch


def _method_prediction(model, item: dict, device: torch.device) -> np.ndarray:
    prediction = model.predict_batch(_batched_item(item, device))
    if not isinstance(prediction, torch.Tensor):
        raise TypeError("Method output does not contain a prediction tensor")
    if prediction.ndim != 2 or prediction.shape[0] != 1:
        raise ValueError(
            f"Expected condition-level predictions [1, genes], got {prediction.shape}"
        )
    return prediction.float().cpu().numpy()[0]


def _finite_mean(values) -> float:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    return float(finite.mean()) if finite.size else float("nan")


def _condition_degs(dataset: MAPDataset, condition_id: int, fdr: float, max_cells, seed: int, mask_fn):
    condition = dataset.conditions.loc[condition_id]
    population = str(condition["population"])
    arrays = dataset._open_population(population)
    # Row-level splits intentionally place different cells from the same
    # condition in train and internal_test.  DEG discovery must use only the
    # cells assigned to the split being evaluated; falling back to the full
    # condition group here would leak training rows into internal_test.
    condition_rows = np.asarray(
        dataset.rows_by_condition.get(
            int(condition_id),
            dataset._group_rows(arrays, "condition", condition_id),
        ),
        dtype=np.int64,
    )
    matching_groups = np.unique(np.asarray(arrays["row_group"][condition_rows]))
    control_rows = np.unique(np.concatenate([
        np.asarray(dataset._group_rows(arrays, "control_group", int(value)), dtype=np.int64)
        for value in matching_groups
    ]))
    rng = np.random.default_rng([seed, int(condition_id), 991])
    if max_cells is not None:
        if len(condition_rows) > max_cells:
            condition_rows = rng.choice(condition_rows, max_cells, replace=False)
        if len(control_rows) > max_cells:
            control_rows = rng.choice(control_rows, max_cells, replace=False)
    condition_values = np.asarray(arrays["hvg"][condition_rows], dtype=np.float32)
    control_values = np.asarray(arrays["hvg"][control_rows], dtype=np.float32)
    mask = mask_fn(control_values, condition_values, fdr)
    true_delta = condition_values.mean(0) - control_values.mean(0)
    return mask, true_delta


def _group_degs(
    dataset: MAPDataset,
    condition_ids: list[int] | tuple[int, ...],
    fdr: float,
    max_cells,
    seed: int,
    mask_fn,
):
    """Discover DEG genes for one cell-line/drug group across all doses."""
    condition_values = []
    control_values = []
    seen_control_rows: dict[str, set[int]] = defaultdict(set)
    for condition_id in condition_ids:
        condition = dataset.conditions.loc[int(condition_id)]
        population = str(condition["population"])
        arrays = dataset._open_population(population)
        rows = np.asarray(
            dataset.rows_by_condition.get(
                int(condition_id), dataset._group_rows(arrays, "condition", int(condition_id))
            ), dtype=np.int64,
        )
        condition_values.append(np.asarray(arrays["hvg"][rows], dtype=np.float32))
        groups = np.unique(np.asarray(arrays["row_group"][rows]))
        for group in groups:
            seen_control_rows.setdefault(population, set()).update(
                int(value) for value in dataset._group_rows(arrays, "control_group", int(group))
            )
    for population, rows in seen_control_rows.items():
        arrays = dataset._open_population(population)
        control_values.append(
            np.asarray(arrays["hvg"][np.asarray(sorted(rows), dtype=np.int64)], dtype=np.float32)
        )
    condition_matrix = np.concatenate(condition_values, axis=0)
    control_matrix = np.concatenate(control_values, axis=0)
    rng = np.random.default_rng([seed, len(condition_ids), 997])
    if max_cells is not None:
        if len(condition_matrix) > max_cells:
            condition_matrix = condition_matrix[rng.choice(len(condition_matrix), max_cells, replace=False)]
        if len(control_matrix) > max_cells:
            control_matrix = control_matrix[rng.choice(len(control_matrix), max_cells, replace=False)]
    return mask_fn(control_matrix, condition_matrix, fdr), condition_matrix.mean(0) - control_matrix.mean(0)


def _select_deg_ids(mask, true_delta, top_k: int) -> np.ndarray:
    """Return up to ``top_k`` significant genes ordered by |true delta|.

    The Method 4.5.2 protocol does not substitute all HVGs when a condition
    has fewer than ``top_k`` significant genes; the condition contributes only
    the significant genes it has.
    """
    mask = np.asarray(mask, dtype=bool)
    delta = np.asarray(true_delta, dtype=np.float64)
    if mask.shape != delta.shape:
        raise ValueError("DEG mask and delta must have the same shape")
    candidates = np.flatnonzero(mask)
    k = min(int(top_k), len(candidates))
    order = np.argsort(-np.abs(delta[candidates]), kind="mergesort")
    return np.asarray(candidates[order[:k]], dtype=np.int64)


def _aggregate_cell_line_drug(records, prediction_rows, dataset, args, seed, mask_fn, condition_metrics_fn):
    """Collapse dose-level rows to the legacy cell-line–drug evaluation unit."""
    grouped = defaultdict(list)
    grouped_rows = defaultdict(list)
    for record, row in zip(records, prediction_rows):
        key = (record["population"], record.get("combination_key", record["drug"]))
        grouped[key].append(record)
        grouped_rows[key].append(row)
    output_records, output_rows = [], []
    for (population, drug), items in sorted(grouped.items()):
        condition_ids = [int(item["condition_id"]) for item in items]
        weights = np.asarray([item["n_condition_cells"] for item in items], dtype=np.float64)
        weights /= weights.sum()
        predicted = np.average(np.stack([item["predicted"] for item in items]), axis=0, weights=weights)
        observed = np.average(np.stack([item["observed"] for item in items]), axis=0, weights=weights)
        control = np.average(np.stack([item["control"] for item in items]), axis=0, weights=weights)
        deg_mask, true_delta = _group_degs(
            dataset, condition_ids, args.deg_fdr, args.deg_max_cells, seed, mask_fn
        )
        all_metrics = condition_metrics_fn(predicted, observed, control, true_deg_mask=deg_mask, deg_top_k=args.deg_top_k)
        deg_ids = _select_deg_ids(deg_mask, true_delta, args.deg_top_k)
        metric_values = {f"hvg_{name}": value for name, value in all_metrics.items()}
        if len(deg_ids):
            for name, value in condition_metrics_fn(predicted, observed, control, deg_ids).items():
                metric_values[f"deg_{name}"] = value
        output_records.append({
            "population": population,
            "drug": drug,
            "combination_key": str(drug),
            "condition_id": condition_ids[0],
            "condition_ids": tuple(condition_ids),
            "n_doses": int(len(condition_ids)),
            "predicted": predicted,
            "observed": observed,
            "control": control,
            "metrics": metric_values,
            "too_few_degs": int(len(np.flatnonzero(deg_mask)) < args.deg_top_k),
        })
        rows = grouped_rows[(population, drug)]
        output_rows.append({
            "condition_id": condition_ids[0],
            "condition_ids": list(condition_ids),
            "population": population,
            "drug": drug,
            "combination_key": str(drug),
            "smiles": str(rows[0]["smiles"]),
            "component_smiles": rows[0].get("component_smiles", [str(rows[0]["smiles"])]),
            "component_doses": rows[0].get("component_doses", [float(rows[0]["dose"])]),
            "dose": float(np.mean([float(row["dose"]) for row in rows])),
            "doses": [float(row["dose"]) for row in rows],
            "predicted": predicted.astype(np.float32).tolist(),
            "observed": observed.astype(np.float32).tolist(),
            "control": control.astype(np.float32).tolist(),
            "true_deg_ids": deg_ids.astype(np.int32).tolist(),
        })
    return output_records, output_rows


def _summarize_records(records, components, elapsed: float, peak_memory_gb: float):
    values = defaultdict(list)
    predicted_by_population = defaultdict(list)
    observed_by_population = defaultdict(list)
    populations, drugs = [], []
    all_predicted, all_observed, all_control = [], [], []
    too_few_degs = 0
    for record in records:
        for name, value in record["metrics"].items():
            values[name].append(value)
        population = record["population"]
        predicted_by_population[population].append(record["predicted"])
        observed_by_population[population].append(record["observed"])
        populations.append(population)
        drugs.append(record["drug"])
        all_predicted.append(record["predicted"])
        all_observed.append(record["observed"])
        all_control.append(record["control"])
        too_few_degs += int(record["too_few_degs"])
    if not records:
        raise ValueError("The selected evaluation split contains no conditions")
    predicted_matrix = np.stack(all_predicted)
    observed_matrix = np.stack(all_observed)
    control_matrix = np.stack(all_control)
    result = {name: _finite_mean(metric) for name, metric in values.items()}
    result["hvg_auroc"] = result.get("hvg_deg_auroc", float("nan"))
    result["hvg_auprc"] = result.get("hvg_deg_auprc", float("nan"))
    result["deg_auroc"] = result["hvg_auroc"]
    result["deg_auprc"] = result["hvg_auprc"]
    result["deg_top_k_overlap"] = result.get("hvg_deg_accuracy", float("nan"))
    for name in (
        "r2", "pcc", "pearson_delta", "pearson_logfc", "direction_accuracy",
        "mse", "wasserstein", "mmd", "sinkhorn_distance",
    ):
        result.setdefault(f"deg_{name}", float("nan"))
    # The paper's PDS is Manhattan/cityblock distance within each cell line,
    # followed by an unweighted mean over cell lines.  Keep Euclidean as an
    # explicitly named diagnostic for backwards comparisons.
    result["hvg_pds"] = _finite_mean([
        components["discrimination"](
            np.stack(predicted_by_population[value]),
            np.stack(observed_by_population[value]),
            metric="cityblock",
        ) for value in predicted_by_population
    ])
    result["hvg_pds_global"] = components["discrimination"](
        predicted_matrix, observed_matrix, metric="cityblock"
    )
    result["hvg_pds_euclidean"] = _finite_mean([
        components["discrimination"](
            np.stack(predicted_by_population[value]),
            np.stack(observed_by_population[value]),
            metric="euclidean",
        ) for value in predicted_by_population
    ])
    result["hvg_pds_euclidean_global"] = components["discrimination"](
        predicted_matrix, observed_matrix, metric="euclidean"
    )
    result["pds"] = result["hvg_pds"]
    result["pds_euclidean"] = result["hvg_pds_euclidean"]
    result["disentanglement_score"] = components["disentanglement"](
        populations, drugs, predicted_matrix - control_matrix
    )
    result["css"] = components["css"](
        predicted_matrix - control_matrix, observed_matrix - control_matrix
    )
    biological = components["biological"](
        populations, predicted_matrix, observed_matrix, control_matrix
    )
    result["biological_score"] = biological["score"]
    result["mean_population_pearson_logfc"] = biological["mean_population_pearson_logfc"]
    result["mean_population_directional_accuracy"] = biological[
        "mean_population_directional_accuracy"
    ]
    result["population_delta_magnitude_pearson"] = biological[
        "population_delta_magnitude_pearson"
    ]
    result["biological_metrics"] = biological
    result["n_conditions"] = len(records)
    result["n_conditions_with_fewer_than_top_k_significant_degs"] = too_few_degs
    result["evaluation_seconds"] = float(elapsed)
    result["computational_efficiency"] = len(records) / max(float(elapsed), 1e-9)
    result["evaluation_efficiency"] = result["computational_efficiency"]
    result["peak_gpu_memory_gb"] = float(peak_memory_gb)
    return result


@torch.inference_mode()
def evaluate_seed(model, args, evaluation_split: str, seed: int, device, components):
    started = time.perf_counter()
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    fields = (
        {"control_gene_ids", "control_expressions", "condition_hvg_vectors", "control_hvg_vectors"}
        if args.model == "map"
        else method_data_fields(
            args.model,
            evaluation=True,
            input_mode=(
                model.hparams.get("input_mode")
                if args.model == "xpert"
                else None
            ),
        )
    )
    dataset_class = CRISPDataset if args.model == "crisp" else MAPDataset
    dataset_options = {}
    if args.model == "crisp":
        dataset_options = {
            # DEG masks affect training loss only and are not needed to predict.
            "deg_mask_mode": "validation",
            "drug_representation": model.hparams.get(
                "drug_representation", "official"
            ),
            "control_representation": model.hparams.get(
                "control_representation", "official"
            ),
            "material_dir": args.material_dir,
        }
    dataset = dataset_class(
        args.data_dir, args.regime, evaluation_split,
        set_size=args.set_size, seed=seed, training=False,
        split_file=args.split_file, populations=args.populations, fields=fields,
        **dataset_options,
    )
    records = []
    prediction_rows = []
    processed_batches = 0
    # Always score each dose-level condition.  The cell-line/drug view is
    # derived from these same predictions below, so one evaluator invocation
    # writes both granular and merged metrics without changing the sampled
    # cells or introducing a second model pass.
    evaluation_indices = range(len(dataset.condition_ids))
    total_evaluations = len(evaluation_indices)
    for index in evaluation_indices:
        processed_batches += 1
        item = dataset[index]
        precision = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device.type == "cuda"
            else nullcontext()
        )
        with precision:
            if args.model == "map":
                component_smiles = item.get("component_smiles", [item["drug_smiles"]])
                component_doses = item.get("component_doses", [item["drug_conc"]])
                dose = torch.tensor([component_doses], device=device, dtype=torch.float32)
                _, predicted_hvg = model(
                    item["control_gene_ids"].unsqueeze(0).to(device),
                    item["control_expressions"].unsqueeze(0).to(device),
                    [component_smiles],
                    dose,
                )
                predicted = predicted_hvg.float().mean(1).cpu().numpy()[0]
            else:
                predicted = _method_prediction(model, item, device)
        observed = item["condition_hvg_vectors"].float().mean(0).numpy()
        control = item["control_hvg_vectors"].float().mean(0).numpy()
        condition_id = int(item["condition_id"])
        population = str(item["population"])
        condition_ids = tuple(int(value) for value in item.get("condition_ids", (condition_id,)))
        condition = dataset.conditions.loc[condition_ids[0]]
        drug = str(item.get("drug", condition["drug"]))
        # Keep the number of rows contributing to this dose-level condition.
        # It is used as the weight when dose-level predictions are collapsed
        # into one cell-line--drug result.  Row-aware splits expose the exact
        # evaluation rows through ``rows_by_condition``; paper-style splits
        # fall back to the complete condition group.
        condition_rows = dataset.rows_by_condition.get(condition_id)
        if condition_rows is None:
            arrays = dataset._open_population(population)
            condition_rows = dataset._group_rows(
                arrays, "condition", condition_id
            )
        n_condition_cells = int(len(condition_rows))
        if n_condition_cells <= 0:
            raise ValueError(
                f"Condition {condition_id} has no cells in evaluation split"
            )
        if args.evaluation_unit == "dose_level_condition":
            # DEG discovery follows Method 4.5.2: all cell-level rows for
            # this condition (or the explicitly selected split rows), not the
            # 24 cells sampled for the model input.
            deg_mask, true_delta = _condition_degs(
                dataset, condition_id, args.deg_fdr, args.deg_max_cells,
                seed, components["deg_mask"],
            )
        else:
            # Optional legacy cell-line/drug analysis pools all doses before
            # DEG discovery, matching the historical evaluator only when the
            # caller explicitly requests it.
            deg_mask, true_delta = _group_degs(
                dataset, condition_ids, args.deg_fdr, args.deg_max_cells,
                seed, components["deg_mask"],
            )
        metrics = components["condition_metrics"](
            predicted, observed, control,
            true_deg_mask=deg_mask, deg_top_k=args.deg_top_k,
        )
        condition_metrics_values = {f"hvg_{name}": value for name, value in metrics.items()}
        deg_ids = _select_deg_ids(deg_mask, true_delta, args.deg_top_k)
        if len(deg_ids):
            for name, value in components["condition_metrics"](
                predicted, observed, control, deg_ids
            ).items():
                condition_metrics_values[f"deg_{name}"] = value
        records.append({
            "condition_id": condition_id,
            "condition_ids": condition_ids,
            "population": population,
            "drug": drug,
            "combination_key": str(item.get("combination_key", drug)),
            "n_condition_cells": n_condition_cells,
            "predicted": predicted,
            "observed": observed,
            "control": control,
            "metrics": condition_metrics_values,
            "too_few_degs": int(len(np.flatnonzero(deg_mask)) < args.deg_top_k),
        })
        prediction_rows.append({
            "condition_id": condition_id,
            "population": population,
            "drug": drug,
            "smiles": str(item["drug_smiles"]),
            "dose": float(item["drug_conc"]),
            "component_smiles": list(item.get("component_smiles", [item["drug_smiles"]])),
            "component_doses": list(item.get("component_doses", [item["drug_conc"]])),
            "combination_key": str(item.get("combination_key", drug)),
            "n_condition_cells": n_condition_cells,
            "predicted": predicted.astype(np.float32).tolist(),
            "observed": observed.astype(np.float32).tolist(),
            "control": control.astype(np.float32).tolist(),
            "condition_ids": list(condition_ids),
            "true_deg_ids": deg_ids.astype(np.int32).tolist(),
        })
        if (index + 1) % 25 == 0:
            print(
                f"seed={seed}: {index + 1}/{total_evaluations} "
                f"{args.evaluation_unit}",
                flush=True,
            )

    elapsed = time.perf_counter() - started
    peak_memory = (
        torch.cuda.max_memory_allocated(device) / 1024**3
        if device.type == "cuda"
        else 0.0
    )
    dose_result = _summarize_records(records, components, elapsed, peak_memory)
    dose_result["n_batches"] = processed_batches
    dose_result["batch_size"] = 1
    dose_per_drug = {
        drug: _summarize_records(
            [record for record in records if record["drug"] == drug],
            components,
            elapsed,
            peak_memory,
        )
        for drug in sorted({record["drug"] for record in records})
    }
    merged_records, merged_prediction_rows = _aggregate_cell_line_drug(
        records, prediction_rows, dataset, args, seed,
        components["deg_mask"], components["condition_metrics"],
    )
    merged_result = _summarize_records(
        merged_records, components, elapsed, peak_memory
    )
    merged_result["n_batches"] = len(merged_records)
    merged_result["batch_size"] = 1
    merged_per_drug = {
        drug: _summarize_records(
            [record for record in merged_records if record["drug"] == drug],
            components, elapsed, peak_memory,
        )
        for drug in sorted({record["drug"] for record in merged_records})
    }
    primary = merged_result if args.evaluation_unit in {"cell_line_drug", "cell_line_combination"} else dose_result
    primary_per_drug = (
        merged_per_drug
        if args.evaluation_unit in {"cell_line_drug", "cell_line_combination"}
        else dose_per_drug
    )
    return primary, prediction_rows, dataset, primary_per_drug, {
        "dose_result": dose_result,
        "dose_prediction_rows": prediction_rows,
        "dose_per_drug": dose_per_drug,
        "dose_records": records,
        "merged_result": merged_result,
        "merged_prediction_rows": merged_prediction_rows,
        "merged_per_drug": merged_per_drug,
        "merged_records": merged_records,
    }


def main() -> None:
    args = parse_args()
    if (
        args.set_size <= 0
        or args.deg_top_k <= 0
        or not 0 < args.deg_fdr <= 1
    ):
        raise ValueError(
            "set_size/deg_top_k must be positive and deg_fdr in (0, 1]"
        )
    if not args.seeds:
        raise ValueError("At least one evaluation seed is required")
    components = _components()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    shapes = json.loads((Path(args.data_dir) / "materialized_shapes.json").read_text())
    if args.model == "cmonge":
        hvg_dimensions = {int(value["hvg_dim"]) for value in shapes.values()}
        if hvg_dimensions != {2000} or int(args.hvg_dim) != 2000:
            raise ValueError(
                "CMonge evaluation requires exactly 2000 HVGs in both the "
                f"materialized data and evaluator; data={sorted(hvg_dimensions)}, "
                f"evaluator={args.hvg_dim}"
            )
    if args.model == "map":
        required = {
            "se_checkpoint": args.se_ckpt,
            "esm_embeddings": args.esm_embeddings,
            "mapkg_checkpoint": args.mapkg_ckpt,
            "mapkg_vocab": args.mapkg_vocab,
            "static_token_cache": args.static_token_cache,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ValueError(f"MAP evaluation is missing frozen inputs: {', '.join(missing)}")
        checkpoint_args = checkpoint.get("args", {})
        fusion = args.combination_fusion or checkpoint_args.get("combination_fusion", "avg_emb")
        max_components = args.max_components or checkpoint_args.get("max_components", 2)
        model = components["model"](
            **required,
            num_gene_tokens=args.num_gene_tokens,
            hvg_dim=args.hvg_dim,
            combination_fusion=fusion,
            max_components=max_components,
        ).to(device)
        if checkpoint.get("format") != "map_method_4_4_v1":
            raise ValueError("Only Method 4.4 checkpoints are supported for MAP")
        model.pert_model.load_state_dict(
            checkpoint["pert_model_state_dict"], strict=True
        )
    else:
        if checkpoint.get("format") not in {"map_method_v2", "map_baseline_v2"} or checkpoint.get("model") != args.model:
            raise ValueError(f"Checkpoint is not a {args.model} method checkpoint")
        checkpoint_args = checkpoint.get("args", {})
        populations = list(args.populations or checkpoint_args.get("populations") or shapes)
        model = load_model(
            args.model, checkpoint, args.data_dir, args.hvg_dim, populations, device,
            material_dir=args.material_dir,
        )
    model.eval()

    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Evaluation directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    preparation = (
        json.loads(Path(args.preparation_config).read_text(encoding="utf-8"))
        if args.preparation_config else {}
    )
    checkpoint_path = Path(args.checkpoint).resolve()
    common_payload = {
        "model": args.model,
        "regime": args.regime,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "preparation_id": preparation.get("preparation_id"),
        "preparation": preparation,
        "set_size": args.set_size,
        "batch_size": 1,
        "seeds": args.seeds,
        "deg_top_k": args.deg_top_k,
        "deg_fdr": args.deg_fdr,
        "deg_max_cells": args.deg_max_cells,
        "metric_catalog": components["catalog"],
        "evaluation_unit": args.evaluation_unit,
        "reports": ["dose_level_condition", "cell_line_drug", "cell_line_combination"],
        "deg_method": "wilcoxon_rank_sum_bh",
        "deg_fallback": "none",
        "pds_metric": "cityblock",
        "pds_scope": "mean_over_cell_lines",
        "pds_euclidean_metric": "euclidean",
        "pds_euclidean_scope": "mean_over_cell_lines",
    }
    root_splits = {}
    for evaluation_split in args.evaluation_splits:
        split_output = output / evaluation_split
        split_output.mkdir(parents=True, exist_ok=False)
        runs = []
        per_drug_runs = []
        dose_runs = []
        merged_runs = []
        dose_per_drug_runs = []
        merged_per_drug_runs = []
        dose_condition_metric_rows = []
        cell_line_drug_metric_rows = []
        last_dataset = None
        for seed in args.seeds:
            result, prediction_rows, last_dataset, per_drug, detail = evaluate_seed(
                model, args, evaluation_split, int(seed), device, components
            )
            result["seed"] = int(seed)
            runs.append(result)
            for drug, metrics in per_drug.items():
                per_drug_runs.append({
                    "drug": drug,
                    "seed": int(seed),
                    **{
                        key: value for key, value in metrics.items()
                        if key != "biological_metrics"
                    },
                })
            for key, target in (("dose_result", dose_runs), ("merged_result", merged_runs)):
                row = dict(detail[key])
                row["seed"] = int(seed)
                target.append(row)
            for record, prediction in zip(
                detail["dose_records"], detail["dose_prediction_rows"]
            ):
                dose_condition_metric_rows.append({
                    "condition_id": int(record["condition_id"]),
                    "population": str(record["population"]),
                    "drug": str(record["drug"]),
                    "dose": float(prediction["dose"]),
                    "n_condition_cells": int(record["n_condition_cells"]),
                    "seed": int(seed),
                    **record["metrics"],
                })
            for record, prediction in zip(
                detail["merged_records"], detail["merged_prediction_rows"]
            ):
                cell_line_drug_metric_rows.append({
                    "condition_id": int(record["condition_id"]),
                    "condition_ids": ",".join(
                        str(value) for value in record.get("condition_ids", ())
                    ),
                    "population": str(record["population"]),
                    "drug": str(record["drug"]),
                    "dose": float(prediction["dose"]),
                    "n_doses": int(len(prediction.get("doses", []))),
                    "seed": int(seed),
                    **record["metrics"],
                })
            for key, target in (("dose_per_drug", dose_per_drug_runs),
                                ("merged_per_drug", merged_per_drug_runs)):
                for drug, metrics in detail[key].items():
                    target.append({
                        "drug": drug, "seed": int(seed),
                        **{name: value for name, value in metrics.items()
                           if name != "biological_metrics"},
                    })
            pq.write_table(
                pa.Table.from_pylist(prediction_rows),
                split_output / f"predictions_seed{int(seed)}.parquet",
                compression="zstd",
            )
            pq.write_table(
                pa.Table.from_pylist(detail["merged_prediction_rows"]),
                split_output / f"predictions_cell_line_drug_seed{int(seed)}.parquet",
                compression="zstd",
            )
        dose_summary = components["summarize"](dose_runs)
        merged_summary = components["summarize"](merged_runs)
        split_payload = {
            **common_payload,
            "evaluation_split": evaluation_split,
            "split_file": str(last_dataset.split_file),
            "split_id": last_dataset.split_id,
            "split_rule": last_dataset.split_manifest.get("rule"),
            "split_seed": last_dataset.split_manifest.get("seed"),
            "runs": runs,
            "summary": components["summarize"](runs),
            "dose_level": {
                "summary": dose_summary,
                "runs": dose_runs,
            },
            "cell_line_drug": {
                "summary": merged_summary,
                "runs": merged_runs,
            },
        }
        pd.DataFrame(dose_runs).drop(
            columns=["biological_metrics"], errors="ignore"
        ).to_csv(split_output / "dose_level_runs.csv", index=False)
        pd.DataFrame(merged_runs).drop(
            columns=["biological_metrics"], errors="ignore"
        ).to_csv(split_output / "cell_line_drug_runs.csv", index=False)
        pd.DataFrame(dose_condition_metric_rows).to_csv(
            split_output / "dose_level_condition_metrics.csv", index=False
        )
        pd.DataFrame(cell_line_drug_metric_rows).to_csv(
            split_output / "cell_line_drug_metrics.csv", index=False
        )
        split_payload["dose_level_runs_file"] = str(
            (split_output / "dose_level_runs.csv").resolve()
        )
        split_payload["cell_line_drug_runs_file"] = str(
            (split_output / "cell_line_drug_runs.csv").resolve()
        )
        split_payload["dose_level_condition_metrics_file"] = str(
            (split_output / "dose_level_condition_metrics.csv").resolve()
        )
        split_payload["cell_line_drug_metrics_file"] = str(
            (split_output / "cell_line_drug_metrics.csv").resolve()
        )
        split_payload["report_files"] = {
            "dose_level_runs": split_payload["dose_level_runs_file"],
            "dose_level_condition_metrics": split_payload[
                "dose_level_condition_metrics_file"
            ],
            "cell_line_drug_runs": split_payload["cell_line_drug_runs_file"],
            "cell_line_drug_metrics": split_payload[
                "cell_line_drug_metrics_file"
            ],
        }
        if args.regime == "unprofiled_drug" and evaluation_split == "external_test":
            per_drug_frame = pd.DataFrame(per_drug_runs)
            numeric = [
                column for column in per_drug_frame.select_dtypes(include=[np.number]).columns
                if column != "seed"
            ]
            per_drug_summary = per_drug_frame.groupby("drug", sort=True)[numeric].mean()
            per_drug_summary["n_seeds"] = per_drug_frame.groupby("drug")["seed"].nunique()
            mean_row = per_drug_summary[numeric].mean(axis=0).to_dict()
            mean_row.update({"drug": "__mean__", "n_seeds": len(args.seeds)})
            per_drug_summary = per_drug_summary.reset_index()
            per_drug_summary = pd.concat(
                [per_drug_summary, pd.DataFrame([mean_row])], ignore_index=True
            )
            per_drug_summary.to_csv(split_output / "per_drug_metrics.csv", index=False)
            split_payload["per_drug_metrics"] = per_drug_summary.to_dict("records")
            split_payload["per_drug_mean"] = mean_row
        for label, rows in (("dose_level", dose_per_drug_runs),
                            ("cell_line_drug", merged_per_drug_runs)):
            frame = pd.DataFrame(rows)
            if frame.empty:
                continue
            numeric = [
                column for column in frame.select_dtypes(include=[np.number]).columns
                if column != "seed"
            ]
            summary = frame.groupby("drug", sort=True)[numeric].mean()
            summary["n_seeds"] = frame.groupby("drug")["seed"].nunique()
            mean_row = summary[numeric].mean(axis=0).to_dict()
            mean_row.update({"drug": "__mean__", "n_seeds": len(args.seeds)})
            summary = pd.concat(
                [summary.reset_index(), pd.DataFrame([mean_row])], ignore_index=True
            )
            filename = f"{label}_per_drug_metrics.csv"
            summary.to_csv(split_output / filename, index=False)
            split_payload[f"{label}_per_drug_metrics"] = summary.to_dict("records")
            split_payload[f"{label}_per_drug_mean"] = mean_row
        (split_output / "evaluation.json").write_text(
            json.dumps(split_payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        pd.DataFrame(runs).drop(columns=["biological_metrics"], errors="ignore").to_csv(
            split_output / "runs.csv", index=False
        )
        root_splits[evaluation_split] = {
            "evaluation_file": str((split_output / "evaluation.json").resolve()),
            "summary": split_payload["summary"],
            "dose_level_summary": dose_summary,
            "cell_line_drug_summary": merged_summary,
            "dose_level_runs_file": str((split_output / "dose_level_runs.csv").resolve()),
            "cell_line_drug_runs_file": str((split_output / "cell_line_drug_runs.csv").resolve()),
            "dose_level_condition_metrics_file": str(
                (split_output / "dose_level_condition_metrics.csv").resolve()
            ),
            "cell_line_drug_metrics_file": str(
                (split_output / "cell_line_drug_metrics.csv").resolve()
            ),
            "dose_level_per_drug_metrics_file": str(
                (split_output / "dose_level_per_drug_metrics.csv").resolve()
            ) if (split_output / "dose_level_per_drug_metrics.csv").is_file() else None,
            "cell_line_drug_per_drug_metrics_file": str(
                (split_output / "cell_line_drug_per_drug_metrics.csv").resolve()
            ) if (split_output / "cell_line_drug_per_drug_metrics.csv").is_file() else None,
            "per_drug_metrics_file": (
                str((split_output / "per_drug_metrics.csv").resolve())
                if "per_drug_metrics" in split_payload else None
            ),
        }
    root_payload = {
        **common_payload,
        "evaluation_splits": list(args.evaluation_splits),
        "splits": root_splits,
    }
    (output / "evaluation.json").write_text(
        json.dumps(root_payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        json.dumps(
            {name: value["summary"] for name, value in root_splits.items()},
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
