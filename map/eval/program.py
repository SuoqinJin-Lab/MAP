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
from ..train.baselines import MODEL_REGISTRY, load_model
from ..train.baselines.utils import baseline_data_fields
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
    parser = argparse.ArgumentParser(description="Evaluate MAP or a bundled baseline")
    parser.add_argument("--model", choices=("map", *MODEL_REGISTRY), default="map")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split-file", required=True)
    parser.add_argument(
        "--evaluation-splits",
        nargs="+",
        choices=("internal_test", "external_test"),
        default=["internal_test", "external_test"],
    )
    parser.add_argument(
        "--regime", choices=("unprofiled_drug", "unseen_combination"), required=True
    )
    parser.add_argument("--se-ckpt")
    parser.add_argument("--esm-embeddings")
    parser.add_argument("--mapkg-ckpt")
    parser.add_argument("--mapkg-vocab")
    parser.add_argument("--static-token-cache")
    parser.add_argument("--preparation-config")
    parser.add_argument("--populations", nargs="+")
    parser.add_argument("--set-size", type=int, default=24)
    parser.add_argument("--num-gene-tokens", type=int, default=2047)
    parser.add_argument("--hvg-dim", type=int, default=2000)
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
    batch = {}
    for key, value in item.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.unsqueeze(0).to(device)
        elif key in {"drug_smiles", "population"}:
            batch[key] = [value]
        elif key == "drug_conc":
            batch[key] = torch.tensor([value], device=device, dtype=torch.float32)
        else:
            batch[key] = value
    return batch


def _baseline_prediction(model, item: dict, device: torch.device) -> np.ndarray:
    prediction = model.predict_batch(_batched_item(item, device))
    if not isinstance(prediction, torch.Tensor):
        raise TypeError("Baseline output does not contain a prediction tensor")
    if prediction.ndim != 2 or prediction.shape[0] != 1:
        raise ValueError(f"Expected condition-level predictions [1, genes], got {prediction.shape}")
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
    result["hvg_pds"] = _finite_mean([
        components["discrimination"](
            np.stack(predicted_by_population[value]),
            np.stack(observed_by_population[value]),
        )
        for value in predicted_by_population
    ])
    result["pds"] = result["hvg_pds"]
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
        else baseline_data_fields(args.model, evaluation=True)
    )
    dataset = MAPDataset(
        args.data_dir, args.regime, evaluation_split,
        set_size=args.set_size, seed=seed, training=False,
        split_file=args.split_file, populations=args.populations, fields=fields,
    )
    records = []
    prediction_rows = []
    for index, condition_id in enumerate(dataset.condition_ids):
        item = dataset[index]
        precision = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device.type == "cuda"
            else nullcontext()
        )
        with precision:
            if args.model == "map":
                dose = torch.tensor(
                    [item["drug_conc"]], device=device, dtype=torch.float32
                )
                _, predicted_hvg = model(
                    item["control_gene_ids"].unsqueeze(0).to(device),
                    item["control_expressions"].unsqueeze(0).to(device),
                    [item["drug_smiles"]],
                    dose,
                )
                predicted = predicted_hvg.float().mean(1).cpu().numpy()[0]
            else:
                predicted = _baseline_prediction(model, item, device)
        observed = item["condition_hvg_vectors"].float().mean(0).numpy()
        control = item["control_hvg_vectors"].float().mean(0).numpy()
        deg_mask, true_delta = _condition_degs(
            dataset, int(condition_id), args.deg_fdr, args.deg_max_cells,
            seed, components["deg_mask"],
        )
        metrics = components["condition_metrics"](
            predicted, observed, control,
            true_deg_mask=deg_mask, deg_top_k=args.deg_top_k,
        )
        condition_metrics_values = {
            f"hvg_{name}": value for name, value in metrics.items()
        }
        deg_ids = np.flatnonzero(deg_mask)
        deg_ids = deg_ids[
            np.lexsort((deg_ids, -np.abs(true_delta[deg_ids])))
        ][:args.deg_top_k]
        if len(deg_ids) < args.deg_top_k:
            too_few_degs = True
        else:
            too_few_degs = False
        if len(deg_ids):
            for name, value in components["condition_metrics"](
                predicted, observed, control, deg_ids
            ).items():
                condition_metrics_values[f"deg_{name}"] = value
        population = str(item["population"])
        condition = dataset.conditions.loc[int(condition_id)]
        drug = str(condition["drug"])
        records.append({
            "population": population,
            "drug": drug,
            "predicted": predicted,
            "observed": observed,
            "control": control,
            "metrics": condition_metrics_values,
            "too_few_degs": too_few_degs,
        })
        prediction_rows.append({
            "condition_id": int(condition_id),
            "population": population,
            "drug": drug,
            "smiles": str(item["drug_smiles"]),
            "dose": float(item["drug_conc"]),
            "predicted": predicted.astype(np.float32).tolist(),
            "observed": observed.astype(np.float32).tolist(),
            "control": control.astype(np.float32).tolist(),
            "true_deg_ids": deg_ids.astype(np.int32).tolist(),
        })
        if (index + 1) % 25 == 0:
            print(f"seed={seed}: {index + 1}/{len(dataset)} conditions", flush=True)

    elapsed = time.perf_counter() - started
    peak_memory = (
        torch.cuda.max_memory_allocated(device) / 1024**3
        if device.type == "cuda"
        else 0.0
    )
    result = _summarize_records(records, components, elapsed, peak_memory)
    per_drug = {
        drug: _summarize_records(
            [record for record in records if record["drug"] == drug],
            components,
            elapsed,
            peak_memory,
        )
        for drug in sorted({record["drug"] for record in records})
    }
    return result, prediction_rows, dataset, per_drug


def main() -> None:
    args = parse_args()
    if args.set_size <= 0 or args.deg_top_k <= 0 or not 0 < args.deg_fdr <= 1:
        raise ValueError("set_size/deg_top_k must be positive and deg_fdr in (0, 1]")
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
        model = components["model"](
            **required,
            num_gene_tokens=args.num_gene_tokens,
            hvg_dim=args.hvg_dim,
        ).to(device)
        if checkpoint.get("format") != "map_method_4_4_v1":
            raise ValueError("Only Method 4.4 checkpoints are supported for MAP")
        model.pert_model.load_state_dict(
            checkpoint["pert_model_state_dict"], strict=True
        )
    else:
        if checkpoint.get("format") != "map_baseline_v2" or checkpoint.get("model") != args.model:
            raise ValueError(f"Checkpoint is not a {args.model} baseline checkpoint")
        checkpoint_args = checkpoint.get("args", {})
        populations = list(args.populations or checkpoint_args.get("populations") or shapes)
        model = load_model(
            args.model, checkpoint, args.data_dir, args.hvg_dim, populations, device
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
        "seeds": args.seeds,
        "deg_top_k": args.deg_top_k,
        "deg_fdr": args.deg_fdr,
        "deg_max_cells": args.deg_max_cells,
        "metric_catalog": components["catalog"],
    }
    root_splits = {}
    for evaluation_split in args.evaluation_splits:
        split_output = output / evaluation_split
        split_output.mkdir(parents=True, exist_ok=False)
        runs = []
        per_drug_runs = []
        last_dataset = None
        for seed in args.seeds:
            result, prediction_rows, last_dataset, per_drug = evaluate_seed(
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
            pq.write_table(
                pa.Table.from_pylist(prediction_rows),
                split_output / f"predictions_seed{int(seed)}.parquet",
                compression="zstd",
            )
        split_payload = {
            **common_payload,
            "evaluation_split": evaluation_split,
            "split_file": str(last_dataset.split_file),
            "split_id": last_dataset.split_id,
            "split_rule": last_dataset.split_manifest.get("rule"),
            "split_seed": last_dataset.split_manifest.get("seed"),
            "runs": runs,
            "summary": components["summarize"](runs),
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
        (split_output / "evaluation.json").write_text(
            json.dumps(split_payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        pd.DataFrame(runs).drop(columns=["biological_metrics"], errors="ignore").to_csv(
            split_output / "runs.csv", index=False
        )
        root_splits[evaluation_split] = {
            "evaluation_file": str((split_output / "evaluation.json").resolve()),
            "summary": split_payload["summary"],
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
