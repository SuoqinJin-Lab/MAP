from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr, ranksums, t as student_t, wasserstein_distance


def safe_pearson(left, right) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if len(left) < 2 or np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return float("nan")
    return float(pearsonr(left, right).statistic)


def safe_r2(predicted, observed) -> float:
    predicted = np.asarray(predicted, dtype=np.float64)
    observed = np.asarray(observed, dtype=np.float64)
    denominator = np.square(observed - observed.mean()).sum()
    if denominator < 1e-12:
        return float("nan")
    return float(1.0 - np.square(predicted - observed).sum() / denominator)


def benjamini_hochberg(p_values) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=np.float64)
    p_values = np.where(np.isfinite(p_values), p_values, 1.0)
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0.0, 1.0)
    return result


def significant_deg_mask(control, condition, fdr: float) -> np.ndarray:
    control = np.asarray(control)
    condition = np.asarray(condition)
    if control.ndim != 2 or condition.ndim != 2:
        raise ValueError("DEG inputs must be cell-by-gene matrices")
    if control.shape[1] != condition.shape[1]:
        raise ValueError("Control and condition gene dimensions differ")
    _, p_values = ranksums(control, condition, axis=0, nan_policy="omit")
    return benjamini_hochberg(p_values) < fdr


def _sinkhorn(predicted, observed, bins=64, epsilon_scale=0.05, iterations=60):
    values = np.concatenate([
        np.asarray(predicted, dtype=np.float64),
        np.asarray(observed, dtype=np.float64),
    ])
    if values.size == 0 or not np.all(np.isfinite(values)):
        return float("nan")
    low, high = float(values.min()), float(values.max())
    if high - low < 1e-12:
        return 0.0
    edges = np.linspace(low, high, bins + 1)
    predicted_hist, _ = np.histogram(predicted, bins=edges)
    observed_hist, _ = np.histogram(observed, bins=edges)
    predicted_hist = predicted_hist.astype(np.float64)
    observed_hist = observed_hist.astype(np.float64)
    predicted_hist /= max(predicted_hist.sum(), 1.0)
    observed_hist /= max(observed_hist.sum(), 1.0)
    centers = (edges[:-1] + edges[1:]) * 0.5
    cost = np.square(centers[:, None] - centers[None, :])
    positive = cost[cost > 0]
    epsilon = max(float(np.median(positive)) * epsilon_scale, 1e-8)
    kernel = np.exp(-cost / epsilon)
    left = np.ones(bins, dtype=np.float64)
    right = np.ones(bins, dtype=np.float64)
    for _ in range(iterations):
        left = predicted_hist / np.maximum(kernel @ right, 1e-30)
        right = observed_hist / np.maximum(kernel.T @ left, 1e-30)
    plan = left[:, None] * kernel * right[None, :]
    return float(np.sqrt(max((plan * cost).sum(), 0.0)))


def maximum_mean_discrepancy(predicted, observed, max_samples: int = 512) -> float:
    """RBF-kernel MMD over expression values; lower is better."""
    predicted = np.asarray(predicted, dtype=np.float64).reshape(-1)
    observed = np.asarray(observed, dtype=np.float64).reshape(-1)
    if (
        predicted.size == 0
        or observed.size == 0
        or not np.isfinite(predicted).all()
        or not np.isfinite(observed).all()
    ):
        return float("nan")

    def quantile_sample(values: np.ndarray) -> np.ndarray:
        values = np.sort(values)
        if len(values) <= max_samples:
            return values
        indices = np.linspace(0, len(values) - 1, max_samples, dtype=np.int64)
        return values[indices]

    predicted = quantile_sample(predicted)
    observed = quantile_sample(observed)
    combined = np.concatenate((predicted, observed))
    distances = np.abs(combined[:, None] - combined[None, :])
    positive = distances[distances > 0]
    if not positive.size:
        return 0.0
    bandwidth = max(float(np.median(positive)), 1e-8)
    scale = 2.0 * bandwidth * bandwidth

    def kernel(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return np.exp(-np.square(left[:, None] - right[None, :]) / scale)

    squared = (
        kernel(predicted, predicted).mean()
        + kernel(observed, observed).mean()
        - 2.0 * kernel(predicted, observed).mean()
    )
    return float(np.sqrt(max(float(squared), 0.0)))


def covariance_structure_score(predicted_delta, observed_delta) -> float:
    """Relative Frobenius error between HVG response covariance matrices."""
    predicted = np.asarray(predicted_delta, dtype=np.float64)
    observed = np.asarray(observed_delta, dtype=np.float64)
    if predicted.ndim != 2 or observed.shape != predicted.shape:
        raise ValueError("CSS inputs must be matching condition-by-HVG matrices")
    if predicted.shape[0] < 2 or not np.isfinite(predicted).all() or not np.isfinite(observed).all():
        return float("nan")
    predicted = predicted - predicted.mean(axis=0, keepdims=True)
    observed = observed - observed.mean(axis=0, keepdims=True)
    denominator = float(predicted.shape[0] - 1)
    predicted_covariance = (predicted.T @ predicted) / denominator
    observed_covariance = (observed.T @ observed) / denominator
    reference = float(np.linalg.norm(observed_covariance, ord="fro"))
    predicted_covariance -= observed_covariance
    return float(
        np.linalg.norm(predicted_covariance, ord="fro") / max(reference, 1e-12)
    )


def _roc_metrics(labels, scores):
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    positive = int(labels.sum())
    negative = int((~labels).sum())
    if positive == 0 or negative == 0 or not np.all(np.isfinite(scores)):
        return float("nan"), float("nan")
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order].astype(np.float64)
    cumulative_positive = np.cumsum(sorted_labels)
    cumulative_negative = np.cumsum(1.0 - sorted_labels)
    tpr = np.concatenate([[0.0], cumulative_positive / positive, [1.0]])
    fpr = np.concatenate([[0.0], cumulative_negative / negative, [1.0]])
    precision = cumulative_positive / np.arange(1, len(labels) + 1)
    return (
        float(np.trapezoid(tpr, fpr)),
        float(precision[labels[order]].sum() / positive),
    )


def condition_metrics(
    predicted,
    observed,
    control,
    indices=None,
    true_deg_mask=None,
    deg_top_k=50,
):
    predicted = np.asarray(predicted, dtype=np.float64)
    observed = np.asarray(observed, dtype=np.float64)
    control = np.asarray(control, dtype=np.float64)
    pred_delta = predicted - control
    true_delta = observed - control
    if indices is not None:
        predicted, observed, control = predicted[indices], observed[indices], control[indices]
        pred_delta, true_delta = pred_delta[indices], true_delta[indices]
    metrics = {
        "r2": safe_r2(predicted, observed),
        "pcc": safe_pearson(predicted, observed),
        "pearson_delta": safe_pearson(pred_delta, true_delta),
        "pearson_logfc": safe_pearson(pred_delta, true_delta),
        "direction_accuracy": float(np.mean(np.sign(pred_delta) == np.sign(true_delta))),
        "mse": float(np.mean(np.square(predicted - observed))),
        "wasserstein": float(wasserstein_distance(predicted, observed)),
        "mmd": maximum_mean_discrepancy(predicted, observed),
        "sinkhorn_distance": _sinkhorn(predicted, observed),
    }
    if true_deg_mask is not None:
        labels = np.asarray(true_deg_mask, dtype=bool)
        auroc, auprc = _roc_metrics(labels, np.abs(pred_delta))
        true_top = np.flatnonzero(labels)
        true_top = true_top[
            np.argsort(-np.abs(true_delta[true_top]), kind="mergesort")
        ][:deg_top_k]
        predicted_top = np.argsort(-np.abs(pred_delta), kind="mergesort")[:deg_top_k]
        metrics.update({
            "deg_auroc": auroc,
            "deg_auprc": auprc,
            "deg_accuracy": float(
                len(set(true_top.tolist()) & set(predicted_top.tolist()))
                / max(len(true_top), 1)
            ),
        })
    return metrics


def perturbation_discrimination(predicted, observed) -> float:
    if len(predicted) == 0:
        return float("nan")
    distances = cdist(predicted, observed, metric="cityblock")
    ranks = [
        float(np.sum(distances[index] < distances[index, index]))
        for index in range(len(predicted))
    ]
    return float(1.0 - 2.0 * np.mean(np.asarray(ranks) / len(predicted)))


def disentanglement_score(populations, drugs, deltas) -> float:
    values = np.asarray(deltas, dtype=np.float64)
    if len(values) < 3 or np.var(values) < 1e-12:
        return float("nan")
    populations = np.asarray(populations)
    drugs = np.asarray(drugs)
    global_mean = values.mean(0)
    additive = np.tile(global_mean, (len(values), 1))
    for population in np.unique(populations):
        mask = populations == population
        additive[mask] += values[mask].mean(0) - global_mean
    for drug in np.unique(drugs):
        mask = drugs == drug
        additive[mask] += values[mask].mean(0) - global_mean
    return float(np.clip(1.0 - np.var(values - additive) / np.var(values), 0.0, 1.0))


def biological_metrics(populations, predicted, observed, control) -> dict:
    populations = np.asarray(populations)
    predicted, observed, control = map(np.asarray, (predicted, observed, control))
    per_population = {}
    correlations, directions = [], []
    for population in np.unique(populations):
        mask = populations == population
        pred_delta = predicted[mask].mean(0) - control[mask].mean(0)
        true_delta = observed[mask].mean(0) - control[mask].mean(0)
        correlation = safe_pearson(pred_delta, true_delta)
        direction = float(np.mean(np.sign(pred_delta) == np.sign(true_delta)))
        per_population[str(population)] = {
            "pearson_logfc": correlation,
            "directional_accuracy": direction,
            "predicted_delta_magnitude": float(np.mean(np.abs(pred_delta))),
            "observed_delta_magnitude": float(np.mean(np.abs(true_delta))),
        }
        if np.isfinite(correlation):
            correlations.append(correlation)
        directions.append(direction)
    predicted_magnitude = np.asarray([
        value["predicted_delta_magnitude"] for value in per_population.values()
    ])
    observed_magnitude = np.asarray([
        value["observed_delta_magnitude"] for value in per_population.values()
    ])
    magnitude_correlation = safe_pearson(predicted_magnitude, observed_magnitude)
    mean_correlation = float(np.nanmean(correlations)) if correlations else float("nan")
    mean_direction = float(np.nanmean(directions))
    return {
        "score": float(np.nanmean([mean_correlation, mean_direction, magnitude_correlation])),
        "mean_population_pearson_logfc": mean_correlation,
        "mean_population_directional_accuracy": mean_direction,
        "population_delta_magnitude_pearson": magnitude_correlation,
        "by_population": per_population,
    }


def summarize_runs(results: list[dict]) -> dict:
    excluded = {
        "seed", "n_conditions", "n_test_conditions",
        "n_conditions_with_fewer_than_top_k_significant_degs",
        "deg_top_k", "evaluation_seconds", "biological_metrics",
    }
    names = sorted(name for name in results[0] if name not in excluded)
    summary = {}
    for name in names:
        values = np.asarray([result[name] for result in results], dtype=np.float64)
        valid = values[np.isfinite(values)]
        mean = float(valid.mean()) if len(valid) else float("nan")
        half_width = (
            float(student_t.ppf(0.975, len(valid) - 1) * np.std(valid, ddof=1) / np.sqrt(len(valid)))
            if len(valid) > 1 else float("nan")
        )
        summary[name] = {
            "mean": mean,
            "std": float(np.std(valid, ddof=1)) if len(valid) > 1 else float("nan"),
            "n": int(len(valid)),
            "ci95_half_width": half_width,
            "lower": mean - half_width,
            "upper": mean + half_width,
        }
    return summary


METRIC_CATALOG = {
    "hvg_pearson_delta": {"label": "HVG Pearson Δ", "direction": "higher"},
    "hvg_direction_accuracy": {"label": "HVG Direction Accuracy", "direction": "higher"},
    "hvg_pds": {"label": "HVG PDS", "direction": "higher"},
    "deg_pearson_delta": {"label": "Top-k DEG Pearson Δ", "direction": "higher"},
    "deg_direction_accuracy": {"label": "Top-k DEG Direction Accuracy", "direction": "higher"},
    "hvg_mse": {"label": "HVG MSE", "direction": "lower"},
    "hvg_wasserstein": {"label": "HVG Wasserstein Distance", "direction": "lower"},
    "hvg_mmd": {"label": "HVG MMD", "direction": "lower"},
    "css": {
        "label": "Covariance Structure Score",
        "direction": "lower",
        "definition": "Relative Frobenius error between predicted and observed HVG response covariance matrices",
    },
    "deg_mse": {"label": "Top-k DEG MSE", "direction": "lower"},
    "deg_wasserstein": {"label": "Top-k DEG Wasserstein Distance", "direction": "lower"},
    "deg_mmd": {"label": "Top-k DEG MMD", "direction": "lower"},
    "deg_auroc": {"label": "DEG AUROC", "direction": "higher"},
    "deg_auprc": {"label": "DEG AUPRC", "direction": "higher"},
    "deg_top_k_overlap": {"label": "Top-k DEG Overlap", "direction": "higher"},
    "disentanglement_score": {"label": "Disentanglement Score", "direction": "higher"},
    "computational_efficiency": {"label": "Conditions per Second", "direction": "higher"},
    "peak_gpu_memory_gb": {"label": "Peak GPU Memory (GiB)", "direction": "lower"},
    "biological_score": {"label": "Biological Score", "direction": "higher"},
}


__all__ = [
    "METRIC_CATALOG", "biological_metrics", "condition_metrics",
    "covariance_structure_score", "maximum_mean_discrepancy",
    "disentanglement_score", "perturbation_discrimination", "significant_deg_mask",
    "summarize_runs",
]
