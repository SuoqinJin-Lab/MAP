from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .._common.feedback import Feedback, StageResult
from .._common.identifiers import file_identity, slug
from .._common.paths import DatasetPaths


def _input_identity(path: str | Path) -> str:
    value = Path(path)
    parent = value.parent.name
    return slug(f"{parent}-{value.stem}" if parent else value.stem, 120)


def _resolve_evaluation_file(paths: DatasetPaths, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if path.is_file():
        return path.resolve()
    return paths.evaluations / path


def summarize_evaluations(
    paths: DatasetPaths,
    regimes: Iterable[str] | None = None,
    evaluation_files: Iterable[str | Path] | None = None,
    output_name: str = "evaluation_summary.json",
) -> StageResult:
    report = Feedback(paths.analysis, "evaluation_summary")
    rows = []
    allowed_regimes = set(regimes) if regimes is not None else None
    candidates = (
        [Path(value) for value in evaluation_files]
        if evaluation_files is not None
        else sorted(paths.evaluations.glob("*/evaluation.json"))
    )
    for path in candidates:
        path = _resolve_evaluation_file(paths, path)
        if not path.is_file():
            report.emit("evaluation missing", path=path)
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        regime = payload.get("regime")
        if allowed_regimes is not None and regime not in allowed_regimes:
            continue
        for metric, values in payload.get("summary", {}).items():
            row = {
                "evaluation_id": path.parent.name,
                "evaluation_file": str(path),
                "model": payload.get("model", "map"),
                "regime": regime,
                "split_id": payload.get("split_id"),
                "split_file": payload.get("split_file"),
                "checkpoint": payload.get("checkpoint"),
                "checkpoint_sha256": payload.get("checkpoint_sha256"),
                "metric": metric,
            }
            if isinstance(values, dict):
                row.update(values)
            else:
                row["value"] = values
            rows.append(row)
        report.emit(
            "evaluation loaded",
            evaluation_id=path.parent.name,
            regime=regime,
            metrics=len(payload.get("summary", {})),
        )
    output = paths.analysis / output_name
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    evaluation_ids = {row["evaluation_id"] for row in rows}
    regimes_found = {row["regime"] for row in rows if row["regime"]}
    return report.finish({
        "evaluations": len(evaluation_ids) if rows else len(candidates),
        "regimes": sorted(regimes_found),
        "metrics": len(rows),
    }, [output])


def analyze_predictions(
    paths: DatasetPaths,
    prediction_file: Path,
    output_name: str | None = None,
) -> StageResult:
    """Analyze exported condition-level arrays without requiring a model."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Prediction analysis requires pyarrow") from exc
    report = Feedback(paths.analysis, "response_analysis")
    rows = pq.read_table(prediction_file).to_pylist()
    metrics = []
    for row in rows:
        predicted = np.asarray(row["predicted"], dtype=np.float64)
        observed = np.asarray(row["observed"], dtype=np.float64)
        control = np.asarray(row["control"], dtype=np.float64)
        pred_delta, true_delta = predicted - control, observed - control
        denominator = np.square(observed - observed.mean()).sum()
        metrics.append({
            "condition_id": row.get("condition_id"),
            "population": row.get("population"),
            "drug": row.get("drug"),
            "dose": row.get("dose"),
            "r2": float(1.0 - np.square(predicted - observed).sum() / denominator) if denominator > 0 else float("nan"),
            "pcc_logfc": float(np.corrcoef(pred_delta, true_delta)[0, 1]) if np.std(pred_delta) > 0 and np.std(true_delta) > 0 else float("nan"),
            "mse": float(np.mean(np.square(predicted - observed))),
            "direction_accuracy": float(np.mean(np.sign(pred_delta) == np.sign(true_delta))),
        })
    analysis_id = _input_identity(prediction_file)
    output = paths.analysis / (output_name or f"response_metrics__{analysis_id}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    report.emit("response metrics", conditions=len(metrics), output=output)
    return report.finish({"conditions": len(metrics)}, [output])


def analyze_degs(
    paths: DatasetPaths,
    prediction_file: Path,
    top_k: tuple[int, ...] = (20, 50, 100),
    output_name: str | None = None,
) -> StageResult:
    """Compare predicted and observed top DEG sets from exported arrays."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    report = Feedback(paths.analysis, "deg_analysis")
    rows = pq.read_table(prediction_file).to_pylist()
    output_rows = []
    for row in rows:
        predicted = np.asarray(row["predicted"], dtype=np.float64)
        observed = np.asarray(row["observed"], dtype=np.float64)
        control = np.asarray(row["control"], dtype=np.float64)
        pred_delta = predicted - control
        true_delta = observed - control
        supplied = row.get("true_deg_ids")
        for k in top_k:
            if supplied:
                true_ids = np.asarray(supplied[:k], dtype=np.int64)
            else:
                true_ids = np.argsort(-np.abs(true_delta), kind="mergesort")[:k]
            pred_ids = np.argsort(-np.abs(pred_delta), kind="mergesort")[:k]
            true_set, pred_set = set(map(int, true_ids)), set(map(int, pred_ids))
            overlap = len(true_set & pred_set) / max(len(true_set), 1)
            direction = float(np.mean(np.sign(pred_delta[true_ids]) == np.sign(true_delta[true_ids]))) if len(true_ids) else float("nan")
            output_rows.append({
                "condition_id": row.get("condition_id"),
                "population": row.get("population"),
                "drug": row.get("drug"),
                "top_k": k,
                "overlap": overlap,
                "direction_accuracy": direction,
                "deg_source": "exported_true_deg_ids" if supplied else "top_abs_observed_delta",
            })
    analysis_id = output_name or _input_identity(prediction_file)
    out = paths.analysis / "degs" / slug(analysis_id, 120)
    out.mkdir(parents=True, exist_ok=True)
    table_path = out / "deg_metrics.parquet"
    if output_rows:
        table = pa.Table.from_pylist(output_rows)
    else:
        table = pa.table({
            "condition_id": pa.array([], type=pa.int64()),
            "population": pa.array([], type=pa.string()),
            "drug": pa.array([], type=pa.string()),
            "top_k": pa.array([], type=pa.int32()),
            "overlap": pa.array([], type=pa.float64()),
            "direction_accuracy": pa.array([], type=pa.float64()),
            "deg_source": pa.array([], type=pa.string()),
        })
    pq.write_table(table, table_path)
    summary_path = out / "deg_summary.json"
    summary = {str(k): {
        "mean_overlap": float(np.nanmean([row["overlap"] for row in output_rows if row["top_k"] == k])) if output_rows else float("nan"),
        "mean_direction_accuracy": float(np.nanmean([row["direction_accuracy"] for row in output_rows if row["top_k"] == k])) if output_rows else float("nan"),
    } for k in top_k}
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report.emit("DEG comparison", conditions=len(rows), top_k=list(top_k))
    return report.finish(summary, [table_path, summary_path])


def analyze_embeddings(
    paths: DatasetPaths,
    embedding_file: Path,
    labels_file: Path | None = None,
    max_points: int = 20_000,
    output_name: str | None = None,
) -> StageResult:
    """Write a bounded deterministic PCA projection for a .npy or raw memmap file."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    report = Feedback(paths.analysis, "embedding_analysis")
    embedding_file = Path(embedding_file)
    if embedding_file.suffix == ".npy":
        matrix = np.load(embedding_file, mmap_mode="r")
        if matrix.ndim != 2:
            raise ValueError(f"Embedding array must be 2-D, got {matrix.shape}")
        total_rows, dimensions = map(int, matrix.shape)
    else:
        dimensions = 2048
        itemsize = np.dtype(np.float16).itemsize
        size = embedding_file.stat().st_size
        if size % (dimensions * itemsize):
            raise ValueError("Raw embedding file size is not divisible by 2048 float16 values")
        total_rows = size // (dimensions * itemsize)
        matrix = np.memmap(embedding_file, dtype=np.float16, mode="r", shape=(total_rows, dimensions))
    if total_rows == 0:
        raise ValueError("Embedding file is empty")
    take = min(int(max_points), total_rows)
    indices = np.linspace(0, total_rows - 1, take, dtype=np.int64) if take < total_rows else np.arange(total_rows)
    values = np.asarray(matrix[indices], dtype=np.float32)
    values = values - values.mean(axis=0, keepdims=True)
    covariance = (values.T @ values) / max(values.shape[0] - 1, 1)
    eigenvalues, components = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1][:2]
    coords = values @ components[:, order]
    analysis_id = output_name or _input_identity(embedding_file)
    output = paths.analysis / "embeddings" / slug(analysis_id, 120)
    output.mkdir(parents=True, exist_ok=True)
    coord_path = output / "pca_coords.parquet"
    rows = [{"row": int(index), "pc1": float(point[0]), "pc2": float(point[1])} for index, point in zip(indices, coords)]
    pq.write_table(pa.Table.from_pylist(rows), coord_path)
    report.emit("PCA complete", rows=len(rows), total_rows=int(total_rows), dimensions=dimensions, method="PCA")
    return report.finish({"rows": len(rows), "total_rows": int(total_rows), "dimensions": dimensions, "method": "PCA", "umap": "optional"}, [coord_path])


def analyze_generalization(
    paths: DatasetPaths,
    evaluation_files: Iterable[str | Path] | None = None,
    output_name: str = "generalization_summary.json",
) -> StageResult:
    report = Feedback(paths.analysis, "generalization_analysis")
    summary = []
    candidates = (
        [Path(value) for value in evaluation_files]
        if evaluation_files is not None
        else sorted(paths.evaluations.glob("*/evaluation.json"))
    )
    for path in candidates:
        path = _resolve_evaluation_file(paths, path)
        if not path.is_file():
            report.emit("evaluation missing", path=path)
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        summary.append({
            "evaluation_id": path.parent.name,
            "model": payload.get("model", "map"),
            "regime": payload.get("regime"),
            "split_id": payload.get("split_id"),
            "checkpoint": payload.get("checkpoint"),
            "path": str(path),
            "summary": payload.get("summary", {}),
        })
    output = paths.analysis / output_name
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report.emit("generalization summary", regimes=len(summary))
    return report.finish({"regimes": len(summary)}, [output])


def _svg_frame(title: str, body: str, subtitle: str = "") -> str:
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 640 300">'
        '<rect width="640" height="300" fill="#f5f7f5"/>'
        f'<text x="24" y="32" font-family="system-ui" font-size="20" fill="#176b4d">{title}</text>'
        f'<text x="24" y="54" font-family="system-ui" font-size="12" fill="#637068">{subtitle}</text>'
        f'<g transform="translate(52 74)">{body}</g></svg>'
    )


def build_report(
    paths: DatasetPaths,
    prediction_file: Path | None = None,
    evaluation_summary_file: Path | None = None,
    output_name: str | None = None,
) -> StageResult:
    analysis_id = (
        output_name
        or (_input_identity(prediction_file) if prediction_file is not None else "all_evaluations")
    )
    root = paths.analysis / "reports" / slug(analysis_id, 120)
    report = Feedback(root, "analysis_report")
    output = root / "report.html"
    summary = Path(evaluation_summary_file) if evaluation_summary_file else paths.analysis / "evaluation_summary.json"
    response = paths.analysis / f"response_metrics__{_input_identity(prediction_file)}.json" if prediction_file is not None else paths.analysis / "response_metrics.json"
    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    prediction_rows = []
    if prediction_file is not None and Path(prediction_file).is_file():
        try:
            import pyarrow.parquet as pq
            prediction_rows = pq.read_table(prediction_file).to_pylist()
        except ImportError:
            prediction_rows = []

    # Small dependency-free SVGs keep the report usable on a login node while
    # remaining real plots rather than blank placeholders.
    scatter = []
    for row in prediction_rows[:32]:
        predicted = np.asarray(row["predicted"], dtype=np.float64)
        observed = np.asarray(row["observed"], dtype=np.float64)
        for x, y in zip(predicted[::max(1, len(predicted) // 40)], observed[::max(1, len(observed) // 40)]):
            scatter.append((float(x), float(y)))
    if scatter:
        values = np.asarray(scatter)
        low, high = float(values.min()), float(values.max())
        scale = 235.0 / max(high - low, 1e-9)
        circles = "".join(
            f'<circle cx="{(x-low)*scale:.1f}" cy="{235-(y-low)*scale:.1f}" r="2" fill="#245e8a" opacity=".35"/>'
            for x, y in scatter[:1200]
        )
        diagonal = f'<line x1="0" y1="235" x2="235" y2="0" stroke="#9dac9f" stroke-dasharray="4 4"/>'
        response_svg = _svg_frame("Predicted vs observed response", circles + diagonal, f"n={len(scatter):,} gene values")
    else:
        response_svg = _svg_frame("Predicted vs observed response", '<text x="0" y="40" fill="#637068">Run analyze(prediction_file=...) to populate this plot.</text>')

    if prediction_rows:
        subset = prediction_rows[:20]
        matrix = np.asarray([np.asarray(row["predicted"]) - np.asarray(row["control"]) for row in subset], dtype=np.float64)
        matrix = matrix[:, : min(40, matrix.shape[1])]
        scale = max(float(np.max(np.abs(matrix))), 1e-9)
        cells = []
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = float(matrix[i, j] / scale)
                color = "#176b4d" if value >= 0 else "#245e8a"
                opacity = min(1.0, abs(value))
                cells.append(f'<rect x="{j*5}" y="{i*9}" width="4" height="8" fill="{color}" opacity="{opacity:.2f}"/>')
        deg_svg = _svg_frame("Predicted response heatmap", "".join(cells), f"conditions={len(subset)}; first 40 HVGs")
        dose_values = [(float(row.get("dose", 0.0)), float(np.mean(np.asarray(row["predicted"]) - np.asarray(row["control"])))) for row in prediction_rows]
        dose_values.sort()
        lo, hi = min(v for _, v in dose_values), max(v for _, v in dose_values)
        dose_points = "".join(f'<circle cx="{24 + i*220/max(len(dose_values)-1,1):.1f}" cy="{210-(v-lo)*150/max(hi-lo,1e-9):.1f}" r="3" fill="#176b4d"/>' for i, (_, v) in enumerate(dose_values[:64]))
        dose_svg = _svg_frame("Dose response", dose_points, f"conditions={len(dose_values)}")
    else:
        deg_svg = _svg_frame("Predicted response heatmap", '<text x="0" y="40" fill="#637068">Prediction parquet not supplied.</text>')
        dose_svg = _svg_frame("Dose response", '<text x="0" y="40" fill="#637068">Prediction parquet not supplied.</text>')
    (figures / "response_scatter.svg").write_text(response_svg, encoding="utf-8")
    (figures / "deg_heatmap.svg").write_text(deg_svg, encoding="utf-8")
    (figures / "dose_response.svg").write_text(dose_svg, encoding="utf-8")
    (figures / "embedding_umap.svg").write_text(_svg_frame("Embedding projection", '<text x="0" y="40" fill="#637068">PCA is computed by analyze_embeddings; UMAP is an optional extension.</text>'), encoding="utf-8")
    tables = root / "analysis_tables.xlsx"
    try:
        import pandas as pd
        if summary.is_file():
            payload = json.loads(summary.read_text(encoding="utf-8"))
            pd.DataFrame(payload).to_excel(tables, index=False)
        else:
            pd.DataFrame({"status": ["evaluation_summary.json not found"]}).to_excel(tables, index=False)
    except ImportError:
        # Keep the pipeline usable in the minimal package environment.
        tables.write_text("Install pandas+openpyxl to generate the Excel workbook.\n", encoding="utf-8")
    output.write_text(
        "<html><body><h1>MAP evaluation analysis</h1>"
        f"<p>evaluation summary: {summary.exists()}</p>"
        f"<p>response metrics: {response.exists()}</p></body></html>",
        encoding="utf-8",
    )
    report.emit("analysis report", figures=figures, tables=tables, prediction_rows=len(prediction_rows))
    return report.finish({"evaluation_summary": summary.exists(), "response_metrics": response.exists(), "figures": 4, "analysis_tables": tables.exists()}, [output, tables, figures])
