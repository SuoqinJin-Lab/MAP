from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .._common.feedback import Feedback, StageResult
from .._common.paths import DatasetPaths


_SHARED_FILES = (
    "materialized_shapes.json",
    "conditions.parquet",
    "materialization_manifest.json",
    "preparation_config.json",
)


def _read_json(path: Path, errors: list[str], label: str) -> dict[str, Any]:
    if not path.is_file():
        errors.append(f"{label}: missing {path}")
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        errors.append(f"{label}: invalid JSON {path} ({error})")
        return {}
    if not isinstance(payload, dict):
        errors.append(f"{label}: expected a JSON object in {path}")
        return {}
    return payload


def _required_file(
    path: Path, files: dict[str, dict[str, Any]], errors: list[str], label: str
) -> bool:
    exists = path.is_file()
    files[label] = {"path": str(path), "exists": exists}
    if not exists:
        errors.append(f"{label}: missing {path}")
    return exists


def _resolve_split_path(paths: DatasetPaths, value: str | Path) -> Path:
    value = Path(value)
    if value.is_absolute():
        return value
    # A bare split id is resolved to the canonical immutable split object.
    candidates = [paths.workspace / value, paths.splits / value]
    if len(value.parts) == 1:
        candidates.append(paths.split_file(str(value)))
    return next(
        (candidate for candidate in candidates if candidate.is_file()),
        candidates[-1],
    )


def _split_id_from_request(paths: DatasetPaths, value: str | Path | None) -> str:
    """Derive the report owner even when a dry-run split is missing."""
    if value is None:
        return "unknown"
    requested = Path(value)
    candidate = requested if requested.is_absolute() else paths.workspace / requested
    if len(requested.parts) == 1 and not requested.is_absolute():
        candidate = paths.split_file(str(value))
    if candidate.name == "split.json":
        return candidate.parent.name
    return candidate.stem


def _population_shapes(
    paths: DatasetPaths,
    shapes: Mapping[str, Any],
    *,
    state_inputs: bool,
    state_embeddings: bool,
    errors: list[str],
    files: dict[str, dict[str, Any]],
    inspect_arrays: bool,
) -> dict[str, dict[str, Any]]:
    checked: dict[str, dict[str, Any]] = {}
    for population, value in shapes.items():
        population = str(population)
        if not isinstance(value, Mapping):
            errors.append(f"population {population}: shape must be an object")
            continue
        try:
            n_cells = int(value["n_cells"])
            token_length = int(value.get("token_length", 2048))
            hvg_dim = int(value.get("hvg_dim", 2000))
        except (KeyError, TypeError, ValueError) as error:
            errors.append(f"population {population}: invalid shape ({error})")
            continue
        if min(n_cells, token_length, hvg_dim) <= 0:
            errors.append(f"population {population}: dimensions must be positive")
            continue
        root = paths.prepared / population
        specs: dict[str, tuple[str, np.dtype, tuple[int, ...]]] = {
            "row_condition": ("row_condition.int32.dat", np.dtype(np.int32), (n_cells,)),
            "row_group": ("row_group.uint16.dat", np.dtype(np.uint16), (n_cells,)),
            "hvg": ("hvg.float16.dat", np.dtype(np.float16), (n_cells, hvg_dim)),
        }
        if state_inputs:
            specs.update({
                "state_gene_ids": ("se_gene_ids.uint16.dat", np.dtype(np.uint16), (n_cells, token_length)),
                "state_expression": ("se_expr.float16.dat", np.dtype(np.float16), (n_cells, token_length)),
            })
        if state_embeddings:
            specs["state_embeddings"] = (
                "state_embeddings.float16.dat", np.dtype(np.float16), (n_cells, 2048)
            )
        population_report = {"n_cells": n_cells, "token_length": token_length, "hvg_dim": hvg_dim, "files": {}}
        for name, (filename, dtype, expected_shape) in specs.items():
            path = root / filename
            exists = _required_file(path, files, errors, f"{population}/{name}")
            size_ok = None
            finite = None
            if exists:
                expected_bytes = int(np.prod(expected_shape)) * dtype.itemsize
                size_ok = path.stat().st_size == expected_bytes
                if not size_ok:
                    errors.append(
                        f"{population}/{name}: expected {expected_bytes} bytes, got {path.stat().st_size}"
                    )
                if inspect_arrays and size_ok and name in {"hvg", "state_expression", "state_embeddings"}:
                    try:
                        array = np.memmap(path, dtype=dtype, mode="r", shape=expected_shape)
                        finite = bool(np.isfinite(np.asarray(array[: min(32, n_cells)])).all())
                    except (OSError, ValueError) as error:
                        errors.append(f"{population}/{name}: cannot inspect ({error})")
                    if finite is False:
                        errors.append(f"{population}/{name}: sample contains non-finite values")
            population_report["files"][name] = {
                "path": str(path), "exists": exists, "size_ok": size_ok,
                "sample_finite": finite,
            }
        checked[population] = population_report
    return checked


def _validate_split(
    paths: DatasetPaths,
    split_file: str | Path | None,
    regime: str | None,
    errors: list[str],
    files: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    if split_file is None:
        errors.append("split: a split_file is required for training")
        return None
    split_path = _resolve_split_path(paths, split_file)
    if not _required_file(split_path, files, errors, "split"):
        return None
    payload = _read_json(split_path, errors, "split")
    rule = payload.get("rule", payload.get("regime"))
    if regime is not None and rule != regime:
        errors.append(f"split: rule {rule!r} does not match regime {regime!r}")
    # A split is an immutable project object.  Do not silently accept an
    # arbitrary JSON file (or a project-level ``splits.json``) and then attach
    # its validation report to a different split directory.  This also keeps
    # validation, training and evaluation on exactly the same path contract.
    requested_split_id = payload.get(
        "split_id",
        split_path.parent.name if split_path.name == "split.json" else split_path.stem,
    )
    try:
        canonical_path = paths.split_file(str(requested_split_id)).resolve()
        if split_path.resolve() != canonical_path:
            errors.append(
                "split: file must be stored as "
                f"splits/{requested_split_id}/split.json"
            )
    except ValueError as error:
        errors.append(f"split: invalid split_id {requested_split_id!r} ({error})")
    names = ("train", "internal_test", "external_test")
    aliases = {"internal_test": "val", "external_test": "test"}
    sets: dict[str, set[int]] = {}
    for name in names:
        source = name if name in payload else aliases.get(name)
        values = payload.get(source, []) if source else []
        try:
            sets[name] = {int(value) for value in values}
        except (TypeError, ValueError):
            sets[name] = set()
            errors.append(f"split: {name} is not an integer id list")
        if not sets[name]:
            errors.append(f"split: {name} is empty")
    if sets.get("external_test", set()) & (sets.get("train", set()) | sets.get("internal_test", set())):
        errors.append("split: external_test overlaps train or internal_test")
    return {
        "path": str(split_path.resolve()),
        "split_id": requested_split_id,
        "rule": rule,
        "seed": payload.get("seed"),
        "counts": {name: len(values) for name, values in sets.items()},
    }


def _matrix_check(
    path: Path,
    expected_shape: tuple[int, ...],
    *,
    label: str,
    files: dict[str, dict[str, Any]],
    errors: list[str],
    inspect_arrays: bool,
) -> dict[str, Any]:
    exists = _required_file(path, files, errors, label)
    shape = None
    finite = None
    if exists:
        try:
            array = np.load(path, mmap_mode="r")
            shape = list(array.shape)
            shape_matches = len(array.shape) == len(expected_shape) and all(
                expected < 0 or actual == expected
                for actual, expected in zip(array.shape, expected_shape)
            )
            if not shape_matches:
                errors.append(f"{label}: expected shape {expected_shape}, got {tuple(array.shape)}")
            if inspect_arrays:
                sample = np.asarray(array.reshape(-1)[: min(array.size, 1024)])
                finite = bool(np.isfinite(sample).all())
                if not finite:
                    errors.append(f"{label}: sample contains non-finite values")
        except (OSError, ValueError) as error:
            errors.append(f"{label}: cannot load ({error})")
    return {"path": str(path), "exists": exists, "shape": shape, "expected_shape": list(expected_shape), "sample_finite": finite}


def _model_manifest(
    paths: DatasetPaths, model: str, errors: list[str], files: dict[str, dict[str, Any]],
    material_root: Path | None = None,
) -> tuple[dict[str, Any], Path | None]:
    if material_root is not None:
        candidates = [Path(material_root) / "manifest.json"]
    else:
        artifact = {
            "chemcpa": "drug_ecfp4", "prnet": "drug_fcfp4",
            "crisp": "drug_rdkit2d", "cmonge": "drug_rdkit2d",
            "xpert": "graph_assets",
        }.get(model)
        candidates = [paths.material_dir(artifact) / "manifest.json"] if artifact else []
    for model_file in candidates:
        if model_file.is_file():
            return _read_json(model_file, errors, f"{model} artifact manifest"), model_file
    model_file = candidates[0] if candidates else paths.prepared / "manifest.json"
    errors.append(f"{model}: required artifact manifest is missing")
    files[f"{model}/manifest"] = {"path": str(model_file), "exists": False}
    return {}, None


def _validate_method_artifacts(
    paths: DatasetPaths,
    model: str,
    regime: str | None,
    options: Mapping[str, Any],
    shapes: Mapping[str, Any],
    drug_count: int | None,
    errors: list[str],
    files: dict[str, dict[str, Any]],
    inspect_arrays: bool,
    material_root: Path | None = None,
) -> dict[str, Any]:
    if model == "trainmean":
        return {"material": "shared cell metadata + HVG expression"}
    manifest, manifest_file = _model_manifest(paths, model, errors, files, material_root)
    root = Path(material_root) if material_root is not None else paths.prepared
    if material_root is None:
        extra_artifacts = {
            "crisp": ("control_means", "deg_masks"),
            "xpert": ("drug_unimol", "expression_bins"),
        }.get(model, ())
        for artifact in extra_artifacts:
            extra_file = paths.material_dir(artifact) / "manifest.json"
            if extra_file.is_file():
                extra = _read_json(extra_file, errors, f"{model}/{artifact} manifest")
                manifest.update(extra)
    root = root / str(manifest.get("directory", "."))
    artifact_roots = {
        "chemcpa": {"drug": paths.material_dir("drug_ecfp4")},
        "prnet": {"drug": paths.material_dir("drug_fcfp4")},
        "crisp": {
            "drug": paths.material_dir("drug_rdkit2d"),
            "means": paths.material_dir("control_means"),
            "masks": paths.material_dir("deg_masks"),
        },
        "cmonge": {"drug": root if material_root is not None else paths.material_dir("drug_rdkit2d")},
        "xpert": {
            "graph": paths.material_dir("graph_assets"),
            "drug": paths.material_dir("drug_unimol"),
            "expression": paths.material_dir("expression_bins"),
        },
    }.get(model, {})
    result: dict[str, Any] = {"manifest": str(manifest_file) if manifest_file else None}
    if drug_count is not None and manifest.get("smiles") is not None:
        if len(manifest["smiles"]) != drug_count:
            errors.append(f"{model}: drug vocabulary has {len(manifest['smiles'])}, expected {drug_count}")
    if model in {"chemcpa", "prnet"}:
        filename = manifest.get("matrix", "ecfp4_1024.float32.npy" if model == "chemcpa" else "fcfp4_1024.float32.npy")
        dimension = 1024
        result["drug_features"] = _matrix_check(artifact_roots["drug"] / str(filename), (int(drug_count or 0), dimension), label=f"{model}/drug_features", files=files, errors=errors, inspect_arrays=inspect_arrays)
    elif model == "crisp":
        filename = manifest.get("matrix", "rdkit2d.float32.npy")
        matrix_shape = manifest.get("shape", [int(drug_count or 0), -1])
        expected_shape = (
            int(drug_count or 0),
            int(matrix_shape[1]) if isinstance(matrix_shape, (list, tuple)) and len(matrix_shape) > 1 else -1,
        )
        result["drug_features"] = _matrix_check(artifact_roots["drug"] / str(filename), expected_shape, label="crisp/drug_features", files=files, errors=errors, inspect_arrays=inspect_arrays)
        means = manifest.get("control_group_means")
        if not isinstance(means, Mapping):
            errors.append("crisp: control_group_means material is missing")
        else:
            mean_report = {}
            for population, shape in shapes.items():
                entry = means.get(str(population))
                if not isinstance(entry, Mapping):
                    errors.append(f"crisp: control means are missing for {population}")
                    continue
                directory = artifact_roots["means"] / str(entry.get("directory", ""))
                group_ids = directory / "group_ids.int64.npy"
                embedding = directory / "embeddings.float16.npy"
                hvg = directory / "hvg.float16.npy"
                _required_file(group_ids, files, errors, f"crisp/{population}/group_ids")
                group_count = None
                if group_ids.is_file():
                    try:
                        group_count = int(np.load(group_ids, mmap_mode="r").shape[0])
                    except (OSError, ValueError) as error:
                        errors.append(f"crisp/{population}/group_ids: cannot load ({error})")
                if group_count is not None:
                    _matrix_check(embedding, (group_count, 2048), label=f"crisp/{population}/embeddings", files=files, errors=errors, inspect_arrays=inspect_arrays)
                    _matrix_check(hvg, (group_count, int(shape.get("hvg_dim", 2000))), label=f"crisp/{population}/hvg", files=files, errors=errors, inspect_arrays=inspect_arrays)
                mean_report[str(population)] = {"groups": group_count}
            result["control_group_means"] = mean_report
        if bool(options.get("use_deg_mask", False)):
            masks = manifest.get("deg_masks")
            if not isinstance(masks, Mapping):
                errors.append("crisp: DEG masks are required when use_deg_mask=True")
            else:
                mask_report = {}
                for population, shape in shapes.items():
                    entry = masks.get(str(population))
                    if not isinstance(entry, Mapping):
                        errors.append(f"crisp: DEG mask is missing for {population}")
                        continue
                    directory = artifact_roots["masks"] / str(entry.get("directory", ""))
                    ids_path = directory / str(entry.get("condition_ids_file", "condition_ids.int64.npy"))
                    mask_path = directory / str(entry.get("mask_file", "mask.bool.npy"))
                    _required_file(ids_path, files, errors, f"crisp/{population}/deg_ids")
                    count = None
                    if ids_path.is_file():
                        count = int(np.load(ids_path, mmap_mode="r").shape[0])
                    if count is not None:
                        _matrix_check(mask_path, (count, int(shape.get("hvg_dim", 2000))), label=f"crisp/{population}/deg_mask", files=files, errors=errors, inspect_arrays=False)
                    mask_report[str(population)] = {"conditions": count}
                result["deg_masks"] = mask_report
    elif model == "cmonge":
        representation = str(options.get("drug_representation", manifest.get("drug_representation", "rdkit"))).casefold()
        if regime == "unseen_combination" and "drug_representation" not in options:
            representation = "moa"
        if representation not in {"rdkit", "moa"}:
            errors.append(f"cmonge: unknown drug representation {representation!r}")
        # A project may contain an older RDKit manifest while the requested
        # unseen-combination run needs the separately materialized MoA table.
        # Never reuse the manifest's matrix unless its representation agrees
        # with the run contract.
        manifest_representation = str(manifest.get("drug_representation", "")).casefold()
        if manifest_representation == representation and manifest.get("matrix"):
            filename = manifest["matrix"]
            matrix_shape = manifest.get("shape", [int(drug_count or 0), -1])
        else:
            filename = "moa.float32.npy" if representation == "moa" else "rdkit2d.float32.npy"
            matrix_shape = [int(drug_count or 0), 10 if representation == "moa" else -1]
        expected_dim = (
            int(matrix_shape[1])
            if isinstance(matrix_shape, (list, tuple)) and len(matrix_shape) > 1
            else (10 if representation == "moa" else -1)
        )
        result["drug_representation"] = representation
        result["drug_features"] = _matrix_check(root / str(filename), (int(drug_count or 0), expected_dim), label=f"cmonge/{representation}", files=files, errors=errors, inspect_arrays=inspect_arrays)
        for population, shape in shapes.items():
            if int(shape.get("hvg_dim", 2000)) != 2000:
                errors.append(f"cmonge: {population} has hvg_dim={shape.get('hvg_dim')}, expected 2000")
    elif model == "xpert":
        input_mode = str(options.get("input_mode", "official")).casefold()
        if input_mode not in {"official", "validation"}:
            errors.append(f"xpert: unknown input mode {input_mode!r}")
        result["input_mode"] = input_mode
        required_names = [
            ("ppi_edges", manifest.get("ppi_edge_file", "ppi_edges.npz")),
            ("dti_edges", manifest.get("dti_edge_file", "dti_edges.npz")),
            ("dds_edges", manifest.get("dds_edge_file", "dds_edges.npz")),
            ("drug_hg_embeddings", manifest.get("drug_hg_embedding_file", "drug_hg_embeddings.float32.npy")),
            ("drug_unimol", manifest.get("drug_unimol_file", "drug_unimol.float16.npy")),
            ("graph_checkpoint", manifest.get("graph_checkpoint", "graph_encoder.pt")),
            ("expression_boundaries", manifest.get("expression_bin_boundaries_file", "expression_bin_boundaries.float32.npy")),
        ]
        if input_mode == "official":
            required_names.append(("ppi_gene_vectors_full", manifest.get("ppi_gene_vector_full_file", "ppi_gene_vectors_full.float32.npy")))
        else:
            required_names.append(("ppi_gene_vectors", manifest.get("ppi_gene_vector_file", "ppi_gene_vectors.float32.npy")))
        for label, filename in required_names:
            if label in {"ppi_edges", "dti_edges", "dds_edges", "graph_checkpoint", "ppi_gene_vectors", "ppi_gene_vectors_full"}:
                path = artifact_roots["graph"] / str(filename)
            elif label in {"drug_hg_embeddings", "drug_unimol"}:
                path = artifact_roots["drug"] / str(filename)
            else:
                path = artifact_roots["expression"] / str(filename)
            if path.suffix == ".npy":
                if label == "drug_unimol":
                    expected = (int(drug_count or 0), -1, 514)
                elif label == "drug_hg_embeddings":
                    expected = (int(drug_count or 0), -1)
                elif label == "ppi_gene_vectors_full":
                    expected = (int(manifest.get("full_gene_count", -1)), -1)
                else:  # validation-mode HVG gene vectors
                    expected = (int(manifest.get("gene_count", -1)), -1)
                result[label] = _matrix_check(path, expected, label=f"xpert/{label}", files=files, errors=errors, inspect_arrays=False)
            else:
                result[label] = {"path": str(path), "exists": _required_file(path, files, errors, f"xpert/{label}")}
        expression = manifest.get("expression")
        if not isinstance(expression, Mapping) or expression.get("format") not in {
            "expression_bins_v1", "xpert_hvg_expression_v2"
        }:
            errors.append("xpert: expression_bins_v1 contract is missing")
        elif int(expression.get("gene_count", -1)) != int(next(iter(shapes.values()), {}).get("hvg_dim", -2)):
            errors.append("xpert: expression contract does not match hvg_dim")
    else:
        errors.append(f"unknown model: {model}")
    return result


def validate_method(
    paths: DatasetPaths,
    method: str,
    regime: str,
    split_file: str | Path,
    options: Mapping[str, Any] | None = None,
    frozen_assets: Iterable[str | Path] | str | Path | None = None,
    strict: bool = True,
    inspect_arrays: bool | None = None,
) -> StageResult:
    """Validate exactly the material required by one method and split.

    ``strict=False`` is intended for command dry-runs: paths and metadata are
    checked and a report is written, but no large array is opened and missing
    inputs do not raise.  A strict validation is performed automatically by
    the public training entry points before a subprocess is launched.
    """
    model = str(method).casefold()
    if model not in {"map", "prnet", "chemcpa", "trainmean", "crisp", "xpert", "cmonge"}:
        raise ValueError(f"Unknown method: {model}")
    options = dict(options or {})
    inspect = bool(strict) if inspect_arrays is None else bool(inspect_arrays)
    errors: list[str] = []
    files: dict[str, dict[str, Any]] = {}
    for name in _SHARED_FILES:
        _required_file(paths.prepared / name, files, errors, name)
    shapes = _read_json(paths.prepared / "materialized_shapes.json", errors, "materialized_shapes")
    if not shapes:
        shapes = {}
    populations = _population_shapes(
        paths,
        shapes,
        state_inputs=model in {"map", "xpert"} and str(options.get("input_mode", "official")).casefold() == "official",
        # CRISP uses its own control-group embedding cache under
        # ``crisp``; it does not consume the shared per-cell
        # SE-600M embedding mmap.  Keeping that cache model-specific is what
        # keeps optional artifact preparation genuinely on demand.
        state_embeddings=model == "map",
        errors=errors,
        files=files,
        inspect_arrays=inspect,
    )
    split = _validate_split(paths, split_file, regime, errors, files)
    split_id = str((split or {}).get("split_id") or _split_id_from_request(paths, split_file))
    drug_count = None
    conditions_info: dict[str, Any] = {}
    conditions_path = paths.prepared / "conditions.parquet"
    if conditions_path.is_file():
        try:
            import pyarrow.parquet as pq

            schema = pq.read_schema(conditions_path)
            names = set(schema.names)
            required_columns = {"condition_id", "canonical_smiles", "dose"}
            if not ({"population", "cell_line"} & names):
                required_columns.add("population")
            missing_columns = sorted(required_columns - names)
            if missing_columns:
                errors.append("conditions.parquet: missing columns " + ", ".join(missing_columns))
            table = pq.read_table(conditions_path, columns=["canonical_smiles"])
            drug_count = len(set(str(value) for value in table.column(0).to_pylist()))
            conditions_info = {"columns": sorted(names), "drugs": drug_count}
        except (OSError, ValueError, KeyError) as error:
            errors.append(f"conditions.parquet: cannot inspect ({error})")
    model_info: dict[str, Any] = {}
    if model == "map":
        static_path = paths.prepared / "knowledge_tokens.pt"
        _required_file(static_path, files, errors, "map/static_token_cache")
        if inspect and static_path.is_file():
            try:
                import torch

                cache = torch.load(static_path, map_location="cpu", weights_only=False)
                if not isinstance(cache, Mapping):
                    raise ValueError("cache must contain a mapping")
                if cache.get("format") != "map_static_token_cache_v1":
                    errors.append("map/static_token_cache: unsupported format")
                if int(cache.get("embedding_dim", -1)) != 1024:
                    errors.append("map/static_token_cache: embedding_dim must be 1024")
                for key in ("gene_tokens", "drug_tokens"):
                    tensor = cache.get(key)
                    if not hasattr(tensor, "ndim") or tensor.ndim != 2 or int(tensor.shape[1]) != 1024:
                        errors.append(f"map/static_token_cache: {key} must have shape [N, 1024]")
                    if key == "drug_tokens" and drug_count is not None and hasattr(tensor, "shape") and int(tensor.shape[0]) != drug_count:
                        errors.append("map/static_token_cache: drug token count does not match conditions")
            except (OSError, RuntimeError, ValueError, KeyError, TypeError) as error:
                errors.append(f"map/static_token_cache: cannot load ({error})")
        assets = frozen_assets
        if assets is None:
            assets = (
                paths.frozen_models / "state" / "se600m.safetensors",
                paths.frozen_models / "state" / "Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt",
                paths.frozen_models / "mapkg" / "mapkg_encoder_v3.pt",
                paths.frozen_models / "mapkg" / "bart_vocab.txt",
            )
        elif isinstance(assets, (str, Path)):
            root = Path(assets)
            assets = (
                root / "state" / "se600m.safetensors",
                root / "state" / "Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt",
                root / "mapkg" / "mapkg_encoder_v3.pt",
                root / "mapkg" / "bart_vocab.txt",
            ) if root.is_dir() else (root,)
        else:
            # Materialize one-shot iterables once so validation and the report
            # see the same asset list.
            assets = tuple(assets)
        for index, asset_path in enumerate(assets):
            _required_file(Path(asset_path), files, errors, f"map/frozen_asset_{index}")
        model_info["frozen_assets"] = [str(Path(value)) for value in assets]
    else:
        representation = str(options.get("drug_representation", "rdkit")).casefold()
        if model == "cmonge" and regime == "unseen_combination" and "drug_representation" not in options:
            representation = "moa"
        material_root = (
            paths.split_material_dir(split_id, "drug_moa")
            if model == "cmonge" and representation == "moa"
            else paths.prepared
        )
        model_info = _validate_method_artifacts(
            paths, model, regime, options, shapes, drug_count, errors, files, inspect,
            material_root,
        )
    payload = {
        "format": "map_training_material_validation_v1",
        "method": model,
        "model": model,
        "regime": regime,
        "strict": bool(strict),
        # ``complete`` is the machine-facing answer to the method/split
        # material question.  ``status`` remains a human-readable alias used
        # by existing stage reports.
        "complete": not errors,
        "status": "pass" if not errors else "fail",
        "errors": list(dict.fromkeys(errors)),
        "shared": {"files": files, "populations": populations, "conditions": conditions_info},
        "split": split,
        "model_material": model_info,
    }
    output = paths.validation_path(split_id, model)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    result = Feedback(output.parent, f"validate_{model}").finish(
        payload, [output], status="ok" if not errors else "failed"
    )
    if errors and strict:
        raise ValueError(
            f"Training material validation failed for {model}; see {output}: "
            + "; ".join(payload["errors"][:4])
        )
    return result


def validate_preparation(
    paths: DatasetPaths,
    require_embeddings: bool = False,
    sample_rows: int = 256,
    split_files: tuple[str | Path, ...] | list[str | Path] | None = None,
) -> StageResult:
    report = Feedback(paths.prepared, "validate_preparation")
    manifest_path = paths.prepared / "materialization_manifest.json"
    shapes_path = paths.prepared / "materialized_shapes.json"
    preparation_config_path = paths.prepared / "preparation_config.json"
    required = [
        manifest_path,
        shapes_path,
        preparation_config_path,
        paths.prepared / "condition_filter.json",
        paths.prepared / "conditions.parquet",
    ]
    if require_embeddings:
        # Explicitly requested knowledge/state caches are checked below; the
        # default project validation remains method-neutral.
        required.append(paths.prepared / "knowledge_tokens.pt")
    manifest = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if split_files is None:
        split_files = sorted(paths.splits.glob("*/split.json"))
    resolved_splits = []
    for split_file in split_files:
        split_path = Path(split_file)
        if not split_path.is_absolute():
            # Split files are immutable objects at splits/<split_id>/split.json.
            candidates = (paths.workspace / split_path, paths.splits / split_path)
            split_path = next(
                (candidate for candidate in candidates if candidate.is_file()),
                candidates[0],
            )
        resolved_splits.append(split_path)
    required.extend(resolved_splits)
    missing = [str(path) for path in required if not path.is_file()]
    checks = []
    core_ready = manifest_path.is_file() and shapes_path.is_file()
    if core_ready:
        shapes = json.loads(shapes_path.read_text(encoding="utf-8"))
        first_shape = next(iter(shapes.values())) if shapes else {}
        token_length = int(first_shape.get("token_length", 2048))
        hvg_dim = int(first_shape.get("hvg_dim", 2000))
        for population in shapes:
            n_cells = int(shapes[population]["n_cells"])
            root = paths.prepared / population
            files = {
                "hvg": (root / "hvg.float16.dat", np.float16, (n_cells, hvg_dim)),
                "condition": (root / "row_condition.int32.dat", np.int32, (n_cells,)),
                "group": (root / "row_group.uint16.dat", np.uint16, (n_cells,)),
            }
            if require_embeddings:
                files["genes"] = (root / "se_gene_ids.uint16.dat", np.uint16, (n_cells, token_length))
                files["expression"] = (root / "se_expr.float16.dat", np.float16, (n_cells, token_length))
                files["embedding"] = (root / "state_embeddings.float16.dat", np.float16, (n_cells, 2048))
            status = {"population": population, "cells": n_cells, "files": {}}
            for name, (path, dtype, shape) in files.items():
                expected = int(np.prod(shape)) * np.dtype(dtype).itemsize
                exists = path.is_file()
                size_ok = exists and path.stat().st_size == expected
                finite = None
                if size_ok and name in {"expression", "hvg", "embedding"}:
                    array = np.memmap(path, dtype=dtype, mode="r", shape=shape)
                    finite = bool(np.isfinite(np.asarray(array[:min(sample_rows, n_cells)])).all())
                status["files"][name] = {"exists": exists, "size_ok": size_ok, "sample_finite": finite}
                if not exists or not size_ok or finite is False:
                    missing.append(str(path))
            checks.append(status)
            report.emit("population checked", population=population, cells=n_cells, status="pass" if not any(not x["exists"] or not x["size_ok"] or x["sample_finite"] is False for x in status["files"].values()) else "fail")
    condition_filter_check = None
    filter_path = paths.prepared / "condition_filter.json"
    if filter_path.is_file() and (paths.prepared / "conditions.parquet").is_file():
        import pyarrow.parquet as pq

        condition_filter = json.loads(filter_path.read_text(encoding="utf-8"))
        condition_cells = np.asarray(
            pq.read_table(
                paths.prepared / "conditions.parquet", columns=["n_cells"]
            ).column(0)
        )
        minimum = int(condition_filter["min_cells"])
        maximum = int(condition_filter["max_cells"])
        within_bounds = bool(
            condition_cells.size
            and np.all(condition_cells >= minimum)
            and np.all(condition_cells <= maximum)
        )
        retained_matches = (
            int(condition_cells.sum())
            == int(condition_filter["retained_condition_cells"])
        )
        control_counts = {}
        if shapes_path.is_file():
            for population, shape in json.loads(
                shapes_path.read_text(encoding="utf-8")
            ).items():
                n_cells = int(shape["n_cells"])
                condition_path = paths.prepared / population / "row_condition.int32.dat"
                if not condition_path.is_file():
                    control_counts[population] = -1
                    continue
                condition_array = np.memmap(
                    condition_path,
                    dtype=np.int32,
                    mode="r",
                    shape=(n_cells,),
                )
                control_counts[population] = int(np.count_nonzero(condition_array < 0))
        controls_within_cap = all(
            0 <= value <= maximum for value in control_counts.values()
        )
        retained_controls_match = (
            not control_counts
            or int(sum(control_counts.values()))
            == int(condition_filter.get("retained_control_cells", sum(control_counts.values())))
        )
        preparation = (
            json.loads(preparation_config_path.read_text(encoding="utf-8"))
            if preparation_config_path.is_file()
            else {}
        )
        id_matches = (
            preparation.get("condition_filter", {}).get("filter_id")
            == condition_filter.get("filter_id")
        )
        condition_filter_check = {
            "filter_id": condition_filter.get("filter_id"),
            "min_cells": minimum,
            "max_cells": maximum,
            "conditions_within_bounds": within_bounds,
            "retained_cells_match": retained_matches,
            "control_cells_by_population": control_counts,
            "controls_within_cap": controls_within_cap,
            "retained_controls_match": retained_controls_match,
            "preparation_id_matches": id_matches,
        }
        if (
            not within_bounds
            or not retained_matches
            or not controls_within_cap
            or not retained_controls_match
            or not id_matches
        ):
            missing.append(str(filter_path))
    split_checks = []
    for split_path in resolved_splits:
        if not split_path.is_file():
            continue
        split_payload = json.loads(split_path.read_text(encoding="utf-8"))
        sets = {
            name: {int(value) for value in split_payload.get(name, [])}
            for name in ("train", "internal_test", "external_test")
        }
        external_disjoint = not (
            sets["external_test"] & (sets["train"] | sets["internal_test"])
        )
        row_membership = {}
        for name in ("train", "internal_test", "external_test"):
            membership = {
                str(population): np.asarray(rows, dtype=np.int64)
                for population, rows in split_payload.get(f"{name}_rows", {}).items()
            }
            for population, filename in split_payload.get(
                f"{name}_rows_files", {}
            ).items():
                row_path = Path(filename)
                if not row_path.is_absolute():
                    row_path = split_path.parent / row_path
                if not row_path.is_file():
                    missing.append(str(row_path))
                    continue
                population = str(population)
                if population in membership:
                    missing.append(
                        f"{split_path} defines duplicate row sources for {name}/{population}"
                    )
                    continue
                membership[population] = np.load(
                    row_path, mmap_mode="r"
                )
            row_membership[name] = membership
        has_row_protocol = any(row_membership[name] for name in row_membership)
        row_disjoint = True
        if has_row_protocol:
            populations = set().union(
                *(set(values) for values in row_membership.values())
            )
            for population in populations:
                arrays = [
                    np.asarray(row_membership[name].get(population, ()), dtype=np.int64)
                    for name in ("train", "internal_test", "external_test")
                ]
                if any(len(np.unique(values)) != len(values) for values in arrays):
                    row_disjoint = False
                    break
                if any(
                    np.intersect1d(arrays[left], arrays[right], assume_unique=False).size
                    for left, right in ((0, 1), (0, 2), (1, 2))
                ):
                    row_disjoint = False
                    break
        valid_sets = (
            all(
                sum(len(rows) for rows in row_membership[name].values()) > 0
                for name in row_membership
            )
            if has_row_protocol
            else all(sets[name] for name in sets)
        )
        if (not row_disjoint if has_row_protocol else not external_disjoint) or not valid_sets:
            missing.append(str(split_path))
        split_checks.append({
            "split_file": str(split_path),
            "split_id": split_payload.get("split_id", split_path.stem),
            "rule": split_payload.get("rule"),
            "seed": split_payload.get("seed"),
            "counts": {name: len(values) for name, values in sets.items()},
            "external_condition_set_disjoint": external_disjoint,
            "row_sets_disjoint": row_disjoint if has_row_protocol else None,
            "row_protocol": has_row_protocol,
            "row_counts": {
                name: sum(
                    len(rows) for rows in row_membership[name].values()
                )
                for name in row_membership
            } if has_row_protocol else None,
        })
    missing = list(dict.fromkeys(missing))
    payload = {
        "status": "pass" if not missing else "fail",
        "missing_or_invalid": missing,
        "populations": checks,
        "splits": split_checks,
        "condition_filter": condition_filter_check,
        "preparation": (
            json.loads(preparation_config_path.read_text(encoding="utf-8"))
            if preparation_config_path.is_file()
            else None
        ),
    }
    output = paths.prepared / "preparation_validation.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    result = report.finish(payload, [output], status="ok" if not missing else "failed")
    if missing:
        raise ValueError(f"Preparation validation failed; see {output}")
    (paths.prepared / "_SUCCESS").write_text(
        json.dumps(
            {
                "status": "ok",
                "validated_split_ids": [item["split_id"] for item in split_checks],
                "preparation_id": payload.get("preparation", {}).get("preparation_id")
                if payload.get("preparation")
                else None,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    return result


__all__ = ["validate_preparation", "validate_method"]
