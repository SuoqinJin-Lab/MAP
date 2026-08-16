from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow.parquet as pq
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from .._common.feedback import Feedback
from .._common.paths import DatasetPaths
from .xpert import prepare_xpert_inputs


BASELINE_MODELS = ("prnet", "chemcpa", "trainmean", "crisp", "xpert", "cmonge")


def _smiles(paths: DatasetPaths) -> list[str]:
    table = pq.read_table(
        paths.prepared / "conditions.parquet", columns=["canonical_smiles"]
    )
    return sorted({str(value) for value in table.column(0).to_pylist()})


def _fingerprints(smiles: list[str], *, features: bool) -> np.ndarray:
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2,
        fpSize=1024,
        atomInvariantsGenerator=(
            rdFingerprintGenerator.GetMorganFeatureAtomInvGen() if features else None
        ),
    )
    output = np.zeros((len(smiles), 1024), dtype=np.float32)
    for index, value in enumerate(smiles):
        molecule = Chem.MolFromSmiles(value)
        if molecule is None:
            raise ValueError(f"Invalid canonical SMILES: {value}")
        DataStructs.ConvertToNumpyArray(generator.GetFingerprint(molecule), output[index])
    return output


def _rdkit2d(smiles: list[str]) -> tuple[np.ndarray, list[str]]:
    try:
        from descriptastorus.descriptors.DescriptorGenerator import MakeGenerator
    except ImportError as error:
        raise ImportError(
            "CRISP and CMonge preparation require descriptastorus"
        ) from error
    generator = MakeGenerator(("RDKit2D",))
    values = np.asarray([generator.process(value) for value in smiles], dtype=np.float64)
    values = values[:, 1:]
    values[~np.isfinite(values)] = 0.0
    standard_deviation = values.std(axis=0, ddof=1)
    keep = np.isfinite(standard_deviation) & (standard_deviation > 0.01)
    if not keep.any():
        raise RuntimeError("RDKit2D preprocessing removed every descriptor")
    values = values[:, keep]
    mean = values.mean(axis=0)
    standard_deviation = values.std(axis=0, ddof=1)
    values = (values - mean) / standard_deviation
    columns = list(generator.GetColumns())[1:]
    names = [str(value) for value, selected in zip(columns, keep) if selected]
    return values.astype(np.float32), names


def _crisp_inputs(
    paths: DatasetPaths,
    directory: Path,
    smiles: list[str],
    descriptor_data: tuple[np.ndarray, list[str]],
    *,
    overwrite: bool,
) -> tuple[dict, list[Path]]:
    matrix, descriptors = descriptor_data
    target = directory / "rdkit2d.float32.npy"
    np.save(target, matrix)
    control_group_means, mean_outputs = _crisp_control_group_means(
        paths, directory, overwrite=overwrite
    )
    payload = {
        "model": "crisp",
        "directory": "crisp",
        "representation": "CRISP RDKit2D standardized descriptors",
        "matrix": target.name,
        "smiles": smiles,
        "shape": list(matrix.shape),
        "descriptors": descriptors,
        "source": "ml4bio/CRISP data preprocessing",
        "control_group_means": control_group_means,
    }
    manifest = directory / "manifest.json"
    manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload, [manifest, target, *mean_outputs]


def _crisp_control_group_means(
    paths: DatasetPaths,
    directory: Path,
    *,
    overwrite: bool,
) -> tuple[dict[str, dict], list[Path]]:
    shapes_file = paths.prepared / "materialized_shapes.json"
    if not shapes_file.is_file():
        raise FileNotFoundError(
            "CRISP control means require materialized_shapes.json"
        )
    shapes = json.loads(shapes_file.read_text(encoding="utf-8"))
    payload: dict[str, dict] = {}
    outputs: list[Path] = []
    for population, shape in shapes.items():
        n_cells = int(shape["n_cells"])
        hvg_dim = int(shape["hvg_dim"])
        source = paths.prepared / str(population)
        required = {
            "group_ids": source / "control_group_ids.npy",
            "offsets": source / "control_group_offsets.npy",
            "rows": source / "control_group_rows.npy",
            "embeddings": source / "state_embeddings.float16.dat",
            "hvg": source / "hvg.float16.dat",
        }
        missing = [str(path) for path in required.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "CRISP control means require completed materialization and STATE "
                "embeddings; missing: " + ", ".join(missing)
            )
        output = directory / "control_group_means" / str(population)
        output.mkdir(parents=True, exist_ok=True)
        targets = {
            "group_ids": output / "group_ids.int64.npy",
            "embeddings": output / "embeddings.float16.npy",
            "hvg": output / "hvg.float16.npy",
        }
        group_ids = np.asarray(np.load(required["group_ids"]), dtype=np.int64)
        reusable = not overwrite and all(path.is_file() for path in targets.values())
        if reusable:
            saved_ids = np.load(targets["group_ids"], mmap_mode="r")
            saved_embeddings = np.load(targets["embeddings"], mmap_mode="r")
            saved_hvg = np.load(targets["hvg"], mmap_mode="r")
            reusable = (
                np.array_equal(saved_ids, group_ids)
                and saved_embeddings.shape == (len(group_ids), 2048)
                and saved_hvg.shape == (len(group_ids), hvg_dim)
            )
        if not reusable:
            offsets = np.asarray(np.load(required["offsets"]), dtype=np.int64)
            rows = np.load(required["rows"], mmap_mode="r")
            if len(offsets) != len(group_ids) + 1:
                raise ValueError(
                    f"Invalid CRISP control-group offsets for {population}"
                )
            embeddings = np.memmap(
                required["embeddings"],
                dtype=np.float16,
                mode="r",
                shape=(n_cells, 2048),
            )
            hvg = np.memmap(
                required["hvg"],
                dtype=np.float16,
                mode="r",
                shape=(n_cells, hvg_dim),
            )
            embedding_means = np.empty((len(group_ids), 2048), dtype=np.float16)
            hvg_means = np.empty((len(group_ids), hvg_dim), dtype=np.float16)
            for index in range(len(group_ids)):
                selected_rows = np.asarray(
                    rows[int(offsets[index]): int(offsets[index + 1])],
                    dtype=np.int64,
                )
                if not len(selected_rows):
                    raise ValueError(
                        f"CRISP control group {group_ids[index]} in {population} is empty"
                    )
                embedding_means[index] = np.asarray(
                    embeddings[selected_rows], dtype=np.float32
                ).mean(axis=0)
                hvg_means[index] = np.asarray(
                    hvg[selected_rows], dtype=np.float32
                ).mean(axis=0)
            np.save(targets["group_ids"], group_ids)
            np.save(targets["embeddings"], embedding_means)
            np.save(targets["hvg"], hvg_means)
        relative = output.relative_to(directory)
        payload[str(population)] = {
            "directory": str(relative),
            "groups": len(group_ids),
            "embedding_dim": 2048,
            "hvg_dim": hvg_dim,
            "dtype": "float16",
        }
        outputs.extend(targets.values())
    return payload, outputs


def _cmonge_inputs(
    directory: Path,
    smiles: list[str],
    descriptor_data: tuple[np.ndarray, list[str]],
) -> tuple[dict, list[Path]]:
    matrix, descriptors = descriptor_data
    target = directory / "rdkit2d.float32.npy"
    np.save(target, matrix)
    payload = {
        "model": "cmonge",
        "directory": "cmonge",
        "hvg_dim": 2000,
        "representation": "standardized RDKit molecular descriptors",
        "matrix": target.name,
        "smiles": smiles,
        "shape": list(matrix.shape),
        "descriptors": descriptors,
        "source": "Conditional Monge Gap RDKit drug context",
    }
    manifest = directory / "manifest.json"
    manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload, [manifest, target]


def prepare_baseline_inputs(
    paths: DatasetPaths,
    *,
    models: Iterable[str] = BASELINE_MODELS,
    overwrite: bool = False,
    xpert_aliases_file=None,
    xpert_hidden_size: int = 256,
    xpert_layers: int = 3,
    xpert_epochs: int = 300,
    xpert_seed: int = 4242,
    xpert_device: str | None = None,
):
    selected = tuple(dict.fromkeys(str(value).casefold() for value in models))
    unknown = sorted(set(selected) - set(BASELINE_MODELS))
    if unknown:
        raise ValueError(f"Unavailable baselines: {', '.join(unknown)}")
    if "cmonge" in selected:
        shapes_file = paths.prepared / "materialized_shapes.json"
        if not shapes_file.is_file():
            raise FileNotFoundError(
                "CMonge preparation requires materialized_shapes.json"
            )
        hvg_dimensions = {
            int(value["hvg_dim"])
            for value in json.loads(shapes_file.read_text(encoding="utf-8")).values()
        }
        if hvg_dimensions != {2000}:
            raise ValueError(
                "CMonge requires exactly 2000 HVGs; "
                f"materialized dimensions={sorted(hvg_dimensions)}"
            )
    root = paths.prepared / "baselines"
    manifest_file = root / "manifest.json"
    existing = (
        json.loads(manifest_file.read_text(encoding="utf-8"))
        if manifest_file.is_file()
        else {}
    )
    smiles = _smiles(paths)
    if existing and existing.get("drug_vocabulary") != smiles and not overwrite:
        raise RuntimeError("Existing baseline inputs use a different drug vocabulary")
    existing_models = {
        name: payload
        for name, payload in existing.get("models", {}).items()
        if name in BASELINE_MODELS
    }
    if (
        not overwrite
        and existing.get("format_version") == 6
        and set(selected).issubset(existing_models)
        and set(existing.get("models", {})).issubset(BASELINE_MODELS)
    ):
        return Feedback(root, "prepare_baselines").finish(
            {**existing, "reused": True}, [manifest_file]
        )

    root.mkdir(parents=True, exist_ok=True)
    ecfp4 = _fingerprints(smiles, features=False) if "chemcpa" in selected else None
    fcfp4 = _fingerprints(smiles, features=True) if "prnet" in selected else None
    rdkit2d = (
        _rdkit2d(smiles)
        if {"crisp", "cmonge"}.intersection(selected)
        else None
    )
    models_payload = existing_models
    outputs: list[Path] = []
    for model in selected:
        directory = root / model
        directory.mkdir(parents=True, exist_ok=True)
        if model == "crisp":
            payload, model_outputs = _crisp_inputs(
                paths, directory, smiles, rdkit2d, overwrite=overwrite
            )
            models_payload[model] = payload
            outputs.extend(model_outputs)
            continue
        if model == "cmonge":
            payload, model_outputs = _cmonge_inputs(directory, smiles, rdkit2d)
            models_payload[model] = payload
            outputs.extend(model_outputs)
            continue
        if model == "xpert":
            payload, model_outputs = prepare_xpert_inputs(
                paths,
                directory,
                aliases_file=xpert_aliases_file,
                hidden_size=xpert_hidden_size,
                layers=xpert_layers,
                epochs=xpert_epochs,
                seed=xpert_seed,
                device=xpert_device,
                overwrite=overwrite,
            )
            models_payload[model] = payload
            outputs.extend(model_outputs)
            continue
        if model == "trainmean":
            payload = {
                "model": model,
                "directory": model,
                "representation": "training condition pseudobulk mean",
                "smiles": smiles,
            }
            (directory / "manifest.json").write_text(
                json.dumps(payload, indent=2), encoding="utf-8"
            )
            models_payload[model] = payload
            outputs.append(directory / "manifest.json")
            continue
        matrix = fcfp4 if model == "prnet" else ecfp4
        filename = (
            "fcfp4_1024.float32.npy"
            if model == "prnet"
            else "ecfp4_1024.float32.npy"
        )
        target = directory / filename
        np.save(target, matrix)
        payload = {
            "model": model,
            "directory": model,
            "representation": "FCFP4-1024" if model == "prnet" else "ECFP4-1024",
            "matrix": filename,
            "smiles": smiles,
            "shape": list(matrix.shape),
        }
        (directory / "manifest.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        models_payload[model] = payload
        outputs.extend((directory / "manifest.json", target))

    manifest = {
        "format_version": 6,
        "drug_vocabulary": smiles,
        "models": models_payload,
    }
    manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    outputs.insert(0, manifest_file)
    return Feedback(root, "prepare_baselines").finish(
        {
            "models": list(selected),
            "drugs": len(smiles),
            "output": str(root),
            "reused": False,
        },
        outputs,
    )


__all__ = ["BASELINE_MODELS", "prepare_baseline_inputs"]
