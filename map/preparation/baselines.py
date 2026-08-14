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
    directory: Path,
    smiles: list[str],
    descriptor_data: tuple[np.ndarray, list[str]],
) -> tuple[dict, list[Path]]:
    matrix, descriptors = descriptor_data
    target = directory / "rdkit2d.float32.npy"
    np.save(target, matrix)
    payload = {
        "model": "crisp",
        "directory": "crisp",
        "representation": "CRISP RDKit2D standardized descriptors",
        "matrix": target.name,
        "smiles": smiles,
        "shape": list(matrix.shape),
        "descriptors": descriptors,
        "source": "ml4bio/CRISP data preprocessing",
    }
    manifest = directory / "manifest.json"
    manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload, [manifest, target]


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
    xpert_hidden_size: int = 128,
    xpert_layers: int = 3,
    xpert_epochs: int = 100,
    xpert_seed: int = 42,
    xpert_device: str | None = None,
):
    selected = tuple(dict.fromkeys(str(value).casefold() for value in models))
    unknown = sorted(set(selected) - set(BASELINE_MODELS))
    if unknown:
        raise ValueError(f"Unavailable baselines: {', '.join(unknown)}")
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
            payload, model_outputs = _crisp_inputs(directory, smiles, rdkit2d)
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
        "format_version": 4,
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
