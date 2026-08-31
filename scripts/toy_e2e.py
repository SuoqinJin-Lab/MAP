#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
import traceback
import zipfile
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from map import eval, preparation, preprocess, train


MODELS = ("map", "prnet", "chemcpa", "trainmean", "crisp", "xpert", "cmonge")
DRUGS = (
    ("ToyDrug-A", "CC", 0.1),
    ("ToyDrug-B", "CCC", 0.3),
    ("ToyDrug-C", "CCO", 1.0),
    ("ToyDrug-D", "CCN", 3.0),
)
REAL_XPERT_DRUGS = (
    (
        "Budesonide",
        "CCCC1OC2CC3C4CCC5=CC(=O)C=CC5(C4C(CC3(C2(O1)C(=O)CO)C)O)C",
        0.5,
    ),
    (
        "Dexamethasone",
        "CC1CC2C3CCC4=CC(=O)C=CC4(C3(C(CC2(C1(C(=O)CO)O)C)O)F)C",
        1.5,
    ),
    (
        "Medroxyprogesterone acetate",
        "CC1CC2C(CCC3(C2CCC3(C(=O)C)OC(=O)C)C)C4(C1=CC(=O)CC4)C",
        2.5,
    ),
    (
        "Betamethasone dipropionate",
        "CCC(=O)OCC(=O)C1(C(CC2C1(CC(C3(C2CCC4=CC(=O)C=CC43C)F)O)C)C)OC(=O)CC",
        3.5,
    ),
)
REAL_XPERT_GENES = (
    "IGF2", "IFNL1", "TENM1", "IFNL2", "SERPINA1", "IFNL3", "NRG3", "IFIT2",
    "ZFP36", "APOB", "ALB", "LSAMP", "AFP", "NLGN1", "PLAT", "ANKRD1",
    "CEMIP", "TF", "PLCB1", "FMN1", "CXCL8", "C3", "RSAD2", "SUCNR1",
    "SERPINA3", "HPSE2", "FN1", "FRMD4A", "IGFBP1", "ADAMTS6", "APOD", "PEX5L",
    "ABI3BP", "PAPPA", "GPC3", "CCL2", "TP63", "AHSG", "CCL5", "LOX",
    "CXCL10", "DIRAS3", "SELE", "PODXL", "H3Y1", "CYP24A1", "NKD1", "CDH13",
    "CALN1", "COL8A1", "CXCL3", "PEG3", "SPARC", "COL1A2", "ZC3HAV1", "SAA1",
    "TSHZ2", "IFIT3", "PMEL", "G0S2", "GPC5", "XIRP1", "APOA2", "CACNA2D3",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run preparation, training, evaluation and analysis on real MAP components"
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--frozen-models", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        choices=("tahoe", "sciplex", "nips"),
        default="tahoe",
    )
    parser.add_argument(
        "--models", nargs="+", choices=MODELS, default=["map"],
        help="Methods to exercise; MAP is the lightweight default",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--hvg-dim",
        type=int,
        default=None,
        help="HVG count (defaults to 2,000 with CMonge, otherwise 32)",
    )
    parser.add_argument("--pad-length", type=int, default=33)
    parser.add_argument("--cells-per-condition", type=int, default=4)
    parser.add_argument("--control-cells", type=int, default=8)
    parser.add_argument(
        "--xpert-assets",
        choices=("synthetic", "real"),
        default="synthetic",
        help="Use generated XPert fixtures or the installed STRING/PrimeKG/UniMol assets",
    )
    args = parser.parse_args()
    if args.hvg_dim is None:
        args.hvg_dim = 2000 if "cmonge" in args.models else 32
    return args


def jsonable(value: Any) -> Any:
    if hasattr(value, "as_dict"):
        return value.as_dict()
    if dataclasses.is_dataclass(value):
        return jsonable(dataclasses.asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


class Recorder:
    def __init__(self, path: Path, configuration: dict[str, Any]) -> None:
        self.path = path
        self.payload = {
            "status": "running",
            "configuration": jsonable(configuration),
            "steps": [],
            "models": {},
        }
        self.write()

    def write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(self.payload, indent=2, sort_keys=True, allow_nan=True),
            encoding="utf-8",
        )
        os.replace(temporary, self.path)

    def run(self, name: str, function: Callable[[], Any]) -> Any:
        started = time.perf_counter()
        try:
            result = function()
        except Exception as error:
            self.payload["status"] = "failed"
            self.payload["failure"] = {
                "step": name,
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            }
            self.write()
            raise
        self.payload["steps"].append({
            "name": name,
            "elapsed_seconds": time.perf_counter() - started,
            "result": jsonable(result),
        })
        self.write()
        return result


def require_frozen_assets(root: Path) -> dict[str, str]:
    alternatives = {
        "se600m": ("state/se600m.safetensors", "se600m.safetensors"),
        "esm2": (
            "state/Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt",
            "Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt",
        ),
        "mapkg": ("mapkg/mapkg_encoder_v3.pt", "mapkg_encoder_v3.pt"),
        "vocab": ("mapkg/bart_vocab.txt", "bart_vocab.txt"),
    }
    resolved = {}
    for name, candidates in alternatives.items():
        path = next((root / value for value in candidates if (root / value).is_file()), None)
        if path is None:
            raise FileNotFoundError(
                f"Frozen asset {name!r} is absent from {root}: {', '.join(candidates)}"
            )
        resolved[name] = str(path.resolve())
    return resolved


def make_tahoe_source(
    source: Path,
    esm2_path: Path,
    *,
    seed: int,
    control_cells: int,
    cells_per_condition: int,
    gene_count: int,
    drugs=DRUGS,
    gene_symbols: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    if source.exists() and any(source.iterdir()):
        raise FileExistsError(f"Toy source directory is not empty: {source}")
    data = source / "data"
    metadata = source / "metadata"
    data.mkdir(parents=True)
    metadata.mkdir(parents=True)

    esm2 = torch.load(esm2_path, map_location="cpu", weights_only=False)
    if not isinstance(esm2, dict) or len(esm2) < gene_count:
        raise ValueError("The released ESM2 table does not contain enough gene symbols")
    if gene_symbols is None:
        symbols = list(esm2)[:gene_count]
    else:
        symbols = list(gene_symbols[:gene_count])
        missing = [value for value in symbols if value not in esm2]
        if len(symbols) != gene_count:
            raise ValueError("The real XPert toy gene vocabulary is too small")
        if missing:
            raise ValueError(
                f"The ESM2 table is missing {len(missing)} real XPert toy genes"
            )
    del esm2

    rng = np.random.default_rng(seed)
    gene_ids = list(range(gene_count))
    base = np.linspace(2.0, 18.0, gene_count)
    rows = []
    samples = [{
        "sample": "toy-control",
        "drug": "DMSO_TF",
        "drugname_drugconc": "[('DMSO_TF', 0.0, 'uM')]",
    }]

    def expression(effect: np.ndarray | None = None) -> list[int]:
        means = base if effect is None else base + effect
        return (rng.poisson(means) + 1).astype(np.int32).tolist()

    for _ in range(control_cells):
        rows.append({
            "genes": gene_ids,
            "expressions": expression(),
            "cell_line_id": "TOY_LINE",
            "sample": "toy-control",
            "drug": "DMSO_TF",
            "canonical_smiles": "",
            "plate": "toy-plate-1",
        })
    for drug_index, (drug, smiles, dose) in enumerate(drugs):
        sample = f"toy-condition-{drug_index}"
        samples.append({
            "sample": sample,
            "drug": drug,
            "drugname_drugconc": repr([(drug, dose, "uM")]),
        })
        effect = np.zeros(gene_count, dtype=np.float64)
        start = drug_index * gene_count // len(drugs)
        end = (drug_index + 1) * gene_count // len(drugs)
        effect[start:end] = 3.0 + drug_index
        for _ in range(cells_per_condition):
            rows.append({
                "genes": gene_ids,
                "expressions": expression(effect),
                "cell_line_id": "TOY_LINE",
                "sample": sample,
                "drug": drug,
                "canonical_smiles": smiles,
                "plate": "toy-plate-1",
            })

    pq.write_table(
        pa.Table.from_pylist(rows), data / "train-00000-of-00001.parquet",
        compression="zstd",
    )
    pq.write_table(
        pa.Table.from_pylist([
            {"token_id": index, "gene_symbol": symbol}
            for index, symbol in enumerate(symbols)
        ]),
        metadata / "gene_metadata.parquet",
        compression="zstd",
    )
    pq.write_table(
        pa.Table.from_pylist(samples),
        metadata / "sample_metadata.parquet",
        compression="zstd",
    )
    return {
        "source": str(source),
        "cells": len(rows),
        "control_cells": control_cells,
        "condition_cells": cells_per_condition * len(drugs),
        "conditions": len(drugs),
        "genes": gene_count,
    }


def make_native_atlas_source(
    source: Path,
    esm2_path: Path,
    *,
    dataset: str,
    seed: int,
    control_cells: int,
    cells_per_condition: int,
    gene_count: int,
) -> dict[str, Any]:
    if source.exists() and any(source.iterdir()):
        raise FileExistsError(f"Toy source directory is not empty: {source}")
    source.mkdir(parents=True)
    esm2 = torch.load(esm2_path, map_location="cpu", weights_only=False)
    if not isinstance(esm2, dict) or len(esm2) < gene_count:
        raise ValueError("The released ESM2 table does not contain enough gene symbols")
    symbols = list(esm2)[:gene_count]
    del esm2
    populations = (
        ("TOY_A549", "TOY_K562")
        if dataset == "sciplex"
        else ("TOY_CD4_T", "TOY_B")
    )
    rng = np.random.default_rng(seed)
    base = np.linspace(2.0, 18.0, gene_count)
    matrix = []
    metadata = []

    def expression(effect: np.ndarray | None = None) -> np.ndarray:
        means = base if effect is None else base + effect
        return (rng.poisson(means) + 1).astype(np.float32)

    for population_index, population in enumerate(populations):
        for control_index in range(control_cells):
            metadata.append({
                "cell_id": f"{population}-control-{control_index}",
                "population": population,
                "drug": "DMSO",
                "smiles": "",
                "dose": 0.0,
                "plate": f"plate-{population_index}",
            })
            matrix.append(expression())
        for drug_index, (drug, smiles, dose) in enumerate(DRUGS):
            effect = np.zeros(gene_count, dtype=np.float64)
            start = drug_index * gene_count // len(DRUGS)
            end = (drug_index + 1) * gene_count // len(DRUGS)
            effect[start:end] = 3.0 + drug_index + population_index
            for cell_index in range(cells_per_condition):
                metadata.append({
                    "cell_id": f"{population}-{drug_index}-{cell_index}",
                    "population": population,
                    "drug": drug,
                    "smiles": smiles,
                    "dose": dose if dataset == "sciplex" else 1.0,
                    "plate": f"plate-{population_index}",
                })
                matrix.append(expression(effect))
    matrix = np.asarray(matrix, dtype=np.float32)

    if dataset == "sciplex":
        from scipy import sparse
        from scipy.io import mmwrite

        mmwrite(source / "matrix.mtx", sparse.csr_matrix(matrix.T))
        pd_rows = [{
            "cell_id": row["cell_id"],
            "cell_line": row["population"],
            "product_name": row["drug"],
            "canonical_smiles": row["smiles"],
            "dose_val": row["dose"],
            "plate": row["plate"],
        } for row in metadata]
        import pandas as pd

        pd.DataFrame(pd_rows).to_csv(
            source / "cell_metadata.tsv", sep="\t", index=False
        )
        pd.DataFrame({
            "gene_symbol": symbols,
            "feature_id": [f"gene-{index}" for index in range(gene_count)],
        }).to_csv(source / "gene_metadata.tsv", sep="\t", index=False)
    else:
        import pandas as pd

        obs = pd.DataFrame([{
            "obs_id": row["cell_id"],
            "cell_type": row["population"],
            "sm_name": row["drug"],
            "SMILES": row["smiles"],
            "dose_uM": row["dose"],
            "plate_name": row["plate"],
        } for row in metadata])
        metadata_csv = source / "adata_obs_meta.csv"
        obs.to_csv(metadata_csv, index=False)
        with zipfile.ZipFile(
            source / "adata_obs_meta.csv.zip", "w", zipfile.ZIP_DEFLATED
        ) as archive:
            archive.write(metadata_csv, metadata_csv.name)
        metadata_csv.unlink()

        expression_rows = []
        for row_index, obs_id in enumerate(obs["obs_id"]):
            counts = matrix[row_index].astype(np.int64)
            total = max(int(counts.sum()), 1)
            for gene_index in np.flatnonzero(counts > 0):
                count = int(counts[gene_index])
                expression_rows.append({
                    "obs_id": str(obs_id),
                    "gene": str(symbols[gene_index]),
                    "count": count,
                    "normalized_count": float(np.log1p(count * 1_000_000 / total)),
                })
        expression_rows.sort(key=lambda row: (row["obs_id"], row["gene"]))
        pq.write_table(
            pa.Table.from_pylist(expression_rows),
            source / "adata_train.parquet",
            compression="zstd",
        )
        with zipfile.ZipFile(
            source / "adata_train.parquet.zip", "w", zipfile.ZIP_DEFLATED
        ) as archive:
            archive.write(source / "adata_train.parquet", "adata_train.parquet")
        (source / "adata_train.parquet").unlink()
        excluded_csv = source / "adata_excluded_ids.csv"
        pd.DataFrame([
            {"obs_id": str(obs.iloc[0]["obs_id"]), "gene": str(symbols[-1])}
        ]).to_csv(excluded_csv, index=False)
        with zipfile.ZipFile(
            source / "adata_excluded_ids.csv.zip", "w", zipfile.ZIP_DEFLATED
        ) as archive:
            archive.write(excluded_csv, excluded_csv.name)
        excluded_csv.unlink()
    return {
        "source": str(source),
        "dataset": dataset,
        "populations": list(populations),
        "cells": int(len(matrix)),
        "control_cells": int(control_cells * len(populations)),
        "condition_cells": int(cells_per_condition * len(DRUGS) * len(populations)),
        "conditions": int(len(DRUGS) * len(populations)),
        "genes": gene_count,
    }


def require_files(paths: list[Path]) -> list[str]:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Expected toy artifacts are missing: " + ", ".join(missing))
    return [str(path) for path in paths]


def make_toy_xpert_inputs(workflow, *, seed: int) -> dict[str, Any]:
    """Create tiny model-input assets so toy runs never need external downloads."""
    root = workflow.prepared
    conditions = pq.read_table(
        workflow.prepared / "conditions.parquet", columns=["canonical_smiles"]
    ).column(0).to_pylist()
    smiles = sorted({str(value) for value in conditions})
    shapes = json.loads(
        (workflow.prepared / "materialized_shapes.json").read_text(encoding="utf-8")
    )
    hvg_dim = int(next(iter(shapes.values()))["hvg_dim"])
    all_gene_symbols = json.loads(
        (workflow.prepared / "state_gene_symbols.json").read_text(encoding="utf-8")
    )
    hvg = json.loads(
        (workflow.prepared / "hvg.json").read_text(encoding="utf-8")
    )
    hvg_indices = np.asarray(hvg["state_ids"], dtype=np.int32)
    full_gene_count = len(all_gene_symbols)
    gene_symbols = all_gene_symbols[:full_gene_count]
    if (
        len(hvg_indices) != hvg_dim
        or len(np.unique(hvg_indices)) != hvg_dim
        or np.any(hvg_indices < 0)
        or np.any(hvg_indices >= full_gene_count)
    ):
        raise ValueError("Toy XPert HVG-to-full-gene mapping is invalid")
    graph_directory = root / "graph_assets"
    unimol_directory = root / "drug_unimol"
    graph_directory.mkdir(parents=True, exist_ok=True)
    unimol_directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    hidden = min(16, max(4, full_gene_count))
    full_gene_vectors = rng.normal(size=(full_gene_count, hidden)).astype(np.float32)
    np.save(
        graph_directory / "ppi_gene_vectors_full.float32.npy",
        full_gene_vectors,
    )
    np.save(
        graph_directory / "ppi_gene_vectors.float32.npy",
        full_gene_vectors[hvg_indices],
    )
    np.save(graph_directory / "hvg_indices.int32.npy", hvg_indices)
    np.save(graph_directory / "drug_hg_embeddings.float32.npy", rng.normal(size=(len(smiles), hidden)).astype(np.float32))
    unimol = np.zeros((len(smiles), 8, 514), dtype=np.float16)
    unimol[:, :4, 0] = 1
    unimol[:, :4, 1] = 20
    unimol[:, :4, 2:] = rng.normal(size=(len(smiles), 4, 512)).astype(np.float16)
    np.save(unimol_directory / "drug_unimol.float16.npy", unimol)
    preparation.prepare_expression_bins(
        workflow,
        input_formats=("official", "validation"),
        expression_bins=128,
        expression_sample_cells=4096,
        overwrite=True,
    )
    hvg_symbols = [gene_symbols[int(index)] for index in hvg_indices]
    expression_manifest = root / "expression_bins" / "manifest.json"
    if not expression_manifest.is_file():
        raise FileNotFoundError(
            "Toy XPert expression-bin manifest was not created"
        )
    expression_payload = json.loads(expression_manifest.read_text(encoding="utf-8"))
    expression_contract = expression_payload.get("expression")
    if not isinstance(expression_contract, dict):
        raise ValueError("Toy XPert expression-bin manifest is invalid")
    payload = {
        "artifact": "graph_assets",
        "format": "map_xpert_assets_v4",
        "input_modes": ["official", "validation"],
        "input_formats": ["per_cell_hvg", "pseudobulk_hvg"],
        "gene_symbols": hvg_symbols,
        "full_gene_symbols": gene_symbols,
        "gene_count": hvg_dim,
        "full_gene_count": full_gene_count,
        "hvg_indices": hvg_indices.tolist(),
        "model_gene_space": "prepared_hvg",
        "smiles": smiles,
        "hidden_size": hidden,
        "max_atoms": 8,
        "ppi_gene_vector_file": "ppi_gene_vectors.float32.npy",
        "ppi_gene_vector_full_file": "ppi_gene_vectors_full.float32.npy",
        "hvg_indices_file": "hvg_indices.int32.npy",
        "drug_hg_embedding_file": "drug_hg_embeddings.float32.npy",
        "drug_unimol_file": "drug_unimol.float16.npy",
        "expression": expression_contract,
        "expression_bins": expression_contract["expression_bins"],
        "expression_min": expression_contract["expression_min"],
        "expression_max": expression_contract["expression_max"],
        "expression_bin_boundaries_file": expression_contract["bin_boundaries_file"],
        "toy": True,
    }
    (graph_directory / "manifest.json").write_text(
        json.dumps({**payload, "consumers": ["xpert"]}, indent=2), encoding="utf-8"
    )
    (unimol_directory / "manifest.json").write_text(
        json.dumps({
            "format": "map_xpert_assets_v4", "artifact": "drug_unimol",
            "smiles": smiles, "consumers": ["xpert"],
            "drug_unimol_file": "drug_unimol.float16.npy", "toy": True,
        }, indent=2), encoding="utf-8"
    )
    expression_payload.update({"smiles": smiles, "consumers": ["xpert"], "toy": True})
    expression_manifest.write_text(json.dumps(expression_payload, indent=2), encoding="utf-8")
    return {
        "model": "xpert",
        "toy": True,
        "drugs": len(smiles),
        "genes": full_gene_count,
        "hvgs": hvg_dim,
        "directory": str(graph_directory),
    }


def main() -> None:
    args = parse_args()
    if args.root.exists() and any(args.root.iterdir()):
        raise FileExistsError(f"Toy root must be new or empty: {args.root}")
    if min(
        args.hvg_dim,
        args.pad_length,
        args.cells_per_condition,
        args.control_cells,
    ) <= 0:
        raise ValueError("Toy dimensions and cell counts must be positive")
    if "cmonge" in args.models and args.hvg_dim != 2000:
        raise ValueError("The CMonge toy run requires --hvg-dim 2000")
    gene_count = max(64, args.hvg_dim, args.pad_length - 1)
    if args.xpert_assets == "real":
        if args.dataset != "tahoe":
            raise ValueError("Real XPert assets are exercised by the Tahoe toy dataset")
        if "xpert" not in args.models:
            raise ValueError("--xpert-assets real requires xpert in --models")
        if gene_count > len(REAL_XPERT_GENES):
            raise ValueError(
                f"Real XPert toy dimensions cannot exceed {len(REAL_XPERT_GENES)} genes"
            )
    assets = require_frozen_assets(args.frozen_models)
    recorder = Recorder(args.root / "toy_e2e_summary.json", {
        **vars(args),
        "root": args.root.resolve(),
        "frozen_models": args.frozen_models.resolve(),
        "frozen_assets": assets,
        "gene_count": gene_count,
    })
    frozen_link = args.root / "frozen_models"
    if frozen_link.resolve() != args.frozen_models.resolve():
        frozen_link.symlink_to(args.frozen_models.resolve(), target_is_directory=True)
    if args.dataset == "tahoe":
        source_root = args.root / "raw_datasets" / "Tahoe-100M"
        populations = ["TOY_LINE"]
        source_drugs = REAL_XPERT_DRUGS if args.xpert_assets == "real" else DRUGS
        source_genes = REAL_XPERT_GENES if args.xpert_assets == "real" else None
        recorder.run("generate_tahoe_source", lambda: make_tahoe_source(
            source_root,
            Path(assets["esm2"]),
            seed=args.seed,
            control_cells=args.control_cells,
            cells_per_condition=args.cells_per_condition,
            gene_count=gene_count,
            drugs=source_drugs,
            gene_symbols=source_genes,
        ))
        handler = preprocess.builtin.tahoe(storage=args.root)
        flow = preprocess.pipeline(handler)
        recorder.run("statistics", lambda: flow.watch_data(batch_size=32))
        selection = recorder.run(
            "fetch_populations",
            lambda: flow.fetch_populations(
                populations, project_name="toy-e2e"
            ),
        )
    elif args.dataset == "sciplex":
        source_root = args.root / "raw_datasets" / "SciPlex3"
        generated = recorder.run(
            "generate_sciplex_source",
            lambda: make_native_atlas_source(
                source_root,
                Path(assets["esm2"]),
                dataset="sciplex",
                seed=args.seed,
                control_cells=args.control_cells,
                cells_per_condition=args.cells_per_condition,
                gene_count=gene_count,
            ),
        )
        populations = generated["populations"]
        handler = preprocess.builtin.sciplex(storage=args.root)
        flow = preprocess.pipeline(handler)
        recorder.run("statistics", flow.watch_data)
        selection = recorder.run(
            "fetch_populations",
            lambda: flow.fetch_populations(
                populations, project_name="toy-e2e"
            ),
        )
    else:
        source_root = args.root / "raw_datasets" / "OP3"
        generated = recorder.run(
            "generate_nips_source",
            lambda: make_native_atlas_source(
                source_root,
                Path(assets["esm2"]),
                dataset="nips",
                seed=args.seed,
                control_cells=args.control_cells,
                cells_per_condition=args.cells_per_condition,
                gene_count=gene_count,
            ),
        )
        populations = generated["populations"]
        handler = preprocess.builtin.nips(storage=args.root)
        flow = preprocess.pipeline(handler)
        recorder.run("statistics", flow.watch_data)
        selection = recorder.run(
            "fetch_populations",
            lambda: flow.fetch_populations(
                populations, project_name="toy-e2e"
            ),
        )
    recorder.run(
        "filter_conditions",
        lambda: flow.filter_conditions(
            selection,
            min_cells=1,
            max_cells=max(1, args.cells_per_condition - 1),
            seed=args.seed,
            workers=1,
        ),
    )
    recorder.run(
        "select_hvg",
        lambda: flow.select_hvg(
            selection,
            n_top_genes=args.hvg_dim, workers=1,
        ),
    )
    paths = recorder.run(
        "create_project",
        lambda: preparation.create_project(
            selection, project_name="toy-e2e"
        ),
    )
    materialization_options = {
        "pad_length": args.pad_length,
        "target_sum": 10_000,
        "workers": 1,
    }
    recorder.run(
        "prepare_cell_metadata",
        lambda: preparation.prepare_cell_metadata(
            paths, **materialization_options
        ),
    )
    recorder.run(
        "prepare_state_inputs",
        lambda: preparation.prepare_state_inputs(
            paths, **materialization_options
        ),
    )
    recorder.run(
        "prepare_hvg_expression",
        lambda: preparation.prepare_hvg_expression(
            paths, **materialization_options
        ),
    )
    workflow = paths
    recorder.run(
        "build_sampling_index",
        lambda: preparation.build_sampling_index(workflow, workers=1),
    )
    split = recorder.run(
        "create_split",
        lambda: preparation.create_split(
            workflow,
            rule="unprofiled_drug",
            external_test_size=1,
            internal_test_fraction=0.2,
            seed=args.seed,
        ),
    )
    split_file = split.summary["split_file"]
    recorder.run(
        "precache_gene_tokens",
        lambda: preparation.prepare_gene_tokens(workflow, batch_size=512),
    )
    recorder.run(
        "precache_drug_tokens",
        lambda: preparation.prepare_knowledge_drug_tokens(workflow, batch_size=4),
    )
    recorder.run(
        "merge_knowledge_tokens",
        lambda: preparation.assemble_knowledge_tokens(workflow),
    )
    recorder.run(
        "prepare_condition_embeddings",
        lambda: preparation.prepare_condition_embeddings(
            workflow, batch_size=8, workers=0
        ),
    )
    recorder.run(
        "merge_state_embeddings",
        lambda: preparation.assemble_condition_embeddings(workflow),
    )
    recorder.run(
        "validate_preparation",
        lambda: preparation.validate(workflow, split_files=[split_file]),
    )
    baseline_models = [model for model in args.models if model not in {"map", "xpert"}]
    for model in baseline_models:
        if model == "prnet":
            function = preparation.prepare_fcfp4_features
        elif model == "chemcpa":
            function = preparation.prepare_ecfp4_features
        elif model == "crisp":
            function = lambda: (
                preparation.prepare_molecular_descriptors(workflow),
                preparation.prepare_control_means(workflow),
                preparation.prepare_deg_masks(workflow, top_k=min(50, args.hvg_dim)),
            )
        elif model == "cmonge":
            function = preparation.prepare_molecular_descriptors
        else:
            function = lambda: None
        recorder.run(f"prepare_{model}_inputs", function)
    if "xpert" in args.models:
        if args.xpert_assets == "real":
            recorder.run(
                "prepare_real_xpert_inputs",
                lambda: (
                    preparation.prepare_unimol_tokens(workflow, overwrite=True),
                    preparation.prepare_graph_assets(
                        workflow, overwrite=True, epochs=1, layers=1,
                        hidden_size=min(16, args.hvg_dim), device="cpu",
                    ),
                    preparation.prepare_expression_bins(
                        workflow, input_formats=("official", "validation"),
                        expression_sample_cells=4096, overwrite=True,
                    ),
                ),
            )
        else:
            recorder.run(
                "prepare_toy_xpert_inputs",
                lambda: make_toy_xpert_inputs(workflow, seed=args.seed),
            )

    for model in args.models:
        train_kwargs = {
            "model": model,
            "regime": "unprofiled_drug",
            "split_file": split_file,
            "gpus": 1,
            "num_workers": 0,
            "run_name": f"toy-{model}",
            "set_size": 1,
            "batch_size": 1,
            "epochs": 1,
            "max_steps": 1,
            "checkpoint_every_epochs": 1,
            "samples_per_epoch": 1,
            "seed": args.seed,
        }
        if model == "map":
            train_kwargs.update({
                "gradient_accumulation_steps": 1,
                "warmup_steps": 0,
            })
        elif model in {"prnet", "chemcpa", "crisp", "xpert"}:
            train_kwargs.update({
                "set_size": 2,
                "max_steps": 2,
                "samples_per_epoch": 2,
            })
            if model == "xpert":
                train_kwargs.update({
                    "hidden_size": 16,
                    "attention_heads": 4,
                    "treated_structure": "CA",
                    "control_structure": "SA",
                    "max_steps": 1,
                    "samples_per_epoch": 1,
                })
        elif model == "cmonge":
            train_kwargs.update({
                "set_size": 2,
                "max_steps": 1,
                "samples_per_epoch": 1,
                "ae_epochs": 1,
                "ae_batch_size": 4,
                "ae_width": 16,
                "latent_dim": 8,
                "context_dim": 8,
                "hidden_sizes": (8, 8),
            })
        run = recorder.run(
            f"train_{model}", lambda values=train_kwargs: train.run(workflow, **values)
        )
        run_dir = Path(run.summary["run_dir"])
        training_files = require_files([
            run_dir / "run_config.json",
            run_dir / "last.pt",
        ])
        evaluation = recorder.run(
            f"evaluate_{model}",
            lambda: eval.run(
                workflow,
                run_name=f"toy-{model}",
                checkpoint=run_dir / "last.pt",
                seeds=(args.seed,),
                set_size=1,
                deg_top_k=min(10, args.hvg_dim),
                evaluation_name=f"toy-eval-{model}",
            ),
        )
        evaluation_dir = Path(evaluation.summary["evaluation_dir"])
        external_evaluation = evaluation_dir / "external_test" / "evaluation.json"
        external_prediction = evaluation_dir / "external_test" / f"predictions_seed{args.seed}.parquet"
        evaluation_files = require_files([
            Path(evaluation.summary["evaluation_file"]),
            external_evaluation,
            external_prediction,
            evaluation_dir / "internal_test" / "evaluation.json",
        ])
        analysis = recorder.run(
            f"analyze_{model}",
            lambda: eval.analyze(
                workflow,
                prediction_file=external_prediction,
                evaluation_files=[external_evaluation],
                output_name=f"toy-{model}",
            ),
        )
        report_file = workflow.workspace / "analysis" / "reports" / f"toy-{model}" / "report.html"
        analysis_files = require_files([report_file])
        evaluation_payload = json.loads(
            external_evaluation.read_text(encoding="utf-8")
        )
        recorder.payload["models"][model] = {
            "status": "pass",
            "checkpoint": run.summary["checkpoint"],
            "global_step": int(torch.load(
                run.summary["checkpoint"], map_location="cpu", weights_only=False
            )["global_step"]),
            "training_files": training_files,
            "evaluation_files": evaluation_files,
            "analysis_files": analysis_files,
            "metric_count": len(evaluation_payload["summary"]),
            "analysis": jsonable(analysis),
        }
        recorder.write()

    recorder.payload["status"] = "pass"
    recorder.payload["completed_models"] = list(args.models)
    recorder.write()
    print(json.dumps({
        "status": "pass",
        "summary": str(recorder.path),
        "models": list(args.models),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
