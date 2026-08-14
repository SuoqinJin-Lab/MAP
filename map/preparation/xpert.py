from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from .._common.paths import DatasetPaths


STRING_LINKS = "9606.protein.links.v12.0.txt.gz"
STRING_INFO = "9606.protein.info.v12.0.txt.gz"
PRIMEKG = "primekg.csv"
UNIMOL_CHECKPOINT = "mol_pre_all_h_220816.pt"
UNIMOL_DICTIONARY = "mol.dict.txt"

_ATOM_TO_ID = {
    "N": 2,
    "Se": 3,
    "I": 4,
    "As": 5,
    "Pt": 6,
    "S": 7,
    "Sn": 8,
    "Mg": 9,
    "P": 10,
    "Cl": 11,
    "Au": 12,
    "F": 13,
    "Co": 14,
    "Br": 15,
    "Si": 16,
    "O": 17,
    "H": 18,
    "B": 19,
    "C": 20,
    "Ca": 21,
    "Hg": 22,
    "Li": 23,
    "[UNK]": 24,
    "Na": 25,
    "hg_molecule": 0,
    "molecule": 1,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> dict:
    path = path.resolve()
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _normalized_name(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    text = re.sub(r"\([^)]*\)", " ", text.casefold())
    return re.sub(r"[^a-z0-9]+", "", text)


def _gene_symbols(paths: DatasetPaths) -> list[str]:
    payload = json.loads((paths.prepared / "hvg.json").read_text(encoding="utf-8"))
    symbols = [str(value) for value in payload["gene_symbols"]]
    if len(symbols) != len(set(symbols)):
        raise ValueError("HVG gene symbols must be unique for XPert")
    return symbols


def _drug_vocabulary(paths: DatasetPaths) -> tuple[list[str], dict[str, list[str]]]:
    table = pq.read_table(
        paths.prepared / "conditions.parquet", columns=["drug", "canonical_smiles"]
    ).to_pandas()
    names: dict[str, set[str]] = {}
    for drug, smiles in zip(table["drug"], table["canonical_smiles"]):
        names.setdefault(str(smiles), set()).add(str(drug))
    smiles = sorted(names)
    return smiles, {value: sorted(names[value]) for value in smiles}


def read_string_ppi(
    links_file: str | Path,
    info_file: str | Path,
    gene_symbols: list[str],
    *,
    score_threshold: int = 700,
    chunksize: int = 1_000_000,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Map STRING v12 human edges to the requested gene-symbol order."""
    links_file, info_file = Path(links_file), Path(info_file)
    gene_to_index = {value: index for index, value in enumerate(gene_symbols)}
    info = pd.read_csv(info_file, sep="\t", usecols=["#string_protein_id", "preferred_name"])
    protein_to_gene = {
        str(protein): gene_to_index[str(symbol)]
        for protein, symbol in zip(info["#string_protein_id"], info["preferred_name"])
        if str(symbol) in gene_to_index
    }
    best: dict[tuple[int, int], float] = {}
    raw_high_confidence = 0
    for chunk in pd.read_csv(links_file, sep=r"\s+", chunksize=chunksize):
        selected = chunk.loc[
            chunk["combined_score"] > int(score_threshold),
            ["protein1", "protein2", "combined_score"],
        ]
        raw_high_confidence += len(selected)
        for first, second, score in selected.itertuples(index=False, name=None):
            source = protein_to_gene.get(str(first))
            target = protein_to_gene.get(str(second))
            if source is None or target is None or source == target:
                continue
            edge = (min(source, target), max(source, target))
            best[edge] = max(best.get(edge, 0.0), float(score) / 1000.0)
    undirected = sorted(best)
    sources = [edge[0] for edge in undirected] + [edge[1] for edge in undirected]
    targets = [edge[1] for edge in undirected] + [edge[0] for edge in undirected]
    weights = [best[edge] for edge in undirected] * 2
    edges = np.asarray([sources, targets], dtype=np.int64)
    return edges, np.asarray(weights, dtype=np.float32), {
        "score_threshold": int(score_threshold),
        "raw_high_confidence_edges": int(raw_high_confidence),
        "mapped_proteins": len(protein_to_gene),
        "mapped_genes": len(set(protein_to_gene.values())),
        "undirected_edges": len(undirected),
        "directed_edges": edges.shape[1],
    }


def _drug_alias_lookup(
    smiles: list[str],
    names: dict[str, list[str]],
    aliases_file: str | Path | None,
) -> tuple[dict[str, int], dict[str, str]]:
    lookup: dict[str, int] = {}
    source: dict[str, str] = {}
    for index, value in enumerate(smiles):
        for name in names[value]:
            normalized = _normalized_name(name)
            if normalized:
                lookup.setdefault(normalized, index)
                source.setdefault(normalized, "condition_table")
    if aliases_file is not None:
        aliases = pd.read_csv(aliases_file, sep=None, engine="python")
        required = {"alias", "canonical_smiles"}
        if not required.issubset(aliases.columns):
            raise ValueError("XPert alias table requires alias and canonical_smiles columns")
        smiles_to_index = {value: index for index, value in enumerate(smiles)}
        for alias, value in aliases[["alias", "canonical_smiles"]].itertuples(
            index=False, name=None
        ):
            if str(value) not in smiles_to_index:
                raise ValueError(f"Alias table contains unknown SMILES: {value}")
            normalized = _normalized_name(alias)
            lookup[normalized] = smiles_to_index[str(value)]
            source[normalized] = "alias_table"
    return lookup, source


def read_primekg_dti(
    primekg_file: str | Path,
    smiles: list[str],
    drug_names: dict[str, list[str]],
    gene_symbols: list[str],
    *,
    aliases_file: str | Path | None = None,
    chunksize: int = 500_000,
) -> tuple[np.ndarray, dict]:
    """Extract PrimeKG drug-protein edges using exact normalized names."""
    primekg_file = Path(primekg_file)
    genes = {value: index for index, value in enumerate(gene_symbols)}
    normalized_genes = {_normalized_name(value): index for value, index in genes.items()}
    drugs, alias_sources = _drug_alias_lookup(smiles, drug_names, aliases_file)
    edges: set[tuple[int, int]] = set()
    matched_drugs: set[int] = set()
    matched_by_alias = 0
    dti_rows = 0
    required = ["x_type", "x_name", "y_type", "y_name"]
    for chunk in pd.read_csv(primekg_file, usecols=required, chunksize=chunksize):
        x_type = chunk["x_type"].astype(str).str.casefold()
        y_type = chunk["y_type"].astype(str).str.casefold()
        x_drug = x_type.str.contains("drug", regex=False)
        y_drug = y_type.str.contains("drug", regex=False)
        x_gene = x_type.str.contains("gene", regex=False) | x_type.str.contains(
            "protein", regex=False
        )
        y_gene = y_type.str.contains("gene", regex=False) | y_type.str.contains(
            "protein", regex=False
        )
        selected = chunk.loc[x_drug & y_gene, ["x_name", "y_name"]]
        reverse = chunk.loc[y_drug & x_gene, ["y_name", "x_name"]]
        reverse.columns = ["x_name", "y_name"]
        dti = pd.concat((selected, reverse), ignore_index=True)
        dti_rows += len(dti)
        for drug_name, gene_name in dti.itertuples(index=False, name=None):
            normalized_drug = _normalized_name(drug_name)
            drug_index = drugs.get(normalized_drug)
            gene_index = normalized_genes.get(_normalized_name(gene_name))
            if drug_index is None or gene_index is None:
                continue
            edges.add((drug_index, gene_index))
            matched_drugs.add(drug_index)
            matched_by_alias += alias_sources.get(normalized_drug) == "alias_table"
    ordered = sorted(edges)
    edge_index = (
        np.asarray(ordered, dtype=np.int64).T
        if ordered
        else np.empty((2, 0), dtype=np.int64)
    )
    return edge_index, {
        "primekg_dti_rows": int(dti_rows),
        "mapped_edges": edge_index.shape[1],
        "mapped_drugs": len(matched_drugs),
        "drug_coverage": len(matched_drugs) / max(len(smiles), 1),
        "alias_matched_edges": int(matched_by_alias),
    }


def build_dds(smiles: list[str], *, threshold: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    matrix = np.zeros((len(smiles), 2048), dtype=np.float32)
    for index, value in enumerate(smiles):
        molecule = Chem.MolFromSmiles(value)
        if molecule is None:
            raise ValueError(f"Invalid canonical SMILES: {value}")
        DataStructs.ConvertToNumpyArray(generator.GetFingerprint(molecule), matrix[index])
    intersection = matrix @ matrix.T
    counts = matrix.sum(1)
    union = counts[:, None] + counts[None, :] - intersection
    similarity = intersection / np.maximum(union, 1.0)
    source, target = np.where((similarity > float(threshold)) & ~np.eye(len(smiles), dtype=bool))
    return (
        np.asarray([source, target], dtype=np.int64),
        similarity[source, target].astype(np.float32),
    )


def build_unimol_tokens(
    smiles: list[str],
    checkpoint: str | Path,
    dictionary: str | Path,
    *,
    max_atoms: int = 122,
    batch_size: int = 32,
    use_cuda: bool = True,
) -> np.ndarray:
    try:
        from unimol_tools import UniMolRepr
    except ImportError as error:
        raise ImportError("XPert preparation requires unimol-tools") from error
    model = UniMolRepr(
        data_type="molecule",
        remove_hs=False,
        batch_size=int(batch_size),
        use_cuda=bool(use_cuda and torch.cuda.is_available()),
        pretrained_model_path=str(checkpoint),
        pretrained_dict_path=str(dictionary),
        max_atoms=max(int(max_atoms) - 2, 1),
    )
    result = model.get_repr(smiles, return_atomic_reprs=True)
    if len(result["cls_repr"]) != len(smiles):
        raise RuntimeError("UniMol did not return one representation per drug")
    output = np.zeros((len(smiles), int(max_atoms), 514), dtype=np.float32)
    for index, (molecule, atoms, symbols) in enumerate(
        zip(result["cls_repr"], result["atomic_reprs"], result["atomic_symbol"])
    ):
        features = np.vstack((np.asarray(molecule), np.asarray(molecule), np.asarray(atoms)))
        labels = ["hg_molecule", "molecule", *[str(value) for value in symbols]]
        length = min(len(features), int(max_atoms))
        output[index, :length, 0] = 1.0
        output[index, :length, 1] = [
            _ATOM_TO_ID.get(value, _ATOM_TO_ID["[UNK]"]) for value in labels[:length]
        ]
        output[index, :length, 2:] = features[:length]
    return output


def _aggregate(
    values: torch.Tensor,
    source: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    size: int,
) -> torch.Tensor:
    output = values.new_zeros((size, values.shape[1]))
    output.index_add_(0, target, values[source] * weights.unsqueeze(1))
    counts = values.new_zeros(size)
    counts.index_add_(0, target, weights)
    return output / counts.clamp_min(1e-6).unsqueeze(1)


class _HeterogeneousEncoder(nn.Module):
    def __init__(self, gene_dim: int, drug_dim: int, hidden_size: int, layers: int):
        super().__init__()
        self.gene_input = nn.Linear(gene_dim, hidden_size)
        self.drug_input = nn.Linear(drug_dim, hidden_size)
        self.gene_self = nn.ModuleList(nn.Linear(hidden_size, hidden_size) for _ in range(layers))
        self.gene_ppi = nn.ModuleList(nn.Linear(hidden_size, hidden_size) for _ in range(layers))
        self.gene_dti = nn.ModuleList(nn.Linear(hidden_size, hidden_size) for _ in range(layers))
        self.drug_self = nn.ModuleList(nn.Linear(hidden_size, hidden_size) for _ in range(layers))
        self.drug_dds = nn.ModuleList(nn.Linear(hidden_size, hidden_size) for _ in range(layers))
        self.drug_dti = nn.ModuleList(nn.Linear(hidden_size, hidden_size) for _ in range(layers))
        self.gene_norm = nn.ModuleList(nn.LayerNorm(hidden_size) for _ in range(layers))
        self.drug_norm = nn.ModuleList(nn.LayerNorm(hidden_size) for _ in range(layers))

    def forward(self, gene_values, drug_values, ppi, dti, dds):
        genes = self.gene_input(gene_values)
        drugs = self.drug_input(drug_values)
        for layer in range(len(self.gene_self)):
            ppi_mean = _aggregate(genes, *ppi, len(genes))
            drug_to_gene = _aggregate(drugs, dti[0], dti[1], dti[2], len(genes))
            dds_mean = _aggregate(drugs, *dds, len(drugs))
            gene_to_drug = _aggregate(genes, dti[1], dti[0], dti[2], len(drugs))
            genes = self.gene_norm[layer](F.relu(
                self.gene_self[layer](genes)
                + self.gene_ppi[layer](ppi_mean)
                + self.gene_dti[layer](drug_to_gene)
            ))
            drugs = self.drug_norm[layer](F.relu(
                self.drug_self[layer](drugs)
                + self.drug_dds[layer](dds_mean)
                + self.drug_dti[layer](gene_to_drug)
            ))
        return genes, drugs


def _edge_loss(source, target, edge_index, maximum, generator):
    source_ids, target_ids = edge_index
    if len(source_ids) == 0:
        return source.sum() * 0.0
    if len(source_ids) > maximum:
        chosen = torch.randint(
            len(source_ids), (maximum,), device=source_ids.device, generator=generator
        )
        source_ids, target_ids = source_ids[chosen], target_ids[chosen]
    negative = torch.randint(
        len(target), target_ids.shape, device=target_ids.device, generator=generator
    )
    scale = math.sqrt(source.shape[1])
    positive = (source[source_ids] * target[target_ids]).sum(1) / scale
    negative_score = (source[source_ids] * target[negative]).sum(1) / scale
    return F.softplus(-positive).mean() + F.softplus(negative_score).mean()


def _tensor_edges(edge_index, weights, device):
    return (
        torch.as_tensor(edge_index[0], dtype=torch.long, device=device),
        torch.as_tensor(edge_index[1], dtype=torch.long, device=device),
        torch.as_tensor(weights, dtype=torch.float32, device=device),
    )


def _esm_matrix(path: Path, symbols: list[str]) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    missing = [value for value in symbols if value not in payload]
    if missing:
        raise ValueError(f"ESM2 asset is missing {len(missing)} XPert HVGs")
    return torch.stack([torch.as_tensor(payload[value], dtype=torch.float32) for value in symbols])


def prepare_xpert_inputs(
    paths: DatasetPaths,
    output_dir: str | Path,
    *,
    reference_dir: str | Path | None = None,
    unimol_dir: str | Path | None = None,
    aliases_file: str | Path | None = None,
    score_threshold: int = 700,
    dds_threshold: float = 0.5,
    hidden_size: int = 128,
    layers: int = 3,
    epochs: int = 100,
    edges_per_type: int = 65_536,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    seed: int = 42,
    max_atoms: int = 122,
    unimol_batch_size: int = 32,
    device: str | None = None,
    overwrite: bool = False,
) -> tuple[dict, list[Path]]:
    output_dir = Path(output_dir)
    reference_dir = Path(reference_dir or paths.frozen_models.parent / "reference_data" / "xpert")
    unimol_dir = Path(unimol_dir or paths.frozen_models / "unimol")
    aliases_path = Path(aliases_file) if aliases_file else reference_dir / "drug_aliases.tsv"
    if not aliases_path.is_file():
        aliases_path = None
    required = {
        "string_links": reference_dir / STRING_LINKS,
        "string_info": reference_dir / STRING_INFO,
        "primekg": reference_dir / PRIMEKG,
        "unimol_checkpoint": unimol_dir / UNIMOL_CHECKPOINT,
        "unimol_dictionary": unimol_dir / UNIMOL_DICTIONARY,
    }
    missing = [str(value) for value in required.values() if not value.is_file()]
    if missing:
        raise FileNotFoundError("XPert reference assets are missing: " + ", ".join(missing))
    if min(hidden_size, layers, epochs, edges_per_type, max_atoms, unimol_batch_size) <= 0:
        raise ValueError("XPert graph dimensions and schedules must be positive")
    if not 0 < dds_threshold <= 1 or score_threshold < 0:
        raise ValueError("XPert graph thresholds are invalid")
    manifest_file = output_dir / "manifest.json"
    if manifest_file.is_file() and not overwrite:
        payload = json.loads(manifest_file.read_text(encoding="utf-8"))
        return payload, [manifest_file]
    output_dir.mkdir(parents=True, exist_ok=True)

    symbols = _gene_symbols(paths)
    smiles, drug_names = _drug_vocabulary(paths)
    ppi_edges, ppi_weights, ppi_stats = read_string_ppi(
        required["string_links"], required["string_info"], symbols,
        score_threshold=score_threshold,
    )
    dti_edges, dti_stats = read_primekg_dti(
        required["primekg"], smiles, drug_names, symbols, aliases_file=aliases_path
    )
    dds_edges, dds_weights = build_dds(smiles, threshold=dds_threshold)
    if ppi_edges.shape[1] == 0:
        raise RuntimeError("No STRING PPI edges mapped to the HVG vocabulary")
    if dti_edges.shape[1] == 0:
        raise RuntimeError(
            "No PrimeKG DTI edges mapped; add exact aliases to drug_aliases.tsv"
        )
    np.savez_compressed(output_dir / "ppi_edges.npz", edge_index=ppi_edges, edge_weight=ppi_weights)
    np.savez_compressed(
        output_dir / "dti_edges.npz",
        edge_index=dti_edges,
        edge_weight=np.ones(dti_edges.shape[1], dtype=np.float32),
    )
    np.savez_compressed(output_dir / "dds_edges.npz", edge_index=dds_edges, edge_weight=dds_weights)

    unimol = build_unimol_tokens(
        smiles,
        required["unimol_checkpoint"],
        required["unimol_dictionary"],
        max_atoms=max_atoms,
        batch_size=unimol_batch_size,
    )
    np.save(output_dir / "drug_unimol.float16.npy", unimol.astype(np.float16))
    esm_path = paths.frozen_models / "Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt"
    if not esm_path.is_file():
        esm_path = paths.frozen_models / "state" / "Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt"
    if not esm_path.is_file():
        esm_path = paths.frozen_models / "state" / "gene_embeddings_esm2.pt"
    if not esm_path.is_file():
        raise FileNotFoundError("XPert graph initialization requires the ESM2 gene asset")
    genes = _esm_matrix(esm_path, symbols)
    drugs = torch.from_numpy(unimol[:, 1, 2:].astype(np.float32))
    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(seed)
    np.random.seed(seed)
    encoder = _HeterogeneousEncoder(
        genes.shape[1], drugs.shape[1], int(hidden_size), int(layers)
    ).to(selected_device)
    genes, drugs = genes.to(selected_device), drugs.to(selected_device)
    ppi = _tensor_edges(ppi_edges, ppi_weights, selected_device)
    dti = _tensor_edges(
        dti_edges, np.ones(dti_edges.shape[1], dtype=np.float32), selected_device
    )
    dds = _tensor_edges(dds_edges, dds_weights, selected_device)
    optimizer = torch.optim.Adam(
        encoder.parameters(), lr=float(lr), weight_decay=float(weight_decay)
    )
    generator = torch.Generator(device=selected_device).manual_seed(seed + 1)
    best_loss = float("inf")
    best_state = None
    for epoch in range(int(epochs)):
        encoder.train()
        gene_output, drug_output = encoder(genes, drugs, ppi, dti, dds)
        loss = (
            _edge_loss(gene_output, gene_output, ppi[:2], edges_per_type, generator)
            + _edge_loss(drug_output, gene_output, dti[:2], edges_per_type, generator)
            + _edge_loss(drug_output, drug_output, dds[:2], edges_per_type, generator)
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        value = float(loss.detach().cpu())
        if value < best_loss:
            best_loss = value
            best_state = {key: tensor.detach().cpu() for key, tensor in encoder.state_dict().items()}
        if (epoch + 1) % max(int(epochs) // 10, 1) == 0:
            print(f"XPert graph epoch {epoch + 1}/{epochs}: loss={value:.6f}", flush=True)
    if best_state is None:
        raise RuntimeError("XPert graph pretraining did not produce a checkpoint")
    encoder.load_state_dict(best_state)
    encoder.to(selected_device).eval()
    with torch.no_grad():
        gene_output, drug_output = encoder(genes, drugs, ppi, dti, dds)
    np.save(output_dir / "ppi_gene_vectors.float32.npy", gene_output.cpu().numpy().astype(np.float32))
    np.save(output_dir / "drug_hg_embeddings.float32.npy", drug_output.cpu().numpy().astype(np.float32))
    torch.save(
        {
            "format": "map_xpert_graph_v1",
            "state_dict": best_state,
            "gene_input_dim": int(genes.shape[1]),
            "drug_input_dim": int(drugs.shape[1]),
            "hidden_size": int(hidden_size),
            "layers": int(layers),
            "best_loss": best_loss,
        },
        output_dir / "graph_encoder.pt",
    )
    (output_dir / "gene_symbols.json").write_text(json.dumps(symbols, indent=2), encoding="utf-8")
    (output_dir / "drug_vocabulary.json").write_text(
        json.dumps({"smiles": smiles, "names": drug_names}, indent=2), encoding="utf-8"
    )
    payload = {
        "model": "xpert",
        "directory": "xpert",
        "format": "map_xpert_assets_v1",
        "gene_symbols": symbols,
        "smiles": smiles,
        "hidden_size": int(hidden_size),
        "max_atoms": int(max_atoms),
        "ppi_gene_vector_file": "ppi_gene_vectors.float32.npy",
        "drug_hg_embedding_file": "drug_hg_embeddings.float32.npy",
        "drug_unimol_file": "drug_unimol.float16.npy",
        "ppi_edge_file": "ppi_edges.npz",
        "dti_edge_file": "dti_edges.npz",
        "dds_edge_file": "dds_edges.npz",
        "graph_checkpoint": "graph_encoder.pt",
        "graph": {
            "ppi": ppi_stats,
            "dti": dti_stats,
            "dds_threshold": float(dds_threshold),
            "dds_directed_edges": int(dds_edges.shape[1]),
            "layers": int(layers),
            "epochs": int(epochs),
            "best_loss": best_loss,
            "seed": int(seed),
        },
        "sources": {name: _file_record(value) for name, value in required.items()},
    }
    if aliases_path is not None:
        payload["sources"]["drug_aliases"] = _file_record(aliases_path)
    manifest_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    outputs = [
        manifest_file,
        output_dir / "ppi_edges.npz",
        output_dir / "dti_edges.npz",
        output_dir / "dds_edges.npz",
        output_dir / "drug_unimol.float16.npy",
        output_dir / "ppi_gene_vectors.float32.npy",
        output_dir / "drug_hg_embeddings.float32.npy",
        output_dir / "graph_encoder.pt",
        output_dir / "gene_symbols.json",
        output_dir / "drug_vocabulary.json",
    ]
    return payload, outputs


__all__ = [
    "PRIMEKG",
    "STRING_INFO",
    "STRING_LINKS",
    "UNIMOL_CHECKPOINT",
    "UNIMOL_DICTIONARY",
    "build_dds",
    "build_unimol_tokens",
    "prepare_xpert_inputs",
    "read_primekg_dti",
    "read_string_ppi",
]
