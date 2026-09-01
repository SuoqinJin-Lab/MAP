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
from .._common.hvg import load_hvg_contract


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
    symbols = [
        str(value) for value in json.loads(
            (paths.prepared / "state_gene_symbols.json").read_text(encoding="utf-8")
        )
    ]
    if len(symbols) != len(set(symbols)):
        raise ValueError("STATE gene symbols must be unique for the graph artifact")
    return symbols


def _hvg_indices(paths: DatasetPaths, gene_count: int) -> np.ndarray:
    payload = json.loads((paths.prepared / "hvg.json").read_text(encoding="utf-8"))
    indices = np.asarray(payload["state_ids"], dtype=np.int64)
    if indices.ndim != 1 or len(indices) != len(payload["gene_symbols"]):
        raise ValueError("The graph artifact requires one STATE id for every HVG")
    if (
        len(np.unique(indices)) != len(indices)
        or np.any(indices < 0)
        or np.any(indices >= gene_count)
    ):
        raise ValueError("HVG-to-full-gene indices are invalid for the graph artifact")
    return indices.astype(np.int32)


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
            raise ValueError("Alias table requires alias and canonical_smiles columns")
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
        raise ImportError("Drug-token preparation requires unimol-tools") from error
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


class _HeterogeneousEncoder(nn.Module):
    def __init__(self, gene_dim: int, drug_dim: int, hidden_size: int, layers: int):
        super().__init__()
        self.gene_input = nn.Linear(gene_dim, hidden_size)
        self.drug_input = nn.Linear(drug_dim, hidden_size)
        self.convs = nn.ModuleList(
            [_HeterogeneousSAGE(hidden_size) for _ in range(int(layers))]
        )

    def forward(self, values, edges):
        values = {
            "gene": self.gene_input(values["gene"]),
            "drug": self.drug_input(values["drug"]),
        }
        for conv in self.convs:
            values = {key: output.relu() for key, output in conv(values, edges).items()}
        return values["gene"], values["drug"]


_PPI = ("gene", "PPI", "gene")
_DDS = ("drug", "DDS", "drug")
_DTI = ("drug", "DTI", "gene")
_DTI_REVERSE = ("gene", "DTI", "drug")


class _RelationSAGE(nn.Module):
    """Mean-aggregation GraphSAGE relation matching PyG's default SAGEConv."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.neighbor = nn.Linear(hidden_size, hidden_size)
        self.root = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, source, target, edge_index):
        output = target.new_zeros(target.shape)
        counts = target.new_zeros(target.shape[0], 1)
        if edge_index.shape[1]:
            source_ids, target_ids = edge_index
            output.index_add_(0, target_ids, source[source_ids])
            counts.index_add_(
                0,
                target_ids,
                torch.ones(
                    len(target_ids), 1, device=target.device, dtype=target.dtype
                ),
            )
        output = output / counts.clamp_min(1.0)
        return self.neighbor(output) + self.root(target)


class _HeterogeneousSAGE(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.ppi = _RelationSAGE(hidden_size)
        self.dds = _RelationSAGE(hidden_size)
        self.dti = _RelationSAGE(hidden_size)
        self.dti_reverse = _RelationSAGE(hidden_size)

    def forward(self, values, edges):
        gene = (
            self.ppi(values["gene"], values["gene"], edges[_PPI])
            + self.dti(values["drug"], values["gene"], edges[_DTI])
        )
        drug = (
            self.dds(values["drug"], values["drug"], edges[_DDS])
            + self.dti_reverse(
                values["gene"], values["drug"], edges[_DTI_REVERSE]
            )
        )
        return {"gene": gene, "drug": drug}


class _GraphBatch:
    def __init__(self, x_dict, edge_index_dict):
        self.x_dict = x_dict
        self.edge_index_dict = edge_index_dict

    def clone(self):
        return _GraphBatch(
            {key: value.clone() for key, value in self.x_dict.items()},
            {key: value.clone() for key, value in self.edge_index_dict.items()},
        )

    def to(self, device):
        self.x_dict = {key: value.to(device) for key, value in self.x_dict.items()}
        self.edge_index_dict = {
            key: value.to(device) for key, value in self.edge_index_dict.items()
        }
        return self

    def __getitem__(self, key):
        if isinstance(key, str):
            values = self.x_dict[key]
            return type("NodeStore", (), {"x": values, "num_nodes": len(values)})()
        return type("EdgeStore", (), {"edge_index": self.edge_index_dict[key]})()


def _graph_data(genes, drugs, ppi_edges, dti_edges, dds_edges):
    return _GraphBatch(
        {"gene": genes, "drug": drugs},
        {
            _PPI: torch.as_tensor(ppi_edges, dtype=torch.long),
            _DDS: torch.as_tensor(dds_edges, dtype=torch.long),
            _DTI: torch.as_tensor(dti_edges, dtype=torch.long),
            _DTI_REVERSE: torch.as_tensor(
                np.stack((dti_edges[1], dti_edges[0])), dtype=torch.long
            ),
        },
    )


class _NeighborSubgraphLoader:
    """Dependency-free heterogeneous neighbor sampler for graph pretraining.

    PyG's ``NeighborLoader`` requires an additional compiled sampler backend
    that is not part of MAP's core environment.  This implements the same
    drug-rooted, per-hop fanout contract in NumPy and returns ordinary
    ``HeteroData`` mini-batches consumable by the same GNN.
    """

    def __init__(self, data, *, num_neighbors, batch_size: int, seed: int):
        self.data = data
        self.num_neighbors = tuple(int(value) for value in num_neighbors)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0
        self.edge_types = (_PPI, _DDS, _DTI, _DTI_REVERSE)
        self.edges = {
            edge_type: data.edge_index_dict[edge_type].cpu().numpy().astype(np.int64)
            for edge_type in self.edge_types
        }
        self.incoming: dict[tuple[str, str, str], dict[int, np.ndarray]] = {}
        for edge_type, edge_index in self.edges.items():
            groups: dict[int, list[int]] = {}
            for source, target in edge_index.T:
                groups.setdefault(int(target), []).append(int(source))
            self.incoming[edge_type] = {
                target: np.asarray(sources, dtype=np.int64)
                for target, sources in groups.items()
            }

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return math.ceil(len(self.data.x_dict["drug"]) / self.batch_size)

    def __iter__(self):
        rng = np.random.default_rng([self.seed, self.epoch])
        drug_ids = np.arange(len(self.data.x_dict["drug"]), dtype=np.int64)
        for start in range(0, len(drug_ids), self.batch_size):
            roots = drug_ids[start : start + self.batch_size]
            selected = {"drug": set(map(int, roots)), "gene": set()}
            sampled_edges = {edge_type: set() for edge_type in self.edge_types}
            for fanout in self.num_neighbors:
                frontier = {key: tuple(values) for key, values in selected.items()}
                additions = {"drug": set(), "gene": set()}
                for edge_type in self.edge_types:
                    source_type, _, target_type = edge_type
                    incoming = self.incoming[edge_type]
                    for target in frontier[target_type]:
                        candidates = incoming.get(int(target))
                        if candidates is None or not len(candidates):
                            continue
                        if len(candidates) > int(fanout):
                            candidates = rng.choice(
                                candidates, int(fanout), replace=False
                            )
                        for source in candidates:
                            source = int(source)
                            additions[source_type].add(source)
                            sampled_edges[edge_type].add((source, int(target)))
                selected["drug"].update(additions["drug"])
                selected["gene"].update(additions["gene"])

            local_ids = {
                node_type: np.asarray(sorted(values), dtype=np.int64)
                for node_type, values in selected.items()
            }
            local_lookup = {
                node_type: {int(value): index for index, value in enumerate(values)}
                for node_type, values in local_ids.items()
            }
            batch = _graph_data(
                self.data.x_dict["gene"][torch.as_tensor(local_ids["gene"])],
                self.data.x_dict["drug"][torch.as_tensor(local_ids["drug"])],
                np.empty((2, 0), dtype=np.int64),
                np.empty((2, 0), dtype=np.int64),
                np.empty((2, 0), dtype=np.int64),
            )
            for edge_type in self.edge_types:
                source_type, _, target_type = edge_type
                local_edges = [
                    (
                        local_lookup[source_type][source],
                        local_lookup[target_type][target],
                    )
                    for source, target in sampled_edges[edge_type]
                ]
                batch.edge_index_dict[edge_type] = (
                    torch.as_tensor(local_edges, dtype=torch.long).T.contiguous()
                    if local_edges
                    else torch.empty((2, 0), dtype=torch.long)
                )
            yield batch


def _graph_loss(outputs, edge_index_dict, maximum, generator, *, negative_samples, temperature):
    gene_output, drug_output = outputs
    relations = (
        (gene_output, gene_output, edge_index_dict[_PPI]),
        (drug_output, gene_output, edge_index_dict[_DTI]),
        (gene_output, drug_output, edge_index_dict[_DTI_REVERSE]),
        (drug_output, drug_output, edge_index_dict[_DDS]),
    )
    return sum(
        _edge_loss(
            source,
            target,
            tuple(edge_index),
            maximum,
            generator,
            negative_samples=negative_samples,
            temperature=temperature,
        )
        for source, target, edge_index in relations
    )


def _edge_loss(
    source,
    target,
    edge_index,
    maximum,
    generator,
    *,
    negative_samples: int,
    temperature: float,
):
    all_source_ids, all_target_ids = edge_index
    source_ids, target_ids = all_source_ids, all_target_ids
    if len(source_ids) == 0:
        return source.sum() * 0.0
    if len(source_ids) > maximum:
        chosen = torch.randint(
            len(source_ids), (maximum,), device=source_ids.device, generator=generator
        )
        source_ids, target_ids = source_ids[chosen], target_ids[chosen]
    negative = torch.randint(
        len(target), (len(target_ids), int(negative_samples)),
        device=target_ids.device, generator=generator,
    )
    positive_codes = torch.unique(
        all_source_ids * len(target) + all_target_ids
    ).sort().values
    negative_source = source_ids[:, None].expand_as(negative)
    for _ in range(8):
        codes = negative_source * len(target) + negative
        positions = torch.searchsorted(positive_codes, codes)
        clipped = positions.clamp_max(len(positive_codes) - 1)
        invalid = (
            (positions < len(positive_codes))
            & (positive_codes[clipped] == codes)
        )
        if not invalid.any():
            break
        negative[invalid] = torch.randint(
            len(target), (int(invalid.sum()),), device=target_ids.device,
            generator=generator,
        )
    anchors = source[source_ids]
    positive = F.cosine_similarity(
        anchors, target[target_ids], dim=-1
    ) / float(temperature)
    negatives = F.cosine_similarity(
        anchors[:, None, :], target[negative], dim=-1
    ) / float(temperature)
    return (
        -positive
        + torch.logsumexp(
            torch.cat((positive[:, None], negatives), dim=1), dim=1
        )
    ).mean()


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
        raise ValueError(f"ESM2 asset is missing {len(missing)} graph genes")
    return torch.stack([torch.as_tensor(payload[value], dtype=torch.float32) for value in symbols])


def _sample_hvg_expression(
    paths: DatasetPaths,
    shapes: dict,
    *,
    sample_cells_per_population: int,
) -> np.ndarray:
    """Return a deterministic sample of positive prepared HVG expression.

    The materialized ``hvg.float16.dat`` is the evaluator's exact
    library-size-normalized ``log1p`` representation.  Sampling rows evenly
    avoids a full 50+ GiB scan while still deriving bins from the real data
    distribution instead of the unrelated STATE soft-expression scale.
    """
    positives: list[np.ndarray] = []
    for population, shape in shapes.items():
        n_cells = int(shape["n_cells"])
        hvg_dim = int(shape["hvg_dim"])
        source = paths.prepared / str(population) / "hvg.float16.dat"
        if not source.is_file():
            raise FileNotFoundError(f"Graph preparation requires prepared HVG expression: {source}")
        values = np.memmap(
            source, dtype=np.float16, mode="r", shape=(n_cells, hvg_dim)
        )
        count = min(n_cells, int(sample_cells_per_population))
        rows = np.linspace(0, n_cells - 1, count, dtype=np.int64)
        sampled = np.asarray(values[rows], dtype=np.float32).reshape(-1)
        sampled = sampled[np.isfinite(sampled) & (sampled > 0)]
        if sampled.size:
            positives.append(sampled)
    if not positives:
        raise RuntimeError("Graph preparation could not find positive prepared HVG expression")
    return np.concatenate(positives)


def _prepare_expression_inputs(
    paths: DatasetPaths,
    output_dir: Path,
    hvg_count: int,
    *,
    input_modes: tuple[str, ...] | list[str] = ("official",),
    expression_bins: int = 128,
    expression_min: float | None = None,
    expression_max: float | None = None,
    sample_cells_per_population: int = 4096,
    overwrite: bool = False,
) -> tuple[dict, list[Path]]:
    """Prepare a scale-consistent binned-expression artifact.

    Both input forms consume the already-prepared HVG log
    expression.  ``official`` keeps the selected cells separate and
    ``validation`` pseudobulks them.  No STATE soft-expression is expanded
    into a fake full-gene profile and no duplicate dense memmap is written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    modes = tuple(dict.fromkeys(str(value).casefold() for value in input_modes))
    if not modes or set(modes) - {"official", "validation"}:
        raise ValueError("Expression modes must be official and/or validation")
    if int(expression_bins) < 3 or int(sample_cells_per_population) <= 0:
        raise ValueError("Expression bins and sample size must be positive")
    if (expression_min is None) != (expression_max is None):
        raise ValueError("expression_min and expression_max must be set together")
    if expression_min is not None and not float(expression_min) < float(expression_max):
        raise ValueError("expression_min must be smaller than expression_max")

    shapes = json.loads(
        (paths.prepared / "materialized_shapes.json").read_text(encoding="utf-8")
    )
    hvg_contract = load_hvg_contract(paths.prepared)
    hvg_fingerprint = hvg_contract["fingerprint"]
    dimensions = {int(value["hvg_dim"]) for value in shapes.values()}
    if dimensions != {int(hvg_count)}:
        raise ValueError(
            "Prepared HVG dimensions are inconsistent: "
            f"prepared={sorted(dimensions)}, requested={hvg_count}"
        )
    manifest = output_dir / "expression_manifest.json"
    boundaries_file = output_dir / "expression_bin_boundaries.float32.npy"
    reusable = not overwrite and manifest.is_file() and boundaries_file.is_file()
    if reusable:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if payload.get("hvg_fingerprint") != hvg_fingerprint:
            raise RuntimeError(
                "Existing expression bins use another HVG contract; rerun "
                "preparation with overwrite=True or use a new project"
            )
        reusable = (
            payload.get("format") in {"expression_bins_v1", "xpert_hvg_expression_v2"}
            and payload.get("gene_count") == int(hvg_count)
            and payload.get("expression_bins") == int(expression_bins)
            and set(modes).issubset(payload.get("input_modes", ()))
            and payload.get("sample_cells_per_population")
            == int(sample_cells_per_population)
            and payload.get("requested_expression_min") == expression_min
            and payload.get("requested_expression_max") == expression_max
            and payload.get("hvg_fingerprint") == hvg_fingerprint
        )
        if reusable:
            boundaries = np.load(boundaries_file)
            reusable = boundaries.shape == (int(expression_bins) - 1,)
        if reusable:
            return payload, [manifest, boundaries_file]

    positives = _sample_hvg_expression(
        paths,
        shapes,
        sample_cells_per_population=int(sample_cells_per_population),
    )
    if expression_min is None:
        # Match the reference digitize convention (zero maps to bin 1), while using
        # the actual positive log-expression distribution for all other bins.
        positive_boundaries = np.quantile(
            positives,
            np.linspace(0.0, 1.0, int(expression_bins) - 2),
            method="linear",
        ).astype(np.float32)
        boundaries = np.concatenate((np.zeros(1, dtype=np.float32), positive_boundaries))
        binning = "zero_plus_positive_expression_quantiles"
    else:
        boundaries = np.linspace(
            float(expression_min),
            float(expression_max),
            int(expression_bins) - 1,
            dtype=np.float32,
        )
        binning = "explicit_linear_range"
    if boundaries.shape != (int(expression_bins) - 1,) or np.any(np.diff(boundaries) < 0):
        raise RuntimeError("Expression bin boundaries are invalid")
    np.save(boundaries_file, boundaries)

    populations = {
        str(population): {
            "source_file": str(
                (paths.prepared / str(population) / "hvg.float16.dat").resolve()
            ),
            "shape": [int(shape["n_cells"]), int(shape["hvg_dim"])],
            "dtype": "float16",
        }
        for population, shape in shapes.items()
    }
    payload = {
        "format": "expression_bins_v1",
        "representation": "library_size_normalized_log1p_hvg_expression",
        "target_sum": sorted({float(value.get("target_sum", 10000.0)) for value in shapes.values()}),
        "gene_space": "prepared_hvg",
        "hvg_protocol": hvg_contract["protocol"],
        "hvg_fingerprint": hvg_fingerprint,
        "gene_count": int(hvg_count),
        "input_modes": sorted(modes),
        "mode_contracts": {
            "official": "per_cell_[batch,set_size,hvg]",
            "validation": "pseudobulk_mean_[batch,hvg]",
        },
        "expression_bins": int(expression_bins),
        "binning": binning,
        "bin_boundaries_file": boundaries_file.name,
        "expression_min": float(boundaries[0]),
        "expression_max": float(boundaries[-1]),
        "requested_expression_min": expression_min,
        "requested_expression_max": expression_max,
        "sample_cells_per_population": int(sample_cells_per_population),
        "sampled_positive_values": int(positives.size),
        "sampled_positive_min": float(positives.min()),
        "sampled_positive_max": float(positives.max()),
        "populations": populations,
    }
    manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload, [manifest, boundaries_file]


def _prepare_graph_assets(
    paths: DatasetPaths,
    output_dir: str | Path,
    *,
    reference_dir: str | Path | None = None,
    unimol_dir: str | Path | None = None,
    aliases_file: str | Path | None = None,
    score_threshold: int = 700,
    dds_threshold: float = 0.5,
    hidden_size: int = 256,
    layers: int = 3,
    epochs: int = 300,
    edges_per_type: int = 65_536,
    negative_samples: int = 5,
    temperature: float = 0.5,
    patience: int = 30,
    lr: float = 5e-4,
    weight_decay: float = 0.0,
    seed: int = 4242,
    max_atoms: int = 122,
    unimol_batch_size: int = 32,
    device: str | None = None,
    # Official sparse per-cell expression is the released default. The
    # validation pseudobulk contract remains available for ablations.
    input_formats: tuple[str, ...] | list[str] = ("official",),
    expression_bins: int = 128,
    expression_min: float | None = None,
    expression_max: float | None = None,
    expression_sample_cells: int = 1024,
    graph_pretraining_mode: str = "neighbor_loader",
    graph_num_neighbors: tuple[int, ...] | list[int] = (35, 20, 10),
    graph_batch_size: int = 4,
    include_expression: bool = True,
    overwrite: bool = False,
) -> tuple[dict, list[Path]]:
    output_dir = Path(output_dir)
    hvg_contract = load_hvg_contract(paths.prepared)
    hvg_fingerprint = hvg_contract["fingerprint"]
    raw_input_formats = (
        (input_formats,) if isinstance(input_formats, str) else input_formats
    )
    requested_formats = tuple(
        dict.fromkeys(str(value).casefold() for value in raw_input_formats)
    )
    aliases = {
        "official": "official",
        "dense_full_gene": "official",
        "validation": "validation",
        "sparse_tokens": "validation",
    }
    unknown_formats = sorted(set(requested_formats) - set(aliases))
    if unknown_formats or not requested_formats:
        raise ValueError("input_formats must contain official and/or validation")
    if int(expression_bins) < 3:
        raise ValueError("Expression bin configuration is invalid")
    if (expression_min is None) != (expression_max is None):
        raise ValueError("expression_min and expression_max must be set together")
    if expression_min is not None and not float(expression_min) < float(expression_max):
        raise ValueError("Expression bin range is invalid")
    input_modes = tuple(dict.fromkeys(aliases[value] for value in requested_formats))
    storage_formats = tuple(
        "per_cell_hvg" if value == "official" else "pseudobulk_hvg"
        for value in input_modes
    )
    graph_pretraining_mode = str(graph_pretraining_mode).casefold()
    if graph_pretraining_mode not in {"neighbor_loader", "full_graph"}:
        raise ValueError("graph_pretraining_mode must be neighbor_loader or full_graph")
    graph_num_neighbors = tuple(int(value) for value in graph_num_neighbors)
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
        raise FileNotFoundError("Graph reference assets are missing: " + ", ".join(missing))
    if min(
        hidden_size, layers, epochs, edges_per_type, negative_samples, patience,
        max_atoms, unimol_batch_size, expression_sample_cells, graph_batch_size,
        *graph_num_neighbors,
    ) <= 0:
        raise ValueError("Graph dimensions and schedules must be positive")
    if not 0 < dds_threshold <= 1 or score_threshold < 0 or temperature <= 0:
        raise ValueError("Graph thresholds are invalid")
    manifest_file = output_dir / "manifest.json"
    if manifest_file.is_file() and not overwrite:
        payload = json.loads(manifest_file.read_text(encoding="utf-8"))
        if payload.get("format") not in {"graph_bundle_v1", "map_xpert_assets_v4"}:
            raise RuntimeError(
                "Existing graph inputs use a legacy STATE-expression contract; "
                "rerun graph preparation with overwrite=True"
            )
        if payload.get("hvg_fingerprint") != hvg_fingerprint:
            raise RuntimeError(
                "Existing graph assets use another HVG contract; rerun graph "
                "preparation with overwrite=True or use a new project"
            )
        expression_outputs = []
        if include_expression:
            expression_payload, expression_outputs = _prepare_expression_inputs(
                paths,
                output_dir,
                len(payload["hvg_indices"]),
                input_modes=input_modes,
                expression_bins=int(expression_bins),
                expression_min=expression_min,
                expression_max=expression_max,
                sample_cells_per_population=int(expression_sample_cells),
                overwrite=False,
            )
            payload["expression"] = expression_payload
        payload["input_modes"] = sorted(
            set(payload.get("input_modes", ())) | set(input_modes)
        )
        payload["input_formats"] = sorted(
            set(payload.get("input_formats", ())) | set(storage_formats)
        )
        if include_expression:
            payload["expression_bins"] = int(expression_bins)
            payload["expression_min"] = expression_payload["expression_min"]
            payload["expression_max"] = expression_payload["expression_max"]
            payload["expression_bin_boundaries_file"] = expression_payload[
                "bin_boundaries_file"
            ]
        manifest_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload, outputs
    output_dir.mkdir(parents=True, exist_ok=True)

    symbols = _gene_symbols(paths)
    hvg_indices = _hvg_indices(paths, len(symbols))
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
        raise RuntimeError("No STRING PPI edges mapped to the prepared gene vocabulary")
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
    np.save(output_dir / "hvg_indices.int32.npy", hvg_indices)

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
        raise FileNotFoundError("Graph initialization requires the ESM2 gene asset")
    genes = _esm_matrix(esm_path, symbols)
    drugs = torch.from_numpy(unimol[:, 1, 2:].astype(np.float32))
    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(seed)
    np.random.seed(seed)
    encoder = _HeterogeneousEncoder(
        genes.shape[1], drugs.shape[1], int(hidden_size), int(layers)
    ).to(selected_device)
    graph_data = _graph_data(genes, drugs, ppi_edges, dti_edges, dds_edges)
    full_graph = graph_data.clone().to(selected_device)
    optimizer = torch.optim.Adam(
        encoder.parameters(), lr=float(lr), weight_decay=float(weight_decay)
    )
    generator = torch.Generator(device=selected_device).manual_seed(seed + 1)
    best_loss = float("inf")
    best_state = None
    stale_epochs = 0
    if graph_pretraining_mode == "neighbor_loader":
        graph_batches = _NeighborSubgraphLoader(
            graph_data,
            num_neighbors=graph_num_neighbors,
            batch_size=int(graph_batch_size),
            seed=int(seed) + 11,
        )
    else:
        graph_batches = (full_graph,)
    for epoch in range(int(epochs)):
        encoder.train()
        if hasattr(graph_batches, "set_epoch"):
            graph_batches.set_epoch(epoch)
        epoch_loss = 0.0
        batch_count = 0
        for graph_batch in graph_batches:
            graph_batch = graph_batch.to(selected_device)
            outputs = encoder(graph_batch.x_dict, graph_batch.edge_index_dict)
            loss = _graph_loss(
                outputs,
                graph_batch.edge_index_dict,
                int(edges_per_type),
                generator,
                negative_samples=int(negative_samples),
                temperature=float(temperature),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())
            batch_count += 1
        if not batch_count:
            raise RuntimeError("Graph NeighborLoader produced no batches")
        value = epoch_loss / batch_count
        if value < best_loss:
            best_loss = value
            best_state = {
                key: tensor.detach().cpu().clone()
                for key, tensor in encoder.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        if (epoch + 1) % max(int(epochs) // 10, 1) == 0:
            print(f"Graph epoch {epoch + 1}/{epochs}: loss={value:.6f}", flush=True)
        if stale_epochs >= int(patience):
            print(f"Graph early stopping at epoch {epoch + 1}", flush=True)
            break
    if best_state is None:
            raise RuntimeError("Graph pretraining did not produce a checkpoint")
    encoder.load_state_dict(best_state)
    encoder.to(selected_device).eval()
    with torch.no_grad():
        gene_output, drug_output = encoder(full_graph.x_dict, full_graph.edge_index_dict)
    # The graph is trained on the full prepared gene graph, while the model's cell
    # sequence is aligned to the evaluator's HVG order.  Persist only those
    # vectors for the Transformer and retain the full graph output separately
    # for provenance/debugging.
    np.save(
        output_dir / "ppi_gene_vectors_full.float32.npy",
        gene_output.cpu().numpy().astype(np.float32),
    )
    hvg_gene_output = gene_output[torch.as_tensor(hvg_indices, device=gene_output.device)]
    np.save(
        output_dir / "ppi_gene_vectors.float32.npy",
        hvg_gene_output.cpu().numpy().astype(np.float32),
    )
    hvg_position_indices = np.arange(len(hvg_indices), dtype=np.int32)
    np.save(output_dir / "model_gene_indices.int32.npy", hvg_position_indices)
    np.save(output_dir / "drug_hg_embeddings.float32.npy", drug_output.cpu().numpy().astype(np.float32))
    torch.save(
        {
            "format": "graph_bundle_v1",
            "state_dict": best_state,
            "gene_input_dim": int(genes.shape[1]),
            "drug_input_dim": int(drugs.shape[1]),
            "hidden_size": int(hidden_size),
            "layers": int(layers),
            "best_loss": best_loss,
            "negative_samples": int(negative_samples),
            "temperature": float(temperature),
        },
        output_dir / "graph_encoder.pt",
    )
    hvg_symbols = [symbols[int(index)] for index in hvg_indices]
    (output_dir / "gene_symbols.json").write_text(
        json.dumps(hvg_symbols, indent=2), encoding="utf-8"
    )
    (output_dir / "graph_gene_symbols.json").write_text(
        json.dumps(symbols, indent=2), encoding="utf-8"
    )
    (output_dir / "drug_vocabulary.json").write_text(
        json.dumps({"smiles": smiles, "names": drug_names}, indent=2), encoding="utf-8"
    )
    payload = {
        "artifact": "graph_assets",
        "format": "graph_bundle_v1",
        "input_modes": sorted(input_modes),
        "input_formats": sorted(storage_formats),
        "expression_bins": int(expression_bins),
        "gene_symbols": hvg_symbols,
        "full_gene_symbols": symbols,
        "gene_count": len(hvg_indices),
        "full_gene_count": len(symbols),
        "hvg_indices": hvg_indices.tolist(),
        "model_gene_space": "prepared_hvg",
        "hvg_protocol": hvg_contract["protocol"],
        "hvg_fingerprint": hvg_fingerprint,
        "smiles": smiles,
        "hidden_size": int(hidden_size),
        "max_atoms": int(max_atoms),
        "ppi_gene_vector_file": "ppi_gene_vectors.float32.npy",
        "ppi_gene_vector_full_file": "ppi_gene_vectors_full.float32.npy",
        "hvg_indices_file": "model_gene_indices.int32.npy",
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
            "negative_samples": int(negative_samples),
            "temperature": float(temperature),
            "patience": int(patience),
            "seed": int(seed),
            "pretraining_mode": graph_pretraining_mode,
            "num_neighbors": list(graph_num_neighbors),
            "batch_size": int(graph_batch_size),
        },
        "sources": {name: _file_record(value) for name, value in required.items()},
    }
    expression_outputs = []
    if include_expression:
        expression_payload, expression_outputs = _prepare_expression_inputs(
            paths,
            output_dir,
            len(hvg_indices),
            input_modes=input_modes,
            expression_bins=int(expression_bins),
            expression_min=expression_min,
            expression_max=expression_max,
            sample_cells_per_population=int(expression_sample_cells),
            overwrite=overwrite,
        )
        payload["expression"] = expression_payload
        payload["expression_min"] = expression_payload["expression_min"]
        payload["expression_max"] = expression_payload["expression_max"]
        payload["expression_bin_boundaries_file"] = expression_payload[
            "bin_boundaries_file"
        ]
    if aliases_path is not None:
        payload["sources"]["drug_aliases"] = _file_record(aliases_path)
    manifest_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    outputs = [
        manifest_file,
        output_dir / "ppi_edges.npz",
        output_dir / "dti_edges.npz",
        output_dir / "dds_edges.npz",
        output_dir / "hvg_indices.int32.npy",
        output_dir / "drug_unimol.float16.npy",
        output_dir / "ppi_gene_vectors.float32.npy",
        output_dir / "ppi_gene_vectors_full.float32.npy",
        output_dir / "model_gene_indices.int32.npy",
        output_dir / "drug_hg_embeddings.float32.npy",
        output_dir / "graph_encoder.pt",
        output_dir / "gene_symbols.json",
        output_dir / "graph_gene_symbols.json",
        output_dir / "drug_vocabulary.json",
        *expression_outputs,
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
    "_prepare_expression_inputs",
    "read_primekg_dti",
    "read_string_ppi",
]
