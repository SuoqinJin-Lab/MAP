from __future__ import annotations

import json

from .._common.contracts import load_contract
from .._common.paths import DatasetPaths
from .._common.runtime import asset
from .embeddings import (
    embed_drugs as _embed_drugs,
    embed_genes as _embed_genes,
    embed_state as _embed_state,
    merge_perturbed_cell_embeddings as _merge_state,
    merge_static_tokens as _merge_static,
)
from .processing import (
    build_condition_index as _build_index,
    generate_splits as _create_split,
)
from .validate import validate_preparation


def build_sampling_index(
    paths: DatasetPaths, *, workers: int = 8, overwrite: bool = False
):
    return _build_index(paths, workers=workers, overwrite=overwrite)


def create_split(
    paths: DatasetPaths,
    *,
    rule: str,
    external_test_size: int | float,
    internal_test_fraction: float = 0.2,
    seed: int = 42,
    disjoint_external_drugs: bool = True,
    external_drugs: list[str] | tuple[str, ...] | None = None,
    overwrite: bool = False,
):
    return _create_split(
        paths,
        rule=rule,
        external_test_size=external_test_size,
        internal_test_fraction=internal_test_fraction,
        seed=seed,
        disjoint_external_drugs=disjoint_external_drugs,
        external_drugs=external_drugs,
        overwrite=overwrite,
    )


def precache_gene_tokens(paths: DatasetPaths, **kwargs):
    return _embed_genes(
        paths,
        asset(paths, "state/Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt", "state/gene_embeddings_esm2.pt"),
        asset(paths, "mapkg/mapkg_encoder_v3.pt", "mapkg/mapkg_encoder.pt", "mapkg/mapkg_model.pt"),
        asset(paths, "mapkg/bart_vocab.txt"),
        **kwargs,
    )


def precache_drug_tokens(paths: DatasetPaths, **kwargs):
    return _embed_drugs(
        paths,
        asset(paths, "state/Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt", "state/gene_embeddings_esm2.pt"),
        asset(paths, "mapkg/mapkg_encoder_v3.pt", "mapkg/mapkg_encoder.pt", "mapkg/mapkg_model.pt"),
        asset(paths, "mapkg/bart_vocab.txt"),
        **kwargs,
    )


def merge_knowledge_tokens(paths: DatasetPaths, **kwargs):
    return _merge_static(paths, **kwargs)


def precache_state_embeddings(paths: DatasetPaths, *, populations=None, **kwargs):
    return _embed_state(
        paths,
        asset(paths, "state/se600m.safetensors"),
        asset(paths, "state/Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt", "state/gene_embeddings_esm2.pt"),
        populations=tuple(populations or _populations(paths)),
        **kwargs,
    )


def merge_state_embeddings(paths: DatasetPaths, *, populations=None, **kwargs):
    return _merge_state(
        paths, populations=tuple(populations or _populations(paths)), **kwargs
    )


def validate(paths: DatasetPaths, *, split_files, **kwargs):
    return validate_preparation(paths, split_files=tuple(split_files), **kwargs)


def _populations(paths: DatasetPaths) -> tuple[str, ...]:
    shapes = paths.prepared / "materialized_shapes.json"
    if shapes.is_file():
        return tuple(json.loads(shapes.read_text(encoding="utf-8")))
    payload = load_contract(paths.workspace)
    populations = tuple(str(value) for value in payload.get("populations", ()))
    if not populations:
        raise ValueError("The project data contract contains no populations")
    return populations
