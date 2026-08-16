from __future__ import annotations

import os
from pathlib import Path

from .paths import DatasetPaths


def asset(paths: DatasetPaths, *relative: str) -> Path:
    candidates = [paths.frozen / name for name in relative]
    # Released assets are commonly distributed either in state/mapkg
    # subdirectories or together in one frozen-model directory.
    candidates.extend(paths.frozen / Path(name).name for name in relative)
    return next((path for path in candidates if path.is_file()), candidates[0])


def configure_frozen_runtime(paths: DatasetPaths) -> None:
    os.environ["MAP_MOLECULE_CKPT"] = str(asset(
        paths, "mapkg/molecule_model.pth", "mapkg/molecule_encoder.pth",
    ))
    os.environ["MAP_BART_VOCAB"] = str(asset(paths, "mapkg/bart_vocab.txt"))
    os.environ["MAP_ESM2_EMBEDDINGS"] = str(asset(
        paths,
        "state/Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt",
        "state/gene_embeddings_esm2.pt",
    ))
