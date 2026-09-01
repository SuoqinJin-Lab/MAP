from __future__ import annotations

import os
from pathlib import Path

from .paths import DatasetPaths


def asset(paths: DatasetPaths, relative: str) -> Path:
    path = paths.frozen_models / relative
    if not path.is_file():
        raise FileNotFoundError(f"Frozen model asset is missing: {path}")
    return path


def configure_frozen_runtime(paths: DatasetPaths) -> None:
    os.environ["MAP_MOLECULE_CKPT"] = str(asset(
        paths, "molecule_model.pth",
    ))
    os.environ["MAP_BART_VOCAB"] = str(asset(paths, "bart_vocab.txt"))
    os.environ["MAP_ESM2_EMBEDDINGS"] = str(asset(
        paths,
        "Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt",
    ))
