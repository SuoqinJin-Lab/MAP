from __future__ import annotations

import json
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


# Fields that may legitimately change when an experiment is resumed without
# altering its semantics.  ``output_dir`` and ``resume`` are always accepted;
# resource-affecting options are accepted when resource changes are allowed.
_RESUME_ALWAYS_ALLOWED = {"resume", "output_dir"}
_RESUME_RESOURCE_ALLOWED = {"num_workers", "compile_mode"}


def validate_resume(
    checkpoint: dict,
    current: dict,
    *,
    allow_resource_change: bool = True,
    extra_allowed: tuple[str, ...] = (),
) -> None:
    """Reject a resume whose requested configuration would change semantics.

    Every trainer stores ``args`` in its checkpoint (MAP and all baselines),
    so one shared check covers the whole method family: parameters absent from
    the checkpoint or listed in the allowed sets never block a resume, while
    any semantic difference raises ``ValueError`` with the exact fields.
    """
    previous = dict(checkpoint.get("args") or {})
    permitted = set(_RESUME_ALWAYS_ALLOWED) | set(extra_allowed)
    if allow_resource_change:
        permitted.update(_RESUME_RESOURCE_ALLOWED)
    incompatible = {
        key: {"checkpoint": previous.get(key), "requested": value}
        for key, value in current.items()
        if key in previous and key not in permitted and previous.get(key) != value
    }
    if incompatible:
        raise ValueError(
            "Resume configuration changes experiment semantics: "
            + json.dumps(incompatible, sort_keys=True, default=str)
        )
