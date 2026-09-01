"""Identity helpers for the shared highly-variable-gene space."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


def hvg_fingerprint(payload: Mapping[str, Any]) -> str:
    """Hash the full gene-space contract, including its ordered HVG ids."""
    identity = {
        "protocol": payload.get("protocol"),
        "flavor": payload.get("flavor"),
        "batch_key": payload.get("batch_key"),
        "cells_per_batch": payload.get("cells_per_batch"),
        "n_top_genes": payload.get("n_top_genes"),
        "seurat_span": payload.get("seurat_span"),
        "merge_rule": payload.get("merge_rule"),
        "populations": list(payload.get("populations", ())),
        "sampled_cells_by_population": payload.get(
            "sampled_cells_by_population", {}
        ),
        "state_ids": list(payload.get("state_ids", ())),
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def load_hvg_contract(prepared: str | Path) -> dict[str, Any]:
    """Load and verify a fingerprinted HVG contract."""
    path = Path(prepared) / "hvg.json"
    if not path.is_file():
        raise FileNotFoundError(f"HVG contract is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    fingerprint = payload.get("fingerprint")
    if not fingerprint:
        raise RuntimeError(
            "HVG contract has no fingerprint (legacy preprocessing). "
            "Regenerate HVGs and all HVG-dependent material for this project."
        )
    expected = hvg_fingerprint(payload)
    if fingerprint != expected:
        raise RuntimeError(
            "HVG contract fingerprint does not match its contents; regenerate "
            "HVGs and all HVG-dependent material for this project."
        )
    return payload


__all__ = ["hvg_fingerprint", "load_hvg_contract"]
