from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import Any

from . import pipeline


def run_stage(
    stage: str,
    *,
    contract: dict[str, Any],
    output_dir: str | Path,
    workers: int,
    esm_embeddings: str | Path | None = None,
    seed: int = 42,
    **kwargs: Any,
) -> None:
    """Stream Tahoe rows into MAP's generic prepared representation."""
    if stage not in {"stats", "hvg", "materialize"}:
        raise ValueError(f"Tahoe source reader does not support preparation stage {stage!r}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    values = {
        "raw_dir": str(contract["native_source"]),
        "output_dir": str(output_dir),
        "workers": int(workers),
        "seed": int(seed),
        **kwargs,
    }
    if stage == "stats":
        values["cell_lines"] = [str(value) for value in contract["populations"]]
        if esm_embeddings is None:
            raise ValueError("stats requires the frozen STATE ESM2 gene table")
    if esm_embeddings is not None:
        values["esm_embeddings"] = str(esm_embeddings)
    getattr(pipeline, f"run_{stage}")(Namespace(**values))


__all__ = ["run_stage"]
