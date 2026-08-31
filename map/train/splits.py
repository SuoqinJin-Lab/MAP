"""Resolve immutable split objects from a project contract."""

from __future__ import annotations

import json
from pathlib import Path

from .._common.paths import DatasetPaths


def resolve_split(
    paths: DatasetPaths, regime: str, split_file: str | Path | None
) -> tuple[Path, dict]:
    """Return one split file and verify it belongs to the requested regime."""
    if split_file is None:
        raise ValueError("split_file is required; projects may contain multiple splits")
    requested = Path(split_file)
    candidates = [requested]
    if not requested.is_absolute():
        candidates.extend((
            paths.workspace / requested,
            paths.splits / requested,
        ))
        if len(requested.parts) == 1:
            candidates.append(paths.split_file(str(split_file)))
    path = next((candidate for candidate in candidates if candidate.is_file()), candidates[-1])
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("rule", payload.get("regime")) != regime:
        raise ValueError(
            f"Split rule {payload.get('rule', payload.get('regime'))!r} does not match {regime!r}"
        )
    split_id = str(payload.get("split_id", path.parent.name))
    if path.resolve() != paths.split_file(split_id).resolve():
        raise ValueError(
            f"Split must be stored as splits/{split_id}/split.json, got {path}"
        )
    return path.resolve(), payload


__all__ = ["resolve_split"]
