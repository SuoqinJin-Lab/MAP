"""Shared normalization for single- and multi-component drug conditions."""

from __future__ import annotations

import ast
import json
import math
from collections.abc import Sequence
from typing import Any


def _list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(text)
            except (TypeError, ValueError, SyntaxError, json.JSONDecodeError):
                continue
            if isinstance(parsed, (list, tuple)):
                return list(parsed)
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return list(value)
    if hasattr(value, "tolist") and not isinstance(value, (bytes, bytearray)):
        converted = value.tolist()
        if isinstance(converted, (list, tuple)):
            return list(converted)
    return [value]


def normalize_components(smiles: Any, doses: Any) -> tuple[list[str], list[float]]:
    """Normalize one condition; scalar single-drug input is a valid special case."""
    smiles_values = _list(smiles)
    if smiles_values and isinstance(smiles_values[0], (list, tuple)):
        smiles_values = _list(smiles_values[0])
    dose_values = _list(doses)
    if dose_values and isinstance(dose_values[0], (list, tuple)):
        dose_values = _list(dose_values[0])
    if not dose_values:
        dose_values = [0.0] * len(smiles_values)
    if len(dose_values) == 1 and len(smiles_values) > 1:
        dose_values *= len(smiles_values)
    if len(smiles_values) != len(dose_values) or not smiles_values:
        raise ValueError("Drug components and doses must be non-empty and have equal length")
    output_smiles = [str(value).strip() for value in smiles_values]
    output_doses = [float(value) for value in dose_values]
    if any(not value for value in output_smiles):
        raise ValueError("Drug component SMILES must be non-empty")
    if any(not math.isfinite(value) or value < 0 for value in output_doses):
        raise ValueError("Drug component doses must be finite and non-negative")
    return output_smiles, output_doses


def normalize_batch_components(smiles: Any, doses: Any, batch_size: int) -> tuple[list[list[str]], list[list[float]]]:
    """Normalize the common DataLoader shapes into ``batch x components`` lists."""
    if isinstance(smiles, str):
        smiles = [smiles]
    values = list(smiles)
    # PyTorch's default collator transposes nested component lists.  Undo that
    # layout (components x batch) before normalizing each condition.
    if values and len(values) != batch_size and all(isinstance(value, (list, tuple)) for value in values):
        if all(len(value) == batch_size for value in values):
            values = [list(row) for row in zip(*values)]
    if len(values) == 1 and batch_size > 1:
        values *= batch_size
    if len(values) != batch_size:
        raise ValueError(f"Expected {batch_size} drug conditions, got {len(values)}")
    if hasattr(doses, "tolist"):
        doses = doses.tolist()
    dose_values = list(doses) if isinstance(doses, (list, tuple)) else [doses]
    if dose_values and len(dose_values) != batch_size and all(isinstance(value, (list, tuple)) for value in dose_values):
        if all(len(value) == batch_size for value in dose_values):
            dose_values = [list(row) for row in zip(*dose_values)]
    if len(dose_values) == 1 and batch_size > 1:
        dose_values *= batch_size
    if len(dose_values) != batch_size:
        raise ValueError(f"Expected {batch_size} dose conditions, got {len(dose_values)}")
    normalized = [normalize_components(value, dose) for value, dose in zip(values, dose_values)]
    return [item[0] for item in normalized], [item[1] for item in normalized]


def combination_key(smiles: Sequence[str]) -> str:
    """Stable order-independent key for a drug combination."""
    return "|".join(sorted(str(value).strip() for value in smiles))
