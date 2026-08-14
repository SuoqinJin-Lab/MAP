from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping


def slug(value: Any, max_length: int = 72) -> str:
    """Return a filesystem-safe, human-readable identifier fragment."""
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value).strip()).strip("-_.")
    text = re.sub(r"-+", "-", text).lower() or "artifact"
    if len(text) <= max_length:
        return text
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    return f"{text[:max_length - 10].rstrip('-_')}--{digest}"


def number_token(value: int | float) -> str:
    """Encode a number without characters that are awkward in file names."""
    if isinstance(value, int) or float(value).is_integer():
        return str(int(value))
    return format(float(value), ".8g").replace("-", "m").replace(".", "p")


def size_token(value: int | float) -> str:
    prefix = "n" if isinstance(value, int) or float(value) >= 1 else "p"
    return prefix + number_token(value)


def compact_identifier(parts: list[str], payload: Mapping[str, Any], max_length: int = 180) -> str:
    """Join readable fragments and add a digest only when the name gets long."""
    readable = "__".join(slug(part, 96).replace("__", "-") for part in parts if str(part).strip())
    if len(readable) <= max_length:
        return readable
    digest = config_digest(payload, 10)
    return f"{readable[:max_length - 12].rstrip('-_')}__{digest}"


def split_identifier(
    rule: str,
    test_size: int | float,
    val_size: int | float,
    seed: int,
    *,
    disjoint_test_drugs: bool = True,
) -> str:
    parts = [
        slug(rule, 30),
        f"test-{size_token(test_size)}",
        f"val-{size_token(val_size)}",
        f"seed-{int(seed)}",
    ]
    if rule == "unseen_combination":
        parts.append("testdisjoint" if disjoint_test_drugs else "testindependent")
    return "__".join(parts)


def preparation_identifier(
    *,
    populations: list[str] | tuple[str, ...],
    n_top_genes: int,
    num_gene_tokens: int,
    target_sum: float,
    log1p: bool,
) -> str:
    payload = {
        "populations": list(populations),
        "n_top_genes": int(n_top_genes),
        "num_gene_tokens": int(num_gene_tokens),
        "target_sum": float(target_sum),
        "log1p": bool(log1p),
    }
    return compact_identifier(
        [
            f"populations-{len(populations)}",
            f"hvg-{int(n_top_genes)}",
            f"tokens-{int(num_gene_tokens)}",
            f"sum-{number_token(target_sum)}",
            "log1p" if log1p else "nolog1p",
            f"cfg-{config_digest(payload, 8)}",
        ],
        payload,
        max_length=120,
    )


def training_identifier(
    regime: str,
    split_id: str,
    *,
    set_size: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    lr: float,
    seed: int,
    extra: Mapping[str, Any] | None = None,
) -> str:
    payload = {
        "regime": regime,
        "split_id": split_id,
        "set_size": int(set_size),
        "batch_size": int(batch_size),
        "gradient_accumulation_steps": int(gradient_accumulation_steps),
        "lr": float(lr),
        "seed": int(seed),
        **dict(extra or {}),
    }
    core_parts = [
        regime,
        f"split-{file_identity(split_id)}",
        f"model-se{int(set_size)}-bs{int(batch_size)}-ga{int(gradient_accumulation_steps)}-lr{number_token(lr)}",
        f"trainseed-{int(seed)}",
    ]
    if extra:
        core_parts.append(f"cfg-{config_digest(payload, 8)}")
    return compact_identifier(core_parts, payload)


def evaluation_identifier(
    regime: str,
    split_id: str,
    checkpoint: str | Path,
    *,
    seeds: tuple[int, ...] | list[int],
    set_size: int,
    deg_top_k: int,
    deg_fdr: float,
    extra: Mapping[str, Any] | None = None,
) -> str:
    payload = {
        "regime": regime,
        "split_id": split_id,
        "checkpoint": str(checkpoint),
        "seeds": [int(seed) for seed in seeds],
        "set_size": int(set_size),
        "deg_top_k": int(deg_top_k),
        "deg_fdr": float(deg_fdr),
        **dict(extra or {}),
    }
    seed_text = "-".join(str(int(seed)) for seed in seeds)
    checkpoint_path = Path(checkpoint)
    checkpoint_id = file_identity(checkpoint_path)
    if checkpoint_path.parent.name:
        checkpoint_id = slug(f"{checkpoint_path.parent.name}-{checkpoint_id}", 96)
    core_parts = [
        regime,
        f"split-{file_identity(split_id)}",
        f"ckpt-{checkpoint_id}",
        f"eval-se{int(set_size)}-seeds{seed_text}-deg{int(deg_top_k)}-fdr{number_token(deg_fdr)}",
    ]
    if extra:
        core_parts.append(f"cfg-{config_digest(payload, 8)}")
    return compact_identifier(core_parts, payload)


def file_identity(path: str | Path) -> str:
    return slug(Path(path).stem, 96)


def config_digest(payload: Mapping[str, Any], length: int = 8) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:length]
