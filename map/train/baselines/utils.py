from __future__ import annotations


_HVG_FIELDS = frozenset({"condition_hvg_vectors", "control_hvg_vectors"})

BASELINE_DATA_FIELDS = {
    "prnet": _HVG_FIELDS,
    "chemcpa": _HVG_FIELDS,
    "trainmean": _HVG_FIELDS,
    "crisp": frozenset({"condition_hvg_vectors", "condition_rows"}),
    "xpert": frozenset({
        "control_gene_ids",
        "control_expressions",
        "condition_gene_ids",
        "condition_expressions",
    }),
    "cmonge": _HVG_FIELDS,
}


def baseline_data_fields(model: str, *, evaluation: bool = False) -> frozenset[str]:
    """Return only the materialized cell fields consumed by a baseline."""

    try:
        fields = BASELINE_DATA_FIELDS[str(model).casefold()]
    except KeyError as error:
        raise ValueError(f"Unknown baseline model: {model}") from error
    if not evaluation:
        return fields
    evaluation_fields = fields | _HVG_FIELDS
    if str(model).casefold() == "crisp":
        evaluation_fields |= {"control_embeddings"}
    return evaluation_fields


__all__ = ["BASELINE_DATA_FIELDS", "baseline_data_fields"]
