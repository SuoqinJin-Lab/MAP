from __future__ import annotations


_HVG_FIELDS = frozenset({"condition_hvg_vectors", "control_hvg_vectors"})

METHOD_DATA_FIELDS = {
    "prnet": _HVG_FIELDS,
    "chemcpa": _HVG_FIELDS,
    "trainmean": _HVG_FIELDS,
    "crisp": frozenset({
        "condition_hvg_vectors",
        "condition_rows",
        "control_embeddings",
        "control_hvg_vectors",
    }),
    # XPert official expands sparse per-cell STATE inputs into the complete
    # prepared gene vocabulary. Validation keeps the evaluator-scale HVGs.
    "xpert": _HVG_FIELDS,
    "cmonge": _HVG_FIELDS,
}


def method_data_fields(
    model: str,
    *,
    evaluation: bool = False,
    input_mode: str | None = None,
) -> frozenset[str]:
    """Return only the prepared cell fields consumed by one method."""

    try:
        normalized_model = str(model).casefold()
        fields = METHOD_DATA_FIELDS[normalized_model]
    except KeyError as error:
        raise ValueError(f"Unknown method: {model}") from error
    if normalized_model == "xpert":
        mode = str(input_mode or "official").casefold()
        if mode not in {"official", "validation"}:
            raise ValueError(f"Unknown XPert input mode: {input_mode}")
        if mode == "official":
            fields = fields | frozenset({
                "condition_gene_ids",
                "condition_expressions",
                "control_gene_ids",
                "control_expressions",
            })
    if not evaluation:
        return fields
    evaluation_fields = fields | _HVG_FIELDS
    if normalized_model == "crisp":
        evaluation_fields |= {"control_embeddings"}
    return evaluation_fields


__all__ = ["METHOD_DATA_FIELDS", "method_data_fields"]
