"""Drug-material resolution shared by training and evaluation.

The per-method defaults are declared as data (``_DRUG_REPRESENTATION_RULES``
and ``_SPLIT_MATERIAL_RULES``) instead of ``if model == ...`` branches, so a
new method with a regime-specific drug material only adds one rule row.
"""

from __future__ import annotations

from pathlib import Path

from .paths import DatasetPaths

# model -> regime -> default drug representation, applied only when the
# caller does not request an explicit representation.
_DRUG_REPRESENTATION_RULES = {
    "cmonge": {"unseen_combination": "moa"},
}

# model -> regime -> (default representation, split-local material subdir).
# The entry is used only when the caller does not request an explicit
# representation (or requests exactly the rule's representation).
_SPLIT_MATERIAL_RULES = {
    "cmonge": {"unseen_combination": ("moa", "drug_moa")},
}


def resolve_drug_representation(
    model: str, regime: str, requested: str | None = None
) -> str:
    """Default a method's drug representation by (model, regime) rule.

    An explicit ``requested`` value always wins; only the implicit default
    changes with the regime.  Other models pass ``requested`` through.
    """
    model = str(model).casefold()
    rule = _DRUG_REPRESENTATION_RULES.get(model, {}).get(regime)
    if rule is not None:
        return str(requested or rule).casefold()
    return str(requested or "rdkit").casefold()


def resolve_material_dir(
    paths: DatasetPaths,
    split_id: str,
    model: str,
    regime: str,
    *,
    drug_representation: str | None = None,
    material_dir: str | Path | None = None,
) -> Path:
    """Resolve where a method reads its neutral artifacts from.

    A (model, regime) entry in ``_SPLIT_MATERIAL_RULES`` makes that
    split-local material the default (CMonge's unseen-combination protocol
    consumes the split-local MOA material); everything else reads the shared
    prepared root unless an explicit ``material_dir`` is supplied.
    """
    model = str(model).casefold()
    representation = str(drug_representation or "").casefold()
    rule = _SPLIT_MATERIAL_RULES.get(model, {}).get(regime)
    if rule is not None and representation in ("", rule[0]):
        return paths.split_material_dir(split_id, rule[1])
    if material_dir is not None:
        return Path(material_dir)
    return paths.prepared


__all__ = ["resolve_drug_representation", "resolve_material_dir"]
