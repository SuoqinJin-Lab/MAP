"""Drug-material resolution shared by training and evaluation.

CMonge consumes the MOA drug material in the unseen-combination regime.
That decision is a train/eval-consistent rule, so it lives here once
instead of being re-implemented inside each entry point.
"""

from __future__ import annotations

from pathlib import Path

from .paths import DatasetPaths


def resolve_drug_representation(
    model: str, regime: str, requested: str | None = None
) -> str:
    """Default CMonge to the MOA drug representation in unseen-combination.

    An explicit ``requested`` value always wins; only the implicit default
    changes with the regime.  Other models pass ``requested`` through.
    """
    model = str(model).casefold()
    if model == "cmonge" and regime == "unseen_combination":
        return str(requested or "moa").casefold()
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

    CMonge's unseen-combination protocol consumes the split-local MOA
    material by default; everything else reads the shared prepared root
    unless an explicit ``material_dir`` is supplied.
    """
    model = str(model).casefold()
    representation = str(drug_representation or "").casefold()
    if (
        model == "cmonge"
        and regime == "unseen_combination"
        and representation in ("", "moa")
    ):
        return paths.split_material_dir(split_id, "drug_moa")
    if material_dir is not None:
        return Path(material_dir)
    return paths.prepared


__all__ = ["resolve_drug_representation", "resolve_material_dir"]
