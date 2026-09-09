"""Split-rule vocabulary shared by preparation, CLI and evaluation.

One place for the rule names every layer switches on; adding a new split
rule must not touch argparse choices scattered across the repository.
"""

from __future__ import annotations

# Rules accepted by ``map preparation create-split``.
TRAIN_RULES: tuple[str, ...] = ("unprofiled_drug", "unseen_combination", "combosciplex")

# Pseudo-rule that generates every TRAIN_RULE split in one invocation.
ALL_RULE: str = "all"

# Every rule the split generator understands.
GENERATE_RULES: tuple[str, ...] = (ALL_RULE,) + TRAIN_RULES

# Regimes accepted by the evaluator (every training rule is evaluable).
EVAL_REGIMES: tuple[str, ...] = TRAIN_RULES

# Splits whose drug context is the MoA material (MoA features are only
# defined for these rules).
MOA_SPLIT_RULES: frozenset[str] = frozenset({"unseen_combination", "combosciplex"})


def require_train_rule(rule: str) -> str:
    if rule not in TRAIN_RULES:
        raise ValueError(f"Unknown split rule: {rule}; choose from {TRAIN_RULES}")
    return rule


__all__ = [
    "ALL_RULE", "EVAL_REGIMES", "GENERATE_RULES", "MOA_SPLIT_RULES",
    "TRAIN_RULES", "require_train_rule",
]
