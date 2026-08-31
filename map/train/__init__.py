"""Dataset-independent MAP training tools."""

from .core import METHOD44, run
from .methods.registry import METHOD_REGISTRY

# One model selector is shared by MAP and every bundled method.
MODEL_REGISTRY = METHOD_REGISTRY

__all__ = ["METHOD44", "MODEL_REGISTRY", "METHOD_REGISTRY", "run"]
