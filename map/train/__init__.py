"""Dataset-independent MAP training tools."""

from .core import METHOD44, run
from .baselines import MODEL_REGISTRY

__all__ = ["METHOD44", "MODEL_REGISTRY", "run"]
