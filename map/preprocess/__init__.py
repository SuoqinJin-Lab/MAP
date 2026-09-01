"""Handler-driven preprocessing into MAP's common data contract."""

from . import builtin
from .core import filter_conditions
from .pipeline import DatasetHandler, PreprocessPipeline, pipeline
from .selection import DataSelection
from .handlers.anndata import AnnDataHandler, anndata, op3
from .handlers.combosciplex import ComboSciPlexHandler, combosciplex

__all__ = [
    "builtin",
    "DataSelection",
    "filter_conditions",
    "DatasetHandler",
    "PreprocessPipeline",
    "pipeline",
    "AnnDataHandler",
    "ComboSciPlexHandler",
    "anndata",
    "combosciplex",
    "op3",
]
