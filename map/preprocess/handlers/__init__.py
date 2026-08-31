"""Dataset handlers implementing the common preprocessing boundary.

Each handler owns only native-layout access and emits the three MAP entities;
the generic pipeline in :mod:`map.preprocess.pipeline` performs all shared
selection and preparation steps.
"""

from .anndata import AnnDataHandler, anndata, combosciplex, op3
from .atlas import AtlasHandler
from .tahoe import TahoeHandler, TahoePaths, tahoe

__all__ = [
    "AnnDataHandler",
    "AtlasHandler",
    "TahoePaths",
    "TahoeHandler",
    "anndata",
    "combosciplex",
    "op3",
    "tahoe",
]
