from __future__ import annotations

from .._common.paths import DatasetPaths
from .._common.runtime import asset
from .engine import METHOD44, train_map
from .baselines import MODEL_REGISTRY, train_baseline


def run(
    paths: DatasetPaths,
    *,
    regime: str,
    split_file,
    run_name: str | None = None,
    model: str = "map",
    **kwargs,
):
    model = str(model).casefold()
    if model != "map":
        if model not in MODEL_REGISTRY:
            choices = ", ".join(("map", *MODEL_REGISTRY))
            raise ValueError(f"Unknown training model {model!r}; choose from {choices}")
        return train_baseline(
            paths, model, regime, split_file=split_file, run_name=run_name, **kwargs
        )
    return train_map(
        paths,
        regime,
        asset(paths, "state/se600m.safetensors"),
        asset(paths, "state/Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt", "state/gene_embeddings_esm2.pt"),
        asset(paths, "mapkg/mapkg_encoder_v3.pt", "mapkg/mapkg_encoder.pt", "mapkg/mapkg_model.pt"),
        asset(paths, "mapkg/bart_vocab.txt"),
        split_file=split_file,
        run_name=run_name,
        **kwargs,
    )
