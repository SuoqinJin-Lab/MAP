from ...._common.runtime import asset
from .engine import train_map


def run(paths, *, regime, split_file, run_name=None, **kwargs):
    """Run MAP through the shared method contract."""
    return train_map(
        paths,
        regime,
        asset(paths, "state/se600m.safetensors"),
        asset(
            paths,
            "state/Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt",
            "state/gene_embeddings_esm2.pt",
        ),
        asset(
            paths,
            "mapkg/mapkg_encoder_v3.pt",
            "mapkg/mapkg_encoder.pt",
            "mapkg/mapkg_model.pt",
        ),
        asset(paths, "mapkg/bart_vocab.txt"),
        split_file=split_file,
        run_name=run_name,
        **kwargs,
    )

__all__ = ["run", "train_map"]
