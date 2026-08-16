from __future__ import annotations

import torch
from torch import nn


class LatentToGeneDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        gene_dim: int,
        hidden_dims: list[int],
        dropout: float = 0.1,
    ):
        super().__init__()
        dimensions = [latent_dim, *hidden_dims, gene_dim]
        layers: list[nn.Module] = []
        for index, (source, target) in enumerate(zip(dimensions[:-1], dimensions[1:])):
            layers.append(nn.Linear(source, target))
            if index < len(dimensions) - 2:
                layers.extend([nn.LayerNorm(target), nn.GELU(), nn.Dropout(dropout)])
        layers.append(nn.ReLU())
        self.decoder = nn.Sequential(*layers)

    @property
    def gene_dim(self) -> int:
        for module in reversed(self.decoder):
            if isinstance(module, nn.Linear):
                return module.out_features
        raise RuntimeError("Decoder has no output layer")

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.decoder(values)
