from __future__ import annotations

import torch
from torch import nn


class ResidualProjector(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = hidden_dim or (input_dim + output_dim) // 2
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )
        self.residual_proj = nn.Linear(input_dim, output_dim)
        self.gate = nn.Sequential(nn.Linear(input_dim, output_dim), nn.Sigmoid())

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        gate = self.gate(values)
        return gate * self.mlp(values) + (1 - gate) * self.residual_proj(values)


class CellProjector(nn.Module):
    def __init__(
        self,
        n_layers: int,
        d_cell: int,
        d_out: int,
        hidden_dim: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_cell = d_cell
        hidden_dim = hidden_dim or max(d_cell, d_out)
        layers: list[nn.Module] = [nn.Linear(d_cell, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
        for _ in range(n_layers - 2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)])
        layers.append(nn.Linear(hidden_dim, d_out))
        self.projector = nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.projector(values)


class ProjectOut(nn.Module):
    def __init__(self, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values)
