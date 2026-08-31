from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TrainMean(nn.Module):
    """Condition-level mean expression estimated from the training split."""

    model_name = "trainmean"

    def __init__(self, hvg_dim: int, populations=()) -> None:
        super().__init__()
        self.hvg_dim = int(hvg_dim)
        self.populations = tuple(str(value) for value in populations)
        self.register_buffer("mean_expression", torch.zeros(self.hvg_dim))

    def fit(self, mean_expression: torch.Tensor) -> None:
        values = mean_expression.detach().reshape(-1).to(self.mean_expression)
        if values.numel() != self.hvg_dim:
            raise ValueError(
                f"TrainMean expected {self.hvg_dim} genes, received {values.numel()}"
            )
        self.mean_expression.copy_(values)

    def forward(self, batch: dict) -> torch.Tensor:
        batch_size = int(batch["control_hvg_vectors"].shape[0])
        return self.mean_expression.unsqueeze(0).expand(batch_size, -1)

    def loss(self, prediction: torch.Tensor, batch: dict) -> torch.Tensor:
        target = batch["condition_hvg_vectors"].float().mean(dim=1)
        return F.mse_loss(prediction.float(), target)

    @torch.no_grad()
    def predict_batch(self, batch: dict) -> torch.Tensor:
        return self(batch)

    def configuration(self) -> dict:
        return {
            "model": self.model_name,
            "hvg_dim": self.hvg_dim,
            "estimator": "equal-weight mean of training condition pseudobulks",
        }


__all__ = ["TrainMean"]
