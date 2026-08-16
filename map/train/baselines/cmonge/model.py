from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common import BaselineAssets

try:
    from geomloss import SamplesLoss
except ImportError as error:  # pragma: no cover - exercised by environment setup
    raise ImportError(
        "CMonge requires geomloss; install requirements-map-baselines.txt"
    ) from error


def _mlp(sizes: list[int], *, activate_last: bool = False) -> nn.Sequential:
    layers: list[nn.Module] = []
    for index, (source, target) in enumerate(zip(sizes[:-1], sizes[1:])):
        layers.append(nn.Linear(source, target))
        if activate_last or index < len(sizes) - 2:
            layers.append(nn.GELU())
    return nn.Sequential(*layers)


class ExpressionAutoencoder(nn.Module):
    def __init__(self, expression_dim: int, width: int = 512, latent_dim: int = 50):
        super().__init__()
        self.encoder = _mlp([int(expression_dim), int(width), int(latent_dim)])
        self.decoder = _mlp([int(latent_dim), int(width), int(expression_dim)])

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        return self.encoder(values)

    def decode(self, values: torch.Tensor) -> torch.Tensor:
        return self.decoder(values)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(values))


class ConditionalTransport(nn.Module):
    def __init__(self, latent_dim: int, context_dim: int, hidden_sizes: tuple[int, ...]):
        super().__init__()
        self.network = _mlp(
            [int(latent_dim) + int(context_dim) + 1, *map(int, hidden_sizes), int(latent_dim)]
        )

    def forward(self, source: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        expanded = context[:, None, :].expand(-1, source.shape[1], -1)
        return self.network(torch.cat((source, expanded), dim=-1))


class CMonge(nn.Module):
    """Conditional Monge Gap on the shared single-cell HVG contract."""

    model_name = "cmonge"

    def __init__(
        self,
        data_dir,
        hvg_dim: int,
        populations=(),
        *,
        ae_width: int = 512,
        latent_dim: int = 50,
        context_dim: int = 50,
        hidden_sizes: tuple[int, ...] = (64, 64, 64, 64),
        fitting_epsilon: float = 1.0,
        regularizer_epsilon: float = 1e-2,
        monge_gap_weight: float = 1e-2,
    ) -> None:
        super().__init__()
        if int(hvg_dim) != 2000:
            raise ValueError(
                f"CMonge requires exactly 2000 HVGs; received hvg_dim={hvg_dim}"
            )
        assets = BaselineAssets(data_dir, self.model_name)
        filename = assets.model_manifest.get("matrix", "rdkit2d.float32.npy")
        descriptors = assets.matrix(filename)
        if descriptors.ndim != 2 or descriptors.shape[0] != len(assets.smiles):
            raise ValueError("CMonge RDKit descriptors do not match the drug vocabulary")
        if min(hvg_dim, ae_width, latent_dim, context_dim, *hidden_sizes) <= 0:
            raise ValueError("CMonge dimensions must be positive")
        if fitting_epsilon <= 0 or regularizer_epsilon <= 0 or monge_gap_weight < 0:
            raise ValueError("CMonge epsilon values must be positive and lambda non-negative")

        self.register_buffer("descriptors", descriptors, persistent=False)
        self.smiles = assets.smiles
        self.smiles_to_index = assets.smiles_to_index
        self.populations = tuple(str(value) for value in populations)
        self.hvg_dim = int(hvg_dim)
        self.ae_width = int(ae_width)
        self.latent_dim = int(latent_dim)
        self.context_dim = int(context_dim)
        self.hidden_sizes = tuple(int(value) for value in hidden_sizes)
        self.fitting_epsilon = float(fitting_epsilon)
        self.regularizer_epsilon = float(regularizer_epsilon)
        self.monge_gap_weight = float(monge_gap_weight)

        descriptor_dim = int(descriptors.shape[1])
        self.autoencoder = ExpressionAutoencoder(
            self.hvg_dim, self.ae_width, self.latent_dim
        )
        self.drug_encoder = nn.Linear(descriptor_dim, self.context_dim)
        self.dose_encoder = nn.Linear(descriptor_dim + 1, 1)
        self.transport = ConditionalTransport(
            self.latent_dim, self.context_dim, self.hidden_sizes
        )
        self.fitting_loss = SamplesLoss(
            loss="sinkhorn",
            p=2,
            blur=math.sqrt(self.fitting_epsilon),
            debias=True,
            backend="tensorized",
        )
        self.regularized_ot = SamplesLoss(
            loss="sinkhorn",
            p=2,
            blur=math.sqrt(self.regularizer_epsilon),
            debias=False,
            backend="tensorized",
        )

    def _drug_indices(self, smiles: list[str], device: torch.device) -> torch.Tensor:
        try:
            values = [self.smiles_to_index[str(value)] for value in smiles]
        except KeyError as error:
            raise KeyError(f"CMonge has no descriptor for SMILES {error.args[0]!r}") from error
        return torch.tensor(values, device=device, dtype=torch.long)

    def condition(self, batch: dict, device: torch.device) -> torch.Tensor:
        descriptors = self.descriptors.index_select(
            0, self._drug_indices(batch["drug_smiles"], device)
        ).to(device)
        dose = torch.as_tensor(batch["drug_conc"], device=device, dtype=descriptors.dtype)
        if (dose <= 0).any():
            raise ValueError("CMonge requires positive doses for log(dose) conditioning")
        log_dose = dose.log().reshape(-1, 1)
        return torch.cat(
            (
                self.drug_encoder(descriptors),
                self.dose_encoder(torch.cat((descriptors, log_dose), dim=1)),
            ),
            dim=1,
        )

    def freeze_autoencoder(self) -> None:
        self.autoencoder.requires_grad_(False)
        self.autoencoder.eval()

    def forward(self, batch, *, stage: str = "transport"):
        if stage == "autoencoder":
            values = batch if isinstance(batch, torch.Tensor) else torch.cat(
                (batch["control_hvg_vectors"], batch["condition_hvg_vectors"]), dim=1
            ).reshape(-1, self.hvg_dim)
            return self.autoencoder(values.float())
        if stage != "transport":
            raise ValueError(f"Unknown CMonge stage: {stage}")

        control = batch["control_hvg_vectors"].float()
        target = batch["condition_hvg_vectors"].float()
        with torch.no_grad():
            source_latent = self.autoencoder.encode(control)
            target_latent = self.autoencoder.encode(target)
        context = self.condition(batch, control.device)
        predicted_latent = source_latent + self.transport(source_latent, context)
        prediction = self.autoencoder.decode(predicted_latent)
        return {
            "source_latent": source_latent,
            "target_latent": target_latent,
            "predicted_latent": predicted_latent,
            "prediction": prediction,
        }

    @staticmethod
    def autoencoder_loss(reconstruction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(reconstruction.float(), target.float())

    def loss(self, output: dict, batch: dict | None = None):
        del batch
        source = output["source_latent"]
        predicted = output["predicted_latent"]
        target = output["target_latent"]
        fitting = self.fitting_loss(predicted, target).mean()
        paired_cost = 0.5 * (source - predicted).square().sum(dim=-1).mean(dim=-1)
        monge_gap = (paired_cost - self.regularized_ot(source, predicted)).mean()
        total = fitting + self.monge_gap_weight * monge_gap
        return total, {"sinkhorn": fitting.detach(), "monge_gap": monge_gap.detach()}

    @torch.no_grad()
    def transport_batch(self, batch: dict) -> torch.Tensor:
        return self(batch)["prediction"]

    @torch.no_grad()
    def predict_batch(self, batch: dict) -> torch.Tensor:
        return self.transport_batch(batch).mean(dim=1)

    def configuration(self) -> dict:
        return {
            "model": self.model_name,
            "hvg_dim": self.hvg_dim,
            "descriptor_dim": int(self.descriptors.shape[1]),
            "ae_width": self.ae_width,
            "latent_dim": self.latent_dim,
            "context_dim": self.context_dim,
            "hidden_sizes": list(self.hidden_sizes),
            "fitting_epsilon": self.fitting_epsilon,
            "regularizer_epsilon": self.regularizer_epsilon,
            "monge_gap_weight": self.monge_gap_weight,
        }


__all__ = ["CMonge", "ConditionalTransport", "ExpressionAutoencoder"]
