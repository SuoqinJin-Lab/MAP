from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common import BaselineAssets


class PerturbationEncoder(nn.Module):
    def __init__(self, layer_sizes: list[int], latent_dim: int, dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for index, (input_dim, output_dim) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            layers.append(nn.Linear(input_dim, output_dim, bias=index != 0))
            if index != 0:
                layers.extend((nn.BatchNorm1d(output_dim), nn.LeakyReLU(0.3), nn.Dropout(dropout)))
        self.network = nn.Sequential(*layers) if layers else nn.Identity()
        self.mean = nn.Linear(layer_sizes[-1], latent_dim)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.mean(self.network(values))


class PerturbationDecoder(nn.Module):
    def __init__(
        self, input_dim: int, hidden_sizes: list[int], output_dim: int, dropout: float
    ) -> None:
        super().__init__()
        sizes = [input_dim, *hidden_sizes, output_dim * 2]
        layers: list[nn.Module] = []
        for index, (source, target) in enumerate(zip(sizes[:-1], sizes[1:])):
            last = index == len(sizes) - 2
            layers.append(nn.Linear(source, target, bias=last))
            if not last:
                layers.extend((nn.BatchNorm1d(target), nn.LeakyReLU(0.3), nn.Dropout(dropout)))
        self.network = nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        output = self.network(values)
        mean, dispersion = output.chunk(2, dim=1)
        return torch.cat((F.relu(mean), dispersion), dim=1)


class PerturbationAdaptor(nn.Module):
    def __init__(
        self, input_dim: int, hidden_sizes: list[int], output_dim: int, dropout: float
    ) -> None:
        super().__init__()
        sizes = [input_dim, *hidden_sizes]
        layers: list[nn.Module] = []
        for index, (source, target) in enumerate(zip(sizes[:-1], sizes[1:])):
            layers.append(nn.Linear(source, target, bias=index != 0))
            if index != 0:
                layers.extend((nn.BatchNorm1d(target), nn.LeakyReLU(0.3), nn.Dropout(dropout)))
        self.network = nn.Sequential(*layers) if layers else nn.Identity()
        self.combination = nn.Linear(sizes[-1], output_dim)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.combination(self.network(values))


class PGM(nn.Module):
    """PRnet perturbation-conditioned generative model."""

    def __init__(
        self,
        expression_dim: int,
        drug_dim: int = 1024,
        hidden_sizes: tuple[int, ...] = (128,),
        latent_dim: int = 64,
        adaptor_sizes: tuple[int, ...] = (128,),
        condition_dim: int = 64,
        noise_dim: int = 10,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.expression_dim = int(expression_dim)
        self.noise_dim = int(noise_dim)
        hidden = list(hidden_sizes)
        self.adaptor = PerturbationAdaptor(
            int(drug_dim), list(adaptor_sizes), int(condition_dim), float(dropout)
        )
        self.encoder = PerturbationEncoder(
            [self.expression_dim + int(condition_dim), *hidden], int(latent_dim), float(dropout)
        )
        self.decoder = PerturbationDecoder(
            int(latent_dim) + int(condition_dim) + self.noise_dim,
            list(reversed(hidden)),
            self.expression_dim,
            float(dropout),
        )

    def forward(
        self, control: torch.Tensor, drug: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        condition = self.adaptor(drug)
        latent = self.encoder(torch.cat((control, condition), dim=1))
        return self.decoder(torch.cat((latent, condition, noise), dim=1))


class PRnet(nn.Module):
    model_name = "prnet"

    def __init__(
        self,
        data_dir,
        hvg_dim: int,
        populations,
        *,
        hidden_sizes: tuple[int, ...] = (128,),
        latent_dim: int = 64,
        adaptor_sizes: tuple[int, ...] = (128,),
        condition_dim: int = 64,
        noise_dim: int = 10,
        dropout: float = 0.05,
        prediction_samples: int = 3,
    ) -> None:
        super().__init__()
        assets = BaselineAssets(data_dir, self.model_name)
        self.register_buffer(
            "fingerprints", assets.matrix("fcfp4_1024.float32.npy"), persistent=False
        )
        if self.fingerprints.shape != (len(assets.smiles), 1024):
            raise ValueError("PRnet FCFP4 matrix does not match the prepared drug vocabulary")
        self.smiles = assets.smiles
        self.smiles_to_index = assets.smiles_to_index
        self.populations = tuple(str(value) for value in populations)
        self.hvg_dim = int(hvg_dim)
        if self.hvg_dim <= 0:
            raise ValueError("hvg_dim must be positive")
        self.prediction_samples = int(prediction_samples)
        self.pgm = PGM(
            self.hvg_dim,
            hidden_sizes=hidden_sizes,
            latent_dim=latent_dim,
            adaptor_sizes=adaptor_sizes,
            condition_dim=condition_dim,
            noise_dim=noise_dim,
            dropout=dropout,
        )
        self._configuration = {
            "model": self.model_name,
            "hvg_dim": self.hvg_dim,
            "hidden_sizes": list(hidden_sizes),
            "latent_dim": int(latent_dim),
            "adaptor_sizes": list(adaptor_sizes),
            "condition_dim": int(condition_dim),
            "noise_dim": int(noise_dim),
            "dropout": float(dropout),
            "prediction_samples": self.prediction_samples,
        }

    def _drug_indices(self, smiles: list[str], device: torch.device) -> torch.Tensor:
        return torch.tensor(
            [self.smiles_to_index[str(value)] for value in smiles],
            device=device,
            dtype=torch.long,
        )

    def forward(self, batch: dict, *, noise: torch.Tensor | None = None) -> dict:
        control = batch["control_hvg_vectors"].float()
        batch_size, set_size, hvg_dim = control.shape
        indices = self._drug_indices(list(batch["drug_smiles"]), control.device)
        dose = torch.log10(batch["drug_conc"].float().clamp_min(0) + 1).unsqueeze(1)
        drug = (self.fingerprints[indices] * dose).repeat_interleave(set_size, dim=0)
        flattened = control.reshape(-1, hvg_dim)
        if noise is None:
            noise = torch.randn(
                flattened.shape[0], self.pgm.noise_dim, device=control.device, dtype=control.dtype
            )
        output = self.pgm(flattened, drug, noise)
        mean, raw_variance = output.chunk(2, dim=1)
        return {
            "mean": mean.reshape(batch_size, set_size, hvg_dim),
            "variance": (F.softplus(raw_variance) + 1e-6).reshape(
                batch_size, set_size, hvg_dim
            ),
        }

    def loss(self, output: dict, batch: dict) -> torch.Tensor:
        distribution = torch.distributions.Normal(
            output["mean"].float(), output["variance"].float().sqrt()
        )
        return -distribution.log_prob(batch["condition_hvg_vectors"].float()).mean()

    @torch.no_grad()
    def predict_batch(self, batch: dict) -> torch.Tensor:
        samples = [self(batch)["mean"].mean(dim=1) for _ in range(self.prediction_samples)]
        return torch.stack(samples).mean(dim=0)

    def configuration(self) -> dict:
        return dict(self._configuration)


__all__ = ["PGM", "PRnet"]
