from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common import MethodAssets


class MLP(nn.Module):
    def __init__(
        self,
        sizes: list[int],
        *,
        batch_norm: bool = True,
        last_activation: str = "linear",
    ) -> None:
        super().__init__()
        layers = OrderedDict()
        position = 0
        for index, (source, target) in enumerate(zip(sizes[:-1], sizes[1:])):
            last = index == len(sizes) - 2
            layers[str(position)] = nn.Linear(source, target)
            position += 1
            if batch_norm and not last:
                layers[str(position)] = nn.BatchNorm1d(target)
                position += 1
            if not last:
                layers[str(position)] = nn.ReLU()
                position += 1
        self.network = nn.Sequential(layers)
        if last_activation not in {"linear", "relu"}:
            raise ValueError("last_activation must be 'linear' or 'relu'")
        self.last_activation = last_activation

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        output = self.network(values)
        if self.last_activation == "relu":
            mean, dispersion = output.chunk(2, dim=1)
            return torch.cat((F.relu(mean), dispersion), dim=1)
        return output


class GeneralizedSigmoid(nn.Module):
    def __init__(self, dimensions: int, nonlinearity: str | None = "logsigm") -> None:
        super().__init__()
        if nonlinearity not in {"sigm", "logsigm", None}:
            raise ValueError(f"Unsupported doser nonlinearity: {nonlinearity}")
        self.nonlinearity = nonlinearity
        self.beta = nn.Parameter(torch.ones(1, dimensions))
        self.bias = nn.Parameter(torch.zeros(1, dimensions))

    def forward(self, dose: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        beta = self.beta[0, index]
        bias = self.bias[0, index]
        if self.nonlinearity is None:
            return dose
        transformed = torch.log1p(dose) if self.nonlinearity == "logsigm" else dose
        return (transformed * beta + bias).sigmoid() - bias.sigmoid()


class ComPert(nn.Module):
    """chemCPA compositional perturbation autoencoder."""

    def __init__(
        self,
        num_genes: int,
        num_drugs: int,
        num_populations: int,
        drug_embeddings: torch.Tensor,
        hparams: dict,
    ) -> None:
        super().__init__()
        self.num_genes = int(num_genes)
        self.num_drugs = int(num_drugs)
        self.num_populations = int(num_populations)
        self.hparams = dict(hparams)
        hidden = [self.hparams["autoencoder_width"]] * self.hparams["autoencoder_depth"]
        self.encoder = MLP([self.num_genes, *hidden, self.hparams["dim"]])
        self.decoder = MLP(
            [self.hparams["dim"], *hidden, self.num_genes * 2],
            last_activation="linear",
        )
        self.drug_embeddings = nn.Embedding.from_pretrained(
            drug_embeddings.float(), freeze=True
        )
        embedding_hidden = [self.hparams["embedding_encoder_width"]] * self.hparams[
            "embedding_encoder_depth"
        ]
        self.drug_embedding_encoder = MLP(
            [self.drug_embeddings.embedding_dim, *embedding_hidden, self.hparams["dim"]]
        )
        self.adversary_drugs = MLP(
            [
                self.hparams["dim"],
                *([self.hparams["adversary_width"]] * self.hparams["adversary_depth"]),
                self.num_drugs,
            ]
        )
        self.adversary_populations = MLP(
            [
                self.hparams["dim"],
                *([self.hparams["adversary_width"]] * self.hparams["adversary_depth"]),
                self.num_populations,
            ]
        )
        self.population_embeddings = nn.Embedding(
            self.num_populations, self.hparams["dim"]
        )
        doser_type = self.hparams["doser_type"]
        if doser_type == "mlp":
            self.dosers = nn.ModuleList(
                MLP(
                    [
                        1,
                        *([self.hparams["dosers_width"]] * self.hparams["dosers_depth"]),
                        1,
                    ],
                    batch_norm=False,
                )
                for _ in range(self.num_drugs)
            )
        else:
            self.dosers = GeneralizedSigmoid(self.num_drugs, doser_type)
        self.doser_type = doser_type
        self.reconstruction_loss = nn.GaussianNLLLoss()
        self.drug_adversary_loss = nn.BCEWithLogitsLoss()
        self.population_adversary_loss = nn.CrossEntropyLoss()

    def _scaled_dose(self, index: torch.Tensor, dose: torch.Tensor) -> torch.Tensor:
        if self.doser_type == "mlp":
            values = [
                self.dosers[int(drug)](value.reshape(1, 1)).sigmoid().reshape(())
                for drug, value in zip(index, dose)
            ]
            return torch.stack(values)
        return self.dosers(dose, index)

    def forward(
        self,
        genes: torch.Tensor,
        drug_index: torch.Tensor,
        dose: torch.Tensor,
        population_index: torch.Tensor,
        *,
        adversary_only: bool = False,
    ) -> dict:
        latent_basal = self.encoder(genes)
        output = {
            "latent_basal": latent_basal,
            "drug_logits": self.adversary_drugs(latent_basal),
            "population_logits": self.adversary_populations(latent_basal),
            "drug_index": drug_index,
            "population_index": population_index,
        }
        if adversary_only:
            return output
        chemical = self.drug_embedding_encoder(self.drug_embeddings(drug_index))
        chemical = chemical * self._scaled_dose(drug_index, dose).unsqueeze(1)
        population = self.population_embeddings(population_index)
        latent_treated = latent_basal + chemical + population
        reconstruction = self.decoder(latent_treated)
        mean, raw_variance = reconstruction.chunk(2, dim=1)
        output.update({
            "mean": mean,
            "variance": F.softplus(raw_variance) + 1e-6,
        })
        return output


class ChemCPA(nn.Module):
    model_name = "chemcpa"

    def __init__(
        self,
        data_dir,
        hvg_dim: int,
        populations,
        material_dir=None,
        **hparams,
    ) -> None:
        super().__init__()
        assets = MethodAssets(data_dir, self.model_name, material_dir)
        self.smiles = assets.smiles
        self.smiles_to_index = assets.smiles_to_index
        self.populations = tuple(str(value) for value in populations)
        self.population_to_index = {
            value: index for index, value in enumerate(self.populations)
        }
        if not self.populations:
            raise ValueError("chemCPA requires at least one population covariate")
        defaults = {
            "dim": 256,
            "autoencoder_width": 512,
            "autoencoder_depth": 4,
            "adversary_width": 128,
            "adversary_depth": 3,
            "adversary_steps": 3,
            "reg_adversary": 5.0,
            "reg_adversary_cov": 1.0,
            "penalty_adversary": 3.0,
            "dosers_width": 64,
            "dosers_depth": 2,
            "doser_type": "logsigm",
            "embedding_encoder_width": 512,
            "embedding_encoder_depth": 0,
        }
        defaults.update(hparams)
        self.hparams = defaults
        self.hvg_dim = int(hvg_dim)
        if self.hvg_dim <= 0:
            raise ValueError("hvg_dim must be positive")
        drug_embeddings = assets.matrix("ecfp4_1024.float32.npy")
        if drug_embeddings.shape != (len(self.smiles), 1024):
            raise ValueError("chemCPA ECFP4 matrix does not match the prepared drug vocabulary")
        self.compert = ComPert(
            self.hvg_dim,
            len(self.smiles),
            len(self.populations),
            drug_embeddings,
            self.hparams,
        )

    def _indices(self, values, lookup, device) -> torch.Tensor:
        return torch.tensor(
            [lookup[str(value)] for value in values], device=device, dtype=torch.long
        )

    def forward(self, batch: dict, *, adversary_only: bool = False) -> dict:
        control = batch["control_hvg_vectors"].float()
        batch_size, set_size, hvg_dim = control.shape
        drug_index = self._indices(
            list(batch["drug_smiles"]), self.smiles_to_index, control.device
        ).repeat_interleave(set_size)
        population_index = self._indices(
            list(batch["population"]), self.population_to_index, control.device
        ).repeat_interleave(set_size)
        dose = batch["drug_conc"].float().repeat_interleave(set_size)
        output = self.compert(
            control.reshape(-1, hvg_dim),
            drug_index,
            dose,
            population_index,
            adversary_only=adversary_only,
        )
        if not adversary_only:
            output["mean"] = output["mean"].reshape(batch_size, set_size, hvg_dim)
            output["variance"] = output["variance"].reshape(
                batch_size, set_size, hvg_dim
            )
        return output

    def loss(self, output: dict, batch: dict) -> torch.Tensor:
        return self.compert.reconstruction_loss(
            output["mean"].float(),
            batch["condition_hvg_vectors"].float(),
            output["variance"].float(),
        )

    @torch.no_grad()
    def predict_batch(self, batch: dict) -> torch.Tensor:
        return self(batch)["mean"].mean(dim=1)

    def configuration(self) -> dict:
        return {
            "model": self.model_name,
            "hvg_dim": self.hvg_dim,
            **self.hparams,
        }


__all__ = ["ChemCPA", "ComPert", "GeneralizedSigmoid", "MLP"]
