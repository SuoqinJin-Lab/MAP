from __future__ import annotations

"""CRISP PertAE adaptation for MAP's materialized data contract.

The architecture and default hyperparameters follow ml4bio/CRISP (MIT,
copyright 2025 Xinyuan Liu). Optimizer ownership and data access are adapted
to the repository's shared baseline runner.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common import BaselineAssets


class MLP(nn.Module):
    def __init__(
        self,
        sizes: list[int],
        dropout: float,
        *,
        batch_norm: bool = True,
        final_relu: bool = False,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for index, (source, target) in enumerate(zip(sizes[:-1], sizes[1:])):
            last = index == len(sizes) - 2
            layers.append(nn.Linear(source, target))
            if not last:
                if batch_norm:
                    layers.append(nn.BatchNorm1d(target))
                layers.extend((nn.ReLU(), nn.Dropout(dropout)))
        self.network = nn.Sequential(*layers)
        self.final_relu = bool(final_relu)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        output = self.network(values)
        return F.relu(output) if self.final_relu else output


def mmd_loss(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    combined = torch.cat((first, second), dim=0).float()
    norms = combined.square().sum(dim=1, keepdim=True)
    squared = (norms + norms.transpose(0, 1) - 2 * (combined @ combined.T))
    squared = squared.clamp_min(0)
    count = squared.shape[0]
    squared = squared.masked_fill(
        torch.eye(count, device=squared.device, dtype=torch.bool), 0
    )
    bandwidth = squared.detach().sum() / max(count * count - count, 1)
    bandwidth = bandwidth.clamp_min(1e-6)
    multipliers = torch.tensor(
        [0.25, 0.5, 1.0, 2.0, 4.0], device=combined.device, dtype=combined.dtype
    )
    kernel = torch.exp(
        -squared.unsqueeze(0) / (bandwidth * multipliers[:, None, None])
    ).sum(0)
    split = len(first)
    return (
        kernel[:split, :split].mean()
        - 2 * kernel[:split, split:].mean()
        + kernel[split:, split:].mean()
    )


class PertAE(nn.Module):
    def __init__(
        self,
        num_genes: int,
        drug_embeddings: torch.Tensor,
        num_celltypes: int,
        *,
        fm_dim: int = 2048,
        latent_dim: int = 128,
        encoder_width: int = 256,
        encoder_depth: int = 4,
        decoder_width: int = 1028,
        decoder_depth: int = 4,
        embedding_encoder_width: int = 128,
        embedding_encoder_depth: int = 4,
        doser_width: int = 64,
        doser_depth: int = 3,
        cell_predictor_width: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_genes = int(num_genes)
        self.latent_dim = int(latent_dim)
        self.encoder_fm = MLP(
            [int(fm_dim), *([int(encoder_width)] * int(encoder_depth)), 2 * self.latent_dim],
            float(dropout),
        )
        drug_dim = int(drug_embeddings.shape[1])
        self.drug_embeddings = nn.Embedding.from_pretrained(
            drug_embeddings.float(), freeze=True
        )
        self.drug_encoder = MLP(
            [drug_dim, *([int(embedding_encoder_width)] * int(embedding_encoder_depth)), self.latent_dim],
            float(dropout),
        )
        self.doser = MLP(
            [drug_dim + 1, *([int(doser_width)] * int(doser_depth)), 1],
            float(dropout),
        )
        treated_dim = self.latent_dim * 2
        self.decoder = MLP(
            [treated_dim, *([int(decoder_width)] * int(decoder_depth)), self.num_genes],
            float(dropout),
            final_relu=True,
        )
        self.cell_predictor = MLP(
            [treated_dim, int(cell_predictor_width), int(num_celltypes)],
            float(dropout),
        )

    def forward(
        self,
        cell_embeddings: torch.Tensor,
        drug_index: torch.Tensor,
        dose: torch.Tensor,
        *,
        sample: bool,
    ) -> dict:
        encoded = self.encoder_fm(cell_embeddings)
        mean, raw_log_variance = encoded.chunk(2, dim=1)
        log_variance = F.relu(raw_log_variance).clamp(max=10)
        basal = (
            mean + torch.randn_like(mean) * torch.exp(0.5 * log_variance)
            if sample
            else mean
        )
        chemical = self.drug_embeddings(drug_index)
        scaled_dose = self.doser(torch.cat((chemical, dose.unsqueeze(1)), dim=1))
        drug_latent = self.drug_encoder(chemical) * scaled_dose
        treated = torch.cat((basal, drug_latent), dim=1)
        return {
            "prediction": self.decoder(treated),
            "treated_latent": treated,
            "mean": mean,
            "log_variance": log_variance,
            "cell_logits": self.cell_predictor(treated),
        }


class CRISP(nn.Module):
    model_name = "crisp"

    def __init__(self, data_dir, hvg_dim: int, populations, **options) -> None:
        super().__init__()
        assets = BaselineAssets(data_dir, self.model_name)
        self.smiles = assets.smiles
        self.smiles_to_index = assets.smiles_to_index
        self.populations = tuple(str(value) for value in populations)
        self.population_to_index = {
            value: index for index, value in enumerate(self.populations)
        }
        defaults = {
            "fm_dim": 2048,
            "latent_dim": 128,
            "encoder_width": 256,
            "encoder_depth": 4,
            "decoder_width": 1028,
            "decoder_depth": 4,
            "embedding_encoder_width": 128,
            "embedding_encoder_depth": 4,
            "doser_width": 64,
            "doser_depth": 3,
            "cell_predictor_width": 128,
            "dropout": 0.2,
            "deg_top_k": 50,
            "mse_weight": 0.75,
            "autofocus_weight": 0.25,
            "celltype_weight": 1.0,
            "contrastive_weight": 0.1,
            "mmd_weight": 0.1,
            "kld_normalizer": 500.0,
        }
        defaults.update(options)
        self.hparams = defaults
        self.hvg_dim = int(hvg_dim)
        matrix = assets.matrix("rdkit2d.float32.npy")
        if matrix.shape[0] != len(self.smiles):
            raise ValueError("CRISP RDKit2D matrix does not match the drug vocabulary")
        self.pertae = PertAE(
            self.hvg_dim,
            matrix,
            len(self.populations),
            **{
                key: defaults[key]
                for key in (
                    "fm_dim",
                    "latent_dim",
                    "encoder_width",
                    "encoder_depth",
                    "decoder_width",
                    "decoder_depth",
                    "embedding_encoder_width",
                    "embedding_encoder_depth",
                    "doser_width",
                    "doser_depth",
                    "cell_predictor_width",
                    "dropout",
                )
            },
        )

    @staticmethod
    def _indices(values, lookup, device) -> torch.Tensor:
        return torch.tensor(
            [lookup[str(value)] for value in values], device=device, dtype=torch.long
        )

    def _forward_condition(self, batch: dict, prefix: str = "") -> dict:
        cell = batch[f"{prefix}control_embeddings"]
        if not torch.is_autocast_enabled(cell.device.type):
            cell = cell.float()
        batch_size, set_size, fm_dim = cell.shape
        drug_index = batch.get(f"{prefix}drug_index")
        if drug_index is None:
            drug_index = self._indices(
                list(batch[f"{prefix}drug_smiles"]),
                self.smiles_to_index,
                cell.device,
            )
        else:
            drug_index = drug_index.to(device=cell.device, dtype=torch.long)
        drug = drug_index.repeat_interleave(set_size)
        dose = batch[f"{prefix}drug_conc"].float().repeat_interleave(set_size)
        output = self.pertae(
            cell.reshape(-1, fm_dim), drug, dose, sample=self.training
        )
        output["prediction"] = output["prediction"].reshape(
            batch_size, set_size, self.hvg_dim
        )
        output["treated_latent"] = output["treated_latent"].reshape(
            batch_size, set_size, -1
        )
        return output

    def forward(self, batch: dict) -> dict:
        output = self._forward_condition(batch)
        if "negative_control_embeddings" in batch:
            negative = self._forward_condition(batch, "negative_")
            output.update({f"negative_{key}": value for key, value in negative.items()})
        return output

    def loss(self, output: dict, batch: dict) -> tuple[torch.Tensor, dict]:
        targets = [batch["condition_hvg_vectors"].float()]
        controls = [batch["control_hvg_vectors"].float()]
        predictions = [output["prediction"].float()]
        means = [output["mean"].float()]
        log_variances = [output["log_variance"].float()]
        cell_logits = [output["cell_logits"].float()]
        target_device = targets[0].device
        population_indices = [
            batch["population_index"].to(device=target_device, dtype=torch.long)
            if "population_index" in batch
            else self._indices(
                list(batch["population"]), self.population_to_index, target_device
            )
        ]
        use_deg_masks = (
            "condition_deg_mask" in batch
            or "negative_condition_deg_mask" in batch
        )
        deg_masks = []
        if use_deg_masks:
            condition_deg_mask = batch.get("condition_deg_mask")
            if condition_deg_mask is None:
                condition_deg_mask = torch.zeros_like(
                    batch["condition_hvg_vectors"], dtype=torch.bool
                )
            deg_masks.append(condition_deg_mask.bool())
        if "negative_prediction" in output:
            targets.append(batch["negative_condition_hvg_vectors"].float())
            controls.append(batch["negative_control_hvg_vectors"].float())
            predictions.append(output["negative_prediction"].float())
            means.append(output["negative_mean"].float())
            log_variances.append(output["negative_log_variance"].float())
            cell_logits.append(output["negative_cell_logits"].float())
            population_indices.append(
                batch["negative_population_index"].to(
                    device=target_device, dtype=torch.long
                )
                if "negative_population_index" in batch
                else self._indices(
                    list(batch["negative_population"]),
                    self.population_to_index,
                    target_device,
                )
            )
            if use_deg_masks:
                negative_deg_mask = batch.get("negative_condition_deg_mask")
                if negative_deg_mask is None:
                    negative_deg_mask = torch.zeros_like(
                        batch["negative_condition_hvg_vectors"], dtype=torch.bool
                    )
                deg_masks.append(negative_deg_mask.bool())
        target = torch.cat(targets, dim=0)
        control = torch.cat(controls, dim=0)
        prediction = torch.cat(predictions, dim=0)
        mse = F.mse_loss(prediction, target)
        if deg_masks:
            deg_mask = torch.cat(deg_masks, dim=0)
            autofocus = (
                (prediction - target).square() * deg_mask.to(prediction.dtype)
            ).sum().div(deg_mask.sum().clamp_min(1))
        else:
            # MAP-validation's CRISP contract passes an all-zero DEG mask.
            # Avoid allocating and transferring two dense masks just to
            # recover the exact zero autofocus term.
            autofocus = prediction.new_zeros(())
        mmd = mmd_loss(
            target.reshape(-1, self.hvg_dim), prediction.reshape(-1, self.hvg_dim)
        )
        latent = output["treated_latent"]
        if "negative_treated_latent" in output:
            negative = output["negative_treated_latent"]
        elif latent.shape[0] > 1:
            negative = latent.roll(1, dims=0)
        else:
            negative = latent.roll(1, dims=1)
        labels = -torch.ones(latent.shape[0] * latent.shape[1], device=latent.device)
        contrastive = F.cosine_embedding_loss(
            latent.reshape(-1, latent.shape[-1]),
            negative.reshape(-1, negative.shape[-1]),
            labels,
        )
        populations = torch.cat(population_indices).repeat_interleave(
            target.shape[1]
        )
        celltype = F.cross_entropy(torch.cat(cell_logits, dim=0), populations)
        mean = torch.cat(means, dim=0)
        log_variance = torch.cat(log_variances, dim=0)
        kld = -0.5 * (
            1 + log_variance - mean.square() - log_variance.exp()
        ).sum(1).mean()
        value = (
            self.hparams["mse_weight"] * mse
            + self.hparams["autofocus_weight"] * autofocus
            + self.hparams["celltype_weight"] * celltype
            + self.hparams["contrastive_weight"] * contrastive
            + self.hparams["mmd_weight"] * mmd
            + kld / (self.hparams["kld_normalizer"] * self.hparams["latent_dim"])
        )
        return value, {
            "mse": mse.detach(),
            "autofocus": autofocus.detach(),
            "celltype": celltype.detach(),
            "contrastive": contrastive.detach(),
            "mmd": mmd.detach(),
            "kld": kld.detach(),
        }

    @torch.no_grad()
    def predict_batch(self, batch: dict) -> torch.Tensor:
        return self(batch)["prediction"].mean(dim=1)

    def configuration(self) -> dict:
        return {
            "model": self.model_name,
            "hvg_dim": self.hvg_dim,
            "deg_mask_contract": "all_zero_map_validation",
            **self.hparams,
        }


__all__ = ["CRISP", "MLP", "PertAE", "mmd_loss"]
