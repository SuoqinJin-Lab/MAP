from __future__ import annotations

"""XPert dual-branch Transformer adapted to MAP's materialized contract.

The branch topology and token construction follow GSanShui/XPert (MIT,
copyright 2025 Guo Yue). Attention uses PyTorch SDPA instead of FlashAttention.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common import BaselineAssets


class Attention(nn.Module):
    def __init__(self, hidden_size: int, heads: int, dropout: float) -> None:
        super().__init__()
        if hidden_size % heads:
            raise ValueError("XPert hidden_size must be divisible by attention_heads")
        self.heads = int(heads)
        self.head_size = int(hidden_size) // self.heads
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.dropout = float(dropout)

    def forward(
        self,
        query_values,
        key_values=None,
        value_values=None,
        key_mask=None,
    ):
        key_values = query_values if key_values is None else key_values
        value_values = key_values if value_values is None else value_values
        batch, query_length, hidden = query_values.shape
        key_length = key_values.shape[1]
        query = self.query(query_values).view(batch, query_length, self.heads, self.head_size).transpose(1, 2)
        key = self.key(key_values).view(batch, key_length, self.heads, self.head_size).transpose(1, 2)
        value = self.value(value_values).view(batch, key_length, self.heads, self.head_size).transpose(1, 2)
        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=(
                key_mask[:, None, None, :].to(dtype=torch.bool)
                if key_mask is not None
                else None
            ),
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        return output.transpose(1, 2).contiguous().view(batch, query_length, hidden)


class Residual(nn.Module):
    def __init__(self, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, output, residual):
        return self.dropout(self.norm(output)) + residual


class FeedForward(nn.Module):
    def __init__(self, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
        )
        self.residual = Residual(hidden_size, dropout)

    def forward(self, values):
        return self.residual(self.network(values), values)


class SelfEncoder(nn.Module):
    def __init__(self, hidden_size: int, heads: int, attention_dropout: float, dropout: float):
        super().__init__()
        self.attention = Attention(hidden_size, heads, attention_dropout)
        self.residual = Residual(hidden_size, dropout)
        self.feed_forward = FeedForward(hidden_size, dropout)

    def forward(self, values, key_mask=None):
        values = self.residual(
            self.attention(values, key_mask=key_mask), values
        )
        return self.feed_forward(values)


class CrossEncoder(nn.Module):
    def __init__(self, hidden_size: int, heads: int, attention_dropout: float, dropout: float):
        super().__init__()
        self.drug_self = SelfEncoder(hidden_size, heads, attention_dropout, dropout)
        self.cell_self = Attention(hidden_size, heads, attention_dropout)
        self.cell_cross = Attention(hidden_size, heads, attention_dropout)
        self.self_residual = Residual(hidden_size, dropout)
        self.cross_residual = Residual(hidden_size, dropout)
        self.feed_forward = FeedForward(hidden_size, dropout)

    def forward(self, cell, drug, drug_mask):
        drug = self.drug_self(drug, drug_mask)
        cell = self.self_residual(self.cell_self(cell), cell)
        cell = self.cross_residual(
            self.cell_cross(cell, drug, drug, key_mask=drug_mask), cell
        )
        return self.feed_forward(cell), drug


class Branch(nn.Module):
    def __init__(
        self,
        structure: str,
        hidden_size: int,
        heads: int,
        attention_dropout: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.types = tuple(structure.split("+"))
        invalid = sorted(set(self.types) - {"CA", "SA"})
        if invalid:
            raise ValueError(
                "XPert branch structure contains invalid layers: "
                + ", ".join(invalid)
            )
        self.layers = nn.ModuleList(
            CrossEncoder(hidden_size, heads, attention_dropout, dropout)
            if kind == "CA"
            else SelfEncoder(hidden_size, heads, attention_dropout, dropout)
            for kind in self.types
        )

    def forward(self, cell, drug=None, drug_mask=None):
        for kind, layer in zip(self.types, self.layers):
            if kind == "CA":
                if drug is None or drug_mask is None:
                    raise ValueError("XPert cross-attention requires drug tokens")
                cell, drug = layer(cell, drug, drug_mask)
            else:
                cell = layer(cell)
        return cell


class GeneEmbedding(nn.Module):
    def __init__(self, vectors, bins: int, hidden_size: int, dropout: float):
        super().__init__()
        vectors = vectors.float()
        self.expression = nn.Embedding(int(bins), int(hidden_size))
        self.projection = (
            nn.Linear(vectors.shape[1], hidden_size)
            if vectors.shape[1] != hidden_size
            else nn.Identity()
        )
        self.pretrained = nn.Parameter(vectors)
        self.norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, expression_bins):
        gene = self.projection(self.pretrained).unsqueeze(0)
        return self.dropout(self.norm(self.expression(expression_bins) + gene))


class DrugEmbedding(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        max_atoms: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.chemical = nn.Linear(512, hidden_size)
        self.position = nn.Embedding(max_atoms, hidden_size)
        self.norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, unimol, graph):
        mask = unimol[:, :, 0].to(dtype=torch.bool)
        tokens = self.chemical(unimol[:, :, 2:].float())
        tokens[:, 0] = graph
        positions = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0)
        tokens = self.dropout(self.norm(tokens + self.position(positions)))
        return tokens, mask


class XPert(nn.Module):
    model_name = "xpert"

    def __init__(self, data_dir, hvg_dim: int, populations, **options) -> None:
        super().__init__()
        del populations
        assets = BaselineAssets(data_dir, self.model_name)
        self.smiles = assets.smiles
        self.smiles_to_index = assets.smiles_to_index
        defaults = {
            "hidden_size": 256,
            "attention_heads": 8,
            "attention_dropout": 0.1,
            "hidden_dropout": 0.1,
            "cell_input_dropout": 0.1,
            "drug_input_dropout": 0.1,
            "treated_structure": "CA+SA+SA+CA",
            "control_structure": "SA+SA+SA+SA",
            "expression_bins": 10,
            "expression_min": 0.0,
            "expression_max": 10.0,
            "treated_weight": 0.2,
            "control_weight": 0.003,
            "delta_weight": 0.2,
            "correlation_weight": 1.0,
        }
        defaults.update(options)
        self.hparams = defaults
        self.hvg_dim = int(hvg_dim)
        if self.hvg_dim <= 0:
            raise ValueError("XPert hvg_dim must be positive")
        hidden = int(defaults["hidden_size"])
        bins = int(defaults["expression_bins"])
        if hidden <= 0 or bins < 2:
            raise ValueError("XPert hidden_size must be positive and expression_bins at least 2")
        if hidden % int(defaults["attention_heads"]):
            raise ValueError("XPert hidden_size must be divisible by attention_heads")
        if not float(defaults["expression_min"]) < float(defaults["expression_max"]):
            raise ValueError("XPert expression_min must be smaller than expression_max")
        genes = assets.matrix("ppi_gene_vectors.float32.npy")
        graph = assets.matrix("drug_hg_embeddings.float32.npy")
        unimol = assets.matrix("drug_unimol.float16.npy")
        if genes.shape[0] != self.hvg_dim:
            raise ValueError("XPert PPI gene vectors do not match the HVG order")
        if graph.shape[0] != len(self.smiles) or unimol.shape[0] != len(self.smiles):
            raise ValueError("XPert drug assets do not match the prepared vocabulary")
        if unimol.ndim != 3 or unimol.shape[2] != 514:
            raise ValueError("XPert UniMol tokens must have shape [drugs, atoms, 514]")
        if not torch.all(unimol[:, 0, 0] > 0):
            raise ValueError("Every XPert drug requires an unmasked graph token")
        self.register_buffer("drug_graph", graph, persistent=False)
        self.register_buffer("drug_unimol", unimol, persistent=False)
        self.graph_projection = (
            nn.Linear(graph.shape[1], hidden) if graph.shape[1] != hidden else nn.Identity()
        )
        self.gene_embedding = GeneEmbedding(
            genes,
            defaults["expression_bins"],
            hidden,
            defaults["cell_input_dropout"],
        )
        self.drug_embedding = DrugEmbedding(
            hidden,
            unimol.shape[1],
            defaults["drug_input_dropout"],
        )
        branch_options = (
            hidden,
            defaults["attention_heads"],
            defaults["attention_dropout"],
            defaults["hidden_dropout"],
        )
        self.treated_branch = Branch(defaults["treated_structure"], *branch_options)
        self.control_branch = Branch(defaults["control_structure"], *branch_options)
        self.treated_head = self._head(hidden)
        self.control_head = self._head(hidden)
        self.delta_head = self._head(hidden)
        self.apply(self._initialize)

    @staticmethod
    def _head(hidden_size: int):
        return nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    @staticmethod
    def _initialize(module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _expression_bins(self, values):
        minimum = float(self.hparams["expression_min"])
        maximum = float(self.hparams["expression_max"])
        boundaries = torch.linspace(
            minimum,
            maximum,
            int(self.hparams["expression_bins"]) - 1,
            device=values.device,
            dtype=values.dtype,
        )
        return torch.bucketize(
            values.clamp(minimum, maximum), boundaries, right=True
        )

    def _drug_indices(self, values, device):
        return torch.tensor(
            [self.smiles_to_index[str(value)] for value in values],
            device=device,
            dtype=torch.long,
        )

    def forward(self, batch: dict) -> dict:
        control = batch["control_hvg_vectors"].float().mean(1)
        target = batch["condition_hvg_vectors"].float().mean(1)
        drug_index = self._drug_indices(list(batch["drug_smiles"]), control.device)
        cell_tokens = self.gene_embedding(self._expression_bins(control))
        drug_tokens, drug_mask = self.drug_embedding(
            self.drug_unimol[drug_index],
            self.graph_projection(self.drug_graph[drug_index]),
        )
        treated_tokens = self.treated_branch(
            cell_tokens, drug_tokens, drug_mask
        )
        control_tokens = self.control_branch(cell_tokens)
        treated_output = self.treated_head(treated_tokens).squeeze(-1)
        control_output = self.control_head(control_tokens).squeeze(-1)
        delta = self.delta_head(treated_tokens - control_tokens).squeeze(-1)
        return {
            "prediction": control + delta,
            "target": target,
            "control": control,
            "treated_output": treated_output,
            "control_output": control_output,
            "delta": delta,
        }

    def loss(self, output: dict) -> tuple[torch.Tensor, dict]:
        true_delta = output["target"] - output["control"]
        treated = F.mse_loss(output["treated_output"], output["target"])
        control = F.mse_loss(output["control_output"], output["control"])
        delta = F.mse_loss(output["delta"], true_delta)
        first = output["delta"] - output["delta"].mean(1, keepdim=True)
        second = true_delta - true_delta.mean(1, keepdim=True)
        correlation = (first * second).sum(1) / (
            first.square().sum(1).mul(second.square().sum(1)).clamp_min(1e-6).sqrt()
        )
        correlation_loss = 1 - correlation.mean()
        value = (
            self.hparams["treated_weight"] * treated
            + self.hparams["control_weight"] * control
            + self.hparams["delta_weight"] * delta
            + self.hparams["correlation_weight"] * correlation_loss
        )
        return value, {
            "treated": treated.detach(),
            "control": control.detach(),
            "delta": delta.detach(),
            "correlation": correlation.detach().mean(),
        }

    @torch.no_grad()
    def predict_batch(self, batch: dict) -> torch.Tensor:
        return self(batch)["prediction"]

    def configuration(self) -> dict:
        return {"model": self.model_name, "hvg_dim": self.hvg_dim, **self.hparams}


__all__ = ["XPert"]
