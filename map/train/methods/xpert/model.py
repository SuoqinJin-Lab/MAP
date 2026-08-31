from __future__ import annotations

"""XPert dual-branch Transformer adapted to MAP's materialized contract.

The branch topology and token construction follow GSanShui/XPert (MIT,
copyright 2025 Guo Yue). Attention uses PyTorch SDPA instead of FlashAttention.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common import MethodAssets


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
    def __init__(
        self,
        vectors,
        bins: int,
        hidden_size: int,
        dropout: float,
        *,
        use_gene_position: bool = False,
    ):
        super().__init__()
        vectors = vectors.float()
        self.expression = nn.Embedding(int(bins), int(hidden_size))
        # XPert's reference implementation materializes the optional
        # dimensionality projection once, then stores the resulting PPI
        # vectors in a trainable embedding table.  Applying a projection on
        # every forward would make it a different model (and changes the
        # meaning of the pretrained vectors during fine-tuning).
        if vectors.shape[1] != hidden_size:
            initializer = nn.Linear(vectors.shape[1], hidden_size)
            with torch.no_grad():
                vectors = initializer(vectors)
        self.pretrained = nn.Embedding.from_pretrained(vectors, freeze=False)
        self.position = (
            nn.Embedding(vectors.shape[0], hidden_size)
            if use_gene_position
            else None
        )
        self.norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, expression_bins):
        positions = torch.arange(
            expression_bins.shape[1], device=expression_bins.device
        )
        gene = self.pretrained(positions).unsqueeze(0)
        output = self.expression(expression_bins) + gene
        if self.position is not None:
            output = output + self.position(positions).unsqueeze(0)
        return self.dropout(self.norm(output))


class DrugEmbedding(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        max_atoms: int,
        dropout: float,
        context_tokens: str = "none",
    ) -> None:
        super().__init__()
        self.chemical = nn.Linear(512, hidden_size)
        self.context_tokens = str(context_tokens).casefold()
        if self.context_tokens not in {"none", "dose", "dose_time"}:
            raise ValueError("XPert context_tokens must be none, dose, or dose_time")
        context_count = {"none": 0, "dose": 1, "dose_time": 2}[self.context_tokens]
        self.context_count = context_count
        self.position = nn.Embedding(max_atoms + context_count, hidden_size)
        self.dose = (
            nn.Embedding(10, hidden_size) if context_count else None
        )
        self.time = (
            nn.Embedding(3, hidden_size)
            if self.context_tokens == "dose_time"
            else None
        )
        self.norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, unimol, graph, dose_index=None, time_index=None):
        mask = unimol[:, :, 0].to(dtype=torch.bool)
        tokens = self.chemical(unimol[:, :, 2:].float())
        tokens[:, 0] = graph
        context = []
        if self.dose is not None:
            if dose_index is None:
                raise ValueError("XPert dose context requires dose indices")
            context.append(self.dose(dose_index).unsqueeze(1))
        if self.time is not None:
            if time_index is None:
                raise ValueError("XPert time context requires time indices")
            context.append(self.time(time_index).unsqueeze(1))
        if context:
            tokens = torch.cat((*context, tokens), dim=1)
            mask = torch.cat(
                (torch.ones(mask.shape[0], len(context), device=mask.device, dtype=torch.bool), mask),
                dim=1,
            )
        positions = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0)
        tokens = self.dropout(self.norm(tokens + self.position(positions)))
        return tokens, mask


class XPert(nn.Module):
    model_name = "xpert"

    def __init__(self, data_dir, hvg_dim: int, populations, material_dir=None, **options) -> None:
        super().__init__()
        self.populations = tuple(str(value) for value in populations)
        self.population_to_index = {
            value: index for index, value in enumerate(self.populations)
        }
        assets = MethodAssets(data_dir, self.model_name, material_dir)
        self.smiles = assets.smiles
        self.smiles_to_index = assets.smiles_to_index
        defaults = {
            "hidden_size": 256,
            "attention_heads": 8,
            "attention_dropout": 0.1,
            "hidden_dropout": 0.1,
            "cell_input_dropout": 0.1,
            "drug_input_dropout": 0.1,
            "input_mode": "official",
            "use_gene_position_embedding": False,
            "context_tokens": "dose",
            "attention_padding_mode": "masked",
            "include_cell_context": False,
            "loss_reduction_scale": "sample_count",
            "treated_structure": "CA+SA+SA+CA",
            "control_structure": "SA+SA+SA+SA",
            "expression_bins": 128,
            "expression_min": None,
            "expression_max": None,
            "treated_weight": 0.2,
            "control_weight": 0.003,
            "delta_weight": 0.2,
            "correlation_weight": 1.0,
        }
        defaults.update(options)
        self.hparams = defaults
        self.input_mode = str(defaults["input_mode"]).casefold()
        if self.input_mode not in {"official", "validation"}:
            raise ValueError("XPert input_mode must be official or validation")
        if defaults["loss_reduction_scale"] not in {"none", "sample_count"}:
            raise ValueError(
                "XPert loss_reduction_scale must be none or sample_count"
            )
        if defaults["attention_padding_mode"] not in {
            "official_unmasked",
            "masked",
        }:
            raise ValueError(
                "XPert attention_padding_mode must be official_unmasked or masked"
            )
        self.hvg_dim = int(hvg_dim)
        if self.hvg_dim <= 0:
            raise ValueError("XPert hvg_dim must be positive")
        hidden = int(defaults["hidden_size"])
        bins = int(defaults["expression_bins"])
        if hidden <= 0 or bins < 2:
            raise ValueError("XPert hidden_size must be positive and expression_bins at least 2")
        if hidden % int(defaults["attention_heads"]):
            raise ValueError("XPert hidden_size must be divisible by attention_heads")
        official = self.input_mode == "official"
        genes = assets.matrix(
            "ppi_gene_vectors_full.float32.npy"
            if official else "ppi_gene_vectors.float32.npy"
        )
        graph = assets.matrix("drug_hg_embeddings.float32.npy")
        unimol = assets.matrix("drug_unimol.float16.npy")
        self.gene_dim = int(genes.shape[0])
        self.full_gene_dim = int(
            assets.model_manifest.get("full_gene_count", self.gene_dim)
        )
        expression_contract = assets.model_manifest.get("expression")
        if expression_contract is None or expression_contract.get("format") not in {
            "expression_bins_v1", "xpert_hvg_expression_v2"
        }:
            raise RuntimeError(
                "XPert assets use the obsolete STATE-expression contract; "
                "rerun graph preparation with overwrite=True"
            )
        if expression_contract.get("representation") != "library_size_normalized_log1p_hvg_expression":
            raise ValueError("XPert expression assets are not in evaluator HVG log-expression scale")
        prepared_modes = set(expression_contract.get("input_modes", ()))
        if self.input_mode not in prepared_modes:
            raise ValueError(
                f"XPert input mode {self.input_mode!r} was not prepared; "
                f"available modes={sorted(prepared_modes)}"
            )
        expected_gene_dim = self.full_gene_dim if official else self.hvg_dim
        if self.gene_dim != expected_gene_dim:
            raise ValueError(
                "XPert gene vectors do not match the selected input mode "
                "(validation vectors must share the HVG dimension): "
                f"mode={self.input_mode}, vectors={self.gene_dim}, "
                f"expected={expected_gene_dim}"
            )
        if int(expression_contract["gene_count"]) != self.hvg_dim:
            raise ValueError(
                "XPert evaluator expression contract does not match the HVG "
                f"dimension: contract={expression_contract['gene_count']}, "
                f"evaluator={self.hvg_dim}"
            )
        hvg_indices = torch.as_tensor(
            assets.model_manifest.get("hvg_indices", ()), dtype=torch.long
        ).reshape(-1)
        if hvg_indices.shape != (self.hvg_dim,):
            raise ValueError("XPert HVG index mapping has the wrong shape")
        if hvg_indices.min() < 0 or hvg_indices.max() >= self.full_gene_dim:
            raise ValueError("XPert HVG index mapping is outside the full gene space")
        self.register_buffer("hvg_indices", hvg_indices, persistent=False)
        prepared_bins = int(expression_contract["expression_bins"])
        if official:
            requested_min = (
                0.0 if defaults["expression_min"] is None
                else float(defaults["expression_min"])
            )
            requested_max = (
                10.0 if defaults["expression_max"] is None
                else float(defaults["expression_max"])
            )
            boundaries = torch.linspace(
                requested_min,
                requested_max,
                int(defaults["expression_bins"]) - 1,
                dtype=torch.float32,
            )
        else:
            if int(defaults["expression_bins"]) != prepared_bins:
                raise ValueError(
                    "XPert expression_bins does not match preparation: "
                    f"prepared={prepared_bins}, requested={defaults['expression_bins']}"
                )
            boundaries = assets.matrix(
                expression_contract["bin_boundaries_file"]
            ).reshape(-1)
        expected_bins = int(defaults["expression_bins"])
        if boundaries.shape != (expected_bins - 1,) or not torch.all(
            boundaries[1:] >= boundaries[:-1]
        ):
            raise ValueError("XPert prepared expression boundaries are invalid")
        requested_min = defaults["expression_min"]
        requested_max = defaults["expression_max"]
        if not official and requested_min is not None and not math.isclose(
            float(requested_min), float(boundaries[0]), rel_tol=0.0, abs_tol=1e-6
        ):
            raise ValueError("XPert expression_min does not match preparation")
        if not official and requested_max is not None and not math.isclose(
            float(requested_max), float(boundaries[-1]), rel_tol=0.0, abs_tol=1e-6
        ):
            raise ValueError("XPert expression_max does not match preparation")
        defaults["expression_min"] = float(boundaries[0])
        defaults["expression_max"] = float(boundaries[-1])
        self.register_buffer("expression_boundaries", boundaries, persistent=False)
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
            use_gene_position=bool(defaults["use_gene_position_embedding"]),
        )
        self.drug_embedding = DrugEmbedding(
            hidden,
            unimol.shape[1],
            defaults["drug_input_dropout"],
            context_tokens=defaults["context_tokens"],
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
        self.cell_context_token = (
            nn.Parameter(torch.randn(1, 1, hidden))
            if defaults["include_cell_context"]
            else None
        )
        self.cell_classifier = (
            nn.Linear(hidden, len(self.populations))
            if defaults["include_cell_context"]
            else None
        )
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
        return torch.bucketize(
            values,
            self.expression_boundaries.to(device=values.device, dtype=values.dtype),
            right=True,
        )

    def _drug_indices(self, values, device):
        return torch.tensor(
            [self.smiles_to_index[str(value)] for value in values],
            device=device,
            dtype=torch.long,
        )

    @staticmethod
    def _dose_indices(values: torch.Tensor) -> torch.Tensor:
        boundaries = values.new_tensor(
            [0.21, 0.41, 0.71, 1.01, 1.51, 3.1, 4.1, 7.1, 12.1]
        )
        return torch.bucketize(values.float().clamp_min(0), boundaries)

    def _official_expression(self, batch: dict, prefix: str) -> torch.Tensor:
        gene_key = f"{prefix}_gene_ids"
        expression_key = f"{prefix}_expressions"
        if gene_key not in batch or expression_key not in batch:
            raise KeyError(
                "XPert official input requires sparse full-gene fields "
                f"{gene_key!r} and {expression_key!r}"
            )
        gene_ids = batch[gene_key].long()
        expression = batch[expression_key].float()
        if gene_ids.shape != expression.shape or gene_ids.ndim != 3:
            raise ValueError(
                "XPert official sparse inputs must have shape [B, S, tokens]"
            )
        valid = (
            (gene_ids >= 0)
            & (gene_ids < self.full_gene_dim)
            & torch.isfinite(expression)
            & (expression > 0)
        )
        dense = expression.new_zeros(
            (*expression.shape[:2], self.full_gene_dim)
        )
        dense.scatter_add_(2, gene_ids.clamp(0, self.full_gene_dim - 1), expression * valid)
        return dense

    def _expression_inputs(self, batch: dict, prefix: str):
        key = f"{prefix}_hvg_vectors"
        if self.input_mode == "official":
            expression = self._official_expression(batch, prefix)
            expected_dim = self.full_gene_dim
        else:
            if key not in batch:
                raise KeyError(f"XPert input is missing evaluator-scale field {key!r}")
            expression = batch[key].float()
            expected_dim = self.hvg_dim
        if expression.ndim not in {2, 3} or expression.shape[-1] != expected_dim:
            raise ValueError(
                f"XPert {prefix} expression must have shape [B, {expected_dim}] "
                f"or [B, S, {expected_dim}]"
            )
        if expression.ndim == 2:
            batch_size, gene_count = expression.shape
            set_size = 1
        else:
            batch_size, set_size, gene_count = expression.shape
            # Both upstream XPert's Tahoe adapter and the validation contract
            # pseudobulk the selected cell set before constructing gene tokens.
            expression = expression.mean(dim=1)
            set_size = 1
        expression = expression.reshape(batch_size * set_size, gene_count)
        return expression, self._expression_bins(expression), batch_size, set_size

    def forward(self, batch: dict) -> dict:
        control, control_bins, batch_size, set_size = self._expression_inputs(
            batch, "control"
        )
        target, _, target_batch_size, target_set_size = self._expression_inputs(
            batch, "condition"
        )
        if (batch_size, set_size) != (target_batch_size, target_set_size):
            raise ValueError("XPert control and condition set shapes do not match")
        drug_index = self._drug_indices(list(batch["drug_smiles"]), control.device)
        if set_size > 1:
            drug_index = drug_index.repeat_interleave(set_size)
        dose = batch["drug_conc"].float().to(control.device)
        if set_size > 1:
            dose = dose.repeat_interleave(set_size)
        context_mode = self.hparams["context_tokens"]
        dose_index = (
            self._dose_indices(dose)
            if context_mode in {"dose", "dose_time"}
            else None
        )
        time_index = (
            torch.zeros_like(dose, dtype=torch.long)
            if context_mode == "dose_time"
            else None
        )
        cell_tokens = self.gene_embedding(control_bins)
        if self.cell_context_token is not None:
            cell_tokens = torch.cat(
                (
                    self.cell_context_token.expand(cell_tokens.shape[0], -1, -1),
                    cell_tokens,
                ),
                dim=1,
            )
        drug_tokens, drug_mask = self.drug_embedding(
            self.drug_unimol[drug_index],
            self.graph_projection(self.drug_graph[drug_index]),
            dose_index,
            time_index,
        )
        attention_mask = (
            drug_mask
            if self.hparams["attention_padding_mode"] == "masked"
            else torch.ones_like(drug_mask)
        )
        treated_tokens = self.treated_branch(cell_tokens, drug_tokens, attention_mask)
        control_tokens = self.control_branch(cell_tokens)
        if self.cell_classifier is not None:
            treated_cell_logits = self.cell_classifier(treated_tokens[:, 0])
            control_cell_logits = self.cell_classifier(control_tokens[:, 0])
            treated_tokens = treated_tokens[:, 1:]
            control_tokens = control_tokens[:, 1:]
            population_indices = torch.tensor(
                [self.population_to_index[str(value)] for value in batch["population"]],
                device=control.device,
                dtype=torch.long,
            ).repeat_interleave(set_size)
        else:
            treated_cell_logits = control_cell_logits = population_indices = None
        treated_output = self.treated_head(treated_tokens).squeeze(-1)
        control_output = self.control_head(control_tokens).squeeze(-1)
        delta = self.delta_head(treated_tokens - control_tokens).squeeze(-1)
        prediction = control + delta
        target_hvg = (
            batch["condition_hvg_vectors"].float()
            if self.input_mode == "official"
            else target
        )
        control_hvg = (
            batch["control_hvg_vectors"].float()
            if self.input_mode == "official"
            else control
        )
        return {
            "prediction": prediction,
            "prediction_hvg": prediction[..., self.hvg_indices],
            "target": target,
            "control": control,
            "target_hvg": target_hvg,
            "control_hvg": control_hvg,
            "treated_output": treated_output,
            "control_output": control_output,
            "delta": delta,
            "treated_cell_logits": treated_cell_logits,
            "control_cell_logits": control_cell_logits,
            "population_indices": population_indices,
            "batch_size": batch_size,
            "set_size": set_size,
        }

    def loss(
        self, output: dict, *, initial_phase: bool = False
    ) -> tuple[torch.Tensor, dict]:
        true_delta = output["target"] - output["control"]
        treated = F.mse_loss(output["treated_output"], output["target"])
        if output["population_indices"] is None:
            control = F.mse_loss(output["control_output"], output["control"])
        else:
            control = F.cross_entropy(
                output["treated_cell_logits"], output["population_indices"]
            ) + F.cross_entropy(
                output["control_cell_logits"], output["population_indices"]
            )
        delta = F.mse_loss(output["delta"], true_delta)
        first = output["delta"] - output["delta"].mean(1, keepdim=True)
        second = true_delta - true_delta.mean(1, keepdim=True)
        correlation = (first * second).sum(1) / (
            first.square().sum(1).mul(second.square().sum(1)).clamp_min(1e-6).sqrt()
        )
        correlation_loss = 1 - correlation.mean()
        if initial_phase:
            regression = (
                treated.sqrt(),
                control if output["population_indices"] is not None else control.sqrt(),
                delta.sqrt(),
            )
        else:
            regression = (treated, control, delta)
        value = (
            self.hparams["treated_weight"] * regression[0]
            + self.hparams["control_weight"] * regression[1]
            + self.hparams["delta_weight"] * regression[2]
            + self.hparams["correlation_weight"] * correlation_loss
        )
        if (
            not initial_phase
            and self.hparams["loss_reduction_scale"] == "sample_count"
        ):
            sample_count = output["target"].shape[0]
            value = value * sample_count
        return value, {
            "treated": treated.detach(),
            "control": control.detach(),
            "delta": delta.detach(),
            "correlation": correlation.detach().mean(),
        }

    @torch.no_grad()
    def predict_batch(self, batch: dict) -> torch.Tensor:
        output = self(batch)
        prediction = output["prediction_hvg"]
        if output["set_size"] > 1:
            batch_size, set_size = output["batch_size"], output["set_size"]
            prediction = prediction.reshape(
                batch_size, set_size, self.gene_dim
            ).mean(dim=1)
        return prediction

    def configuration(self) -> dict:
        return {
            "model": self.model_name,
            "hvg_dim": self.hvg_dim,
            "gene_dim": self.gene_dim,
            "full_gene_dim": self.full_gene_dim,
            **self.hparams,
        }


__all__ = ["XPert"]
