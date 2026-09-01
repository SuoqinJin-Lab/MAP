from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import nn


class SkipBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.intermediate_dense = nn.Linear(width, width * 2)
        self.dense = nn.Linear(width * 2, width)
        self.activation = nn.ReLU()
        self.layer_norm = nn.LayerNorm(width)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = values
        values = self.dense(self.activation(self.intermediate_dense(values)))
        return self.layer_norm(values + residual)


class FlashTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dropout = dropout
        self.qkv_proj = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout_layer = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

    def forward(self, src: torch.Tensor, **_: object) -> torch.Tensor:
        residual = src
        query, key, value = torch.chunk(self.qkv_proj(src), 3, dim=-1)
        head_dim = self.d_model // self.nhead
        shape = (src.size(0), src.size(1), self.nhead, head_dim)
        query = query.view(shape).transpose(1, 2)
        key = key.view(shape).transpose(1, 2)
        value = value.view(shape).transpose(1, 2)
        attention = F.scaled_dot_product_attention(
            query, key, value,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        attention = attention.transpose(1, 2).contiguous().view_as(src)
        src = self.norm1(residual + self.dropout_layer(self.out_proj(attention)))
        residual = src
        feed_forward = self.linear2(self.dropout_layer(F.gelu(self.linear1(src))))
        return self.norm2(residual + self.dropout_layer(feed_forward))


class FlashTransformerEncoder(nn.Module):
    def __init__(self, layers: list[nn.Module]):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, src: torch.Tensor, **kwargs: object) -> torch.Tensor:
        for layer in self.layers:
            src = layer(src, **kwargs)
        return src


class StateEmbeddingModel(nn.Module):
    """Inference-only SE-600M graph using the released parameter names."""

    def __init__(self, *, token_dim: int = 5120, d_model: int = 2048, nlayers: int = 16):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, token_dim))
        self.encoder = nn.Sequential(
            nn.Linear(token_dim, d_model), nn.LayerNorm(d_model), nn.SiLU()
        )
        self.transformer_encoder = FlashTransformerEncoder([
            FlashTransformerEncoderLayer(d_model, 16, d_model, 0.1)
            for _ in range(nlayers)
        ])
        self.decoder = nn.Sequential(SkipBlock(d_model), nn.Linear(d_model, d_model))
        self.bin_encoder = nn.Embedding(10, d_model)
        self.count_encoder = nn.Sequential(
            nn.Linear(1, 512), nn.LeakyReLU(), nn.Linear(512, 10)
        )
        self.dataset_token = nn.Parameter(torch.randn(1, token_dim))
        self.dataset_embedder = nn.Linear(d_model, 10)
        self.dataset_encoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.LayerNorm(d_model),
            nn.Dropout(0.1),
            nn.Linear(d_model, 14_420),
        )
        auxiliary_width = d_model + d_model + 11
        self.binary_decoder = nn.Sequential(
            SkipBlock(auxiliary_width),
            SkipBlock(auxiliary_width),
            nn.Linear(auxiliary_width, 1),
        )
        self.register_buffer("_bin_indices_cached", torch.arange(10))
        self.pe_embedding: nn.Embedding | None = None

    def forward(self, src: torch.Tensor, counts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        src = self.encoder(src) * (self.encoder[0].out_features ** 0.5)
        weights = F.softmax(self.count_encoder(counts.unsqueeze(-1)), dim=-1)
        count_embedding = torch.matmul(weights, self.bin_encoder(self._bin_indices_cached))
        dataset_count = count_embedding.new_zeros(count_embedding.size(0), 1, count_embedding.size(2))
        src = src + torch.cat([count_embedding, dataset_count], dim=1)
        output = self.transformer_encoder(src)
        gene_output = self.decoder(output)
        embedding = F.normalize(gene_output[:, 0, :], dim=1)
        return gene_output, embedding


def load_gene_embeddings(path: str | Path) -> torch.Tensor:
    value = torch.load(path, map_location="cpu", weights_only=False)
    matrix = torch.vstack(list(value.values())) if isinstance(value, dict) else value
    if not isinstance(matrix, torch.Tensor) or matrix.ndim != 2 or matrix.shape[1] != 5120:
        raise ValueError(f"Expected the STATE ESM2 gene table [genes, 5120], got {type(value)!r}")
    return matrix.float().contiguous()


class StateEncoder(nn.Module):
    def __init__(self, model: StateEmbeddingModel, gene_embeddings: torch.Tensor):
        super().__init__()
        self.model = model
        self.model.pe_embedding = nn.Embedding.from_pretrained(gene_embeddings, freeze=True)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str | Path,
        gene_embeddings: str | Path,
    ) -> "StateEncoder":
        model = StateEmbeddingModel()
        checkpoint_state = load_file(str(checkpoint))
        wanted = set(model.state_dict())
        state = OrderedDict(
            (key, value) for key, value in checkpoint_state.items() if key in wanted
        )
        aliases = (
            ("gene_embedding_layer.0.weight", "encoder.0.weight"),
            ("gene_embedding_layer.0.bias", "encoder.0.bias"),
            ("gene_embedding_layer.1.weight", "encoder.1.weight"),
            ("gene_embedding_layer.1.bias", "encoder.1.bias"),
        )
        for source, target in aliases:
            if source in checkpoint_state and target in wanted and target not in state:
                state[target] = checkpoint_state[source]
        missing, unexpected = model.load_state_dict(state, strict=False)
        optional = {"pe_embedding.weight"}
        critical = [key for key in missing if key not in optional]
        if critical or unexpected:
            raise RuntimeError(
                "SE-600M checkpoint does not match the inference graph: "
                f"missing={critical}, unexpected={unexpected}"
            )
        return cls(model, load_gene_embeddings(gene_embeddings))

    @property
    def gene_embedding_table(self) -> torch.Tensor:
        assert self.model.pe_embedding is not None
        return self.model.pe_embedding.weight

    def encode(
        self,
        gene_ids: torch.Tensor,
        expressions: torch.Tensor,
        *,
        return_gene_tokens: bool = False,
    ):
        if gene_ids.ndim != 2 or expressions.shape != gene_ids.shape:
            raise ValueError("gene_ids and expressions must have shape [cells, tokens]")
        assert self.model.pe_embedding is not None
        src = F.normalize(self.model.pe_embedding(gene_ids), dim=2)
        src = torch.cat([self.model.cls_token.expand(src.size(0), 1, -1), src[:, 1:]], dim=1)
        src = torch.cat([src, self.model.dataset_token.expand(src.size(0), 1, -1)], dim=1)
        gene_output, embedding = self.model(src, expressions)
        if return_gene_tokens:
            return gene_output[:, 1:gene_ids.shape[1], :], embedding
        return embedding

    def forward(self, gene_ids: torch.Tensor, expressions: torch.Tensor):
        return self.encode(gene_ids, expressions)
