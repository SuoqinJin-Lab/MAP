from __future__ import annotations

import re
from pathlib import Path

import torch
import torch.nn.functional as F
from rdkit import Chem
from torch import nn

from .components import ResidualProjector


SMILES_REGEX = r"\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9]"


class MolTokenizer:
    def __init__(self, vocab_path: str | Path):
        tokens = [value for value in Path(vocab_path).read_text(encoding="utf-8").splitlines() if value]
        self.vocab = {token: index for index, token in enumerate(tokens)}
        self.pad_token = tokens[0]
        self.unk_id = self.vocab[tokens[1]]
        extras = tokens[6:272]
        prefix = "|".join(re.escape(token) for token in extras)
        pattern = f"({prefix + '|' if prefix else ''}{SMILES_REGEX}|.)"
        self.pattern = re.compile(pattern)
        self.begin = tokens[2]
        self.end = tokens[3]

    def encode(self, values: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        sequences = []
        for value in values:
            tokens = [self.begin, *self.pattern.findall(value), self.end]
            sequences.append([self.vocab.get(token, self.unk_id) for token in tokens])
        length = max(len(value) for value in sequences)
        ids = [value + [self.vocab[self.pad_token]] * (length - len(value)) for value in sequences]
        masks = [[False] * len(value) + [True] * (length - len(value)) for value in sequences]
        return torch.tensor(ids, dtype=torch.long).T, torch.tensor(masks, dtype=torch.bool).T


class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.attn_dropout = nn.Dropout(dropout)
        self.query_key_value = nn.Linear(embed_dim, 3 * embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.key_value = nn.Linear(embed_dim, 2 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query: torch.Tensor, key_padding_mask: torch.Tensor | None = None):
        length, batch_size, _ = query.shape
        query, key, value = torch.split(self.query_key_value(query), self.embed_dim, dim=-1)
        query = (query * self.scaling).contiguous().view(length, batch_size * self.num_heads, self.head_dim).transpose(0, 1)
        key = key.contiguous().view(length, batch_size * self.num_heads, self.head_dim).transpose(0, 1)
        value = value.contiguous().view(length, batch_size * self.num_heads, self.head_dim).transpose(0, 1)
        weights = torch.bmm(query, key.transpose(1, 2))
        if key_padding_mask is not None:
            weights = weights.view(batch_size, self.num_heads, length, length)
            weights = weights.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
            weights = weights.view(batch_size * self.num_heads, length, length)
        probabilities = self.attn_dropout(F.softmax(weights, dim=-1))
        output = torch.bmm(probabilities, value).transpose(0, 1).contiguous()
        output = output.view(length, batch_size, self.embed_dim)
        return self.out_proj(output)


class EncoderLayer(nn.Module):
    def __init__(self, embed_dim: int = 256, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiheadAttention(embed_dim, num_heads, dropout)
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.activation_dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(embed_dim, 4 * embed_dim)
        self.fc2 = nn.Linear(4 * embed_dim, embed_dim)
        self.final_layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, values: torch.Tensor, encoder_padding_mask: torch.Tensor | None = None):
        residual = values
        values = self.self_attn_layer_norm(values)
        values = residual + self.attn_dropout(self.self_attn(values, encoder_padding_mask))
        residual = values
        values = self.fc2(self.activation_dropout(F.gelu(self.fc1(self.final_layer_norm(values)))))
        return residual + self.attn_dropout(values)


class MoleculeEncoderStack(nn.Module):
    def __init__(self, layers: int = 4, width: int = 256):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(width) for _ in range(layers)])
        self.norm = nn.LayerNorm(width)

    def forward(self, values: torch.Tensor, padding: torch.Tensor | None = None):
        for layer in self.layers:
            values = layer(values, encoder_padding_mask=padding)
        return self.norm(values)


class MoleculeTransformer(nn.Module):
    def __init__(self, vocab_size: int = 523, width: int = 256, max_length: int = 512):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, width)
        self.encoder = MoleculeEncoderStack(width=width)
        self.register_buffer("pos_emb", torch.zeros(max_length, width))

    def encode(self, ids: torch.Tensor, padding: torch.Tensor) -> torch.Tensor:
        values = self.emb(ids) + self.pos_emb[:ids.shape[0], None, :]
        return self.encoder(values, padding.T)


class MolSTMEncoder(nn.Module):
    def __init__(self, vocab_path: str | Path):
        super().__init__()
        self.tokenizer = MolTokenizer(vocab_path)
        self._model = MoleculeTransformer(vocab_size=len(self.tokenizer.vocab))

    def forward(self, smiles: list[str]) -> torch.Tensor:
        normalized = []
        for value in smiles:
            molecule = Chem.MolFromSmiles(value)
            normalized.append(Chem.MolToSmiles(molecule) if molecule is not None else "C")
        ids, padding = self.tokenizer.encode(normalized)
        device = next(self.parameters()).device
        ids, padding = ids.to(device), padding.to(device)
        memory = self._model.encode(ids[:512], padding[:512])
        valid = (~padding[:512]).float()
        weights = valid / (valid.sum(dim=0, keepdim=True) + 1e-9)
        return (memory * weights.unsqueeze(-1)).sum(dim=0)


class MAPKGEncoder(nn.Module):
    def __init__(
        self,
        *,
        vocab_path: str | Path,
        d_model: int = 1024,
    ):
        super().__init__()
        self.smiles_encoder = MolSTMEncoder(vocab_path)
        self.smiles_projector = ResidualProjector(256, d_model, 512)
        self.gene_projector = ResidualProjector(5120, d_model, 1536)

    def load_from_full_model(self, checkpoint: str | Path) -> "MAPKGEncoder":
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        source = payload.get("model_state_dict", payload)
        source = {key.removeprefix("module."): value for key, value in source.items()}
        wanted = set(self.state_dict())
        state = {key: value for key, value in source.items() if key in wanted}
        missing, unexpected = self.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                "MAP-KG checkpoint does not match the frozen entity encoders: "
                f"missing={missing}, unexpected={unexpected}"
            )
        return self

    def forward(self, smiles: list[str]) -> torch.Tensor:
        return self.smiles_projector(self.smiles_encoder(smiles))

    def encode_genes(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.gene_projector(embeddings)
