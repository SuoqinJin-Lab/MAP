from __future__ import annotations

import torch
from torch import nn
from transformers import LlamaConfig, LlamaModel


class NoRoPE(nn.Module):
    def __init__(self, head_dim: int):
        super().__init__()
        self.head_dim = head_dim

    def forward(self, hidden_states: torch.Tensor, position_ids: torch.Tensor):
        batch_size, sequence_length, _ = hidden_states.shape
        return (
            hidden_states.new_ones(batch_size, sequence_length, self.head_dim),
            hidden_states.new_zeros(batch_size, sequence_length, self.head_dim),
        )


class LlamaBidirectionalModel(LlamaModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.rotary_emb = NoRoPE(config.head_dim)

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor | None,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values,
        output_attentions: bool = False,
    ):
        return None


def map_transformer(num_gene_tokens: int, width: int = 1024, num_drug_tokens: int = 1) -> nn.Module:
    # The paper omits the head count; the released MAP implementation fixes it at 8.
    heads = 8
    config = LlamaConfig(
        max_position_embeddings=num_gene_tokens + int(num_drug_tokens) + 1,
        hidden_size=width,
        intermediate_size=2688,
        num_hidden_layers=4,
        num_attention_heads=heads,
        num_key_value_heads=heads,
        head_dim=width // heads,
        use_cache=False,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        rms_norm_eps=1e-6,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
    )
    model = LlamaBidirectionalModel(config)
    model.embed_tokens.weight.requires_grad = False
    model.embed_tokens.weight.data.zero_()
    return model
