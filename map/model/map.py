from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .components import CellProjector, ProjectOut, ResidualProjector
from .decoder import LatentToGeneDecoder
from .molecule import MAPKGEncoder
from .state import StateEncoder
from .transformer import map_transformer
from .._common.components import normalize_batch_components


class PerturbationEncoder(nn.Module):
    def __init__(
        self,
        *,
        mapkg_checkpoint: str | Path,
        mapkg_vocab: str | Path,
        static_token_cache: str | Path | None,
        num_gene_tokens: int = 2048,
        hvg_dim: int = 2000,
        combination_fusion: str = "avg_emb",
        max_components: int = 2,
    ):
        super().__init__()
        self.dim_emb = 1024
        self.num_gene_tokens = int(num_gene_tokens)
        self.combination_fusion = str(combination_fusion).casefold()
        if self.combination_fusion not in {"avg_emb", "two_tokens"}:
            raise ValueError("combination_fusion must be avg_emb or two_tokens")
        # ``avg_emb`` consumes any number of components through one fused token;
        # ``two_tokens`` keeps a fixed-width token block and therefore needs a
        # padding limit.
        self.max_components = int(max_components)
        if self.max_components <= 0:
            raise ValueError("max_components must be positive")
        self.cell_projector = CellProjector(4, 2048, self.dim_emb, dropout=0.1)
        self.mapkg_encoder = MAPKGEncoder(
            vocab_path=mapkg_vocab,
            d_model=self.dim_emb,
        )
        self.mapkg_encoder.load_from_full_model(mapkg_checkpoint).eval()
        for parameter in self.mapkg_encoder.parameters():
            parameter.requires_grad = False
        self.gene_tokens_projector = ResidualProjector(
            2048 + self.dim_emb, self.dim_emb, 2048
        )
        self.transformer_backbone = map_transformer(
            self.num_gene_tokens, self.dim_emb, self.max_components
        )
        self.project_out = ProjectOut(self.dim_emb, 2048, dropout=0.1)
        self.gene_decoder = LatentToGeneDecoder(2048, hvg_dim, [512, 1024], 0.1)
        self._gene_knowledge_table = None
        self._cached_gene_knowledge_cpu = None
        self._cached_drug_tokens_cpu = None
        self._cached_drug_tokens_device = None
        self._cached_smiles_to_index = None
        if static_token_cache is not None:
            self._load_static_token_cache(static_token_cache)

    def _load_static_token_cache(self, path: str | Path) -> None:
        cache = torch.load(path, map_location="cpu", weights_only=False)
        if cache.get("format") != "map_static_token_cache_v1":
            raise ValueError(f"Unsupported MAP static-token cache: {path}")
        gene_tokens = cache.get("gene_tokens")
        drug_tokens = cache.get("drug_tokens")
        drug_smiles = cache.get("drug_smiles")
        if not isinstance(gene_tokens, torch.Tensor) or gene_tokens.ndim != 2:
            raise ValueError("Static cache gene_tokens must be rank 2")
        if not isinstance(drug_tokens, torch.Tensor) or drug_tokens.ndim != 2:
            raise ValueError("Static cache drug_tokens must be rank 2")
        if gene_tokens.shape[1] != 1024 or drug_tokens.shape[1] != 1024:
            raise ValueError("Static token width must be 1024")
        if not isinstance(drug_smiles, list) or len(drug_smiles) != len(drug_tokens):
            raise ValueError("Static cache drug_smiles does not match drug_tokens")
        self._cached_gene_knowledge_cpu = gene_tokens.contiguous()
        self._cached_drug_tokens_cpu = drug_tokens.contiguous()
        self._cached_smiles_to_index = {value: index for index, value in enumerate(drug_smiles)}

    def encode_drug(self, smiles: list[str], reference: torch.Tensor) -> torch.Tensor:
        if self._cached_smiles_to_index is not None:
            indices = [self._cached_smiles_to_index.get(value) for value in smiles]
            if all(index is not None for index in indices):
                if (
                    self._cached_drug_tokens_device is None
                    or self._cached_drug_tokens_device.device != reference.device
                    or self._cached_drug_tokens_device.dtype != reference.dtype
                ):
                    self._cached_drug_tokens_device = self._cached_drug_tokens_cpu.to(
                        device=reference.device, dtype=reference.dtype
                    )
                return self._cached_drug_tokens_device.index_select(
                    0,
                    torch.tensor(indices, device=reference.device, dtype=torch.long),
                )
        with torch.no_grad():
            return self.mapkg_encoder(smiles).to(dtype=reference.dtype)

    def _component_tokens(self, smiles, doses, reference: torch.Tensor):
        batch_size = len(smiles) if not isinstance(smiles, str) else 1
        smile_batch, dose_batch = normalize_batch_components(smiles, doses, batch_size)
        flat_smiles = [value for row in smile_batch for value in row]
        encoded = self.encode_drug(flat_smiles, reference)
        offsets = []
        cursor = 0
        for row in smile_batch:
            offsets.append(encoded[cursor:cursor + len(row)])
            cursor += len(row)
        scaled = [tokens * torch.log10(torch.tensor(row_doses, device=reference.device, dtype=reference.dtype) + 1).unsqueeze(-1)
                  for tokens, row_doses in zip(offsets, dose_batch)]
        return scaled

    def _gene_table(self, esm_embeddings: torch.Tensor) -> torch.Tensor:
        if self._gene_knowledge_table is None or self._gene_knowledge_table.device != esm_embeddings.device:
            if self._cached_gene_knowledge_cpu is not None:
                if len(self._cached_gene_knowledge_cpu) != len(esm_embeddings):
                    raise ValueError("Static gene cache does not match the STATE gene table")
                self._gene_knowledge_table = self._cached_gene_knowledge_cpu.to(esm_embeddings.device)
            else:
                with torch.no_grad():
                    self._gene_knowledge_table = torch.cat([
                        self.mapkg_encoder.encode_genes(chunk)
                        for chunk in esm_embeddings.split(512)
                    ]).detach()
        return self._gene_knowledge_table

    def forward(
        self,
        gene_tokens: torch.Tensor,
        cell_embeddings: torch.Tensor,
        gene_ids: torch.Tensor,
        esm_embeddings: torch.Tensor,
        smiles: list[str],
        doses: torch.Tensor,
    ):
        batch_size = len(smiles)
        bulk_cells = cell_embeddings.shape[0]
        if bulk_cells % batch_size:
            raise ValueError("Cell set is not divisible by the condition batch")
        set_size = bulk_cells // batch_size
        cell_tokens = self.cell_projector(cell_embeddings)
        component_tokens = self._component_tokens(smiles, doses, cell_embeddings)
        if self.combination_fusion == "avg_emb":
            drug_tokens = torch.stack([tokens.mean(dim=0) for tokens in component_tokens])
        else:
            if any(len(tokens) > self.max_components for tokens in component_tokens):
                raise ValueError(f"Input has more than max_components={self.max_components}")
        knowledge = torch.nn.functional.embedding(gene_ids.long(), self._gene_table(esm_embeddings))
        fused = self.gene_tokens_projector(torch.cat([gene_tokens, knowledge.to(gene_tokens.dtype)], dim=-1))
        if self.combination_fusion == "avg_emb":
            drug_tokens = drug_tokens[:, None, :].expand(batch_size, set_size, -1).reshape(bulk_cells, -1)
            combined = torch.cat([drug_tokens[:, None], cell_tokens[:, None], fused], dim=1)
            cell_position = 1
        else:
            padded = cell_embeddings.new_zeros((batch_size, self.max_components, self.dim_emb))
            component_mask = torch.zeros(
                (batch_size, self.max_components), device=cell_embeddings.device,
                dtype=torch.long,
            )
            for index, tokens in enumerate(component_tokens):
                padded[index, :len(tokens)] = tokens
                component_mask[index, :len(tokens)] = 1
            drug_tokens = padded[:, None].expand(batch_size, set_size, -1, -1).reshape(
                bulk_cells, self.max_components, self.dim_emb
            )
            combined = torch.cat([drug_tokens, cell_tokens[:, None], fused], dim=1)
            cell_position = self.max_components
            attention_mask = torch.cat([
                component_mask[:, None].expand(batch_size, set_size, -1).reshape(bulk_cells, self.max_components),
                torch.ones((bulk_cells, 1 + fused.shape[1]), device=cell_embeddings.device, dtype=torch.long),
            ], dim=1)
        updated_cell = self.transformer_backbone(
            inputs_embeds=combined,
            attention_mask=attention_mask if self.combination_fusion == "two_tokens" else None,
        ).last_hidden_state[:, cell_position]
        # Method 4.4 cell-token residual is part of the canonical MAP model.
        cell_update = updated_cell + cell_tokens
        predicted_embeddings = self.project_out(cell_update)
        predicted_expression = self.gene_decoder(predicted_embeddings)
        return (
            predicted_embeddings.reshape(batch_size, set_size, -1),
            predicted_expression.reshape(batch_size, set_size, -1),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.mapkg_encoder.eval()
        return self


class MAPModel(nn.Module):
    """MAP initialized with the frozen components from Method 4.4."""

    def __init__(
        self,
        *,
        se_checkpoint: str | Path,
        esm_embeddings: str | Path,
        mapkg_checkpoint: str | Path,
        mapkg_vocab: str | Path,
        static_token_cache: str | Path | None,
        num_gene_tokens: int = 2048,
        hvg_dim: int = 2000,
        combination_fusion: str = "avg_emb",
        max_components: int = 2,
    ):
        super().__init__()
        self.num_gene_tokens = int(num_gene_tokens)
        self.state = StateEncoder.from_pretrained(se_checkpoint, esm_embeddings)
        self.pert_model = PerturbationEncoder(
            mapkg_checkpoint=mapkg_checkpoint,
            mapkg_vocab=mapkg_vocab,
            static_token_cache=static_token_cache,
            num_gene_tokens=num_gene_tokens,
            hvg_dim=hvg_dim,
            combination_fusion=combination_fusion,
            max_components=max_components,
        )

    def encode_state(self, gene_ids, expressions, return_gene_tokens=False):
        return self.state.encode(
            gene_ids, expressions, return_gene_tokens=return_gene_tokens
        )

    def forward(self, control_gene_ids, control_expressions, smiles, doses):
        batch_size, set_size, token_length = control_gene_ids.shape
        expected_length = self.num_gene_tokens + 1
        if token_length != expected_length:
            raise ValueError(
                "STATE input length must include one SPECIAL token and "
                f"{self.num_gene_tokens} gene tokens; expected {expected_length}, "
                f"got {token_length}"
            )
        gene_ids = control_gene_ids.reshape(batch_size * set_size, token_length)
        expressions = control_expressions.reshape(batch_size * set_size, token_length)
        with torch.no_grad():
            gene_tokens, cell_embeddings = self.encode_state(
                gene_ids, expressions, return_gene_tokens=True
            )
        return self.pert_model(
            gene_tokens,
            cell_embeddings,
            gene_ids[:, 1:],
            self.state.gene_embedding_table,
            smiles,
            doses,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.state.eval()
        return self
