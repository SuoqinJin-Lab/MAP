from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

from map.preparation.xpert import build_dds, read_primekg_dti, read_string_ppi
from map.train.baselines.cmonge.model import CMonge
from map.train.baselines.crisp.model import CRISP
from map.train.baselines.xpert.model import XPert


def test_string_and_primekg_mapping(tmp_path):
    info = tmp_path / "protein.info.gz"
    links = tmp_path / "protein.links.gz"
    primekg = tmp_path / "primekg.csv"
    aliases = tmp_path / "aliases.tsv"
    pd.DataFrame({
        "#string_protein_id": ["9606.ENSP1", "9606.ENSP2", "9606.ENSP3"],
        "preferred_name": ["TP53", "EGFR", "NOT_IN_HVG"],
    }).to_csv(info, sep="\t", index=False, compression="gzip")
    pd.DataFrame({
        "protein1": ["9606.ENSP1", "9606.ENSP1", "9606.ENSP2"],
        "protein2": ["9606.ENSP2", "9606.ENSP3", "9606.ENSP1"],
        "combined_score": [701, 999, 700],
    }).to_csv(links, sep=" ", index=False, compression="gzip")
    edges, weights, stats = read_string_ppi(links, info, ["TP53", "EGFR"])
    assert edges.tolist() == [[0, 1], [1, 0]]
    assert np.allclose(weights, [0.701, 0.701])
    assert stats["undirected_edges"] == 1

    pd.DataFrame([
        {"x_type": "drug", "x_name": "Drug Alpha", "y_type": "gene/protein", "y_name": "TP53"},
        {"x_type": "gene/protein", "x_name": "EGFR", "y_type": "drug", "y_name": "Beta salt"},
        {"x_type": "disease", "x_name": "ignored", "y_type": "gene/protein", "y_name": "TP53"},
    ]).to_csv(primekg, index=False)
    pd.DataFrame({
        "alias": ["Beta salt"],
        "canonical_smiles": ["CCC"],
    }).to_csv(aliases, sep="\t", index=False)
    smiles = ["CC", "CCC"]
    names = {"CC": ["Drug Alpha"], "CCC": ["different name"]}
    dti, dti_stats = read_primekg_dti(
        primekg, smiles, names, ["TP53", "EGFR"], aliases_file=aliases
    )
    assert dti.tolist() == [[0, 1], [0, 1]]
    assert dti_stats["mapped_drugs"] == 2
    assert dti_stats["drug_coverage"] == 1.0


def test_dds_is_directed_and_excludes_self_edges():
    edges, weights = build_dds(["CC", "CCC", "c1ccccc1"], threshold=0.1)
    assert not np.any(edges[0] == edges[1])
    assert edges.shape[1] == len(weights)
    assert {(int(a), int(b)) for a, b in edges.T} == {
        (int(b), int(a)) for a, b in edges.T
    }


def _write_assets(tmp_path):
    root = tmp_path / "baselines"
    (root / "crisp").mkdir(parents=True)
    (root / "cmonge").mkdir(parents=True)
    (root / "xpert").mkdir(parents=True)
    smiles = ["CC", "CCC"]
    np.save(root / "crisp" / "rdkit2d.float32.npy", np.arange(12, dtype=np.float32).reshape(2, 6))
    np.save(root / "cmonge" / "rdkit2d.float32.npy", np.arange(12, dtype=np.float32).reshape(2, 6))
    np.save(root / "xpert" / "ppi_gene_vectors.float32.npy", np.arange(16, dtype=np.float32).reshape(4, 4))
    np.save(root / "xpert" / "drug_hg_embeddings.float32.npy", np.arange(8, dtype=np.float32).reshape(2, 4))
    unimol = np.zeros((2, 5, 514), dtype=np.float16)
    unimol[:, :3, 0] = 1
    unimol[:, :3, 2:] = 0.25
    np.save(root / "xpert" / "drug_unimol.float16.npy", unimol)
    (root / "manifest.json").write_text(json.dumps({
        "format_version": 3,
        "drug_vocabulary": smiles,
        "models": {
            "crisp": {"model": "crisp", "directory": "crisp"},
            "cmonge": {
                "model": "cmonge", "directory": "cmonge",
                "matrix": "rdkit2d.float32.npy",
            },
            "xpert": {"model": "xpert", "directory": "xpert"},
        },
    }))
    return smiles


def _batch(smiles):
    return {
        "control_embeddings": torch.randn(2, 2, 3),
        "control_hvg_vectors": torch.rand(2, 2, 4),
        "condition_hvg_vectors": torch.rand(2, 2, 4),
        "drug_smiles": smiles,
        "drug_conc": torch.tensor([0.3, 1.0]),
        "population": ["A", "B"],
    }


def test_crisp_pertae_forward_and_loss(tmp_path):
    smiles = _write_assets(tmp_path)
    model = CRISP(
        tmp_path,
        4,
        ["A", "B"],
        fm_dim=3,
        latent_dim=4,
        encoder_width=8,
        encoder_depth=1,
        decoder_width=8,
        decoder_depth=1,
        embedding_encoder_width=8,
        embedding_encoder_depth=1,
        doser_width=8,
        doser_depth=1,
        cell_predictor_width=8,
        deg_top_k=2,
    )
    batch = _batch(smiles)
    output = model(batch)
    loss, metrics = model.loss(output, batch)
    assert output["prediction"].shape == (2, 2, 4)
    assert torch.isfinite(loss)
    assert set(metrics) == {"mse", "autofocus", "celltype", "contrastive", "mmd", "kld"}
    model.eval()
    assert model.predict_batch(batch).shape == (2, 4)

    batch.update({
        "negative_control_embeddings": batch["control_embeddings"].flip(0),
        "negative_condition_hvg_vectors": batch["condition_hvg_vectors"].flip(0),
        "negative_control_hvg_vectors": batch["control_hvg_vectors"].flip(0),
        "negative_drug_smiles": list(reversed(smiles)),
        "negative_drug_conc": batch["drug_conc"].flip(0),
        "negative_population": ["B", "A"],
        "negative_condition_id": torch.tensor([1, 0]),
    })
    model.train()
    paired_output = model(batch)
    paired_loss, _ = model.loss(paired_output, batch)
    assert "negative_prediction" in paired_output
    assert torch.isfinite(paired_loss)


def test_xpert_forward_and_loss(tmp_path):
    smiles = _write_assets(tmp_path)
    model = XPert(
        tmp_path,
        4,
        ["A", "B"],
        hidden_size=4,
        attention_heads=2,
        treated_structure="CA+SA",
        control_structure="SA+SA",
        expression_bins=4,
        expression_max=2.0,
    )
    batch = _batch(smiles)
    output = model(batch)
    loss, metrics = model.loss(output)
    assert output["prediction"].shape == (2, 4)
    assert torch.isfinite(loss)
    assert set(metrics) == {"treated", "control", "delta", "correlation"}
    model.eval()
    assert model.predict_batch(batch).shape == (2, 4)


def test_cmonge_distribution_transport_and_loss(tmp_path):
    smiles = _write_assets(tmp_path)
    model = CMonge(
        tmp_path,
        4,
        ["A", "B"],
        ae_width=8,
        latent_dim=3,
        context_dim=3,
        hidden_sizes=(8, 8),
    )
    batch = _batch(smiles)
    reconstruction = model(batch, stage="autoencoder")
    assert reconstruction.shape == (8, 4)
    assert torch.isfinite(model.autoencoder_loss(
        reconstruction,
        torch.cat((batch["control_hvg_vectors"], batch["condition_hvg_vectors"]), dim=1).reshape(-1, 4),
    ))
    model.freeze_autoencoder()
    output = model(batch)
    loss, metrics = model.loss(output, batch)
    assert output["prediction"].shape == (2, 2, 4)
    assert torch.isfinite(loss)
    assert set(metrics) == {"sinkhorn", "monge_gap"}
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.transport.parameters())
    assert all(parameter.grad is None for parameter in model.autoencoder.parameters())
    model.eval()
    assert model.transport_batch(batch).shape == (2, 2, 4)
    assert model.predict_batch(batch).shape == (2, 4)


def test_xpert_expression_bins_match_official_digitize(tmp_path):
    smiles = _write_assets(tmp_path)
    model = XPert(
        tmp_path,
        4,
        ["A", "B"],
        hidden_size=4,
        attention_heads=2,
        treated_structure="SA",
        control_structure="SA",
        expression_bins=4,
        expression_min=0.0,
        expression_max=3.0,
    )
    values = torch.tensor([[0.0, 0.5, 1.5, 3.0]])
    assert model._expression_bins(values).tolist() == [[1, 1, 2, 3]]


def test_xpert_padding_tokens_do_not_change_prediction(tmp_path):
    smiles = _write_assets(tmp_path)
    model = XPert(
        tmp_path,
        4,
        ["A", "B"],
        hidden_size=4,
        attention_heads=2,
        treated_structure="CA+SA",
        control_structure="SA",
        expression_bins=4,
        expression_max=2.0,
    ).eval()
    batch = _batch(smiles)
    first = model.predict_batch(batch)
    model.drug_unimol[:, 3:, 2:] = 1000
    second = model.predict_batch(batch)
    assert torch.allclose(first, second, atol=1e-6, rtol=1e-6)
