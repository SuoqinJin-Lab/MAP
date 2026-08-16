from __future__ import annotations

from argparse import Namespace
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from map import DatasetPaths, eval, experiment_paths, preparation, preprocess, train
from map.preprocess.sources.tahoe import TahoeSource
from map.preprocess.sources.tahoe.paths import TahoePaths
from map._common.identifiers import evaluation_identifier, split_identifier, training_identifier
from map.eval.analysis import analyze_degs, analyze_embeddings, analyze_predictions, build_report
from map.preprocess.sources.tahoe.data import (
    create_tahoe_project,
    fetch_cell_lines,
    watch_data,
)
from map.preprocess.sources.tahoe.pipeline import run_filter
from map.preparation.embeddings import (
    embed_state,
    merge_perturbed_cell_embeddings,
    merge_static_tokens,
)
from map.preparation.processing import generate_splits
from map.train.engine import train_map
from map.train.program import (
    MAP_COMPILE_PARTS,
    compile_map_parts,
    load_checkpoint,
    save_checkpoint,
)
from map.train.baselines import build_model, load_model
from map._common.contracts import load_contract
from map._common.runtime import asset
from map.eval.program import _baseline_prediction


def make_tahoe(root: Path) -> TahoePaths:
    paths = TahoePaths(dataset="Tahoe-100M", source=root / "source", workspace=root / "workspace", frozen_models=root / "models")
    paths.raw_data.mkdir(parents=True)
    paths.raw_metadata.mkdir(parents=True)
    rows = [
        {"genes": [0, 1, 2, 3], "expressions": [10, 0, 2, 1], "cell_line_id": "CVCL_0131", "sample": "s1", "drug": "DMSO_TF", "canonical_smiles": "", "plate": "p1"},
        {"genes": [0, 1, 2, 3], "expressions": [4, 3, 0, 2], "cell_line_id": "CVCL_0131", "sample": "s2", "drug": "DrugA", "canonical_smiles": "CC", "plate": "p1"},
        {"genes": [0, 1, 2, 3], "expressions": [1, 1, 1, 1], "cell_line_id": "CVCL_1056", "sample": "s3", "drug": "DrugB", "canonical_smiles": "CCC", "plate": "p2"},
    ]
    pq.write_table(pa.Table.from_pylist(rows), paths.raw_data / "train-00000-of-00001.parquet")
    pq.write_table(pa.Table.from_pylist([
        {"token_id": 0, "gene_symbol": "ACTB"},
        {"token_id": 1, "gene_symbol": "MT-ND1"},
        {"token_id": 2, "gene_symbol": "GAPDH"},
        {"token_id": 3, "gene_symbol": "TP53"},
    ]), paths.raw_metadata / "gene_metadata.parquet")
    pq.write_table(pa.Table.from_pylist([
        {"sample": "s1", "drugname_drugconc": "[('DMSO_TF', 0, 'uM')]"},
        {"sample": "s2", "drugname_drugconc": "[('DrugA', 1, 'uM')]"},
        {"sample": "s3", "drugname_drugconc": "[('DrugB', 1, 'uM')]"},
    ]), paths.raw_metadata / "sample_metadata.parquet")
    return paths


def test_tahoe_source_is_only_the_native_data_boundary(tmp_path):
    source = tmp_path / "incoming" / "tahoe-native"
    projects = tmp_path / "experiments"
    frozen = tmp_path / "shared-models"
    native = TahoeSource(source, projects, frozen_models=frozen)
    assert native.source == source
    assert native.projects == projects
    assert native.frozen_models == frozen
    for name in ("watch_data", "fetch_cell_line", "prepare"):
        assert hasattr(native, name)
    assert not hasattr(native, "watch_cell_line")
    assert not hasattr(native, "watch_condition")
    assert not hasattr(native, "qc")
    assert not hasattr(native, "fetch_split")
    assert not hasattr(native, "train")
    assert not hasattr(native, "evaluate")


def test_internal_reader_keeps_explicit_roots(tmp_path):
    source = tmp_path / "incoming" / "tahoe-native"
    projects = tmp_path / "experiments"
    frozen = tmp_path / "shared-models"
    native = TahoeSource(source, projects, frozen_models=frozen)
    raw_data = source / "data"
    raw_metadata = source / "metadata"
    raw_data.mkdir(parents=True)
    raw_metadata.mkdir(parents=True)
    rows = [{"genes": [0], "expressions": [1], "cell_line_id": "P1", "sample": "s1", "drug": "DMSO_TF", "canonical_smiles": "", "plate": "g1"}]
    pq.write_table(pa.Table.from_pylist(rows), raw_data / "part.parquet")
    pq.write_table(pa.Table.from_pylist([{"token_id": 0, "gene_symbol": "ACTB"}]), raw_metadata / "gene_metadata.parquet")
    pq.write_table(pa.Table.from_pylist([{"sample": "s1", "drugname_drugconc": "[('DMSO_TF', 0, 'uM')]"}]), raw_metadata / "sample_metadata.parquet")
    selection = native.fetch_cell_line(["P1"])
    assert selection.source == source
    assert selection.frozen_models == frozen
    assert selection.cache_workspace.parent == projects
    assert not (projects / "paper-run").exists()


def test_explicit_roots_are_all_or_nothing(tmp_path):
    paths = experiment_paths(
        tmp_path / "workspace", tmp_path / "models", source=tmp_path / "source"
    )
    assert paths.source == tmp_path / "source"
    assert paths.prepared == tmp_path / "workspace" / "materialized"


def test_dataset_statistics_caches_are_source_specific(tmp_path):
    first = TahoePaths(
        dataset="Tahoe-100M",
        source=tmp_path / "raw" / "first",
        workspace=tmp_path / "projects" / "first",
        frozen_models=tmp_path / "models",
    )
    second = TahoePaths(
        dataset="Tahoe-100M",
        source=tmp_path / "raw" / "second",
        workspace=tmp_path / "projects" / "second",
        frozen_models=tmp_path / "models",
    )
    assert first.dataset_cache != second.dataset_cache
    assert first.dataset_cache == first.source
    assert second.dataset_cache == second.source


def test_four_stage_modules_are_public():
    assert preprocess.tahoe.statistics
    assert preprocess.tahoe.fetch_cell_line
    assert preprocess.tahoe.filter_conditions
    assert preprocess.tahoe.select_hvg
    assert preprocess.tahoe.materialize
    assert not hasattr(preparation, "stats")
    assert not hasattr(preparation, "select_hvg")
    assert not hasattr(preparation, "materialize_cells")
    assert preparation.create_workflow
    assert preparation.build_sampling_index
    assert preparation.create_split
    assert preparation.precache_gene_tokens
    assert preparation.precache_drug_tokens
    assert preparation.precache_state_embeddings
    assert train.run
    assert eval.run


def test_generic_preparation_train_and_eval_sources_are_dataset_independent():
    root = Path(__file__).parents[1] / "map"
    paths = [
        *(root / "preparation").glob("*.py"),
        *(root / "train").glob("*.py"),
        *(root / "eval").glob("*.py"),
    ]
    for path in paths:
        if path.name in {"__init__.py", "program.py"}:
            continue
        source = path.read_text(encoding="utf-8").casefold()
        assert "tahoe" not in source, path
        assert "paper_cell_lines" not in source, path


def test_package_has_no_reference_repository_dependency(tmp_path):
    source_root = Path(__file__).parents[1]
    copied = tmp_path / "standalone"
    shutil.copytree(source_root / "map", copied / "map")
    code = "import map; from map import preparation, train, eval; print('ok')"
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=copied,
        check=True,
        text=True,
        capture_output=True,
        env={"PYTHONPATH": str(copied)},
    )
    assert completed.stdout.strip() == "ok"
    forbidden = (
        "find_reference_root",
        "precompute_static_tokens.py",
        "from model.",
        "from data.",
        "/absolute/reference/map",
    )
    for path in (copied / "map").rglob("*.py"):
        text = path.read_text(encoding="utf-8").casefold()
        for value in forbidden:
            assert value.casefold() not in text, (path, value)


def test_watch_fetch_and_create_project(tmp_path, monkeypatch):
    paths = make_tahoe(tmp_path)
    projects = paths.workspace.parent
    overview = watch_data(paths.source, projects, batch_size=2)
    assert overview["cells"] == 3
    assert overview["cells_by_cell_line"]["CVCL_0131"] == 2
    assert overview["control_cells"] == 1
    assert overview["condition_cells"] == 2
    assert overview["drugs"] == 2
    assert overview["conditions"] == 2
    assert overview["excluded_perturbation_cells"] == 0
    assert overview["cell_line_ids"] == ["CVCL_0131", "CVCL_1056"]
    assert (paths.source / "data_summary.json").is_file()
    assert not (paths.source / "dose_distribution.svg").exists()
    assert not (paths.source / "cells_by_cell_line.svg").exists()

    from map.preprocess.sources.tahoe import data as tahoe_data

    monkeypatch.setattr(
        tahoe_data,
        "watch_data",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected rescan")),
    )
    fetched = fetch_cell_lines(paths, ["CVCL_0131", "CVCL_1056"], batch_size=2)
    assert fetched["cells"] == 3
    contract = create_tahoe_project(paths)
    assert set(contract["entities"]) == {"condition", "control_cell", "condition_cell"}
    reopened = experiment_paths(paths.workspace, paths.frozen_models)
    assert reopened.source == paths.source.resolve()
    assert load_contract(paths.workspace)["populations"] == ["CVCL_0131", "CVCL_1056"]
    assert contract["condition_summary"] == str(paths.condition_summary.resolve())
    rows = pq.read_table(paths.condition_summary).to_pylist()
    assert sum(row["cells"] for row in rows if row["kind"] == "control") == 1
    assert not list(paths.staging.rglob("*.html"))


def test_projects_are_independent_and_created_by_materialization(tmp_path, monkeypatch):
    seed_paths = make_tahoe(tmp_path / "seed")
    native = TahoeSource(
        seed_paths.source, tmp_path / "projects", frozen_models=tmp_path / "models"
    )
    first = native.fetch_cell_line(["CVCL_0131", "CVCL_1056"])
    second = native.fetch_cell_line(["CVCL_0131"])
    assert first.cache_workspace != second.cache_workspace
    assert first.contract_payload["populations"] == ["CVCL_0131", "CVCL_1056"]
    assert second.contract_payload["populations"] == ["CVCL_0131"]
    first.reserve_project("two-lines")
    assert (tmp_path / "projects" / "two-lines" / "preprocess.json").is_file()

    first.prepared.mkdir(parents=True, exist_ok=True)
    (first.prepared / "condition_filter.json").write_text(
        json.dumps(
            {
                "filter_id": "toy-filter",
                "min_cells": 1,
                "max_cells": 5000,
                "seed": 42,
                "retained_conditions": 1,
                "retained_condition_cells": 1,
            }
        ),
        encoding="utf-8",
    )
    (first.prepared / "hvg.json").write_text("{}", encoding="utf-8")
    captured = {}

    def fake_materialize(*args, **kwargs):
        captured.update(kwargs)
        return "ok"

    monkeypatch.setattr("map.preprocess.core._materialize", fake_materialize)
    assert preprocess.core.materialize_selection(first, project_name="two-lines") == "ok"
    assert captured["num_gene_tokens"] == 2047
    workspace = tmp_path / "projects" / "two-lines"
    assert (workspace / "materialized").is_dir()
    assert "native_split" not in load_contract(workspace)


def test_fixed_tahoe_facade_reserves_project_before_materialization(tmp_path):
    seed = make_tahoe(tmp_path / "seed")
    storage = tmp_path / "storage"
    raw = storage / "raw_datasets" / "Tahoe-100M"
    shutil.copytree(seed.source, raw)

    summary = preprocess.tahoe.statistics(storage=storage, batch_size=2)
    assert summary["cells"] == 3
    preprocess.tahoe.fetch_cell_line(
        ["CVCL_0131"], project_name="a549-test", storage=storage, batch_size=2
    )
    record = storage / "projects" / "a549-test" / "preprocess.json"
    payload = json.loads(record.read_text(encoding="utf-8"))
    assert payload["project_name"] == "a549-test"
    assert payload["populations"] == ["CVCL_0131"]
    assert (raw / "data_summary.json").is_file()
    assert (record.parent / "selection.json").is_file()
    assert (record.parent / "conditions.parquet").is_file()
    assert not (storage / "projects" / ".preprocess").exists()
    assert not (record.parent / "workflow.json").exists()


def test_condition_filter_is_independent_per_population_and_caps_cells(tmp_path):
    raw = tmp_path / "raw"
    data = raw / "data"
    metadata = raw / "metadata"
    output = tmp_path / "prepared"
    data.mkdir(parents=True)
    metadata.mkdir(parents=True)
    rows = []
    samples = [{
        "sample": "control",
        "drug": "DMSO_TF",
        "drugname_drugconc": "[('DMSO_TF', 0, 'uM')]",
    }]
    for population in ("P1", "P2"):
        rows.append({
            "genes": [0], "expressions": [1], "cell_line_id": population,
            "sample": "control", "drug": "DMSO_TF", "canonical_smiles": "",
            "plate": "plate-1",
        })
    specifications = (("P1", "DrugA", "CC", 4), ("P1", "DrugB", "CCC", 1), ("P2", "DrugA", "CC", 3))
    for sample_index, (population, drug, smiles, count) in enumerate(specifications):
        sample = f"sample-{sample_index}"
        samples.append({
            "sample": sample, "drug": drug,
            "drugname_drugconc": repr([(drug, 1.0, "uM")]),
        })
        rows.extend({
            "genes": [0], "expressions": [1], "cell_line_id": population,
            "sample": sample, "drug": drug, "canonical_smiles": smiles,
            "plate": "plate-1",
        } for _ in range(count))
    pq.write_table(pa.Table.from_pylist(rows), data / "part.parquet")
    pq.write_table(pa.Table.from_pylist(samples), metadata / "sample_metadata.parquet")

    run_filter(Namespace(
        raw_dir=str(raw), output_dir=str(output), workers=1,
        cell_lines=["P1", "P2"], min_cells=2, max_cells=3, seed=17,
        overwrite=False,
    ))
    manifest = json.loads((output / "condition_filter.json").read_text(encoding="utf-8"))
    table = pq.read_table(output / "condition_filter.parquet").to_pandas()
    assert manifest["retained_conditions"] == 2
    assert manifest["dropped_conditions"] == 1
    assert manifest["capped_conditions"] == 1
    assert manifest["retained_condition_cells"] == 6
    assert set(table["population"]) == {"P1", "P2"}
    assert table["retained_cells"].tolist() == [3, 3]


def test_unprofiled_split_uses_external_drugs_and_internal_cell_holdout(tmp_path):
    paths = TahoePaths(
        dataset="toy", source=tmp_path / "source",
        workspace=tmp_path / "workspace", frozen_models=tmp_path / "models",
    )
    paths.prepared.mkdir(parents=True)
    population = "P1"
    base = paths.prepared / population
    base.mkdir()
    conditions = [
        {
            "condition_id": condition_id,
            "population": population,
            "drug": drug,
            "canonical_smiles": "C",
            "dose": 1.0,
        }
        for condition_id, drug in enumerate(("DrugA", "DrugB", "DrugC"))
    ]
    pq.write_table(pa.Table.from_pylist(conditions), paths.prepared / "conditions.parquet")
    np.save(base / "condition_ids.npy", np.arange(3, dtype=np.int32))
    np.save(base / "condition_offsets.npy", np.asarray([0, 10, 20, 30], dtype=np.int64))
    np.save(base / "condition_rows.npy", np.arange(30, dtype=np.int64))
    (paths.prepared / "materialized_shapes.json").write_text(
        json.dumps({population: {"n_cells": 30}}), encoding="utf-8"
    )
    (paths.prepared / "stats_manifest.json").write_text("{}", encoding="utf-8")

    result = generate_splits(
        paths,
        rule="unprofiled_drug",
        external_test_size=1,
        internal_test_fraction=0.2,
        seed=42,
        external_drugs=["DrugB", "AbsentDrug"],
    )
    payload = json.loads(Path(result.summary["split_file"]).read_text(encoding="utf-8"))
    assert payload["external_test_drugs"] == ["DrugB"]
    assert payload["missing_external_test_drugs"] == ["AbsentDrug"]
    assert payload["external_test"] == [1]
    assert payload["train"] == [0, 2]
    assert payload["internal_test"] == [0, 2]
    assert payload["row_counts"] == {
        "train": 16, "internal_test": 4, "external_test": 10
    }
    train_rows = set(payload["train_rows"][population])
    internal_rows = set(payload["internal_test_rows"][population])
    external_rows = set(payload["external_test_rows"][population])
    assert not train_rows & internal_rows
    assert not train_rows & external_rows
    assert not internal_rows & external_rows
    assert external_rows == set(range(10, 20))


def test_workflow_opens_only_after_generic_materialization(tmp_path):
    seed = make_tahoe(tmp_path / "seed")
    projects = tmp_path / "storage" / "projects"
    selection = TahoeSource(
        seed.source, projects, frozen_models=tmp_path / "storage" / "frozen_models"
    ).fetch_cell_line(["CVCL_0131"])
    selection.reserve_project("generic-project")
    selection.prepared.mkdir(parents=True)
    paths = selection.create_project("generic-project")
    (paths.prepared / "condition_filter.json").write_text(
        json.dumps({"filter_id": "toy-filter"}), encoding="utf-8"
    )
    (paths.prepared / "materialized_shapes.json").write_text(
        json.dumps({"CVCL_0131": {"n_cells": 1}}), encoding="utf-8"
    )
    (paths.prepared / "preparation_config.json").write_text("{}", encoding="utf-8")
    pq.write_table(pa.Table.from_pylist([{"condition_id": 0}]), paths.prepared / "conditions.parquet")

    workflow = preparation.create_workflow(
        "generic-project", storage=tmp_path / "storage"
    )
    assert workflow.workspace == projects / "generic-project"
    assert workflow.prepared == workflow.workspace / "materialized"
    assert (workflow.workspace / "workflow.json").is_file()


def test_prediction_and_embedding_analysis(tmp_path):
    paths = TahoePaths(dataset="Tahoe-100M", source=tmp_path / "source", workspace=tmp_path / "workspace", frozen_models=tmp_path / "models")
    paths.ensure_outputs()
    prediction_file = tmp_path / "predictions.parquet"
    rows = [{
        "condition_id": 7,
        "population": "CVCL_0131",
        "drug": "DrugA",
        "dose": 1.0,
        "predicted": [1.0, 3.0, 2.0, 4.0],
        "observed": [1.0, 2.5, 2.2, 3.8],
        "control": [1.0, 1.0, 2.0, 2.0],
        "true_deg_ids": [1, 3],
    }]
    pq.write_table(pa.Table.from_pylist(rows), prediction_file)

    response = analyze_predictions(paths, prediction_file)
    deg = analyze_degs(paths, prediction_file, top_k=(2,))
    assert response.summary["conditions"] == 1
    assert deg.summary["2"]["mean_overlap"] == 1.0
    assert (paths.evaluations / "degs" / f"{tmp_path.name}-predictions" / "deg_metrics.parquet").is_file()
    report = build_report(paths, prediction_file)
    assert report.summary["figures"] == 4
    assert (
        paths.evaluations / "reports" / f"{tmp_path.name}-predictions" / "figures" / "response_scatter.svg"
    ).is_file()

    embedding_file = tmp_path / "embeddings.npy"
    import numpy as np
    np.save(embedding_file, np.arange(48, dtype=np.float32).reshape(12, 4))
    embedding = analyze_embeddings(paths, embedding_file, max_points=6)
    assert embedding.summary["rows"] == 6
    assert embedding.summary["total_rows"] == 12
    assert (
        paths.evaluations / "embeddings" / f"{tmp_path.name}-embeddings" / "pca_coords.parquet"
    ).is_file()


def test_state_embedding_resume_is_per_cell_line(tmp_path):
    paths = TahoePaths(dataset="Tahoe-100M", source=tmp_path / "source", workspace=tmp_path / "workspace", frozen_models=tmp_path / "models")
    paths.prepared.mkdir(parents=True)
    (paths.prepared / "materialized_shapes.json").write_text(
        json.dumps({"CVCL_0131": {"n_cells": 2}}), encoding="utf-8"
    )
    base = paths.prepared / "CVCL_0131"
    base.mkdir()
    (base / "state_embeddings.float16.dat").write_bytes(bytes(2 * 2048 * 2))
    (base / "embedding_complete.json").write_text("{}", encoding="utf-8")
    result = embed_state(
        paths,
        Path("se600m.safetensors"),
        Path("gene_embeddings_esm2.pt"),
        populations=("CVCL_0131",),
        resume=True,
    )
    assert result.summary["skipped"] == ["CVCL_0131"]
    assert result.summary["completed"] == []


def test_partitioned_embeddings_merge_in_global_row_order(tmp_path):
    paths = TahoePaths(dataset="Tahoe-100M", source=tmp_path / "source", workspace=tmp_path / "workspace", frozen_models=tmp_path / "models")
    paths.prepared.mkdir(parents=True)
    cell_line = "CVCL_0131"
    (paths.prepared / "materialized_shapes.json").write_text(
        json.dumps({cell_line: {"n_cells": 5}}), encoding="utf-8"
    )
    part_root = paths.prepared / cell_line / "state_embedding_parts"
    part_root.mkdir(parents=True)
    source = {"se_checkpoint": {"sha256": "example"}}
    for index, (start, end) in enumerate(((0, 2), (2, 5))):
        stem = f"part-{index:05d}-of-00002"
        values = np.full((end - start, 2048), index + 1, dtype=np.float16)
        values.tofile(part_root / f"{stem}.float16.dat")
        (part_root / f"{stem}.json").write_text(
            json.dumps({
                "n_cells": end - start,
                "total_cells": 5,
                "row_start": start,
                "row_end": end,
                "partition_index": index,
                "num_partitions": 2,
                "embedding_dim": 2048,
                "dtype": "float16",
                "source": source,
            }),
            encoding="utf-8",
        )

    result = merge_perturbed_cell_embeddings(
        paths, populations=(cell_line,), num_partitions=2
    )
    merged = np.memmap(
        paths.prepared / cell_line / "state_embeddings.float16.dat",
        dtype=np.float16,
        mode="r",
        shape=(5, 2048),
    )
    assert np.all(merged[:2] == 1)
    assert np.all(merged[2:] == 2)
    assert result.summary["merged"] == [cell_line]


def test_gene_and_drug_token_partitions_merge_in_global_order(tmp_path):
    paths = TahoePaths(dataset="Tahoe-100M", source=tmp_path / "source", workspace=tmp_path / "workspace", frozen_models=tmp_path / "models")
    source = {"conditions": "conditions", "esm_embeddings": "esm", "mapkg_ckpt": "mapkg"}
    cases = {
        "genes": {
            "total": 4,
            "indices": ([0, 2], [1, 3]),
            "labels": (["G0", "G2"], ["G1", "G3"]),
        },
        "drugs": {
            "total": 3,
            "indices": ([0, 2], [1]),
            "labels": (["C", "CCC"], ["CC"]),
        },
    }
    for kind, case in cases.items():
        root = paths.prepared / "static_token_parts" / kind
        root.mkdir(parents=True, exist_ok=True)
        prefix = "gene" if kind == "genes" else "drug"
        for index in range(2):
            indices = list(case["indices"][index])
            tokens = torch.stack([
                torch.full((1024,), float(global_index), dtype=torch.bfloat16)
                for global_index in indices
            ])
            torch.save({
                "format": "map_static_token_cache_v1",
                "kind": kind,
                "embedding_dim": 1024,
                "dtype": "bfloat16",
                "source": source,
                "partition_index": index,
                "num_partitions": 2,
                f"{prefix}_tokens": tokens,
                f"{prefix}_indices": indices,
                f"{prefix}_total": case["total"],
                "gene_symbols" if kind == "genes" else "drug_smiles": list(case["labels"][index]),
            }, root / f"part-{index:05d}-of-00002.pt")

    result = merge_static_tokens(paths, gene_partitions=2, drug_partitions=2)
    cache = torch.load(
        paths.prepared / "map_static_tokens.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert cache["gene_symbols"] == ["G0", "G1", "G2", "G3"]
    assert cache["drug_smiles"] == ["C", "CC", "CCC"]
    assert cache["gene_tokens"][:, 0].float().tolist() == [0.0, 1.0, 2.0, 3.0]
    assert cache["drug_tokens"][:, 0].float().tolist() == [0.0, 1.0, 2.0]
    assert result.summary["genes"] == 4


def test_identifiers_expose_high_value_parameters():
    split_id = split_identifier("unseen_combination", 0.05, 0.2, 42)
    assert split_id == (
        "unseen_combination__external-p0p05__internal-p0p2__seed-42__externaldisjoint"
    )
    run_id = training_identifier(
        "unprofiled_drug",
        "unprofiled_drug__external-n16__internal-p0p2__seed-42",
        set_size=24,
        batch_size=1,
        gradient_accumulation_steps=2,
        lr=1e-5,
        seed=7,
    )
    assert "model-se24-bs1-ga2-lr1em05" in run_id
    assert "trainseed-7" in run_id
    evaluation_id = evaluation_identifier(
        "unprofiled_drug",
        "unprofiled_drug__external-n16__internal-p0p2__seed-42",
        "runs/example/last.pt",
        seeds=(42, 43, 44, 45, 46),
        set_size=24,
        deg_top_k=50,
        deg_fdr=0.05,
    )
    assert "ckpt-example-last" in evaluation_id
    assert "seeds42-43-44-45-46" in evaluation_id


def test_parameterized_split_and_train_dry_run(tmp_path):
    paths = TahoePaths(dataset="Tahoe-100M", source=tmp_path / "source", workspace=tmp_path / "workspace", frozen_models=tmp_path / "models")
    paths.prepared.mkdir(parents=True)
    cell_line = "CVCL_0131"
    base = paths.prepared / cell_line
    base.mkdir()
    np.save(base / "condition_ids.npy", np.arange(6, dtype=np.int32))
    np.save(base / "condition_offsets.npy", np.asarray([0, 5, 10, 15, 20, 26, 32]))
    np.save(base / "condition_rows.npy", np.arange(32, dtype=np.int64))
    (paths.prepared / "materialized_shapes.json").write_text(
        json.dumps({
            cell_line: {
                "n_cells": 32,
                "token_length": 2048,
                "hvg_dim": 2000,
                "target_sum": 10000.0,
            }
        }),
        encoding="utf-8",
    )
    (paths.prepared / "stats_manifest.json").write_text(
        json.dumps({"populations": [cell_line], "source_parts": []}),
        encoding="utf-8",
    )
    conditions = [
        {
            "condition_id": index,
            "population": cell_line,
            "drug": f"drug-{index}",
            "canonical_smiles": "C" * (index + 1),
            "dose": 1.0,
        }
        for index in range(6)
    ]
    pq.write_table(
        pa.Table.from_pylist(conditions), paths.prepared / "conditions.parquet"
    )

    split = generate_splits(
        paths,
        rule="unprofiled_drug",
        external_test_size=1,
        internal_test_fraction=0.2,
        seed=123,
    )
    split_path = Path(split.summary["split_file"])
    assert split_path.name == (
        "unprofiled_drug__external-n1__internal-p0p2__seed-123.json"
    )
    payload = json.loads(split_path.read_text(encoding="utf-8"))
    assert payload["split_id"] == split_path.stem
    assert payload["counts"] == {
        "train": 5, "internal_test": 5, "external_test": 1
    }
    manifest = json.loads(
        (paths.prepared / "manifest.json").read_text(encoding="utf-8")
    )
    assert payload["split_id"] in manifest["split_registry"]

    run = train_map(
        paths,
        "unprofiled_drug",
        Path("se.safetensors"),
        Path("esm.pt"),
        Path("mapkg.pt"),
        Path("bart_vocab.txt"),
        split_file=split_path,
        gpus=2,
        num_workers=6,
        seed=7,
        dry_run=True,
    )
    assert run.summary["split_id"] == payload["split_id"]
    assert "trainseed-7" in run.summary["run_id"]
    assert run.summary["gpus"] == 2
    assert run.summary["train_split"] == "train"
    assert run.summary["params"]["compile_mode"] == "default"
    assert "val_split" not in run.summary
    assert run.summary["checkpoint"].endswith("/last.pt")


def test_segmented_compile_checkpoint_is_eager_compatible(tmp_path):
    class State(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Linear(2, 2)
            self.model.requires_grad_(False)

    class Perturbation(torch.nn.Module):
        def __init__(self):
            super().__init__()
            for name in (
                "cell_projector",
                "gene_tokens_projector",
                "transformer_backbone",
                "project_out",
                "gene_decoder",
            ):
                setattr(self, name, torch.nn.Linear(2, 2))

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.state = State()
            self.pert_model = Perturbation()

    source = Model()
    original_keys = tuple(source.state_dict())
    assert compile_map_parts(source, "default") == MAP_COMPILE_PARTS
    assert tuple(source.state_dict()) == original_keys
    assert not any("_orig_mod" in key for key in source.state_dict())

    optimizer = torch.optim.Adam(source.pert_model.parameters(), lr=1e-3)
    for parameter in source.pert_model.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    checkpoint = tmp_path / "compiled.pt"
    save_checkpoint(
        checkpoint,
        source,
        optimizer,
        scheduler,
        scaler,
        Namespace(compile_mode="default"),
        4,
        17,
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert not any("_orig_mod" in key for key in payload["pert_model_state_dict"])

    eager = Model()
    eager_optimizer = torch.optim.Adam(eager.pert_model.parameters(), lr=1e-3)
    eager_scheduler = torch.optim.lr_scheduler.LambdaLR(
        eager_optimizer, lambda _: 1.0
    )
    eager_scaler = torch.amp.GradScaler("cpu", enabled=False)
    restored = load_checkpoint(
        checkpoint, eager, eager_optimizer, eager_scheduler, eager_scaler
    )
    assert restored["epoch"] == 4
    assert restored["global_step"] == 17
    for key, value in source.pert_model.state_dict().items():
        assert torch.equal(value, eager.pert_model.state_dict()[key])
    assert len(eager_optimizer.state) == len(optimizer.state)


def test_small_gene_space_fails_before_loess(tmp_path):
    from map.preprocess.sources.tahoe.pipeline import _fit_seurat_clip

    means = np.arange(1, 11, dtype=np.float64)
    np.savez(
        tmp_path / "global_count_stats.npz",
        sums=means * 10,
        sumsq=(means * means + 1) * 10,
        maxima=means,
        n_cells=np.asarray(10, dtype=np.int64),
    )
    import pytest

    with pytest.raises(ValueError, match="at least 20 non-constant"):
        _fit_seurat_clip(tmp_path)


def test_sciplex_h5ad_source_creates_generic_contract(tmp_path):
    import anndata as ad
    from scipy import sparse

    source = tmp_path / "sciplex.h5ad"
    data = ad.AnnData(
        X=sparse.csr_matrix(np.asarray([
            [1, 0, 3], [0, 2, 1], [4, 0, 0], [1, 1, 1],
        ], dtype=np.float32)),
        obs={
            "cell_line": ["A549", "A549", "MCF7", "MCF7"],
            "product_name": ["Vehicle", "DrugA", "Vehicle", "DrugA"],
            "dose": [0.0, 1.0, 0.0, 1.0],
            "canonical_smiles": ["", "CC", "", "CC"],
            "plate": ["p1", "p1", "p2", "p2"],
        },
        var={"gene_symbol": ["ACTB", "GAPDH", "TP53"]},
    )
    data.write_h5ad(source)
    source = preprocess.anndata(
        source,
        tmp_path / "projects",
        frozen_models=tmp_path / "models",
        dataset="SciPlex3",
    )
    summary = source.watch_data()
    assert summary["cells"] == 4
    assert summary["populations"] == 2
    assert summary["control_cells"] == 2
    selection = source.fetch_cell_line(["A549"])
    contract = selection.contract_payload
    assert contract["counts"]["selected_cells"] == 2
    assert contract["source_format"] == "anndata"
    assert contract["populations"] == ["A549"]
    assert contract["preparation_backend"] == "map.preprocess.sources.anndata_backend"


def test_native_sciplex_matrix_market_adapter(tmp_path):
    from scipy import sparse
    from scipy.io import mmwrite
    from map.preprocess.sources.atlas_backend import _canonical_source

    storage = tmp_path / "storage"
    raw = storage / "raw_datasets" / "SciPlex3"
    raw.mkdir(parents=True)
    matrix = np.asarray([
        [3, 0, 1],
        [0, 4, 2],
        [5, 1, 0],
        [0, 2, 6],
    ], dtype=np.float32)
    mmwrite(raw / "matrix.mtx", sparse.csr_matrix(matrix.T))
    pd.DataFrame({
        "cell_id": ["a0", "a1", "m0", "m1"],
        "cell_line": ["A549", "A549", "MCF7", "MCF7"],
        "product_name": ["Vehicle", "DrugA", "Vehicle", "DrugA"],
        "canonical_smiles": ["", "CC", "", "CC"],
        "dose_val": [0.0, 1.0, 0.0, 1.0],
        "plate": ["p1", "p1", "p2", "p2"],
    }).to_csv(raw / "cell_metadata.tsv", sep="\t", index=False)
    pd.DataFrame({
        "gene_symbol": ["ACTB", "GAPDH", "TP53"],
        "feature_id": ["g0", "g1", "g2"],
    }).to_csv(raw / "gene_metadata.tsv", sep="\t", index=False)

    summary = preprocess.sciplex.statistics(storage=storage)
    assert summary["cells"] == 4
    assert summary["cell_line_ids"] == ["A549", "MCF7"]
    selection = preprocess.sciplex.fetch_cell_line(
        ["A549"], project_name="sciplex-a549", storage=storage
    )
    canonical = _canonical_source(
        selection.contract_payload, selection.prepared, batch_size=2, shard_size=2
    )
    rows = pq.read_table(canonical / "data" / "part-00000.parquet").to_pylist()
    assert len(rows) == 2
    assert rows[0]["cell_line_id"] == "A549"
    assert rows[0]["genes"] == [0, 2]
    assert rows[0]["expressions"] == [3.0, 1.0]
    assert rows[1]["drug"] == "DrugA"
    assert selection.contract_payload["source_format"] == "sciplex"


def test_native_nips_long_parquet_adapter(tmp_path):
    from map.preprocess.sources.atlas_backend import _canonical_source

    storage = tmp_path / "storage"
    raw = storage / "raw_datasets" / "OP3"
    raw.mkdir(parents=True)
    metadata = pd.DataFrame({
        "obs_id": ["c0", "c1", "c2", "c3"],
        "cell_type": ["T cells", "T cells", "B cells", "B cells"],
        "sm_name": ["DMSO", "DrugA", "DMSO", "DrugA"],
        "SMILES": ["", "CC", "", "CC"],
        "dose_uM": [14.1, 1.0, 14.1, 1.0],
        "plate_name": ["p0", "p0", "p1", "p1"],
        "control": [True, False, True, False],
    })
    metadata_csv = raw / "adata_obs_meta.csv"
    metadata.to_csv(metadata_csv, index=False)
    with zipfile.ZipFile(raw / "adata_obs_meta.csv.zip", "w") as archive:
        archive.write(metadata_csv, metadata_csv.name)
    metadata_csv.unlink()

    expression_rows = [
        {"obs_id": "c0", "gene": "ACTB", "count": 3, "normalized_count": 6.0},
        {"obs_id": "c0", "gene": "GAPDH", "count": 2, "normalized_count": 5.5},
        {"obs_id": "c1", "gene": "ACTB", "count": 1, "normalized_count": 5.0},
        {"obs_id": "c1", "gene": "TP53", "count": 4, "normalized_count": 6.2},
        {"obs_id": "c2", "gene": "GAPDH", "count": 5, "normalized_count": 6.4},
    ]
    parquet = raw / "adata_train.parquet"
    pq.write_table(pa.Table.from_pylist(expression_rows), parquet)
    with zipfile.ZipFile(raw / "adata_train.parquet.zip", "w") as archive:
        archive.write(parquet, parquet.name)
    parquet.unlink()

    excluded = raw / "adata_excluded_ids.csv"
    pd.DataFrame([{"obs_id": "c0", "gene": "GAPDH"}]).to_csv(
        excluded, index=False
    )
    with zipfile.ZipFile(raw / "adata_excluded_ids.csv.zip", "w") as archive:
        archive.write(excluded, excluded.name)
    excluded.unlink()

    summary = preprocess.nips.statistics(storage=storage)
    assert summary["source_cells"] == 4
    assert summary["cells"] == 3
    assert summary["excluded_cells"] == 1
    assert summary["cell_type_ids"] == ["B cells", "T cells"]
    selection = preprocess.nips.fetch_cell_type(
        ["T cells"], project_name="nips-t", storage=storage
    )
    canonical = _canonical_source(
        selection.contract_payload, selection.prepared, batch_size=2, shard_size=2
    )
    rows = pq.read_table(canonical / "data" / "part-00000.parquet").to_pylist()
    assert len(rows) == 2
    assert rows[0]["genes"] == [0]
    assert rows[0]["expressions"] == [3.0]
    assert rows[1]["genes"] == [0, 2]
    assert rows[1]["drug"] == "DrugA"
    assert selection.contract_payload["source_format"] == "nips"


def test_anndata_without_population_column_uses_one_population(tmp_path):
    import anndata as ad

    source = tmp_path / "single-population.h5ad"
    ad.AnnData(
        X=np.asarray([[1, 0], [0, 1]], dtype=np.float32),
        obs={
            "condition": ["control", "DrugA"],
            "dose": [0.0, 1.0],
            "canonical_smiles": ["", "CC"],
        },
        var={"gene_symbol": ["ACTB", "TP53"]},
    ).write_h5ad(source)
    source = preprocess.anndata(
        source, tmp_path / "projects", frozen_models=tmp_path / "models"
    )
    summary = source.watch_data()
    assert summary["cells_by_population"] == {"__ALL__": 2}
    selection = source.prepare()
    assert selection.contract_payload["populations"] == ["__ALL__"]


def test_select_hvg_owns_internal_statistics(monkeypatch, tmp_path):
    paths = TahoePaths(
        dataset="Tahoe-100M",
        source=tmp_path / "source",
        workspace=tmp_path / "workspace",
        frozen_models=tmp_path / "models",
    )
    calls = []
    monkeypatch.setattr("map.preprocess.core._ensure_stats", lambda *args, **kwargs: calls.append("stats"))
    monkeypatch.setattr("map.preprocess.core._fit_hvg", lambda *args, **kwargs: calls.append("hvg") or "ok")
    assert preprocess.core.select_hvg(paths, n_top_genes=2000, workers=3) == "ok"
    assert calls == ["stats", "hvg"]


def test_train_registry_defaults_to_map_and_baselines_dry_run(tmp_path):
    paths = TahoePaths(
        dataset="Tahoe-100M",
        source=tmp_path / "source",
        workspace=tmp_path / "workspace",
        frozen_models=tmp_path / "models",
    )
    paths.prepared.mkdir(parents=True)
    population = "A549"
    (paths.prepared / "materialized_shapes.json").write_text(
        json.dumps({population: {"n_cells": 32, "token_length": 2048, "hvg_dim": 2000}}),
        encoding="utf-8",
    )
    split_file = paths.prepared / "split.json"
    split_file.write_text(json.dumps({
        "split_id": "split", "rule": "unprofiled_drug", "seed": 42,
        "train": [0], "internal_test": [1], "external_test": [2],
    }), encoding="utf-8")
    (paths.prepared / "manifest.json").write_text(json.dumps({
        "splits": {"unprofiled_drug": "split.json"},
        "materialized_shapes": {population: {"n_cells": 32, "token_length": 2048, "hvg_dim": 2000}},
    }), encoding="utf-8")
    baseline_root = paths.prepared / "baselines"
    baseline_root.mkdir()
    (baseline_root / "manifest.json").write_text(
        json.dumps({"models": {model: {} for model in train.MODEL_REGISTRY}}),
        encoding="utf-8",
    )

    assert tuple(train.MODEL_REGISTRY) == (
        "prnet", "chemcpa", "trainmean", "crisp", "xpert", "cmonge"
    )
    for model in train.MODEL_REGISTRY:
        result = train.run(
            paths,
            model=model,
            regime="unprofiled_drug",
            split_file=split_file,
            populations=[population],
            gpus=2,
            run_name=f"paper-{model}",
            dry_run=True,
        )
        assert result.summary["model"] == model
        assert result.summary["dry_run"] is True
        assert result.summary["run_id"] == f"paper-{model}"


def test_baseline_inputs_are_prepared_once_before_training(tmp_path):
    paths = DatasetPaths(
        dataset="toy",
        source=tmp_path / "source",
        workspace=tmp_path / "project",
        frozen_models=tmp_path / "models",
    )
    paths.prepared.mkdir(parents=True)
    (paths.prepared / "materialized_shapes.json").write_text(
        json.dumps({"A549": {"n_cells": 2, "hvg_dim": 2000}}),
        encoding="utf-8",
    )
    population_root = paths.prepared / "A549"
    population_root.mkdir()
    np.stack((np.ones(2048), np.full(2048, 3))).astype(np.float16).tofile(
        population_root / "state_embeddings.float16.dat"
    )
    np.stack((np.ones(2000), np.full(2000, 5))).astype(np.float16).tofile(
        population_root / "hvg.float16.dat"
    )
    np.save(population_root / "control_group_ids.npy", np.asarray([7]))
    np.save(population_root / "control_group_offsets.npy", np.asarray([0, 2]))
    np.save(population_root / "control_group_rows.npy", np.asarray([0, 1]))
    smiles = ["CC", "CCC"]
    pq.write_table(
        pa.Table.from_pylist([
            {"canonical_smiles": value} for value in smiles
        ]),
        paths.prepared / "conditions.parquet",
    )
    hvg_symbols = ["ACTB", "GAPDH", "TP53"]
    (paths.prepared / "hvg.json").write_text(
        json.dumps({"gene_symbols": hvg_symbols}), encoding="utf-8"
    )
    selected_models = ("prnet", "chemcpa", "trainmean", "crisp", "cmonge")
    result = preparation.prepare_baseline_inputs(
        paths,
        models=selected_models,
    )
    assert result.summary["models"] == list(selected_models)
    mean_root = paths.prepared / "baselines" / "crisp" / "control_group_means" / "A549"
    assert np.array_equal(
        np.load(mean_root / "group_ids.int64.npy"), np.asarray([7])
    )
    assert np.allclose(np.load(mean_root / "embeddings.float16.npy"), 2)
    assert np.allclose(np.load(mean_root / "hvg.float16.npy"), 3)
    batch = {
        "control_hvg_vectors": torch.ones(1, 2, 3),
        "condition_hvg_vectors": torch.full((1, 2, 3), 2.0),
        "control_embeddings": torch.ones(1, 2, 2048),
        "drug_smiles": ["CC"],
        "drug_conc": torch.tensor([1.0]),
        "population": ["A549"],
    }
    for model in selected_models:
        assert (paths.prepared / "baselines" / model / "manifest.json").is_file()
        model_hvg_dim = 2000 if model == "cmonge" else 3
        model_batch = batch
        if model == "cmonge":
            model_batch = {
                **batch,
                "control_hvg_vectors": torch.ones(1, 2, 2000),
                "condition_hvg_vectors": torch.full((1, 2, 2000), 2.0),
            }
        instance = build_model(
            model, paths.prepared, model_hvg_dim, ["A549"]
        )
        loss = instance.loss(instance(model_batch), model_batch)
        if isinstance(loss, tuple):
            loss = loss[0]
        assert torch.isfinite(loss)
        checkpoint = {
            "model_configuration": instance.configuration(),
            "model_state_dict": instance.state_dict(),
        }
        restored = load_model(
            model, checkpoint, paths.prepared, model_hvg_dim,
            ["A549"], torch.device("cpu")
        )
        restored.eval()
        assert restored.predict_batch(model_batch).shape == (1, model_hvg_dim)


def test_eval_registry_supports_map_and_baselines_dry_run(tmp_path):
    paths = TahoePaths(
        dataset="toy", source=tmp_path / "source",
        workspace=tmp_path / "workspace", frozen_models=tmp_path / "models",
    )
    paths.prepared.mkdir(parents=True)
    population = "A549"
    shapes = {population: {"n_cells": 8, "token_length": 33, "hvg_dim": 32}}
    (paths.prepared / "materialized_shapes.json").write_text(
        json.dumps(shapes), encoding="utf-8"
    )
    split_file = paths.prepared / "split.json"
    split_file.write_text(json.dumps({
        "split_id": "toy-split", "rule": "unprofiled_drug", "seed": 42,
        "train": [0], "internal_test": [1], "external_test": [2],
    }), encoding="utf-8")
    (paths.prepared / "manifest.json").write_text(json.dumps({
        "splits": {"unprofiled_drug": "split.json"},
        "materialized_shapes": shapes,
    }), encoding="utf-8")
    for model in ("map", *train.MODEL_REGISTRY):
        run_name = f"paper-{model}"
        run_dir = paths.runs / run_name
        run_dir.mkdir(parents=True)
        (run_dir / "run_config.json").write_text(
            json.dumps({
                "model": model,
                "regime": "unprofiled_drug",
                "split_file": str(split_file),
            }),
            encoding="utf-8",
        )
        result = eval.run(
            paths, run_name=run_name, checkpoint=run_dir / "last.pt",
            seeds=(42,), set_size=1,
            evaluation_name=f"eval-{model}", dry_run=True,
        )
        assert result.summary["model"] == model
        assert result.summary["dry_run"] is True
        assert result.summary["evaluation_splits"] == [
            "internal_test", "external_test"
        ]
        assert set(result.summary["split_evaluation_files"]) == {
            "internal_test", "external_test"
        }


def test_baseline_prediction_uses_condition_level_protocol():
    class Example(torch.nn.Module):
        def __init__(self, output):
            super().__init__()
            self.output = output

        def predict_batch(self, _batch):
            return self.output

    item = {
        "drug_smiles": "CC", "population": "A549", "drug_conc": 1.0,
        "control_hvg_vectors": torch.zeros(2, 3),
    }
    condition_level = torch.arange(3, dtype=torch.float32).reshape(1, 3)
    assert np.allclose(
        _baseline_prediction(Example(condition_level), item, torch.device("cpu")),
        condition_level.numpy()[0],
    )


def test_frozen_assets_may_use_release_root_or_named_subdirectories(tmp_path):
    paths = TahoePaths(
        dataset="toy", source=tmp_path / "source",
        workspace=tmp_path / "workspace", frozen_models=tmp_path / "models",
    )
    paths.frozen_models.mkdir()
    flat = paths.frozen_models / "se600m.safetensors"
    flat.touch()
    assert asset(paths, "state/se600m.safetensors") == flat
    nested = paths.frozen_models / "mapkg" / "bart_vocab.txt"
    nested.parent.mkdir()
    nested.touch()
    assert asset(paths, "mapkg/bart_vocab.txt") == nested


def test_map_rejects_state_input_without_special_plus_configured_genes():
    import pytest

    from map.model.map import MAPModel

    model = MAPModel.__new__(MAPModel)
    torch.nn.Module.__init__(model)
    model.num_gene_tokens = 2
    gene_ids = torch.zeros(1, 1, 4, dtype=torch.long)
    expressions = torch.zeros(1, 1, 4)
    with pytest.raises(ValueError, match="expected 3, got 4"):
        model(gene_ids, expressions, ["CC"], torch.ones(1))


def test_map_adds_original_cell_to_updated_cell_before_project_out(monkeypatch):
    from types import SimpleNamespace

    from map.model.map import PerturbationEncoder

    class ZeroGeneProjector(torch.nn.Module):
        def forward(self, values):
            return values.new_zeros(*values.shape[:-1], 1024)

    class ConstantCellUpdate(torch.nn.Module):
        def forward(self, *, inputs_embeds):
            hidden = torch.zeros_like(inputs_embeds)
            hidden[:, 1] = 3
            return SimpleNamespace(last_hidden_state=hidden)

    encoder = PerturbationEncoder.__new__(PerturbationEncoder)
    torch.nn.Module.__init__(encoder)
    encoder.cell_projector = torch.nn.Identity()
    encoder.gene_tokens_projector = ZeroGeneProjector()
    encoder.transformer_backbone = ConstantCellUpdate()
    encoder.project_out = torch.nn.Identity()
    encoder.gene_decoder = torch.nn.Identity()
    monkeypatch.setattr(
        PerturbationEncoder,
        "encode_drug",
        lambda self, smiles, reference: reference.new_zeros(len(smiles), 1024),
    )
    monkeypatch.setattr(
        PerturbationEncoder,
        "_gene_table",
        lambda self, esm_embeddings: esm_embeddings.new_zeros(4, 1024),
    )

    predicted_embeddings, _ = encoder(
        gene_tokens=torch.zeros(1, 2, 2048),
        cell_embeddings=torch.full((1, 1024), 2.0),
        gene_ids=torch.tensor([[0, 1]]),
        esm_embeddings=torch.zeros(4, 5120),
        smiles=["CC"],
        doses=torch.ones(1),
    )
    assert torch.equal(predicted_embeddings, torch.full((1, 1, 1024), 5.0))
