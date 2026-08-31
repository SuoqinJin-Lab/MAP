# Preparation Reference

Preparation is incremental. Generate the neutral artifacts required by the run. Paths below are relative to `storage/projects/<project_name>/`.

One project can keep multiple immutable splits. Select one by its `split_file`; training and evaluation records always retain that identifier.

## Shared artifacts

| Artifact | Generator | Files | Consumers |
|---|---|---|---|
| Cell metadata | preparation.prepare_cell_metadata(paths) | conditions.parquet; <population>/row_condition.int32.dat; <population>/row_group.uint16.dat | all methods and evaluation |
| STATE inputs | preparation.prepare_state_inputs(paths) | <population>/se_gene_ids.uint16.dat; <population>/se_expr.float16.dat | STATE-based inputs |
| HVG expression | preparation.prepare_hvg_expression(paths) | <population>/hvg.float16.dat | all methods and evaluation |
| Sampling index | preparation.build_sampling_index(paths) | condition/group index files recorded in index_manifest.json | all trainers and evaluation |
| Split | preparation.create_split(paths, ...) | splits/<split_id>/split.json | selected train/evaluation run |
| Knowledge tokens | prepare_gene_tokens(), prepare_knowledge_drug_tokens(), assemble_knowledge_tokens() | materialize/knowledge_tokens.pt and partition manifests | downstream token consumers |
| Condition embeddings | prepare_condition_embeddings(), assemble_condition_embeddings() | materialize/<population>/state_embeddings.float16.dat | downstream condition-embedding consumers |

Preparation functions write one neutral artifact directory under `materialize/`. Split-dependent artifacts are written under `splits/<split_id>/materialize/`.

### Typical shared sequence

~~~python
from map import preparation

preparation.prepare_cell_metadata(paths, workers=8)
preparation.prepare_state_inputs(paths, workers=8)
preparation.prepare_hvg_expression(paths, workers=8)
preparation.build_sampling_index(paths, workers=8)
split = preparation.create_split(
    paths, rule="unseen_combination",
    external_test_size=0.2, internal_test_fraction=0.2, seed=42,
)

# Knowledge and condition representations (generate when requested by a run)
preparation.prepare_gene_tokens(paths, num_partitions=4)
preparation.prepare_knowledge_drug_tokens(paths, num_partitions=4)
preparation.assemble_knowledge_tokens(paths, gene_partitions=4, drug_partitions=4)
preparation.prepare_condition_embeddings(paths, num_partitions=4, workers=8)
preparation.assemble_condition_embeddings(paths, num_partitions=4)
~~~

## Optional artifacts (on demand)

Generate only the artifacts needed by the selected experiment. Each function owns one artifact directory; preparation never creates a method or run directory.

| Artifact | Generator | Main files | Consumer |
|---|---|---|---|
| ECFP4 features | prepare_ecfp4_features(paths) | materialize/drug_ecfp4/ | chemCPA |
| FCFP4 features | prepare_fcfp4_features(paths) | materialize/drug_fcfp4/ | PRnet |
| Molecular descriptors | prepare_molecular_descriptors(paths) | materialize/drug_rdkit2d/ | CRISP, CMonge |
| Control means | prepare_control_means(paths) | materialize/control_means/ | CRISP |
| DEG masks | prepare_deg_masks(paths, top_k=50, mask_mode="official") | materialize/deg_masks/ | CRISP |
| MoA features | prepare_moa_features(paths, split_file=...) | splits/<split_id>/materialize/drug_moa/ | CMonge |
| UniMol drug representation | prepare_unimol_tokens(paths) | materialize/drug_unimol/ | XPert |
| Graph assets | prepare_graph_assets(paths) | materialize/graph_assets/ | XPert |
| Expression bins | prepare_expression_bins(paths, input_formats=("official",)) | materialize/expression_bins/ | XPert |

Some methods consume only the shared cell metadata and HVG expression, so no optional artifact is needed for them.

## Validation before training

Every public training entry point performs strict validation automatically. Run it explicitly to inspect a project before submitting a local or scheduled job:

~~~python
from map.preparation import validate_method

report = validate_method(
    paths,
    method="map",
    regime="unprofiled_drug",
    split_file="splits/unprofiled_drug-r0.2-s42.json",
)
~~~

The report is written to `splits/<split_id>/methods/<method>/validation.json`. A failed report lists the missing or mismatched file, shape, split rule, model representation, or frozen asset. Use `strict=False` for a non-blocking dry-run.

Method-level checks include:

- MAP: static token cache, SE-600M, ESM2, MAP-KG checkpoint and vocabulary.
- CRISP: control-group means and DEG masks when use_deg_mask=True.
- CMonge: MoA representation is selected automatically for unseen_combination.
- XPert: official full-gene or validation-mode graph/expression contract.
- PRnet and chemCPA: their own 1,024-dimensional fingerprint matrix.
- TrainMean: shared metadata and HVG expression only.

After validation, train directly with `train.run(paths, model=..., split_file=..., run_name=...)`. Runs are stored under `splits/<split_id>/methods/<method>/runs/<run_name>/`; evaluations are stored below the selected run.
