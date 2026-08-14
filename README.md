# MAP

Self-contained preprocessing, preparation, training, evaluation, and analysis for MAP,
PRnet, chemCPA, TrainMean, CRISP, XPert, and CMonge. The package does not import
another MAP or baseline checkout.

```text
map/
├── preprocess/   # dataset adapters, selection, HVG, materialization
├── preparation/  # generic workflow, splits and frozen-model caches
├── model/        # MAP Method 4.4 model
├── train/        # MAP by default; optional baseline registry
└── eval/         # shared metrics and downstream reports

storage/
├── raw_datasets/
├── frozen_models/
├── reference_data/
├── projects/
└── toy_runs/
```

`storage/` may be a symlink to a large-data location. All examples and default
entry points use repository-relative paths below it.

Install MAP:

```bash
python -m pip install -r requirements-map.txt
python -m pip install -e .
```

Install MAP and the baselines:

```bash
python -m pip install -r requirements-map-baselines.txt
python -m pip install -e .
```

```python
from map import eval, preparation, preprocess, train

project_name = "six-lines-unprofiled"
preprocess.tahoe.statistics(batch_size=8192)
preprocess.tahoe.fetch_cell_line([
    "CVCL_1098", "CVCL_1056", "CVCL_0131",
    "CVCL_0069", "CVCL_0023", "CVCL_0480",
], project_name=project_name)
preprocess.tahoe.select_hvg(project_name, n_top_genes=2_000, workers=8)
preprocess.tahoe.materialize(
    project_name,
    num_gene_tokens=2_048,
    target_sum=10_000,
    workers=8,
)
workflow = preparation.create_workflow(project_name)
preparation.build_sampling_index(workflow, workers=8)
split = preparation.create_split(
    workflow, rule="unprofiled_drug", test_size=16, val_size=16, seed=42,
)

for partition_index in range(4):
    preparation.precache_gene_tokens(
        workflow, partition_index=partition_index, num_partitions=4,
    )
    preparation.precache_drug_tokens(
        workflow, partition_index=partition_index, num_partitions=4,
    )
    preparation.precache_state_embeddings(
        workflow, partition_index=partition_index, num_partitions=4,
        batch_size=32, workers=8,
    )

preparation.merge_knowledge_tokens(
    workflow, gene_partitions=4, drug_partitions=4,
)
preparation.merge_state_embeddings(workflow, num_partitions=4)
preparation.validate(workflow, split_files=[split.summary["split_file"]])

run = train.run(
    workflow,
    run_name="map-unprofiled-seed42",
    regime="unprofiled_drug",
    split_file=split.summary["split_file"],
)
evaluation = eval.run(
    workflow,
    run_name="map-unprofiled-seed42",
    seeds=(42, 43, 44, 45, 46),
)
```

Native SciPlex3 and OP3/NIPS inputs use the same downstream workflow:

```python
# storage/raw_datasets/SciPlex3/GSM4150378_sciPlex3_cds_all_cells.RDS
preprocess.sciplex.export_rds()
summary = preprocess.sciplex.statistics(smiles_map="sciplex_smiles.tsv")
preprocess.sciplex.fetch_cell_line(
    ["A549", "MCF7", "K562"], project_name="sciplex3",
)

# storage/raw_datasets/OP3/{adata_obs_meta.csv.zip,adata_train.parquet.zip,
#                           adata_excluded_ids.csv.zip}
summary = preprocess.nips.statistics()
preprocess.nips.fetch_cell_type(
    ["B cells", "Myeloid cells", "NK cells", "T cells CD4+",
     "T cells CD8+", "T regulatory cells"],
    project_name="op3",
)
```

Continue with the selected adapter's `select_hvg()` and `materialize()` methods,
then use the dataset-independent `preparation`, `train`, and `eval` modules above.

CRISP and CMonge generate standardized RDKit2D descriptors from the local SMILES
vocabulary. CMonge uses the paper's frozen HVG autoencoder and conditional
Monge Gap training scheme through PyTorch and GeomLoss. XPert requires these
immutable assets:

```text
storage/frozen_models/unimol/
├── mol.dict.txt
└── mol_pre_all_h_220816.pt

storage/reference_data/xpert/
├── 9606.protein.info.v12.0.txt.gz
├── 9606.protein.links.v12.0.txt.gz
├── primekg.csv
└── drug_aliases.tsv                 # optional exact aliases
```

Download locations:

- [STRING human protein links](https://stringdb-downloads.org/download/protein.links.v12.0/9606.protein.links.v12.0.txt.gz)
- [STRING human protein names](https://stringdb-downloads.org/download/protein.info.v12.0/9606.protein.info.v12.0.txt.gz)
- [PrimeKG knowledge graph](https://dataverse.harvard.edu/api/access/datafile/6180620)
- [UniMol checkpoint](https://huggingface.co/dptech/Uni-Mol-Models/resolve/main/mol_pre_all_h_220816.pt)
- [UniMol dictionary](https://huggingface.co/dptech/Uni-Mol-Models/resolve/main/mol.dict.txt)

Baseline inputs are prepared once, outside training. XPert preparation maps
STRING with `combined_score > 700`, extracts PrimeKG drug-protein edges,
constructs Morgan-Tanimoto DDS edges at `> 0.5`, generates UniMol tokens, and
pretrains the three-layer heterogeneous graph encoder.

```python
preparation.prepare_baseline_inputs(
    workflow,
    models=("prnet", "chemcpa", "trainmean", "crisp", "xpert", "cmonge"),
    xpert_epochs=100,
)
```

Run the generated Tahoe toy chain with:

```bash
./toy.sh --dataset tahoe --models map prnet chemcpa trainmean crisp xpert cmonge
```

The toy XPert run uses generated miniature tensors and does not require the
external graph assets above. A real XPert preparation reports all missing file
paths before graph construction starts.

To exercise the installed STRING, PrimeKG, and UniMol files on a small Tahoe
dataset with real drug names, run:

```bash
./toy.sh --dataset tahoe --models xpert --xpert-assets real \
  --hvg-dim 64 --num-gene-tokens 64
```

Each run is written to `storage/toy_runs/toy-<timestamp>-<id>/`.
The complete tutorial is [docs/index.html](docs/index.html).
