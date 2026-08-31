# MAP

一个自包含的单细胞扰动建模工具包。MAP、PRnet、chemCPA、TrainMean、CRISP、XPert 和 CMonge 共用同一套项目、split、训练和评估接口。

## 项目布局

```text
storage/projects/<project>/
├── contract.json
├── materialize/                  # 按物料类型组织的共享产物
│   ├── drug_ecfp4/
│   ├── graph_assets/
│   └── ...
└── splits/<split_id>/             # 每个 split 独立维护方法状态
    ├── split.json
    ├── materialize/cmonge/        # 仅 split 相关物料
    └── methods/<method>/          # 仅训练/校验时按需创建
        ├── validation.json
        └── runs/<run_name>/evaluations/<evaluation_name>/
```

`storage` 可以指向独立的数据盘。原始数据放在 `storage/raw_datasets/`，冻结模型放在 `storage/frozen_models/`；它们不会复制进项目。

## 安装

```bash
python -m pip install -r requirements-map.txt
python -m pip install -e .
```

需要额外方法时：

```bash
python -m pip install -r requirements-map-baselines.txt
python -m pip install -e .
```

两份 requirements 使用同一组核心版本边界；第二份只增加额外方法所需依赖。

## 数据到项目

数据集的 native 读取由 handler 提供，之后进入通用 pipeline：

```python
from map import preprocess, preparation

flow = preprocess.pipeline(preprocess.builtin.tahoe(storage="storage"))
flow.watch_data(batch_size=8192)
selection = flow.fetch_populations(
    ["CVCL_1098", "CVCL_1056", "CVCL_0131"],
    project_name="six-lines",
)
flow.filter_conditions(selection, min_cells=500, max_cells=5_000, seed=42)
flow.select_hvg(selection, n_top_genes=2_000)
paths = preparation.create_project(selection, project_name="six-lines")
```

SciPlex、OP3/NIPS 和 AnnData 使用同一个 handler 边界；只需替换 `preprocess.builtin` 的 handler。

## 通用物料与 split

每个物料单独生成，可中断后继续。训练前只生成当前实验需要的物料。

```python
preparation.prepare_cell_metadata(paths)
preparation.prepare_state_inputs(paths)
preparation.prepare_hvg_expression(paths)
preparation.build_sampling_index(paths)

split = preparation.create_split(
    paths,
    rule="unseen_combination",
    external_test_size=0.05,
    internal_test_fraction=0.20,
    seed=42,
    output_name="unseen-comb__ext-p0p05__int-p0p20__seed-42.json",
)
```

同一项目可创建任意多个 split；训练和评估始终显式接收 `split_file`，也可直接传入已登记的 `split_id`。

## 统一训练与评估

所有方法都通过 `train.run()` 进入，各方法只负责自己的 model/trainer 和按需物料：

```python
run = train.run(
    paths,
    model="map",                  # 也可以是 prnet / chemcpa / ...
    regime="unseen_combination",
    split_file=split.summary["split_file"],
    run_name="map-unseen-seed42",
    epochs=100,
    checkpoint_every_epochs=5,
    gpus=2,
)

report = eval.run(
    paths,
    run_name="map-unseen-seed42",
    checkpoint=run.summary["checkpoint"],
    seeds=(42,),
)
```

训练入口会先执行方法/split 级物料校验，并写入 `splits/<split_id>/methods/<method>/validation.json`。只有使用某个方法时，该方法目录才会创建。评估主报告按 `cell_line_drug` 汇总，同时保存每个 dose-level 的明细。

本地运行和调度运行使用同一 Python 调用；资源由用户自己的 shell 或 Slurm 脚本决定，包本身不提交作业。

完整的逐步示例见 [`docs/index.html`](docs/index.html)，物料清单见 [`docs/preparation.md`](docs/preparation.md)。
