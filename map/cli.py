from __future__ import annotations

import argparse

from . import eval, preparation, preprocess, train


def _size(value: str) -> int | float:
    return int(value) if value.isdigit() else float(value)


def _storage(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--storage", default="storage")


def _project(parser: argparse.ArgumentParser) -> None:
    _storage(parser)
    parser.add_argument("--project-name", required=True)


def _partition(parser: argparse.ArgumentParser, *, batch_size: int) -> None:
    parser.add_argument("--partition-index", type=int, default=0)
    parser.add_argument("--num-partitions", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=batch_size)


def _native_statistics(parser: argparse.ArgumentParser, population_option: str) -> None:
    _storage(parser)
    parser.add_argument("--smiles-map")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument(population_option)
    parser.add_argument("--perturbation-key")
    parser.add_argument("--dose-key")
    parser.add_argument("--smiles-key")
    parser.add_argument("--control-key")
    parser.add_argument("--group-key")


def _native_project_commands(dataset_parser, fetch_name: str, population_flag: str):
    commands = dataset_parser.add_subparsers(dest="command", required=True)
    statistics = commands.add_parser("statistics")
    _native_statistics(
        statistics,
        "--cell-line-key" if population_flag == "cell_lines" else "--cell-type-key",
    )
    fetch = commands.add_parser(fetch_name)
    _project(fetch)
    fetch.add_argument(
        "--cell-lines" if population_flag == "cell_lines" else "--cell-types",
        nargs="+",
        required=True,
    )
    fetch.add_argument("--smiles-map")
    filtering = commands.add_parser("filter-conditions")
    _project(filtering)
    filtering.add_argument("--min-cells", type=int, default=500)
    filtering.add_argument("--max-cells", type=int, default=5_000)
    filtering.add_argument("--seed", type=int, default=42)
    filtering.add_argument("--workers", type=int, default=8)
    filtering.add_argument("--overwrite", action="store_true")
    hvg = commands.add_parser("select-hvg")
    _project(hvg)
    hvg.add_argument("--n-top-genes", type=int, default=2_000)
    hvg.add_argument("--workers", type=int, default=8)
    materialize = commands.add_parser("materialize")
    _project(materialize)
    materialize.add_argument("--pad-length", type=int, default=2_048)
    materialize.add_argument("--target-sum", type=float, default=10_000)
    materialize.add_argument("--workers", type=int, default=8)
    return commands


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MAP preprocess, preparation, train and eval")
    stages = parser.add_subparsers(dest="stage", required=True)

    preprocess_stage = stages.add_parser("preprocess")
    datasets = preprocess_stage.add_subparsers(dest="dataset", required=True)
    tahoe = datasets.add_parser("tahoe")
    commands = tahoe.add_subparsers(dest="command", required=True)
    statistics = commands.add_parser("statistics")
    _storage(statistics)
    statistics.add_argument("--batch-size", type=int, default=8192)
    statistics.add_argument("--refresh", action="store_true")
    fetch = commands.add_parser("fetch-cell-line")
    _project(fetch)
    fetch.add_argument("--cell-lines", nargs="+", required=True)
    fetch.add_argument("--batch-size", type=int, default=8192)
    filtering = commands.add_parser("filter-conditions")
    _project(filtering)
    filtering.add_argument("--min-cells", type=int, default=500)
    filtering.add_argument("--max-cells", type=int, default=5_000)
    filtering.add_argument("--seed", type=int, default=42)
    filtering.add_argument("--workers", type=int, default=8)
    filtering.add_argument("--overwrite", action="store_true")
    hvg = commands.add_parser("select-hvg")
    _project(hvg)
    hvg.add_argument("--n-top-genes", type=int, default=2_000)
    hvg.add_argument("--workers", type=int, default=8)
    materialize = commands.add_parser("materialize")
    _project(materialize)
    materialize.add_argument("--pad-length", type=int, default=2_048)
    materialize.add_argument("--target-sum", type=float, default=10_000)
    materialize.add_argument("--workers", type=int, default=8)

    sciplex = datasets.add_parser("sciplex")
    sciplex_commands = _native_project_commands(
        sciplex, "fetch-cell-line", "cell_lines"
    )
    export = sciplex_commands.add_parser("export-rds")
    _storage(export)
    export.add_argument("--input-file")
    export.add_argument("--overwrite", action="store_true")
    nips = datasets.add_parser("nips")
    _native_project_commands(nips, "fetch-cell-type", "cell_types")

    preparation_stage = stages.add_parser("preparation")
    commands = preparation_stage.add_subparsers(dest="command", required=True)
    workflow = commands.add_parser("create-workflow")
    _project(workflow)
    index = commands.add_parser("build-sampling-index")
    _project(index)
    index.add_argument("--workers", type=int, default=8)
    split = commands.add_parser("create-split")
    _project(split)
    split.add_argument("--rule", choices=("unprofiled_drug", "unseen_combination"), required=True)
    split.add_argument("--external-test-size", type=_size, required=True)
    split.add_argument("--internal-test-fraction", type=float, default=0.2)
    split.add_argument(
        "--external-drugs", nargs="+",
        help="Explicit drug names for the unprofiled-drug external test set",
    )
    split.add_argument("--seed", type=int, default=42)
    gene = commands.add_parser("precache-gene-tokens")
    _project(gene)
    _partition(gene, batch_size=512)
    drug = commands.add_parser("precache-drug-tokens")
    _project(drug)
    _partition(drug, batch_size=32)
    merge_knowledge = commands.add_parser("merge-knowledge-tokens")
    _project(merge_knowledge)
    merge_knowledge.add_argument("--gene-partitions", type=int, default=1)
    merge_knowledge.add_argument("--drug-partitions", type=int, default=1)
    state = commands.add_parser("precache-state-embeddings")
    _project(state)
    _partition(state, batch_size=32)
    state.add_argument("--workers", type=int, default=8)
    state.add_argument("--populations", nargs="+")
    merge_state = commands.add_parser("merge-state-embeddings")
    _project(merge_state)
    merge_state.add_argument("--num-partitions", type=int, required=True)
    merge_state.add_argument("--populations", nargs="+")
    validate = commands.add_parser("validate")
    _project(validate)
    validate.add_argument("--split-files", nargs="+", required=True)
    baselines = commands.add_parser("prepare-baselines")
    _project(baselines)
    baselines.add_argument(
        "--models",
        nargs="+",
        choices=train.MODEL_REGISTRY,
        default=list(train.MODEL_REGISTRY),
    )
    baselines.add_argument("--xpert-aliases-file")
    baselines.add_argument("--xpert-hidden-size", type=int, default=256)
    baselines.add_argument("--xpert-layers", type=int, default=3)
    baselines.add_argument("--xpert-epochs", type=int, default=300)
    baselines.add_argument("--xpert-seed", type=int, default=4242)
    baselines.add_argument("--xpert-device")

    training = stages.add_parser("train")
    _project(training)
    training.add_argument(
        "--model",
        choices=("map", *train.MODEL_REGISTRY),
        default="map",
    )
    training.add_argument("--regime", required=True)
    training.add_argument("--split-file", required=True)
    training.add_argument("--run-name", required=True)
    training.add_argument("--gpus", type=int, default=4)
    training.add_argument("--num-workers", type=int, default=6)
    training.add_argument("--set-size", type=int)
    training.add_argument("--batch-size", type=int)
    training.add_argument("--gradient-accumulation-steps", type=int)
    training.add_argument("--lr", type=float)
    training.add_argument("--hvg-loss-weight", type=float)
    training.add_argument("--epochs", type=int)
    training.add_argument("--max-steps", type=int)
    training.add_argument("--warmup-steps", type=int)
    training.add_argument("--checkpoint-every-epochs", type=int)
    training.add_argument("--compile-mode", choices=("none", "default"))
    training.add_argument("--resume")
    training.add_argument("--seed", type=int)
    training.add_argument("--dry-run", action="store_true")

    evaluation = stages.add_parser("eval")
    _project(evaluation)
    commands = evaluation.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--run-name")
    run.add_argument(
        "--model", choices=("map", *train.MODEL_REGISTRY)
    )
    run.add_argument("--regime")
    run.add_argument("--checkpoint", required=True)
    run.add_argument("--split-file")
    run.add_argument("--evaluation-name")
    run.add_argument(
        "--evaluation-splits", nargs="+",
        choices=("internal_test", "external_test"),
        default=["internal_test", "external_test"],
    )
    run.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    run.add_argument("--dry-run", action="store_true")
    analyze = commands.add_parser("analyze")
    analyze.add_argument("--prediction-file")
    analyze.add_argument("--evaluation-files", nargs="+")
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if args.stage == "preprocess":
        if args.dataset == "tahoe":
            if args.command == "statistics":
                preprocess.tahoe.statistics(
                    storage=args.storage, batch_size=args.batch_size, refresh=args.refresh
                )
            elif args.command == "fetch-cell-line":
                preprocess.tahoe.fetch_cell_line(
                    args.cell_lines,
                    project_name=args.project_name,
                    storage=args.storage,
                    batch_size=args.batch_size,
                )
            elif args.command == "filter-conditions":
                preprocess.tahoe.filter_conditions(
                    args.project_name,
                    storage=args.storage,
                    min_cells=args.min_cells,
                    max_cells=args.max_cells,
                    seed=args.seed,
                    workers=args.workers,
                    overwrite=args.overwrite,
                )
            elif args.command == "select-hvg":
                preprocess.tahoe.select_hvg(
                    args.project_name,
                    storage=args.storage,
                    n_top_genes=args.n_top_genes,
                    workers=args.workers,
                )
            else:
                preprocess.tahoe.materialize(
                    args.project_name,
                    storage=args.storage,
                    pad_length=args.pad_length,
                    target_sum=args.target_sum,
                    workers=args.workers,
                )
            return
        facade = preprocess.sciplex if args.dataset == "sciplex" else preprocess.nips
        if args.command == "export-rds":
            facade.export_rds(
                storage=args.storage,
                input_file=args.input_file,
                overwrite=args.overwrite,
            )
        elif args.command == "statistics":
            population_key = (
                args.cell_line_key if args.dataset == "sciplex" else args.cell_type_key
            )
            facade.statistics(
                storage=args.storage,
                smiles_map=args.smiles_map,
                refresh=args.refresh,
                **({"cell_line_key": population_key} if args.dataset == "sciplex" else {"cell_type_key": population_key}),
                perturbation_key=args.perturbation_key,
                dose_key=args.dose_key,
                smiles_key=args.smiles_key,
                control_key=args.control_key,
                group_key=args.group_key,
            )
        elif args.command in {"fetch-cell-line", "fetch-cell-type"}:
            populations = (
                args.cell_lines if args.dataset == "sciplex" else args.cell_types
            )
            fetch = (
                facade.fetch_cell_line
                if args.dataset == "sciplex"
                else facade.fetch_cell_type
            )
            fetch(
                populations,
                project_name=args.project_name,
                storage=args.storage,
                smiles_map=args.smiles_map,
            )
        elif args.command == "filter-conditions":
            facade.filter_conditions(
                args.project_name,
                storage=args.storage,
                min_cells=args.min_cells,
                max_cells=args.max_cells,
                seed=args.seed,
                workers=args.workers,
                overwrite=args.overwrite,
            )
        elif args.command == "select-hvg":
            facade.select_hvg(
                args.project_name,
                storage=args.storage,
                n_top_genes=args.n_top_genes,
                workers=args.workers,
            )
        else:
            facade.materialize(
                args.project_name,
                storage=args.storage,
                pad_length=args.pad_length,
                target_sum=args.target_sum,
                workers=args.workers,
            )
        return

    workflow = preparation.create_workflow(
        args.project_name, storage=args.storage
    )
    if args.stage == "preparation":
        if args.command == "create-workflow":
            print(workflow.workspace)
        elif args.command == "build-sampling-index":
            preparation.build_sampling_index(workflow, workers=args.workers)
        elif args.command == "create-split":
            preparation.create_split(
                workflow, rule=args.rule,
                external_test_size=args.external_test_size,
                internal_test_fraction=args.internal_test_fraction,
                seed=args.seed,
                external_drugs=args.external_drugs,
            )
        elif args.command == "precache-gene-tokens":
            preparation.precache_gene_tokens(
                workflow, partition_index=args.partition_index,
                num_partitions=args.num_partitions, batch_size=args.batch_size,
            )
        elif args.command == "precache-drug-tokens":
            preparation.precache_drug_tokens(
                workflow, partition_index=args.partition_index,
                num_partitions=args.num_partitions, batch_size=args.batch_size,
            )
        elif args.command == "merge-knowledge-tokens":
            preparation.merge_knowledge_tokens(
                workflow, gene_partitions=args.gene_partitions,
                drug_partitions=args.drug_partitions,
            )
        elif args.command == "precache-state-embeddings":
            preparation.precache_state_embeddings(
                workflow, populations=args.populations,
                partition_index=args.partition_index,
                num_partitions=args.num_partitions,
                batch_size=args.batch_size, workers=args.workers,
            )
        elif args.command == "merge-state-embeddings":
            preparation.merge_state_embeddings(
                workflow, populations=args.populations,
                num_partitions=args.num_partitions,
            )
        elif args.command == "prepare-baselines":
            preparation.prepare_baseline_inputs(
                workflow,
                models=args.models,
                xpert_aliases_file=args.xpert_aliases_file,
                xpert_hidden_size=args.xpert_hidden_size,
                xpert_layers=args.xpert_layers,
                xpert_epochs=args.xpert_epochs,
                xpert_seed=args.xpert_seed,
                xpert_device=args.xpert_device,
            )
        else:
            preparation.validate(workflow, split_files=args.split_files)
        return

    if args.stage == "train":
        common = dict(
            model=args.model, gpus=args.gpus, num_workers=args.num_workers,
            run_name=args.run_name, resume=args.resume, dry_run=args.dry_run,
        )
        for name in (
            "set_size", "batch_size", "lr", "epochs", "max_steps",
            "checkpoint_every_epochs", "seed",
        ):
            value = getattr(args, name)
            if value is not None:
                common[name] = value
        if args.model == "map":
            for name in (
                "gradient_accumulation_steps", "hvg_loss_weight", "warmup_steps",
                "compile_mode",
            ):
                value = getattr(args, name)
                if value is not None:
                    common[name] = value
        train.run(
            workflow, regime=args.regime, split_file=args.split_file, **common
        )
        return
    if args.command == "run":
        eval.run(
            workflow, run_name=args.run_name, model=args.model, regime=args.regime,
            checkpoint=args.checkpoint, split_file=args.split_file,
            evaluation_name=args.evaluation_name, seeds=tuple(args.seeds),
            evaluation_splits=tuple(args.evaluation_splits),
            dry_run=args.dry_run,
        )
    else:
        eval.analyze(
            workflow, prediction_file=args.prediction_file,
            evaluation_files=args.evaluation_files,
        )


if __name__ == "__main__":
    main()
