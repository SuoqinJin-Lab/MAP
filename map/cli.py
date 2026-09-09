from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

from . import eval, preparation, preprocess, train
from ._common import splits


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


def _handler_factory(value: str):
    if hasattr(preprocess.builtin, value):
        return getattr(preprocess.builtin, value)
    if ":" not in value:
        raise ValueError(
            "handler must be a built-in name or package.module:factory"
        )
    module_name, attribute = value.split(":", 1)
    return getattr(importlib.import_module(module_name), attribute)


def _experiment_paths(args, *, frozen=None):
    """Resolve project paths for preparation/train/eval commands."""
    from ._common.paths import experiment_paths

    frozen_root = Path(frozen) if frozen is not None else Path(args.storage) / "frozen_models"
    return experiment_paths(
        Path(args.storage) / "projects" / args.project_name,
        frozen_root,
    )

def _preprocess_flow(args):
    options = {}
    if args.handler_config:
        options = json.loads(
            Path(args.handler_config).read_text(encoding="utf-8")
        )
        if not isinstance(options, dict):
            raise ValueError("handler config must contain one JSON object")
    factory = _handler_factory(args.handler)
    handler = (
        factory(storage=args.storage, **options)
        if callable(factory)
        else factory
    )
    return preprocess.pipeline(handler)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MAP preprocess, preparation, train and eval")
    stages = parser.add_subparsers(dest="stage", required=True)

    preprocess_stage = stages.add_parser("preprocess")
    _storage(preprocess_stage)
    preprocess_stage.add_argument("--handler", required=True)
    preprocess_stage.add_argument("--handler-config")
    commands = preprocess_stage.add_subparsers(dest="command", required=True)
    statistics = commands.add_parser("statistics")
    statistics.add_argument("--batch-size", type=int)
    statistics.add_argument("--refresh", action="store_true")
    fetch = commands.add_parser("fetch-populations")
    fetch.add_argument("--project-name", required=True)
    fetch.add_argument("--populations", nargs="+", required=True)
    fetch.add_argument("--batch-size", type=int)
    filtering = commands.add_parser("filter-conditions")
    filtering.add_argument("--project-name", required=True)
    filtering.add_argument("--min-cells", type=int, default=500)
    filtering.add_argument("--max-cells", type=int, default=5_000)
    filtering.add_argument("--seed", type=int, default=42)
    filtering.add_argument("--workers", type=int, default=8)
    filtering.add_argument("--overwrite", action="store_true")
    hvg = commands.add_parser("select-hvg")
    hvg.add_argument("--project-name", required=True)
    hvg.add_argument("--n-top-genes", type=int, default=2_000)
    hvg.add_argument("--workers", type=int, default=8)
    hvg.add_argument(
        "--batch-key", choices=("none", "population"), default="none",
        help=("HVG protocol: none fits one global gene set (default); "
              "population fits within-cell-line statistics and merges one "
              "shared gene set"),
    )

    preparation_stage = stages.add_parser("preparation")
    commands = preparation_stage.add_subparsers(dest="command", required=True)
    index = commands.add_parser("build-sampling-index")
    _project(index)
    index.add_argument("--workers", type=int, default=8)
    split = commands.add_parser("create-split")
    _project(split)
    split.add_argument("--rule", choices=splits.TRAIN_RULES, required=True)
    split.add_argument("--external-test-size", type=_size, required=True)
    split.add_argument("--internal-test-fraction", type=float, default=0.2)
    split.add_argument(
        "--external-drugs", nargs="+",
        help="Explicit drug names for the unprofiled-drug external test set",
    )
    split.add_argument("--seed", type=int, default=42)
    split.add_argument(
        "--output-name",
        help="Optional JSON filename under projects/<project>/splits/",
    )
    split.add_argument("--overwrite", action="store_true")
    split.add_argument(
        "--disjoint-external-drugs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For unseen-combination, keep external drugs disjoint across populations",
    )
    gene = commands.add_parser("prepare-gene-tokens")
    _project(gene)
    _partition(gene, batch_size=512)
    drug = commands.add_parser("prepare-knowledge-drug-tokens")
    _project(drug)
    _partition(drug, batch_size=32)
    merge_knowledge = commands.add_parser("assemble-knowledge-tokens")
    _project(merge_knowledge)
    merge_knowledge.add_argument("--gene-partitions", type=int, default=1)
    merge_knowledge.add_argument("--drug-partitions", type=int, default=1)
    state = commands.add_parser("prepare-condition-embeddings")
    _project(state)
    _partition(state, batch_size=32)
    state.add_argument("--workers", type=int, default=8)
    state.add_argument("--populations", nargs="+")
    merge_state = commands.add_parser("assemble-condition-embeddings")
    _project(merge_state)
    merge_state.add_argument("--num-partitions", type=int, required=True)
    merge_state.add_argument("--populations", nargs="+")
    validate = commands.add_parser("validate")
    _project(validate)
    validate.add_argument("--split-files", nargs="+", required=True)
    validate_method = commands.add_parser("validate-method")
    _project(validate_method)
    validate_method.add_argument(
        "--method", choices=train.MODEL_REGISTRY, default="map"
    )
    validate_method.add_argument("--regime", required=True)
    validate_method.add_argument("--split-file", required=True)
    validate_method.add_argument("--frozen-models")
    validate_method.add_argument("--use-deg-mask", action="store_true")
    validate_method.add_argument(
        "--crisp-deg-mask-mode", choices=("official", "validation")
    )
    validate_method.add_argument(
        "--cmonge-drug-representation", choices=("rdkit", "moa")
    )
    validate_method.add_argument(
        "--xpert-input-mode", choices=("official", "validation")
    )
    validate_method.add_argument(
        "--strict", action=argparse.BooleanOptionalAction, default=True
    )
    artifact_specs = {
        "prepare-ecfp4-features": (),
        "prepare-fcfp4-features": (),
        "prepare-control-means": (),
        "prepare-deg-masks": (),
        "prepare-molecular-descriptors": (),
        "prepare-moa-features": (),
        "prepare-unimol-tokens": (),
        "prepare-graph-assets": (),
        "prepare-expression-bins": (),
    }
    for command_name in artifact_specs:
        artifact = commands.add_parser(command_name)
        _project(artifact)
        artifact.add_argument("--overwrite", action="store_true")
        if command_name == "prepare-deg-masks":
            artifact.add_argument("--crisp-deg-top-k", type=int, default=50)
            artifact.add_argument(
                "--crisp-deg-mask-mode",
                choices=("official", "validation"),
                default="official",
            )
        elif command_name == "prepare-expression-bins":
            artifact.add_argument(
                "--input-formats", nargs="+",
                choices=("official", "validation"), default=["official"],
            )
            artifact.add_argument("--expression-bins", type=int, default=128)
            artifact.add_argument("--expression-min", type=float)
            artifact.add_argument("--expression-max", type=float)
            artifact.add_argument("--expression-sample-cells", type=int, default=1024)
        elif command_name == "prepare-moa-features":
            artifact.add_argument("--split-file", required=True)
            artifact.add_argument("--max-cells-per-condition", type=int, default=32)
            artifact.add_argument("--moa-components", type=int, default=10)
            artifact.add_argument("--seed", type=int, default=42)
        elif command_name == "prepare-unimol-tokens":
            artifact.add_argument("--unimol-batch-size", type=int, default=32)
            artifact.add_argument("--max-atoms", type=int, default=122)
        elif command_name == "prepare-graph-assets":
            artifact.add_argument("--aliases-file")
            artifact.add_argument("--graph-hidden-size", type=int, default=256)
            artifact.add_argument("--graph-layers", type=int, default=3)
            artifact.add_argument("--graph-epochs", type=int, default=300)
            artifact.add_argument("--graph-seed", type=int, default=4242)
            artifact.add_argument("--graph-device")
            artifact.add_argument(
                "--graph-pretraining-mode",
                choices=("neighbor_loader", "full_graph"),
                default="neighbor_loader",
            )
            artifact.add_argument(
                "--graph-num-neighbors", type=int, nargs="+", default=[35, 20, 10]
            )
            artifact.add_argument("--graph-batch-size", type=int, default=4)
    for command_name in (
        "prepare-cell-metadata",
        "prepare-state-inputs",
        "prepare-hvg-expression",
    ):
        artifact = commands.add_parser(command_name)
        _project(artifact)
        artifact.add_argument("--pad-length", type=int, default=2_048)
        artifact.add_argument("--target-sum", type=float, default=10_000)
        artifact.add_argument("--workers", type=int, default=8)
        artifact.add_argument("--overwrite", action="store_true")

    training = stages.add_parser("train")
    _project(training)
    training.add_argument(
        "--model", choices=train.MODEL_REGISTRY, default="map",
    )
    training.add_argument("--regime", required=True)
    training.add_argument("--split-file", required=True)
    training.add_argument("--run-name", required=True)
    # Keep the default allocation conservative; callers remain free to choose
    # the resources in their local or scheduled wrapper.
    training.add_argument("--gpus", type=int, default=2)
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
    training.add_argument(
        "--xpert-input-mode",
        choices=("official", "validation"),
        help=(
            "XPert prepared input contract; official is the default"
        ),
    )
    training.add_argument(
        "--xpert-context-tokens", choices=("none", "dose", "dose_time")
    )
    training.add_argument(
        "--xpert-attention-padding-mode",
        choices=("masked", "official_unmasked"),
    )
    training.add_argument(
        "--xpert-use-gene-position-embedding",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    training.add_argument(
        "--xpert-loss-reduction-scale", choices=("none", "sample_count")
    )
    training.add_argument("--xpert-expression-bins", type=int)
    training.add_argument("--xpert-expression-min", type=float)
    training.add_argument("--xpert-expression-max", type=float)
    training.add_argument(
        "--crisp-deg-mask-mode", choices=("official", "validation")
    )
    training.add_argument("--crisp-deg-top-k", type=int)
    training.add_argument(
        "--crisp-drug-representation", choices=("official", "validation")
    )
    training.add_argument(
        "--crisp-control-representation", choices=("official", "validation")
    )
    training.add_argument(
        "--cmonge-drug-representation", choices=("rdkit", "moa")
    )
    training.add_argument("--resume")
    training.add_argument("--seed", type=int)
    training.add_argument("--dry-run", action="store_true")

    evaluation = stages.add_parser("eval")
    _project(evaluation)
    commands = evaluation.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--run-name")
    run.add_argument(
        "--model", choices=train.MODEL_REGISTRY
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
    run.add_argument("--set-size", type=int, default=24)
    run.add_argument(
        "--evaluation-unit",
        choices=("dose_level_condition", "cell_line_drug"),
        default="cell_line_drug",
        help="Primary report unit; both units are always written",
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
        flow = _preprocess_flow(args)
        if args.command == "statistics":
            options = {"refresh": args.refresh}
            if args.batch_size is not None:
                options["batch_size"] = args.batch_size
            flow.watch_data(**options)
        elif args.command == "fetch-populations":
            options = {}
            if args.batch_size is not None:
                options["batch_size"] = args.batch_size
            flow.fetch_populations(
                args.populations,
                project_name=args.project_name,
                **options,
            )
        elif args.command == "filter-conditions":
            flow.filter_conditions(
                flow.open_project(args.project_name),
                min_cells=args.min_cells,
                max_cells=args.max_cells,
                seed=args.seed,
                workers=args.workers,
                overwrite=args.overwrite,
            )
        else:
            flow.select_hvg(
                flow.open_project(args.project_name),
                n_top_genes=args.n_top_genes,
                workers=args.workers,
                batch_key=None if args.batch_key == "none" else args.batch_key,
            )
        return

    # Each preparation command writes one artifact after the handler has
    # created the project contract.
    if args.stage == "preparation" and args.command in {
        "prepare-cell-metadata",
        "prepare-state-inputs",
        "prepare-hvg-expression",
    }:
        artifact_paths = _experiment_paths(args)
        function = {
            "prepare-cell-metadata": preparation.prepare_cell_metadata,
            "prepare-state-inputs": preparation.prepare_state_inputs,
            "prepare-hvg-expression": preparation.prepare_hvg_expression,
        }[args.command]
        function(
            artifact_paths,
            workers=args.workers,
            target_sum=args.target_sum,
            pad_length=args.pad_length,
            overwrite=args.overwrite,
        )
        return

    artifact_commands = {
        "prepare-ecfp4-features",
        "prepare-fcfp4-features",
        "prepare-control-means",
        "prepare-deg-masks",
        "prepare-molecular-descriptors",
        "prepare-moa-features",
        "prepare-unimol-tokens",
        "prepare-graph-assets",
        "prepare-expression-bins",
    }
    if args.stage == "preparation" and args.command in artifact_commands:
        artifact_paths = _experiment_paths(args)
        if args.command == "prepare-ecfp4-features":
            preparation.prepare_ecfp4_features(
                artifact_paths, overwrite=args.overwrite
            )
        elif args.command == "prepare-fcfp4-features":
            preparation.prepare_fcfp4_features(
                artifact_paths, overwrite=args.overwrite
            )
        elif args.command == "prepare-control-means":
            preparation.prepare_control_means(
                artifact_paths, overwrite=args.overwrite
            )
        elif args.command == "prepare-deg-masks":
            preparation.prepare_deg_masks(
                artifact_paths,
                top_k=args.crisp_deg_top_k,
                mask_mode=args.crisp_deg_mask_mode,
                overwrite=args.overwrite,
            )
        elif args.command == "prepare-molecular-descriptors":
            preparation.prepare_molecular_descriptors(
                artifact_paths, overwrite=args.overwrite
            )
        elif args.command == "prepare-moa-features":
            preparation.prepare_moa_features(
                artifact_paths,
                split_file=args.split_file,
                max_cells_per_condition=args.max_cells_per_condition,
                n_components=args.moa_components,
                seed=args.seed,
                overwrite=args.overwrite,
            )
        elif args.command == "prepare-unimol-tokens":
            preparation.prepare_unimol_tokens(
                artifact_paths,
                overwrite=args.overwrite,
                max_atoms=args.max_atoms,
                batch_size=args.unimol_batch_size,
            )
        elif args.command == "prepare-graph-assets":
            preparation.prepare_graph_assets(
                artifact_paths,
                overwrite=args.overwrite,
                aliases_file=args.aliases_file,
                hidden_size=args.graph_hidden_size,
                layers=args.graph_layers,
                epochs=args.graph_epochs,
                seed=args.graph_seed,
                device=args.graph_device,
                graph_pretraining_mode=args.graph_pretraining_mode,
                graph_num_neighbors=args.graph_num_neighbors,
                graph_batch_size=args.graph_batch_size,
            )
        else:
            preparation.prepare_expression_bins(
                artifact_paths,
                input_formats=args.input_formats,
                expression_bins=args.expression_bins,
                expression_min=args.expression_min,
                expression_max=args.expression_max,
                expression_sample_cells=args.expression_sample_cells,
                overwrite=args.overwrite,
            )
        return

    if args.stage == "preparation" and args.command == "validate-method":
        artifact_paths = _experiment_paths(args, frozen=args.frozen_models)
        options = {
            key: value
            for key, value in {
                "use_deg_mask": args.use_deg_mask,
                "deg_mask_mode": args.crisp_deg_mask_mode,
                "drug_representation": args.cmonge_drug_representation,
                "input_mode": args.xpert_input_mode,
            }.items()
            if value is not None
        }
        preparation.validate_method(
            artifact_paths,
            method=args.method,
            regime=args.regime,
            split_file=args.split_file,
            options=options,
            frozen_assets=args.frozen_models,
            strict=args.strict,
        )
        return
    if args.stage == "preparation":
        paths = _experiment_paths(args)
        if args.command == "build-sampling-index":
            preparation.build_sampling_index(paths, workers=args.workers)
        elif args.command == "create-split":
            preparation.create_split(
                paths, rule=args.rule,
                external_test_size=args.external_test_size,
                internal_test_fraction=args.internal_test_fraction,
                seed=args.seed,
                external_drugs=args.external_drugs,
                output_name=args.output_name,
                overwrite=args.overwrite,
                disjoint_external_drugs=args.disjoint_external_drugs,
            )
        elif args.command == "prepare-gene-tokens":
            preparation.prepare_gene_tokens(
                paths, partition_index=args.partition_index,
                num_partitions=args.num_partitions, batch_size=args.batch_size,
            )
        elif args.command == "prepare-knowledge-drug-tokens":
            preparation.prepare_knowledge_drug_tokens(
                paths, partition_index=args.partition_index,
                num_partitions=args.num_partitions, batch_size=args.batch_size,
            )
        elif args.command == "assemble-knowledge-tokens":
            preparation.assemble_knowledge_tokens(
                paths, gene_partitions=args.gene_partitions,
                drug_partitions=args.drug_partitions,
            )
        elif args.command == "prepare-condition-embeddings":
            preparation.prepare_condition_embeddings(
                paths, populations=args.populations,
                partition_index=args.partition_index,
                num_partitions=args.num_partitions,
                batch_size=args.batch_size, workers=args.workers,
            )
        elif args.command == "assemble-condition-embeddings":
            preparation.assemble_condition_embeddings(
                paths, populations=args.populations,
                num_partitions=args.num_partitions,
            )
        else:
            preparation.validate(paths, split_files=args.split_files)
        return

    if args.stage == "train":
        paths = _experiment_paths(args)
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
        elif args.model == "xpert":
            xpert_options = {
                "input_mode": args.xpert_input_mode,
                "context_tokens": args.xpert_context_tokens,
                "attention_padding_mode": args.xpert_attention_padding_mode,
                "use_gene_position_embedding": (
                    args.xpert_use_gene_position_embedding
                ),
                "loss_reduction_scale": args.xpert_loss_reduction_scale,
                "expression_bins": args.xpert_expression_bins,
                "expression_min": args.xpert_expression_min,
                "expression_max": args.xpert_expression_max,
            }
            common.update({
                name: value
                for name, value in xpert_options.items()
                if value is not None
            })
        elif args.model == "crisp":
            crisp_options = {
                "deg_mask_mode": args.crisp_deg_mask_mode,
                "deg_top_k": args.crisp_deg_top_k,
                "drug_representation": args.crisp_drug_representation,
                "control_representation": args.crisp_control_representation,
            }
            common.update({
                name: value
                for name, value in crisp_options.items()
                if value is not None
            })
        elif args.model == "cmonge" and args.cmonge_drug_representation is not None:
            common["drug_representation"] = args.cmonge_drug_representation
        train.run(
            paths, regime=args.regime, split_file=args.split_file, **common
        )
        return
    if args.command == "run":
        paths = _experiment_paths(args)
        eval.run(
            paths, run_name=args.run_name, model=args.model, regime=args.regime,
            checkpoint=args.checkpoint, split_file=args.split_file,
            evaluation_name=args.evaluation_name, seeds=tuple(args.seeds),
            evaluation_splits=tuple(args.evaluation_splits),
            set_size=args.set_size,
            evaluation_unit=args.evaluation_unit,
            dry_run=args.dry_run,
        )
    else:
        paths = _experiment_paths(args)
        eval.analyze(
            paths, prediction_file=args.prediction_file,
            evaluation_files=args.evaluation_files,
        )


if __name__ == "__main__":
    main()
