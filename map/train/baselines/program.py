from __future__ import annotations

import argparse

from .engine import DEFAULTS
from .registry import MODEL_REGISTRY, add_model_arguments, train_model


def parse_args(argv=None):
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--model", choices=MODEL_REGISTRY, required=True)
    selected, _ = selector.parse_known_args(argv)
    defaults = DEFAULTS[selected.model]
    parser = argparse.ArgumentParser(
        description=f"Train the {selected.model} baseline on MAP mmap data",
        parents=[selector],
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--regime", choices=("unprofiled_drug", "unseen_combination"), required=True
    )
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--populations", nargs="+", required=True)
    parser.add_argument("--set-size", type=int, default=defaults["set_size"])
    parser.add_argument("--batch-size", type=int, default=defaults["batch_size"])
    parser.add_argument("--epochs", type=int, default=defaults["epochs"])
    parser.add_argument("--max-steps", type=int, default=defaults["max_steps"])
    parser.add_argument(
        "--checkpoint-every-epochs",
        type=int,
        default=defaults["checkpoint_every_epochs"],
    )
    parser.add_argument("--samples-per-epoch", type=int)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--lr", type=float, default=defaults["lr"])
    parser.add_argument(
        "--amp-dtype",
        choices=("fp32", "bf16"),
        default=defaults.get("amp_dtype", "fp32"),
    )
    parser.add_argument("--seed", type=int, default=defaults["seed"])
    parser.add_argument("--resume")
    add_model_arguments(selected.model, parser)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if min(
        args.set_size,
        args.batch_size,
        args.epochs,
        args.max_steps,
        args.checkpoint_every_epochs,
    ) <= 0:
        raise ValueError("Training sizes and schedules must be positive")
    if args.lr <= 0 or args.num_workers < 0:
        raise ValueError("lr must be positive and num_workers non-negative")
    if args.samples_per_epoch is not None and args.samples_per_epoch <= 0:
        raise ValueError("samples_per_epoch must be positive when provided")
    train_model(args.model, args)


if __name__ == "__main__":
    main()
