"""CLI entry point: ``python -m majorana_acp.cli.evaluate <ckpt-or-dir>``.

The positional argument accepts either a ``.pt`` checkpoint file or a
training run directory; in the latter case the latest ``epoch_*.pt`` is
used. By default outputs land in ``<ckpt-parent>/eval/``; pass ``--out``
to override.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from majorana_acp.eval.evaluator import evaluate


def _setup_console_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Majorana PSD classifier on the test set."
    )
    parser.add_argument(
        "checkpoint",
        type=Path,
        help="Path to an epoch_*.pt file or to a training run directory.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Output directory (default: <checkpoint-parent>/eval for "
            "--split test, or <checkpoint-parent>/eval_train for --split train)."
        ),
    )
    parser.add_argument(
        "--split",
        choices=("train", "test"),
        default="test",
        help=(
            "Which Majorana split to score. Use 'train' to produce a "
            "predictions.h5 over training files (e.g. for cut_acceptance "
            "CNP training); 'test' (default) keeps the historical behaviour."
        ),
    )
    parser.add_argument(
        "--subset-portion",
        type=float,
        default=None,
        help=(
            "Override ``data.subset_portion`` for this evaluation only. "
            "The training-time subset is a *training* concern (how much "
            "labelled data the classifier saw); the test split is "
            "held-out and should normally be evaluated at 1.0 regardless "
            "of what the training config used. Pass 1.0 to score the "
            "full split. Default: use the YAML's value (legacy)."
        ),
    )
    args = parser.parse_args(argv)

    _setup_console_logging()
    evaluate(
        args.checkpoint,
        out_dir=args.out,
        split=args.split,
        subset_portion=args.subset_portion,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
