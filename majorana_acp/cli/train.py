"""CLI entry point: ``python -m majorana_acp.cli.train configs/foo.yaml``.

Loads and validates the YAML config, then hands off to
:func:`majorana_acp.training.trainer.train`.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from majorana_acp.training.config import load_config
from majorana_acp.training.trainer import train


def _setup_console_logging() -> None:
    """Stream majorana_acp logs to stderr at INFO level."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train a Majorana PSD classifier from a YAML experiment config."
    )
    parser.add_argument("config", type=Path, help="Path to the experiment YAML.")
    args = parser.parse_args(argv)

    _setup_console_logging()
    cfg = load_config(args.config)
    train(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
