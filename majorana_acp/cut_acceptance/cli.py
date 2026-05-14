"""CLI entry point for the binned-CNP cut-acceptance pipeline.

Usage::

    python -m majorana_acp.cut_acceptance.cli configs/cut_acceptance/foo.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from majorana_acp.cut_acceptance.config import load_config
from majorana_acp.cut_acceptance.pipeline import run_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(prog="majorana_acp.cut_acceptance.cli")
    parser.add_argument("config", type=Path, help="Path to a CutAcceptanceConfig YAML file.")
    parser.add_argument(
        "--seed", type=int, default=0, help="Master seed (multiplied + offset for sub-RNGs)."
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log = logging.getLogger("cut_acceptance")

    cfg = load_config(args.config)
    log.info(
        "config: %s  target_class=%s  bin_width=%.0f keV  out_dir=%s",
        cfg.name,
        cfg.target_class,
        cfg.energy_bin_width,
        cfg.out_dir,
    )

    summary = run_pipeline(cfg, seed=args.seed)

    log.info("done: %s", summary.out_dir)
    log.info(
        "  pool: train_events=%d  bins_used=%d  validation_events=%d",
        summary.n_train_events,
        summary.n_bins_used,
        summary.n_validation_events,
    )
    log.info(
        "  T*=%.4f  cnp_final_loss=%.4f  (run scripts/diagnostics/cnp_test_inference for coverage)",
        summary.youden_T_star,
        summary.cnp_final_train_loss,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
