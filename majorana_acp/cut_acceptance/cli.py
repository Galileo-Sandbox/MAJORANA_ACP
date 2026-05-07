"""CLI entry point for the cut-acceptance pipeline.

Usage::

    python -m majorana_acp.cut_acceptance.cli configs/cut_acceptance/foo.yaml

Optional overrides for the MFGP-prep batch sizes and master RNG seed.
The pipeline writes everything under ``cfg.out_dir``.
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
        "--n-mfgp-lf-trials",
        type=int,
        default=200,
        help="Number of LF θ samples used to build the MFGP datasets.",
    )
    parser.add_argument(
        "--n-mfgp-hf-trials",
        type=int,
        default=100,
        help="Number of HF θ samples used to build the MFGP datasets.",
    )
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
    log.info("config: %s  target_class=%d  out_dir=%s", cfg.name, cfg.target_class, cfg.out_dir)

    summary = run_pipeline(
        cfg,
        n_mfgp_lf_trials=args.n_mfgp_lf_trials,
        n_mfgp_hf_trials=args.n_mfgp_hf_trials,
        seed=args.seed,
    )

    log.info("done: %s", summary.out_dir)
    log.info(
        "  pools: lf=%d  hf_train=%d  hf_holdout=%d",
        summary.n_lf,
        summary.n_hf_train,
        summary.n_hf_holdout,
    )
    log.info(
        "  coverage: 1σ=%.3f  2σ=%.3f  3σ=%.3f  (n_test=%d)",
        summary.coverage_1sigma,
        summary.coverage_2sigma,
        summary.coverage_3sigma,
        summary.holdout_n_test,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
