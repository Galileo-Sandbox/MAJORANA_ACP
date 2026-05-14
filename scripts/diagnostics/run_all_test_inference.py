"""Run cnp_test_inference on every (model, binning, class) cell.

Skips configs whose trained CNP isn't on disk yet.
"""
from __future__ import annotations

import argparse
from pathlib import Path

# Headless backend for the CLI sweep so figures save without an X display.
import matplotlib  # noqa: E402

matplotlib.use("Agg")

from majorana_acp.cut_acceptance.config import load_config  # noqa: E402
from scripts.diagnostics.cnp_test_inference import (  # noqa: E402
    DEFAULT_MAX_CONTEXT_PER_PASS,
    DEFAULT_N_DENSE,
)
from scripts.diagnostics.cnp_test_inference import (
    run as run_cell,
)

CFG_ROOT = Path("configs/cut_acceptance").resolve()
OUT_ROOT = Path("analysis/cnp_audit").resolve()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-fraction", type=float, default=1.0)
    ap.add_argument("--context-fraction", type=float, default=0.20)
    ap.add_argument("--n-mc", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-dense", type=int, default=DEFAULT_N_DENSE)
    ap.add_argument(
        "--max-context-per-pass", type=int, default=DEFAULT_MAX_CONTEXT_PER_PASS,
    )
    args = ap.parse_args()

    n_done = n_skipped = 0
    for cfg_path in sorted(CFG_ROOT.rglob("*.yaml")):
        if "_legacy" in cfg_path.parts:
            continue
        try:
            cfg = load_config(cfg_path)
        except Exception as exc:
            print(f"[skip] {cfg_path}: cfg parse failed ({exc})")
            n_skipped += 1
            continue
        if not (Path(cfg.out_dir) / "cnp.ckpt").is_file():
            print(f"[skip] {cfg_path.name}: no cnp.ckpt")
            n_skipped += 1
            continue
        rel = cfg_path.relative_to(CFG_ROOT)
        out_dir = OUT_ROOT / rel.parent / rel.stem
        run_cell(
            cfg_path, out_dir,
            test_fraction=args.test_fraction,
            context_fraction=args.context_fraction,
            n_mc=args.n_mc,
            seed=args.seed,
            n_dense=args.n_dense,
            max_context_per_pass=args.max_context_per_pass,
        )
        n_done += 1
        print(f"[done] {rel} → {out_dir}\n")
    print(f"summary: {n_done} done / {n_skipped} skipped")


if __name__ == "__main__":
    main()
