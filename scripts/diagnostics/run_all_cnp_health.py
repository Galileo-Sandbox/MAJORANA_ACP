"""Run cnp_health_report on every cut_acceptance config found.

Skips configs whose pipeline outputs aren't on disk yet (so it's safe
to run mid-way through a 9-cell sweep without crashing).
"""
from __future__ import annotations

from pathlib import Path

from majorana_acp.cut_acceptance.config import load_config
from scripts.diagnostics.cnp_health_report import run as run_cell

CFG_ROOT = Path("configs/cut_acceptance").resolve()
OUT_ROOT = Path("analysis/cnp_audit").resolve()


def main() -> None:
    n_total = n_done = n_skipped = 0
    for cfg_path in sorted(CFG_ROOT.rglob("*.yaml")):
        if "_legacy" in cfg_path.parts:
            continue
        n_total += 1
        try:
            cfg = load_config(cfg_path)
        except Exception as exc:
            print(f"[skip] {cfg_path}: cfg parse failed ({exc})")
            n_skipped += 1
            continue
        if not (Path(cfg.out_dir) / "cnp.ckpt").is_file():
            print(f"[skip] {cfg_path.name}: no cnp.ckpt at {cfg.out_dir}")
            n_skipped += 1
            continue
        rel = cfg_path.relative_to(CFG_ROOT)
        out_dir = OUT_ROOT / rel.parent / rel.stem
        try:
            run_cell(cfg_path, out_dir)
            n_done += 1
            print(f"[done] {rel} → {out_dir}")
        except Exception as exc:
            print(f"[fail] {rel}: {exc}")
    print(f"\nsummary: {n_done} done / {n_skipped} skipped / {n_total} total")


if __name__ == "__main__":
    main()
