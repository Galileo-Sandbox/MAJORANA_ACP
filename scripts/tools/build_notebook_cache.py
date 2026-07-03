"""Persist compact per-paradigm inference caches for the flagship notebook.

For every canonical cell in :data:`CANONICAL_PARADIGMS`, this script

* re-runs :func:`scripts.diagnostics.cnp_test_inference.infer_and_evaluate`
  with the same knobs the notebook uses (``TEST_FRACTION=1.0``,
  ``CONTEXT_N_POINTS=2000``, ``N_MC=50``, ``seed=0``);
* replays the notebook's ``load_dt_events`` / ``load_train_events``
  helpers to snapshot the raw D_C / D_T / D_train event arrays;
* dumps everything to
  ``cache/inference/<model>/<paradigm>/inclusive_bin10.npz`` plus a
  sibling ``inclusive_bin10_audit.json`` carrying the sawtooth +
  peak metrics.

A fresh clone renders ``notebooks/data_visualization.ipynb`` §8.4
end-to-end from these two files alone — no ``cnp.ckpt`` or upstream
``predictions.h5`` required.

Usage
-----
Run after training any of the canonical paradigms::

    uv run python -m scripts.tools.build_notebook_cache

or restrict to a single paradigm::

    uv run python -m scripts.tools.build_notebook_cache sweeps/cell17
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np

from majorana_acp.analysis.metrics import analyze_sawtooth_suite
from majorana_acp.cut_acceptance.config import load_config
from majorana_acp.cut_acceptance.pipeline import resolve_out_dir
from scripts.diagnostics.cnp_test_inference import (
    _filter_test_events,
    infer_and_evaluate,
    split_test_data,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CFG_ROOT = REPO_ROOT / "configs" / "cut_acceptance"
CACHE_ROOT = REPO_ROOT / "cache" / "inference"

# Kept in sync with the notebook's §8.4.1 default knobs.
MODEL = "simple_cnn_small"
TEST_FRACTION = 1.0
CONTEXT_N_POINTS = 2000
N_MC = 50
SEED = 0

# The five canonical storytellers referenced by README + notebook.
CANONICAL_PARADIGMS: list[str] = [
    "true_cnp",
    "sweeps/base1_matched",
    "sweeps/base3_matched",
    "sweeps/cell15_v5",
    "sweeps/cell17",
]

# Sawtooth-diagnostic Compton windows — identical to write_report.
SAWTOOTH_WINDOWS: list[tuple[float, float]] = [(1700.0, 2000.0), (2200.0, 2400.0)]


def _load_train_events(cfg) -> tuple[np.ndarray, np.ndarray]:
    """Mirror the notebook's ``load_train_events`` filter block."""
    with h5py.File(cfg.train_predictions_path, "r") as f:
        e_full = f["energy"][:].astype(np.float64)
        s_full = f["score"][:].astype(np.float64)
        l_full = f["label"][:].astype(np.int64)
    if cfg.target_class == "all":
        cls_mask = np.ones_like(l_full, dtype=bool)
    else:
        cls_mask = l_full == int(cfg.target_class)
    e_lo, e_hi = cfg.energy_range
    keep = cls_mask & (e_full >= e_lo) & (e_full <= e_hi)
    return e_full[keep], s_full[keep]


def build_cache_for(paradigm: str) -> Path | None:
    cfg_path = CFG_ROOT / MODEL / paradigm / "bin10" / "inclusive.yaml"
    if not cfg_path.is_file():
        print(f"  [skip] {paradigm}: config not found at {cfg_path}")
        return None
    cfg = load_config(cfg_path)
    ckpt = resolve_out_dir(cfg) / "cnp.ckpt"
    if not ckpt.is_file():
        print(f"  [skip] {paradigm}: no trained checkpoint at {ckpt}")
        return None

    # Split D_C / D_T with the exact fractions the notebook uses.
    energy_all, score_all, T_star = _filter_test_events(cfg)
    n_total = energy_all.size
    ctx_frac = min(CONTEXT_N_POINTS / max(n_total, 1), 0.999)
    e_C, s_C, e_T, s_T = split_test_data(
        energy_all,
        score_all,
        test_fraction=TEST_FRACTION,
        context_fraction=ctx_frac,
        seed=SEED,
    )

    # Full training pool, same filters as the model saw.
    e_train, s_train = _load_train_events(cfg)

    # Live inference — same fractions as get_inference().
    res = infer_and_evaluate(
        cfg,
        test_fraction=TEST_FRACTION,
        context_fraction=ctx_frac,
        n_mc=N_MC,
        seed=SEED,
    )

    # Sawtooth metrics — mirror write_report.
    sawtooth_metrics = {
        f"region_{int(lo)}_{int(hi)}": analyze_sawtooth_suite(
            res.dense_energies, res.dense_beta, (lo, hi)
        )
        for (lo, hi) in SAWTOOTH_WINDOWS
    }

    out_dir = CACHE_ROOT / MODEL / paradigm
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / "inclusive_bin10.npz"
    audit_path = out_dir / "inclusive_bin10_audit.json"

    np.savez_compressed(
        npz_path,
        # Bin-center arrays.
        bin_centers=res.bin_centers,
        rate=res.rate,
        rate_lo=res.rate_lo,
        rate_hi=res.rate_hi,
        n_target_per_bin=res.n_target_per_bin,
        beta=res.beta,
        beta_std=res.beta_std,
        # Dense grid (red curve).
        dense_energies=res.dense_energies,
        dense_beta=res.dense_beta,
        dense_beta_std=res.dense_beta_std,
        # Raw event arrays for the D_T / D_C / D_train plot toggle.
        e_C=e_C,
        s_C=s_C,
        e_T=e_T,
        s_T=s_T,
        e_train=e_train,
        s_train=s_train,
        # Scalars.
        T_star=np.float64(T_star),
        n_context_per_pass=np.int64(res.n_context_per_pass),
        context_window_kev=np.float64(res.context_window_kev),
        n_total=np.int64(res.n_total),
        n_context_total=np.int64(res.n_context_total),
        n_target_total=np.int64(res.n_target_total),
        pearson_r=np.float64(res.pearson_r),
        mean_offset=np.float64(res.mean_offset),
    )

    audit = dict(
        paradigm=paradigm,
        T_star=res.T_star,
        n_total=res.n_total,
        n_context_total=res.n_context_total,
        n_target_total=res.n_target_total,
        pearson_r=res.pearson_r,
        mean_offset=res.mean_offset,
        coverage_cnp=res.coverage_cnp,
        coverage_combined=res.coverage_combined,
        sawtooth_metrics=sawtooth_metrics,
        peak_metrics=[
            {
                "peak_name": pm.peak_name,
                "peak_energy_kev": pm.peak_energy_kev,
                "half_window_kev": pm.half_window_kev,
                "n_bins_in_window": pm.n_bins_in_window,
                "chi2_DC": pm.chi2_DC,
                "z_DC": pm.z_DC,
                "p_DC": pm.p_DC,
                "n_valid_DC": pm.n_valid_DC,
                "chi2_DT": pm.chi2_DT,
                "z_DT": pm.z_DT,
                "p_DT": pm.p_DT,
                "n_valid_DT": pm.n_valid_DT,
            }
            for pm in res.peak_metrics
        ],
    )
    audit_path.write_text(json.dumps(audit, indent=2))

    size_kb = npz_path.stat().st_size / 1024
    print(f"  [ok]   {paradigm}: wrote {npz_path.name} ({size_kb:.0f} KB)")
    return npz_path


def main() -> None:
    paradigms = sys.argv[1:] or CANONICAL_PARADIGMS
    print(f"Building notebook cache for {len(paradigms)} paradigm(s)…")
    for p in paradigms:
        build_cache_for(p)


if __name__ == "__main__":
    main()
