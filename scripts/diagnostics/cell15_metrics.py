"""Print the Cell 15 sweep metrics block for one paradigm.

Usage:
    python -m scripts.diagnostics.cell15_metrics <paradigm-path-relative-to-hybrid_scale>

Pipes straight into the cell15_vN section of
``analysis/cnp_audit/scanning_cell15_hyperpar.md``.

Computes the six lines:
  1. MASD — mean of (1.7–2.0 MeV) and (2.2–2.4 MeV) sawtooth.
  2. overall: pooled-binomial z + cov 1σ/2σ/3σ over [500, 3000] keV.
  3. FE 2614:  pooled-binomial z + cov 1σ/2σ/3σ over ±10 keV window.
  4. SE 2103:  ditto.
  5. DEP 1592: ditto.
  6. Bi 1620:  ditto.

Uses the SAME formula as the notebook's §8.4.6 (`Φ(k − z) − Φ(−k − z)`
with σ_combined = √(σ_CNP² + σ_emp²)). Reads from the full-test
predictions file ``runs/.../eval_full_test/predictions.h5``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_curve

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from majorana_acp.cut_acceptance.config import load_config
from scripts.diagnostics.cnp_test_inference import infer_and_evaluate

FULL_TEST = _REPO_ROOT / "runs/small_data_configs/simple_cnn_small/eval_full_test/predictions.h5"
PEAKS = (("FE 2614", 2614.0), ("SE 2103", 2103.0),
         ("DEP 1592", 1592.0), ("Bi 1620", 1620.0))


def _pooled(e, s, T_star, dense_E, dense_beta, dense_sigma):
    N = e.size
    if N == 0:
        return None
    p_hat = float((s >= T_star).mean())
    sigma_emp = float(np.sqrt(max(p_hat * (1 - p_hat), 1e-12) / N))
    bp = np.interp(e, dense_E, dense_beta)
    sp = np.interp(e, dense_E, dense_sigma)
    beta_avg = float(bp.mean())
    sigma_cnp_avg = float(sp.mean())
    sigma_comb = float(np.sqrt(sigma_cnp_avg**2 + sigma_emp**2))
    z = (p_hat - beta_avg) / max(sigma_comb, 1e-12)
    cov = lambda k: float(norm.cdf(k - z) - norm.cdf(-k - z))
    return N, z, cov(1), cov(2), cov(3)


def metrics(paradigm: str) -> str:
    """Return the markdown table block for one paradigm."""
    os.chdir(_REPO_ROOT)
    cfg_path = (
        _REPO_ROOT
        / "configs/cut_acceptance/simple_cnn_small"
        / paradigm
        / "bin10/inclusive.yaml"
    )
    cfg = load_config(cfg_path)
    cfg = cfg.model_copy(update={"validation_predictions_path": FULL_TEST})

    # MC-Dropout inference for σ_CNP at each query.
    res = infer_and_evaluate(
        cfg, test_fraction=1.0, context_fraction=2000 / 141474,
        n_mc=50, seed=0,
    )

    # Pool events.
    with h5py.File(FULL_TEST, "r") as f:
        e_full = f["energy"][:].astype(np.float64)
        s_full = f["score"][:].astype(np.float64)
        l_full = f["label"][:].astype(np.int64)
    fpr, tpr, thr = roc_curve(l_full, s_full)
    T_star = float(thr[int(np.argmax(tpr - fpr))])
    keep = (e_full >= 500) & (e_full <= 3000)
    e_all, s_all = e_full[keep], s_full[keep]

    # MASD from test_set_audit.json (must exist; run inference first).
    audit = (
        _REPO_ROOT / "analysis/cnp_audit/simple_cnn_small"
        / paradigm / "bin10/inclusive/test_set_audit.json"
    )
    if audit.exists():
        d = json.loads(audit.read_text())
        masd_lo = d["sawtooth_metrics"]["region_1700_2000"]["masd"]
        masd_hi = d["sawtooth_metrics"]["region_2200_2400"]["masd"]
        masd = 0.5 * (masd_lo + masd_hi)
    else:
        masd_lo = masd_hi = masd = float("nan")

    overall = _pooled(e_all, s_all, T_star,
                      res.dense_energies, res.dense_beta, res.dense_beta_std)
    peak_rows = []
    for label, e_pk in PEAKS:
        m = np.abs(e_all - e_pk) <= 10.0
        pk = _pooled(e_all[m], s_all[m], T_star,
                     res.dense_energies, res.dense_beta, res.dense_beta_std)
        peak_rows.append((label, pk))

    # Compose the markdown block.
    lines = []
    lines.append(f"| metric | value |")
    lines.append(f"|---|---|")
    lines.append(
        f"| `MASD` | **{masd:.4f}** "
        f"(1.7-2.0: {masd_lo:.4f} / 2.2-2.4: {masd_hi:.4f}) |"
    )
    if overall is None:
        lines.append("| `overall` | (no events) |")
    else:
        _, z, c1, c2, c3 = overall
        lines.append(
            f"| `overall`   | z = {z:+.3f}  ·  cov = {c1:.2f}/{c2:.2f}/{c3:.2f} |"
        )
    for label, pk in peak_rows:
        if pk is None:
            lines.append(f"| `{label}` | (no events in window) |")
        else:
            _, z, c1, c2, c3 = pk
            lines.append(
                f"| `{label}`  | z = {z:+.3f}  ·  cov = {c1:.2f}/{c2:.2f}/{c3:.2f} |"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "paradigm",
        help="Paradigm path under configs/.../simple_cnn_small/, e.g. "
             "'hybrid_scale/flat_stratified_..._hardfilter_xfeed_sl1_sg50_pedetach'",
    )
    args = p.parse_args(argv)
    print(metrics(args.paradigm))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
