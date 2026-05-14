"""CNP-only health audit for one binned cut_acceptance run.

Reads the artifacts the binned-CNP pipeline writes
(``validation_binned.npz``, ``cnp_predictions.npz``, ``cnp.ckpt``,
``run_summary.json``) and produces:

  - **PNG**: empirical A(E, T*) (binned + binomial errors) overlaid
    with CNP β(E, T*).  Vertical lines mark the known Tl-208 peaks /
    DEP region — purely visual annotation, no physics filter is
    applied.
  - **TXT** + **JSON**: numeric report covering
      * Range Recovery Ratio per energy slab
      * Pearson r between β_CNP(E, T*) and binned A(E, T*)
      * Local Variance Ratio (binned vs predicted spectral wiggles)
      * Endpoint bias |β − A_emp| at T ∈ {0.05, 0.95}
      * Peak sensitivity at known background peaks

The report uses the saved prediction grids — no need to re-run the
CNP or refit anything.

Usage::

    python -m scripts.diagnostics.cnp_health_report \\
        configs/cut_acceptance/simple_cnn_small/bin10/inclusive.yaml \\
        [--out-dir analysis/cnp_audit/simple_cnn_small/bin10/inclusive]

The PNG / TXT / JSON go under ``<out-dir>/`` so they can be re-opened
later without re-running the diagnostic.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import torch

from core import build_cnp
from majorana_acp.cut_acceptance.binned_sampler import BinnedSampler, load_events
from majorana_acp.cut_acceptance.config import load_config
from schemas.data_models import InputMode, StandardBatch

# Peak windows used purely as visual annotation in the plot. Numbers from the
# Th-228 calibration line catalog (Bi-214 single-escape, Tl-208 DEP, etc.).
PEAK_ANNOTATIONS = [
    ("Tl-208 FE", 2614.0),
    ("Tl-208 SE", 2103.0),
    ("Tl-208 DEP", 1592.0),
    ("Bi-214 1620", 1620.0),
]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def load_cnp(cfg, ckpt_path: Path):
    """Rebuild the CNP with cfg.decoder_hidden_dims and load state_dict.

    The upstream ``load_checkpoint`` rebuilds with the default decoder
    dims, which mismatches our deeper override — so go manual.
    """
    cnp = build_cnp(
        cfg.encoder, dim_theta=2, dim_phi=None,
        decoder_hidden_dims=list(cfg.decoder_hidden_dims),
    )
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cnp.load_state_dict(state["model_state"])
    cnp.eval()
    return cnp


def cnp_beta_at(cnp, sampler: BinnedSampler, cfg, e_center: float, T: float,
                n_ctx: int = 32, seed: int = 0) -> float:
    """β(E_center, T) using a context drawn from the same bin pool."""
    # Find the bin in the sampler that contains e_center.
    centers = sampler.bin_centers
    i = int(np.argmin(np.abs(centers - e_center)))
    ev = sampler._index.bin_events[i]
    if ev.size == 0:
        return float("nan")
    rng = np.random.default_rng(int(seed))
    ctx_scores = sampler._index.score[ev]
    picks = rng.integers(0, ctx_scores.size, size=n_ctx)
    ctx_labels = (ctx_scores[picks] >= T).astype(np.int8)[None, :]
    e_lo, e_hi = cfg.energy_range
    t_lo, t_hi = cfg.threshold_range
    theta = np.array(
        [[(centers[i] - e_lo) / (e_hi - e_lo), (T - t_lo) / (t_hi - t_lo)]],
        dtype=np.float64,
    )
    ctx = StandardBatch(mode=InputMode.DESIGN_ONLY, theta=theta, phi=None, labels=ctx_labels)
    tgt = StandardBatch(mode=InputMode.DESIGN_ONLY, theta=theta, phi=None, labels=ctx_labels.copy())
    with torch.no_grad():
        return float(cnp.predict_beta(ctx, tgt).cpu().numpy().mean())


def empirical_slab_A(energies: np.ndarray, scores: np.ndarray, T: float,
                     e_lo: float, e_hi: float) -> tuple[float, int]:
    """Empirical A(T) for events with E ∈ [e_lo, e_hi]."""
    m = (energies >= e_lo) & (energies <= e_hi)
    if not m.any():
        return float("nan"), 0
    return float((scores[m] >= T).mean()), int(m.sum())


# --------------------------------------------------------------------------- #
# Main report
# --------------------------------------------------------------------------- #


def run(cfg_path: Path, out_dir: Path) -> dict:
    cfg = load_config(cfg_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    art = Path(cfg.out_dir)
    cnp = load_cnp(cfg, art / "cnp.ckpt")

    # Re-build the training sampler so we can draw real-bin contexts for
    # the per-slab β probes.  We only re-load the training pool here —
    # the CNP itself is the trained one.
    train_e, train_s = load_events(
        cfg.train_predictions_path,
        target_class=cfg.target_class,
        energy_range=cfg.energy_range,
    )
    sampler = BinnedSampler(
        train_e, train_s,
        energy_range=cfg.energy_range,
        energy_bin_width=cfg.energy_bin_width,
        threshold_range=cfg.threshold_range,
        n_per_trial=cfg.n_per_trial,
        min_events_per_bin=cfg.min_events_per_bin,
    )

    # Load validation slabs from the *test* split (filtered the same way
    # the pipeline filtered them).
    with h5py.File(cfg.validation_predictions_path, "r") as f:
        v_e_full = f["energy"][:].astype(np.float64)
        v_s_full = f["score"][:].astype(np.float64)
        v_l_full = f["label"][:].astype(np.int64)
    if cfg.target_class == "all":
        m = np.ones_like(v_l_full, dtype=bool)
    else:
        m = v_l_full == int(cfg.target_class)
    e_lo, e_hi = cfg.energy_range
    m &= (v_e_full >= e_lo) & (v_e_full <= e_hi)
    v_e = v_e_full[m]
    v_s = v_s_full[m]

    # Load the pipeline's saved arrays.
    val_arr = np.load(art / "validation_binned.npz")
    bin_centers = val_arr["bin_centers"]
    A_emp = val_arr["rate"]
    A_err = val_arr["rate_err"]
    T_star = float(val_arr["T_star"])

    preds = np.load(art / "cnp_predictions.npz")
    beta_E = preds["beta_at_T_star"]
    beta_std_E = preds["beta_std_at_T_star"] if "beta_std_at_T_star" in preds.files else None

    # ----- metrics ----- #
    valid = ~np.isnan(A_emp) & ~np.isnan(beta_E)
    pearson_r = (
        float(np.corrcoef(beta_E[valid], A_emp[valid])[0, 1])
        if valid.sum() > 1
        else float("nan")
    )
    var_pred = float(np.var(beta_E[valid])) if valid.any() else float("nan")
    var_emp = float(np.var(A_emp[valid])) if valid.any() else float("nan")
    lvr = var_pred / var_emp if var_emp > 0 else float("nan")
    mean_off = float(np.mean(A_emp[valid] - beta_E[valid])) if valid.any() else float("nan")

    # Coverage of empirical inside CNP's ±kσ band — two flavors:
    #   * cnp_only:  |β_emp − β_pred| < k · σ_CNP   (model's stated band)
    #   * combined:  |β_emp − β_pred| < k · √(σ_CNP² + σ_emp²)
    # Combined accounts for the binomial noise on the *empirical* side
    # too — it answers "do the two bands overlap?" rather than "does the
    # empirical fall inside the model's confidence interval?". CNP-only
    # is the stricter of the two.
    cov_cnp = {"1sigma": float("nan"), "2sigma": float("nan"), "3sigma": float("nan")}
    cov_comb = {"1sigma": float("nan"), "2sigma": float("nan"), "3sigma": float("nan")}
    if beta_std_E is not None and valid.any():
        resid = np.abs(A_emp[valid] - beta_E[valid])
        sigma_cnp = np.maximum(beta_std_E[valid], 1e-9)
        sigma_emp = np.maximum(A_err[valid], 1e-9)
        z_cnp = resid / sigma_cnp
        z_comb = resid / np.sqrt(sigma_cnp**2 + sigma_emp**2)
        cov_cnp = {f"{k}sigma": float(np.mean(z_cnp <= k)) for k in (1, 2, 3)}
        cov_comb = {f"{k}sigma": float(np.mean(z_comb <= k)) for k in (1, 2, 3)}
    coverage = {"cnp_only": cov_cnp, "combined": cov_comb}

    # Range Recovery Ratio per slab.  Pick slabs that exist in the cfg's
    # energy range; ±50 keV for continuum, ±25 keV for peak windows.
    slabs = [
        ("continuum 800±100",  (700, 900)),
        ("continuum 1500±100", (1400, 1600)),
        ("DEP-ish 1592±25",    (1567, 1617)),
        ("Tl-208 SE 2103±25",  (2078, 2128)),
        ("Tl-208 FE 2614±25",  (2589, 2639)),
    ]
    rrr_rows: list[dict] = []
    for name, (lo, hi) in slabs:
        if hi < e_lo or lo > e_hi:
            continue
        A05, n = empirical_slab_A(v_e, v_s, 0.05, lo, hi)
        A95, _ = empirical_slab_A(v_e, v_s, 0.95, lo, hi)
        if n < cfg.min_events_per_bin or np.isnan(A05) or np.isnan(A95):
            continue
        e_c = 0.5 * (lo + hi)
        b05 = cnp_beta_at(cnp, sampler, cfg, e_c, 0.05)
        b95 = cnp_beta_at(cnp, sampler, cfg, e_c, 0.95)
        emp_rng = A05 - A95
        cnp_rng = b05 - b95
        rrr = cnp_rng / emp_rng if emp_rng != 0 else float("nan")
        rrr_rows.append(dict(
            slab=name, A05=A05, A95=A95, emp_rng=emp_rng,
            b05=b05, b95=b95, cnp_rng=cnp_rng, rrr=rrr, n=n,
        ))

    # Endpoint bias against a continuum reference (1500 keV ± 100).
    ref_lo, ref_hi = 1400.0, 1600.0
    ref_A05, _ = empirical_slab_A(v_e, v_s, 0.05, ref_lo, ref_hi)
    ref_A95, _ = empirical_slab_A(v_e, v_s, 0.95, ref_lo, ref_hi)
    ref_b05 = cnp_beta_at(cnp, sampler, cfg, 1500.0, 0.05)
    ref_b95 = cnp_beta_at(cnp, sampler, cfg, 1500.0, 0.95)
    epb_low = abs(ref_b05 - ref_A05) if not np.isnan(ref_A05) else float("nan")
    epb_high = abs(ref_b95 - ref_A95) if not np.isnan(ref_A95) else float("nan")

    # Peak sensitivity.
    peak_drops = []
    for label_, peak_E in [("Tl-208 SE 2103", 2103.0), ("Tl-208 FE 2614", 2614.0)]:
        if not (e_lo <= peak_E <= e_hi):
            continue
        b_peak = cnp_beta_at(cnp, sampler, cfg, peak_E, T_star)
        a_peak, n_peak = empirical_slab_A(v_e, v_s, T_star, peak_E - 25, peak_E + 25)
        peak_drops.append(dict(label=label_, energy=peak_E,
                               b_peak=b_peak, a_peak=a_peak, n=n_peak))

    # ----- plot ----- #
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.errorbar(
        bin_centers, A_emp, yerr=A_err,
        fmt="o", ms=4, capsize=2, color="steelblue",
        label=f"empirical A(E)  (binomial err, N_total={int(m.sum())})",
    )
    if beta_std_E is not None:
        mu_clip = np.clip(beta_E, 0.0, 1.0)
        ax.fill_between(
            bin_centers,
            np.clip(mu_clip - beta_std_E, 0.0, 1.0),
            np.clip(mu_clip + beta_std_E, 0.0, 1.0),
            color="firebrick", alpha=0.20, label="CNP ±1σ",
        )
    ax.plot(bin_centers, beta_E, color="firebrick", lw=1.7, label="CNP β(E)")
    for label_, e_pk in PEAK_ANNOTATIONS:
        if e_lo <= e_pk <= e_hi:
            ax.axvline(e_pk, color="gray", ls="--", lw=0.6, alpha=0.7)
            ax.text(e_pk, 1.02, label_.split()[-1], fontsize=8, ha="center", color="gray")
    ax.set_xlabel("Energy [keV]")
    ax.set_ylabel(f"acceptance at T* = {T_star:.4f}")
    ax.set_xlim(e_lo, e_hi)
    ax.set_ylim(-0.05, 1.10)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(
        f"{cfg.name}   (bin = {cfg.energy_bin_width:.0f} keV, "
        f"target_class={cfg.target_class!r})\n"
        f"Pearson r = {pearson_r:.3f}    LVR = {lvr:.2f}    "
        f"offset = {mean_off:+.3f}    "
        f"cov₁σ CNP={cov_cnp['1sigma']:.2f}  comb={cov_comb['1sigma']:.2f}",
        fontsize=10,
    )
    fig.tight_layout()
    plot_path = out_dir / "cnp_only_audit.png"
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)

    # ----- text + json ----- #
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append(f"CNP-only health audit  ·  {cfg.name}")
    lines.append("=" * 78)
    lines.append(f"  config:                     {cfg_path}")
    lines.append(f"  train predictions:          {cfg.train_predictions_path}")
    lines.append(f"  validation predictions:     {cfg.validation_predictions_path}")
    lines.append(f"  artifacts:                  {art}")
    lines.append(f"  target_class:               {cfg.target_class!r}")
    lines.append(f"  energy_bin_width [keV]:     {cfg.energy_bin_width:.1f}")
    lines.append(f"  bins used:                  {sampler.n_bins}")
    lines.append(f"  n_events (val, filtered):   {int(m.sum())}")
    lines.append(f"  Youden-J best T*:           {T_star:.4f}")
    lines.append("")
    lines.append("[1] Range Recovery Ratio per energy slab")
    lines.append(
        f"    {'slab':25s} {'N':>5s} {'A(0.05)':>9s} {'A(0.95)':>9s} {'emp_rng':>9s} "
        f"{'β(0.05)':>9s} {'β(0.95)':>9s} {'CNP_rng':>9s} {'RRR':>7s}"
    )
    for r in rrr_rows:
        lines.append(
            f"    {r['slab']:25s} {r['n']:>5d} "
            f"{r['A05']:.3f}     {r['A95']:.3f}     {r['emp_rng']:+.3f}     "
            f"{r['b05']:.3f}     {r['b95']:.3f}     {r['cnp_rng']:+.3f}     {r['rrr']:+.3f}"
        )
    lines.append("")
    lines.append(f"[2] Energy fidelity (CNP vs binned at T*)")
    lines.append(f"    Pearson r        = {pearson_r:+.4f}")
    lines.append(f"    mean offset      = {mean_off:+.4f}  (binned − CNP)")
    lines.append("")
    lines.append(f"[3] Local Variance Ratio")
    lines.append(f"    Var(β_CNP)       = {var_pred:.5f}")
    lines.append(f"    Var(A_emp)       = {var_emp:.5f}")
    lines.append(f"    LVR              = {lvr:.3f}    (1.0 = matches; <1 = oversmoothed)")
    lines.append("")
    lines.append(f"[4] Endpoint bias (1500 ±100 keV continuum reference)")
    lines.append(f"    |β(0.05) − A(0.05)| = {epb_low:.4f}")
    lines.append(f"    |β(0.95) − A(0.95)| = {epb_high:.4f}")
    lines.append("")
    lines.append(
        "[5] Coverage at T*  (Gaussian targets: 1σ=0.683  2σ=0.954  3σ=0.997)"
    )
    lines.append(
        f"    cnp_only :   1σ={cov_cnp['1sigma']:.3f}   2σ={cov_cnp['2sigma']:.3f}   "
        f"3σ={cov_cnp['3sigma']:.3f}      "
        "(|β_emp − β_pred| < k · σ_CNP)"
    )
    lines.append(
        f"    combined :   1σ={cov_comb['1sigma']:.3f}   2σ={cov_comb['2sigma']:.3f}   "
        f"3σ={cov_comb['3sigma']:.3f}      "
        "(|β_emp − β_pred| < k · √(σ_CNP² + σ_emp²))"
    )
    lines.append("")
    lines.append(f"[6] Peak sensitivity (at T* = {T_star:.4f})")
    for d in peak_drops:
        lines.append(
            f"    {d['label']:18s} (E={d['energy']:.0f} keV, N={d['n']})  "
            f"β_CNP = {d['b_peak']:.3f}    A_emp(±25 keV) = {d['a_peak']:.3f}"
        )
    lines.append("")
    lines.append(f"plot: {plot_path}")
    text = "\n".join(lines)
    (out_dir / "cnp_only_audit.txt").write_text(text + "\n")

    metrics = dict(
        config=str(cfg_path),
        name=cfg.name,
        target_class=str(cfg.target_class),
        energy_bin_width=cfg.energy_bin_width,
        T_star=T_star,
        n_validation_events=int(m.sum()),
        n_bins_used=int(sampler.n_bins),
        pearson_r=pearson_r,
        mean_offset=mean_off,
        var_beta=var_pred,
        var_emp=var_emp,
        lvr=lvr,
        epb_low=epb_low,
        epb_high=epb_high,
        coverage=coverage,
        rrr_per_slab=rrr_rows,
        peak_drops=peak_drops,
    )
    (out_dir / "cnp_only_audit.json").write_text(json.dumps(metrics, indent=2))

    print(text)
    print(f"\nSaved:\n  {plot_path}\n  {out_dir / 'cnp_only_audit.txt'}"
          f"\n  {out_dir / 'cnp_only_audit.json'}")
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Defaults to analysis/cnp_audit/<config-parent>/<config-stem>/.")
    args = ap.parse_args()
    out_dir = args.out_dir
    if out_dir is None:
        # e.g. configs/cut_acceptance/simple_cnn_small/bin10/inclusive.yaml
        #      → analysis/cnp_audit/simple_cnn_small/bin10/inclusive/
        rel = args.config.relative_to("configs/cut_acceptance") if args.config.is_absolute() is False \
            else args.config
        out_dir = Path("analysis/cnp_audit") / args.config.parent.relative_to(
            "configs/cut_acceptance"
        ) / args.config.stem
    run(args.config, out_dir)


if __name__ == "__main__":
    main()
