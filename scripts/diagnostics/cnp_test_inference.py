"""Test-set inference for binned-CNP cut acceptance.

Implements the D_C / D_T protocol the project switched to so that the
calibration check has no information leakage:

* Sample a configurable fraction of the **test** events (default 100%).
* Split the sample into a Context set ``D_C`` (default 20%) and an
  independent Target set ``D_T`` (default 80%).
* Bin both sets at the trained model's energy grid.
* CNP β(E) is computed by **conditioning on D_C** at each bin: for
  each bin we draw ``n_per_trial`` events with replacement from that
  bin's D_C pool, binarise their scores at the queried T, and run
  ``n_mc`` forward passes with **dropout active** (MC Dropout). β_mean
  and σ_CNP come from the sample over those passes.
* Empirical pass rate uses **only D_T**, with **Wilson** 1σ errorbars
  (handles small-N / k=0 / k=N correctly — unlike the naive
  √(p(1-p)/n) form).
* Coverage is reported two ways:
    - cnp_only : |β_emp − β_pred| < k · σ_CNP
    - combined : |β_emp − β_pred| < k · √(σ_CNP² + σ_emp²)
  with σ_emp = (rate_hi − rate_lo) / 2 (the Wilson half-width).

Bins where the D_C pool is empty are skipped (CNP β set to NaN);
bins where D_T is empty get NaN rate (no empirical point plotted).

The script can be run on the CLI for one config, or imported and
called from a notebook (the notebook exposes TEST_FRACTION /
CONTEXT_FRACTION / N_MC / SEED variables that pass through to
``infer_and_evaluate``).
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_curve

from core import build_cnp
from majorana_acp.cut_acceptance.config import CutAcceptanceConfig, load_config
from majorana_acp.cut_acceptance.pipeline import wilson_interval
from schemas.data_models import InputMode, StandardBatch


PEAK_MARKERS = [
    ("Tl-208 FE", 2614.0),
    ("Tl-208 SE", 2103.0),
    ("Tl-208 DEP", 1592.0),
    ("Bi-214 1620", 1620.0),
]


@dataclass(frozen=True)
class InferenceResult:
    """Per-cell artifacts produced by :func:`infer_and_evaluate`."""

    bin_centers: np.ndarray
    # Empirical (D_T only) — rate is k/N, lo/hi are Wilson 1σ bounds.
    rate: np.ndarray
    rate_lo: np.ndarray
    rate_hi: np.ndarray
    n_target_per_bin: np.ndarray
    # CNP prediction (conditioned on D_C).
    beta: np.ndarray
    beta_std: np.ndarray
    n_context_per_bin: np.ndarray
    # Headline scalars.
    T_star: float
    n_total: int
    n_context_total: int
    n_target_total: int
    pearson_r: float
    mean_offset: float
    coverage_cnp: dict[str, float]
    coverage_combined: dict[str, float]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _bin_edges_from_centers(bin_centers: np.ndarray, bin_width: float) -> np.ndarray:
    """Pipeline writes bin *centers*; reconstruct the edge array."""
    half = 0.5 * bin_width
    return np.concatenate([bin_centers - half, [bin_centers[-1] + half]])


def _load_cnp(cfg: CutAcceptanceConfig, ckpt_path: Path) -> torch.nn.Module:
    cnp = build_cnp(
        cfg.encoder, dim_theta=2, dim_phi=None,
        decoder_hidden_dims=list(cfg.decoder_hidden_dims),
    )
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cnp.load_state_dict(state["model_state"])
    cnp.eval()
    return cnp


def _filter_test_events(
    cfg: CutAcceptanceConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Load test predictions.h5 and apply (class, energy_range) filters.

    Returns (energy, score, label_full, score_full, T_star) — the
    label_full + score_full are kept around for the ROC curve, which
    uses *all* labels (not just the class-filtered ones) to pick the
    Youden-J optimal threshold.
    """
    with h5py.File(cfg.validation_predictions_path, "r") as f:
        e_full = f["energy"][:].astype(np.float64)
        s_full = f["score"][:].astype(np.float64)
        l_full = f["label"][:].astype(np.int64)
    fpr, tpr, thr = roc_curve(l_full, s_full)
    T_star = float(thr[int(np.argmax(tpr - fpr))])

    if cfg.target_class == "all":
        cls_mask = np.ones_like(l_full, dtype=bool)
    else:
        cls_mask = l_full == int(cfg.target_class)
    e_lo, e_hi = cfg.energy_range
    keep = cls_mask & (e_full >= e_lo) & (e_full <= e_hi)
    return e_full[keep], s_full[keep], l_full, s_full, T_star


def split_test_data(
    energy: np.ndarray, score: np.ndarray,
    *, test_fraction: float, context_fraction: float, seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(e_C, s_C, e_T, s_T)`` for the configured split.

    ``test_fraction`` ∈ (0, 1] picks a subset of the test events
    (seeded); ``context_fraction`` ∈ (0, 1) splits that subset into
    Context (D_C) and Target (D_T) halves.
    """
    if not 0.0 < test_fraction <= 1.0:
        raise ValueError(f"test_fraction must be in (0, 1], got {test_fraction}")
    if not 0.0 < context_fraction < 1.0:
        raise ValueError(f"context_fraction must be in (0, 1), got {context_fraction}")
    rng = np.random.default_rng(int(seed))
    n = energy.size
    perm = rng.permutation(n)
    n_use = int(round(test_fraction * n))
    perm_use = perm[:n_use]
    n_ctx = int(round(context_fraction * n_use))
    ctx_idx = perm_use[:n_ctx]
    tgt_idx = perm_use[n_ctx:]
    return energy[ctx_idx], score[ctx_idx], energy[tgt_idx], score[tgt_idx]


def _bin_events(
    energy: np.ndarray, edges: np.ndarray
) -> list[np.ndarray]:
    """Return a per-bin list of *indices into ``energy``*."""
    bin_idx = np.clip(np.digitize(energy, edges) - 1, 0, edges.size - 2)
    return [np.flatnonzero(bin_idx == b) for b in range(edges.size - 1)]


def _empirical_with_wilson(
    energies: np.ndarray, scores: np.ndarray, T: float, edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-bin (rate, rate_lo, rate_hi, counts) on the supplied event set."""
    total, _ = np.histogram(energies, bins=edges)
    pcnt, _ = np.histogram(energies[scores >= T], bins=edges)
    rate = np.divide(pcnt, total, out=np.full(total.shape, np.nan), where=total > 0)
    lo, hi = wilson_interval(pcnt, total, z=1.0)
    return rate, lo, hi, total.astype(np.int64)


def _cnp_infer_per_bin(
    cnp: torch.nn.Module,
    cfg: CutAcceptanceConfig,
    bin_centers: np.ndarray,
    bin_context_scores: list[np.ndarray],
    T: float,
    *,
    n_mc: int,
    seed: int,
    max_context: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-bin (β_mean, β_std) using MC Dropout with **variable-N** context.

    For each bin we condition the CNP on the actual events in
    ``D_C[bin]`` — **without replacement** — capped at
    ``max_context`` (defaults to ``cfg.n_per_trial``). Bins with an
    empty D_C pool are NaN. This is the key robustness change: the
    CNP receives a context whose size honestly reflects how much data
    that bin has. Combined with low ``cnp.n_context_min`` during
    training, σ_CNP should grow when the bin is data-poor.

    Skipping a bin means: empty D_C → no possible prediction.
    """
    if max_context is None:
        max_context = cfg.n_per_trial
    rng = np.random.default_rng(int(seed))
    n_e = bin_centers.size
    beta_mean = np.full(n_e, np.nan, dtype=np.float64)
    beta_std = np.full(n_e, np.nan, dtype=np.float64)
    e_lo, e_hi = cfg.energy_range
    t_lo, t_hi = cfg.threshold_range
    t_norm = (T - t_lo) / (t_hi - t_lo)

    cnp.train()  # enable dropout
    try:
        with torch.no_grad():
            for i, e_center in enumerate(bin_centers):
                ctx_pool = bin_context_scores[i]
                if ctx_pool.size == 0:
                    continue
                # Effective context size honors the bin's actual D_C
                # population (no padding-by-replacement). If the bin
                # has fewer events than max_context, the CNP sees a
                # smaller trial and *should* return a wider σ.
                n_ctx = int(min(ctx_pool.size, max_context))
                e_norm = (e_center - e_lo) / (e_hi - e_lo)
                theta = np.array([[e_norm, t_norm]], dtype=np.float64)
                samples = np.empty(n_mc, dtype=np.float64)
                for m in range(n_mc):
                    if n_ctx >= ctx_pool.size:
                        # Use every available D_C event in the bin
                        # (still permute so order-sensitive parts of
                        # the encoder see slight variation).
                        order = rng.permutation(ctx_pool.size)
                        picks = order[:n_ctx]
                    else:
                        picks = rng.choice(ctx_pool.size, size=n_ctx, replace=False)
                    ctx_labels = (ctx_pool[picks] >= T).astype(np.int8)[None, :]
                    ctx_batch = StandardBatch(
                        mode=InputMode.DESIGN_ONLY, theta=theta, phi=None, labels=ctx_labels
                    )
                    tgt_batch = StandardBatch(
                        mode=InputMode.DESIGN_ONLY, theta=theta, phi=None,
                        labels=ctx_labels.copy(),
                    )
                    beta = cnp.predict_beta(ctx_batch, tgt_batch).cpu().numpy()
                    samples[m] = float(beta.mean())
                beta_mean[i] = float(samples.mean())
                beta_std[i] = float(samples.std(ddof=1))
    finally:
        cnp.eval()
    return beta_mean, beta_std


def _coverage_two_ways(
    rate: np.ndarray, rate_lo: np.ndarray, rate_hi: np.ndarray,
    beta: np.ndarray, beta_std: np.ndarray,
) -> tuple[dict[str, float], dict[str, float], float, float]:
    """Return (cnp_only, combined, mean_offset, pearson_r) at 1σ/2σ/3σ."""
    valid = ~np.isnan(rate) & ~np.isnan(beta)
    cov_cnp: dict[str, float] = {f"{k}sigma": float("nan") for k in (1, 2, 3)}
    cov_comb: dict[str, float] = {f"{k}sigma": float("nan") for k in (1, 2, 3)}
    mean_off = float("nan")
    pearson_r = float("nan")
    if valid.any():
        mean_off = float(np.mean(rate[valid] - beta[valid]))
        if valid.sum() > 1:
            pearson_r = float(np.corrcoef(rate[valid], beta[valid])[0, 1])
        sigma_emp = 0.5 * (rate_hi[valid] - rate_lo[valid])
        sigma_cnp = np.maximum(beta_std[valid], 1e-9)
        sigma_tot = np.sqrt(sigma_cnp**2 + sigma_emp**2)
        resid = np.abs(rate[valid] - beta[valid])
        z_cnp = resid / sigma_cnp
        z_comb = resid / np.maximum(sigma_tot, 1e-9)
        cov_cnp = {f"{k}sigma": float(np.mean(z_cnp <= k)) for k in (1, 2, 3)}
        cov_comb = {f"{k}sigma": float(np.mean(z_comb <= k)) for k in (1, 2, 3)}
    return cov_cnp, cov_comb, mean_off, pearson_r


# --------------------------------------------------------------------------- #
# Top-level inference call
# --------------------------------------------------------------------------- #


def infer_and_evaluate(
    cfg: CutAcceptanceConfig,
    *,
    test_fraction: float = 1.0,
    context_fraction: float = 0.20,
    n_mc: int = 50,
    seed: int = 0,
) -> InferenceResult:
    """Load the trained CNP and run the D_C / D_T protocol end-to-end."""
    # 1. Test events filtered by (class, energy range), and Youden-J T*.
    energy_all, score_all, _l_full, _s_full, T_star = _filter_test_events(cfg)

    # 2. Random D_C / D_T split (after taking an overall test_fraction).
    e_C, s_C, e_T, s_T = split_test_data(
        energy_all, score_all,
        test_fraction=test_fraction,
        context_fraction=context_fraction,
        seed=seed,
    )

    # 3. Bin grid from the pipeline's saved training_pool (so the CNP
    #    sees energies it was trained on).
    pool = np.load(Path(cfg.out_dir) / "training_pool.npz")
    bin_centers = pool["bin_centers"]
    edges = _bin_edges_from_centers(bin_centers, cfg.energy_bin_width)

    # 4. Empirical from D_T only.
    rate, rate_lo, rate_hi, counts_T = _empirical_with_wilson(e_T, s_T, T_star, edges)
    # Skip rates for bins whose D_T count is below min_events_per_bin
    # (these are dominated by sample noise — see project rationale).
    sparse_T = counts_T < cfg.min_events_per_bin
    rate[sparse_T] = np.nan
    rate_lo[sparse_T] = np.nan
    rate_hi[sparse_T] = np.nan

    # 5. CNP context pool per bin (D_C events binned at the same grid).
    ctx_bin_indices = _bin_events(e_C, edges)
    bin_context_scores = [s_C[idx] for idx in ctx_bin_indices]
    counts_C = np.array([p.size for p in bin_context_scores], dtype=np.int64)

    # 6. MC-Dropout CNP inference per bin, conditioned on D_C.
    cnp = _load_cnp(cfg, Path(cfg.out_dir) / "cnp.ckpt")
    beta, beta_std = _cnp_infer_per_bin(
        cnp, cfg, bin_centers, bin_context_scores, T_star,
        n_mc=n_mc, seed=seed,
    )

    # 7. Headline metrics (Pearson, offset, two coverage flavors).
    cov_cnp, cov_comb, mean_off, pearson_r = _coverage_two_ways(
        rate, rate_lo, rate_hi, beta, beta_std
    )

    return InferenceResult(
        bin_centers=bin_centers,
        rate=rate, rate_lo=rate_lo, rate_hi=rate_hi,
        n_target_per_bin=counts_T,
        beta=beta, beta_std=beta_std,
        n_context_per_bin=counts_C,
        T_star=T_star,
        n_total=int(energy_all.size),
        n_context_total=int(e_C.size),
        n_target_total=int(e_T.size),
        pearson_r=pearson_r,
        mean_offset=mean_off,
        coverage_cnp=cov_cnp,
        coverage_combined=cov_comb,
    )


# --------------------------------------------------------------------------- #
# Plot + report
# --------------------------------------------------------------------------- #


def plot_inference(
    cfg: CutAcceptanceConfig,
    res: InferenceResult,
    out_path: Path,
    *,
    show_band: bool = True,
) -> None:
    """Save the canonical 1-axis figure for one cell.

    Blue: D_T binned points with Wilson errorbars.
    Red:  CNP β(E) (conditioned on D_C).
    Band: ±1σ combined ( √(σ_CNP² + σ_emp²) ).
    """
    fig, ax = plt.subplots(figsize=(11, 4.8))
    # Asymmetric Wilson errorbars; defensively clamp at 0 in case of FP
    # rounding so matplotlib doesn't reject the input.
    yerr_lo = np.maximum(np.where(np.isnan(res.rate), 0.0, res.rate - res.rate_lo), 0.0)
    yerr_hi = np.maximum(np.where(np.isnan(res.rate), 0.0, res.rate_hi - res.rate), 0.0)
    ax.errorbar(
        res.bin_centers, res.rate, yerr=[yerr_lo, yerr_hi],
        fmt="o", ms=4, capsize=2, color="steelblue",
        label=f"D_T binned (Wilson 1σ)  N_target={res.n_target_total}",
    )
    mu = np.clip(res.beta, 0.0, 1.0)
    if show_band:
        sigma_emp = 0.5 * (res.rate_hi - res.rate_lo)
        sigma_emp = np.where(np.isnan(sigma_emp), 0.0, sigma_emp)
        sigma_cnp = np.where(np.isnan(res.beta_std), 0.0, res.beta_std)
        sigma_tot = np.sqrt(sigma_cnp**2 + sigma_emp**2)
        ax.fill_between(
            res.bin_centers,
            np.clip(mu - sigma_tot, 0.0, 1.0),
            np.clip(mu + sigma_tot, 0.0, 1.0),
            color="firebrick", alpha=0.20,
            label="combined ±1σ  (√(σ_CNP² + σ_emp²))",
        )
    ax.plot(
        res.bin_centers, mu, color="firebrick", lw=1.6,
        label=f"CNP β(E) | D_C   N_context={res.n_context_total}",
    )
    e_lo, e_hi = cfg.energy_range
    for label_, e_pk in PEAK_MARKERS:
        if e_lo <= e_pk <= e_hi:
            ax.axvline(e_pk, color="gray", ls="--", lw=0.6, alpha=0.7)
            ax.text(e_pk, 1.02, label_.split()[-1], fontsize=7, ha="center", color="gray")
    ax.set_xlabel("Energy [keV]")
    ax.set_ylabel(f"acceptance at T* = {res.T_star:.4f}")
    ax.set_xlim(e_lo, e_hi)
    ax.set_ylim(-0.05, 1.10)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(
        f"{cfg.name}   (bin = {cfg.energy_bin_width:.0f} keV, "
        f"target_class={cfg.target_class!r})\n"
        f"r = {res.pearson_r:.3f}    offset = {res.mean_offset:+.3f}    "
        f"cov₁σ  CNP={res.coverage_cnp['1sigma']:.2f}  "
        f"combined={res.coverage_combined['1sigma']:.2f}",
        fontsize=10,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def write_report(
    cfg: CutAcceptanceConfig,
    res: InferenceResult,
    out_dir: Path,
    *,
    test_fraction: float, context_fraction: float, n_mc: int, seed: int,
) -> dict:
    """Persist text + JSON reports + the diagnostic figure."""
    out_dir.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        "=" * 78,
        f"Test-set inference  ·  {cfg.name}",
        "=" * 78,
        f"  config:                       {Path(cfg.out_dir).name}",
        f"  trained CNP:                  {Path(cfg.out_dir) / 'cnp.ckpt'}",
        f"  test predictions:             {cfg.validation_predictions_path}",
        f"  target_class:                 {cfg.target_class!r}",
        f"  energy_bin_width [keV]:       {cfg.energy_bin_width:.1f}",
        f"  test_fraction:                {test_fraction}",
        f"  context_fraction (of subset): {context_fraction}",
        f"  n_mc passes (MC Dropout):     {n_mc}",
        f"  seed:                         {seed}",
        f"  Youden-J best T*:             {res.T_star:.4f}",
        "",
        f"  N_total (after class+E filter): {res.n_total}",
        f"  |D_C| = {res.n_context_total}    |D_T| = {res.n_target_total}",
        f"  mean events / bin   D_C = {float(res.n_context_per_bin.mean()):.2f}    "
        f"D_T = {float(res.n_target_per_bin.mean()):.2f}",
        f"  bins with empty D_C: {int((res.n_context_per_bin == 0).sum())}",
        f"  bins with empty D_T: {int((res.n_target_per_bin == 0).sum())}",
        "",
        f"[1] Energy fidelity (D_T vs CNP β(E))",
        f"    Pearson r        = {res.pearson_r:+.4f}",
        f"    mean offset (D_T − CNP) = {res.mean_offset:+.4f}",
        "",
        "[2] Coverage at T*  (Gaussian targets: 1σ=0.683  2σ=0.954  3σ=0.997)",
        f"    cnp_only :   1σ={res.coverage_cnp['1sigma']:.3f}   "
        f"2σ={res.coverage_cnp['2sigma']:.3f}   3σ={res.coverage_cnp['3sigma']:.3f}",
        f"    combined :   1σ={res.coverage_combined['1sigma']:.3f}   "
        f"2σ={res.coverage_combined['2sigma']:.3f}   3σ={res.coverage_combined['3sigma']:.3f}",
        "",
    ]
    plot_path = out_dir / "test_set_audit.png"
    plot_inference(cfg, res, plot_path)
    lines.append(f"plot: {plot_path}")
    text = "\n".join(lines) + "\n"
    (out_dir / "test_set_audit.txt").write_text(text)

    metrics = dict(
        config=cfg.name,
        target_class=str(cfg.target_class),
        energy_bin_width=cfg.energy_bin_width,
        test_fraction=test_fraction,
        context_fraction=context_fraction,
        n_mc=n_mc,
        seed=seed,
        T_star=res.T_star,
        n_total=res.n_total,
        n_context_total=res.n_context_total,
        n_target_total=res.n_target_total,
        bins_empty_D_C=int((res.n_context_per_bin == 0).sum()),
        bins_empty_D_T=int((res.n_target_per_bin == 0).sum()),
        pearson_r=res.pearson_r,
        mean_offset=res.mean_offset,
        coverage_cnp=res.coverage_cnp,
        coverage_combined=res.coverage_combined,
    )
    (out_dir / "test_set_audit.json").write_text(json.dumps(metrics, indent=2))
    print(text)
    return metrics


def run(
    cfg_path: Path, out_dir: Path,
    *,
    test_fraction: float = 1.0, context_fraction: float = 0.20,
    n_mc: int = 50, seed: int = 0,
) -> dict:
    cfg = load_config(cfg_path)
    res = infer_and_evaluate(
        cfg,
        test_fraction=test_fraction, context_fraction=context_fraction,
        n_mc=n_mc, seed=seed,
    )
    return write_report(
        cfg, res, out_dir,
        test_fraction=test_fraction, context_fraction=context_fraction,
        n_mc=n_mc, seed=seed,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--test-fraction", type=float, default=1.0)
    ap.add_argument("--context-fraction", type=float, default=0.20)
    ap.add_argument("--n-mc", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out_dir = args.out_dir
    if out_dir is None:
        rel = args.config.relative_to(Path("configs/cut_acceptance"))
        out_dir = Path("analysis/cnp_audit") / rel.parent / rel.stem
    run(args.config, out_dir,
        test_fraction=args.test_fraction,
        context_fraction=args.context_fraction,
        n_mc=args.n_mc,
        seed=args.seed)


if __name__ == "__main__":
    main()
