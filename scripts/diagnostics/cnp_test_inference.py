"""Test-set inference for the True-CNP cut-acceptance pipeline.

The trained CNP is a 1D-regression CNP in ``InputMode.EVENT_ONLY``:
each context event carries its own ``phi_i = (E_i_norm, T_norm)``. As
a true stochastic process, β(E) is a continuous function of energy —
we evaluate it on a dense grid by aggregating **all** D_C events
(capped at :data:`MAX_CONTEXT_PER_PASS`) into a single global
representation ``r``, then decoding at every query point in one
forward call.

Pipeline
--------
* Sample a configurable fraction of the **test** events (default 100%).
* Split into a Context set ``D_C`` (default 20%) and a disjoint Target
  set ``D_T`` (default 80%).
* Empirical pass rate (blue) comes from **D_T only**, binned at the
  pipeline's saved bin grid, with Wilson 1σ errorbars.
* CNP β(E) (red) comes from MC Dropout: ``n_mc`` forward passes, each
  with a fresh D_C subsample of up to :data:`MAX_CONTEXT_PER_PASS`
  events. Each pass produces one global ``r`` aggregated across the
  entire spectrum, then decodes at the bin centers AND a dense grid in
  one shot.
* Coverage is reported at the **bin-center** queries (matching D_T
  binning):
    - cnp_only : |β_emp − β_pred| < k · σ_CNP
    - combined : |β_emp − β_pred| < k · √(σ_CNP² + σ_emp²)
  with σ_emp = (rate_hi − rate_lo) / 2 (the Wilson half-width).
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import h5py

# NB: we do *not* force the matplotlib backend at module-load time —
# that would override the inline backend when this module is imported
# from a Jupyter notebook (the bug that hid all §8.4 plots). The CLI
# entry point flips to Agg explicitly before saving any figure.
import matplotlib.pyplot as plt
import numpy as np
import torch
from core import build_cnp
from schemas.data_models import InputMode, StandardBatch
from sklearn.metrics import roc_curve

from majorana_acp.cut_acceptance.config import CutAcceptanceConfig, load_config
from majorana_acp.cut_acceptance.pipeline import wilson_interval

PEAK_MARKERS = [
    ("Tl-208 FE", 2614.0),
    ("Tl-208 SE", 2103.0),
    ("Tl-208 DEP", 1592.0),
    ("Bi-214 1620", 1620.0),
]

# Defaults exposed through the CLI / notebook for tuning.
DEFAULT_N_DENSE = 800
DEFAULT_MAX_CONTEXT_PER_PASS = 256


@dataclass(frozen=True)
class InferenceResult:
    """Per-cell artifacts produced by :func:`infer_and_evaluate`.

    Two parallel β estimates:

    * **Bin-center** (``bin_centers``, ``beta``, ``beta_std``) — one
      query per pipeline-saved bin. Used for coverage metrics vs D_T.
    * **Dense grid** (``dense_energies``, ``dense_beta``,
      ``dense_beta_std``) — uniform sampling for plotting a smooth red
      curve.

    Both come out of the same MC-Dropout passes — a single shared
    target_batch carries both sets of query coordinates, so they are
    consistent with each other up to floating-point.
    """

    bin_centers: np.ndarray
    # Empirical (D_T only) — rate is k/N, lo/hi are Wilson 1σ bounds.
    rate: np.ndarray
    rate_lo: np.ndarray
    rate_hi: np.ndarray
    n_target_per_bin: np.ndarray
    # CNP prediction at bin centers (used for coverage vs D_T).
    beta: np.ndarray
    beta_std: np.ndarray
    # CNP prediction on a dense grid (smooth red curve).
    dense_energies: np.ndarray
    dense_beta: np.ndarray
    dense_beta_std: np.ndarray
    # Knobs.
    n_context_per_pass: int   # actual cap used (min(MAX_CONTEXT_PER_PASS, |D_C|)).
    context_window_kev: float  # always np.inf for True CNP — kept for notebook compat.
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
    """Reconstruct the EVENT_ONLY CNP architecture and load weights."""
    cnp = build_cnp(
        cfg.encoder, dim_theta=None, dim_phi=2,
        decoder_hidden_dims=list(cfg.decoder_hidden_dims),
    )
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cnp.load_state_dict(state["model_state"])
    cnp.eval()
    return cnp


def _filter_test_events(
    cfg: CutAcceptanceConfig,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Load test predictions.h5; apply (class, energy_range) filters; pick T*.

    T* comes from the Youden-J point of the ROC over **all** labels
    (not just the class-filtered ones).
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
    return e_full[keep], s_full[keep], T_star


def split_test_data(
    energy: np.ndarray, score: np.ndarray,
    *, test_fraction: float, context_fraction: float, seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(e_C, s_C, e_T, s_T)`` for the configured split."""
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


def _empirical_with_wilson(
    energies: np.ndarray, scores: np.ndarray, T: float, edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-bin (rate, rate_lo, rate_hi, counts) on the supplied event set."""
    total, _ = np.histogram(energies, bins=edges)
    pcnt, _ = np.histogram(energies[scores >= T], bins=edges)
    rate = np.divide(pcnt, total, out=np.full(total.shape, np.nan), where=total > 0)
    lo, hi = wilson_interval(pcnt, total, z=1.0)
    return rate, lo, hi, total.astype(np.int64)


# --------------------------------------------------------------------------- #
# Global-aggregation MC-Dropout inference
# --------------------------------------------------------------------------- #


def _cnp_infer_global(
    cnp: torch.nn.Module,
    cfg: CutAcceptanceConfig,
    query_energies: np.ndarray,
    ctx_energies: np.ndarray,
    ctx_scores: np.ndarray,
    T: float,
    *,
    n_mc: int,
    seed: int,
    max_context_per_pass: int = DEFAULT_MAX_CONTEXT_PER_PASS,
) -> tuple[np.ndarray, np.ndarray, int]:
    """β(E) at arbitrary query energies via global-aggregation MC Dropout.

    Each MC pass:
      1. Sample (without replacement) up to ``max_context_per_pass``
         events from D_C → that pass's context.
      2. Run **one** CNP forward: aggregate across all context events
         → single global ``r``; decode at every query energy in one
         shot.

    The CNP is a 1D-regression CNP in EVENT_ONLY mode, so each context
    event contributes its own ``(E_i_norm, T_norm)`` and each target
    query is just another phi row.

    Returns ``(beta_mean, beta_std, n_context_per_pass_used)`` where
    the third element is ``min(|D_C|, max_context_per_pass)``.
    """
    rng = np.random.default_rng(int(seed))
    n_q = query_energies.size
    if ctx_energies.size == 0:
        return (
            np.full(n_q, np.nan, dtype=np.float64),
            np.full(n_q, np.nan, dtype=np.float64),
            0,
        )

    e_lo, e_hi = cfg.energy_range
    t_lo, t_hi = cfg.threshold_range
    t_norm = float((T - t_lo) / (t_hi - t_lo))
    q_norm = (query_energies - e_lo) / (e_hi - e_lo)
    # Target batch is identical for every MC pass: one trial, N_q
    # queries, each at (E*_i_norm, T_norm). Labels are unused at
    # inference but required by the StandardBatch validator.
    tgt_phi = np.stack([q_norm, np.full_like(q_norm, t_norm)], axis=-1)[None, :, :]
    tgt_labels = np.zeros((1, n_q), dtype=np.int8)
    tgt_batch = StandardBatch(
        mode=InputMode.EVENT_ONLY, theta=None, phi=tgt_phi, labels=tgt_labels,
    )

    n_ctx = int(min(ctx_energies.size, max_context_per_pass))
    ctx_e_norm_all = (ctx_energies - e_lo) / (e_hi - e_lo)
    ctx_x_all = (ctx_scores >= T).astype(np.int8)

    # NOTE: context sampling is **natural density**, not bin-stratified
    # (despite training using bin-stratified). The asymmetry is intentional:
    # training balance forces the model to learn β(E) at sparse energies as
    # well as dense ones; at inference, an energy region with more events
    # genuinely carries more information about β there, and we let the
    # encoder's representation reflect that. Do not "fix" this to match
    # training without thinking through the information-flow implications.
    samples = np.empty((n_mc, n_q), dtype=np.float64)
    cnp.train()  # enable dropout
    try:
        with torch.no_grad():
            for m in range(n_mc):
                if n_ctx >= ctx_energies.size:
                    picks = rng.permutation(ctx_energies.size)
                else:
                    picks = rng.choice(ctx_energies.size, size=n_ctx, replace=False)
                ctx_e_norm = ctx_e_norm_all[picks]
                ctx_phi = np.stack(
                    [ctx_e_norm, np.full(n_ctx, t_norm)], axis=-1
                )[None, :, :]
                ctx_labels = ctx_x_all[picks][None, :]
                ctx_batch = StandardBatch(
                    mode=InputMode.EVENT_ONLY, theta=None,
                    phi=ctx_phi, labels=ctx_labels,
                )
                beta = cnp.predict_beta(ctx_batch, tgt_batch).cpu().numpy()
                samples[m] = beta[0]
    finally:
        cnp.eval()

    beta_mean = samples.mean(axis=0)
    beta_std = samples.std(axis=0, ddof=1)
    return beta_mean, beta_std, n_ctx


def pull_arrays(res: InferenceResult) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(z_cnp_only, z_combined)`` signed pulls for valid bins.

    A "pull" is the signed standardized residual
    ``z = (rate_emp − β_pred) / σ`` per D_T bin. Two flavors:

    * ``z_cnp_only`` uses ``σ = σ_CNP`` (MC-Dropout std alone) —
      should appear wide (under-covered) because binomial scatter
      isn't included.
    * ``z_combined`` uses ``σ = √(σ_CNP² + σ_emp²)`` — should be
      ≈ ``N(0, 1)`` if the model is calibrated against the
      binomial-noise-corrupted empirical estimate.

    Only bins where both ``rate`` and ``beta`` are finite contribute.
    """
    valid = ~np.isnan(res.rate) & ~np.isnan(res.beta)
    if not valid.any():
        empty = np.empty(0, dtype=np.float64)
        return empty, empty
    rate = res.rate[valid]
    beta = res.beta[valid]
    sigma_cnp = np.maximum(res.beta_std[valid], 1e-9)
    sigma_emp = 0.5 * (res.rate_hi[valid] - res.rate_lo[valid])
    sigma_combined = np.sqrt(sigma_cnp**2 + sigma_emp**2)
    resid = rate - beta
    return resid / sigma_cnp, resid / np.maximum(sigma_combined, 1e-9)


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
    n_dense: int = DEFAULT_N_DENSE,
    max_context_per_pass: int = DEFAULT_MAX_CONTEXT_PER_PASS,
) -> InferenceResult:
    """Load the trained CNP and run the D_C / D_T protocol end-to-end."""
    energy_all, score_all, T_star = _filter_test_events(cfg)

    e_C, s_C, e_T, s_T = split_test_data(
        energy_all, score_all,
        test_fraction=test_fraction,
        context_fraction=context_fraction,
        seed=seed,
    )

    # The pipeline's saved bin grid is still the canonical D_T binning
    # for the blue Wilson points + coverage. The CNP itself doesn't
    # know about it.
    pool = np.load(Path(cfg.out_dir) / "training_pool.npz")
    bin_centers = pool["bin_centers"]
    edges = _bin_edges_from_centers(bin_centers, cfg.energy_bin_width)
    e_lo, e_hi = cfg.energy_range
    dense_energies = np.linspace(e_lo, e_hi, int(n_dense))

    rate, rate_lo, rate_hi, counts_T = _empirical_with_wilson(e_T, s_T, T_star, edges)
    sparse_T = counts_T < cfg.min_events_per_bin
    rate[sparse_T] = np.nan
    rate_lo[sparse_T] = np.nan
    rate_hi[sparse_T] = np.nan

    cnp = _load_cnp(cfg, Path(cfg.out_dir) / "cnp.ckpt")
    n_bin = bin_centers.size
    query_e = np.concatenate([bin_centers, dense_energies])
    beta_all, std_all, n_ctx_used = _cnp_infer_global(
        cnp, cfg, query_e, e_C, s_C, T_star,
        n_mc=n_mc, seed=seed,
        max_context_per_pass=max_context_per_pass,
    )
    beta_bin, std_bin = beta_all[:n_bin], std_all[:n_bin]
    beta_dense, std_dense = beta_all[n_bin:], std_all[n_bin:]

    cov_cnp, cov_comb, mean_off, pearson_r = _coverage_two_ways(
        rate, rate_lo, rate_hi, beta_bin, std_bin
    )

    return InferenceResult(
        bin_centers=bin_centers,
        rate=rate, rate_lo=rate_lo, rate_hi=rate_hi,
        n_target_per_bin=counts_T,
        beta=beta_bin, beta_std=std_bin,
        dense_energies=dense_energies,
        dense_beta=beta_dense, dense_beta_std=std_dense,
        n_context_per_pass=int(n_ctx_used),
        context_window_kev=float("inf"),
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

    Blue: D_T binned points with Wilson 1σ errorbars.
    Red:  CNP β(E) on the **dense grid** (continuous; no per-bin gaps).
    Band: dense ±σ_CNP — the MC-Dropout predictive uncertainty.
    """
    fig, ax = plt.subplots(figsize=(11, 4.8))
    yerr_lo = np.maximum(np.where(np.isnan(res.rate), 0.0, res.rate - res.rate_lo), 0.0)
    yerr_hi = np.maximum(np.where(np.isnan(res.rate), 0.0, res.rate_hi - res.rate), 0.0)
    ax.errorbar(
        res.bin_centers, res.rate, yerr=[yerr_lo, yerr_hi],
        fmt="o", ms=4, capsize=2, color="steelblue",
        label=f"D_T binned (Wilson 1σ)  N_target={res.n_target_total}",
    )

    mu = np.clip(res.dense_beta, 0.0, 1.0)
    if show_band:
        sigma_cnp = np.where(np.isnan(res.dense_beta_std), 0.0, res.dense_beta_std)
        ax.fill_between(
            res.dense_energies,
            np.clip(mu - sigma_cnp, 0.0, 1.0),
            np.clip(mu + sigma_cnp, 0.0, 1.0),
            color="firebrick", alpha=0.20,
            label="CNP ±σ_CNP  (MC Dropout)",
        )
    ax.plot(
        res.dense_energies, mu, color="firebrick", lw=1.6,
        label=(
            f"CNP β(E) | D_C   N_context={res.n_context_total}"
            f"   n_ctx/pass={res.n_context_per_pass}"
        ),
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


def plot_pulls(
    ax,
    z: np.ndarray,
    *,
    title: str,
    n_bins: int = 30,
    x_range: tuple[float, float] = (-5.0, 5.0),
) -> None:
    """Draw a single pull histogram with the ideal ``N(0, 1)`` overlay.

    The histogram bars are counts; the red curve is the standard
    normal PDF scaled by ``N · bin_width`` so the two integrate to the
    same area. A calibrated model produces bars that follow the curve
    closely with empirical ``μ ≈ 0`` and ``σ ≈ 1``.
    """
    z = z[np.isfinite(z)]
    n = z.size
    if n == 0:
        ax.text(0.5, 0.5, "no valid bins", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title(title, fontsize=10)
        return
    counts, edges = np.histogram(z, bins=n_bins, range=x_range)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_width = edges[1] - edges[0]
    ax.bar(
        centers, counts, width=bin_width,
        color="steelblue", alpha=0.65, edgecolor="white", linewidth=0.5,
        label=f"observed (N={n})",
    )
    x = np.linspace(x_range[0], x_range[1], 400)
    pdf = np.exp(-0.5 * x**2) / np.sqrt(2.0 * np.pi)
    ax.plot(x, pdf * n * bin_width, color="firebrick", lw=1.6, label="ideal N(0, 1)")
    ax.axvline(0.0, color="gray", ls=":", alpha=0.6, lw=0.8)
    mu = float(z.mean())
    sigma = float(z.std(ddof=1)) if n > 1 else float("nan")
    ax.set_xlim(*x_range)
    ax.set_xlabel("z = (rate − β) / σ")
    ax.set_ylabel("count")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(f"{title}\nemp μ = {mu:+.2f}    σ = {sigma:.2f}", fontsize=10)


def plot_coverage(
    cfg: CutAcceptanceConfig,
    res: InferenceResult,
    out_path: Path,
) -> None:
    """Save the canonical calibration figure: combined-σ pulls vs N(0, 1).

    Uses ``σ_combined = √(σ_CNP² + σ_emp²)`` — the only flavor with a
    meaningful Gaussian-target interpretation. The σ_CNP-only pulls
    are over-wide by construction (binomial scatter isn't included)
    and don't carry calibration information, so we don't draw them.
    """
    _, z_comb = pull_arrays(res)
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    plot_pulls(
        ax, z_comb,
        title=f"pulls / σ_combined   (cov₁σ = {res.coverage_combined['1sigma']:.2f})",
    )
    fig.suptitle(
        f"{cfg.name}   ·   bin = {cfg.energy_bin_width:.0f} keV   ·   "
        f"target_class = {cfg.target_class!r}",
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
        f"  n_dense (red curve points):   {res.dense_energies.size}",
        f"  n_mc passes (MC Dropout):     {n_mc}",
        f"  n_context per MC pass:        {res.n_context_per_pass}",
        f"  seed:                         {seed}",
        f"  Youden-J best T*:             {res.T_star:.4f}",
        "",
        f"  N_total (after class+E filter): {res.n_total}",
        f"  |D_C| = {res.n_context_total}    |D_T| = {res.n_target_total}",
        f"  bins with empty D_T:          {int((res.n_target_per_bin == 0).sum())}",
        "",
        "[1] Energy fidelity (D_T vs CNP β(E) at bin centers)",
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
    coverage_path = out_dir / "coverage_audit.png"
    plot_coverage(cfg, res, coverage_path)
    lines.append(f"plot: {plot_path}")
    lines.append(f"coverage: {coverage_path}")
    text = "\n".join(lines) + "\n"
    (out_dir / "test_set_audit.txt").write_text(text)

    metrics = dict(
        config=cfg.name,
        target_class=str(cfg.target_class),
        energy_bin_width=cfg.energy_bin_width,
        test_fraction=test_fraction,
        context_fraction=context_fraction,
        n_dense=int(res.dense_energies.size),
        n_mc=n_mc,
        n_context_per_pass=res.n_context_per_pass,
        seed=seed,
        T_star=res.T_star,
        n_total=res.n_total,
        n_context_total=res.n_context_total,
        n_target_total=res.n_target_total,
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
    n_dense: int = DEFAULT_N_DENSE,
    max_context_per_pass: int = DEFAULT_MAX_CONTEXT_PER_PASS,
) -> dict:
    cfg = load_config(cfg_path)
    res = infer_and_evaluate(
        cfg,
        test_fraction=test_fraction, context_fraction=context_fraction,
        n_mc=n_mc, seed=seed,
        n_dense=n_dense,
        max_context_per_pass=max_context_per_pass,
    )
    return write_report(
        cfg, res, out_dir,
        test_fraction=test_fraction, context_fraction=context_fraction,
        n_mc=n_mc, seed=seed,
    )


def main() -> None:
    # CLI entry — headless. Notebook callers leave the backend alone
    # so plt.show() keeps working inline.
    import matplotlib
    matplotlib.use("Agg")

    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--test-fraction", type=float, default=1.0)
    ap.add_argument("--context-fraction", type=float, default=0.20)
    ap.add_argument("--n-mc", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--n-dense", type=int, default=DEFAULT_N_DENSE,
        help="Number of points in the smooth red β(E) curve.",
    )
    ap.add_argument(
        "--max-context-per-pass", type=int, default=DEFAULT_MAX_CONTEXT_PER_PASS,
        help="Cap on D_C events fed to the CNP per MC pass (without replacement).",
    )
    args = ap.parse_args()
    out_dir = args.out_dir
    if out_dir is None:
        rel = args.config.relative_to(Path("configs/cut_acceptance"))
        out_dir = Path("analysis/cnp_audit") / rel.parent / rel.stem
    run(args.config, out_dir,
        test_fraction=args.test_fraction,
        context_fraction=args.context_fraction,
        n_mc=args.n_mc,
        seed=args.seed,
        n_dense=args.n_dense,
        max_context_per_pass=args.max_context_per_pass)


if __name__ == "__main__":
    main()
