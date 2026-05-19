"""Test-set inference for the True-CNP cut-acceptance pipeline.

The trained CNP is a 1D-regression CNP in ``InputMode.EVENT_ONLY``:
each context event carries its own ``phi_i = (E_i_norm, T_norm)``. As
a true stochastic process, β(E) is a continuous function of energy —
we evaluate it on a dense grid by aggregating **all** D_C events into
a single global representation ``r`` and decoding at every query point
in one forward call. ``DEFAULT_MAX_CONTEXT_PER_PASS`` is set high
enough that the full D_C is used by default; the cap exists only for
memory-constrained experiments and can be lowered explicitly.

Pipeline
--------
* Sample a configurable fraction of the **test** events (default 100%).
* Split into a Context set ``D_C`` (default 20%) and a disjoint Target
  set ``D_T`` (default 80%).
* Empirical pass rate (blue) comes from **D_T only**, binned at the
  pipeline's saved bin grid, with Wilson 1σ errorbars.
* CNP β(E) (red) comes from MC Dropout: ``n_mc`` forward passes that
  share the **same full D_C context** (built once outside the loop),
  varying only the dropout mask. Each pass produces one global ``r``
  aggregated across the entire spectrum, then decodes at the bin
  centers AND a dense grid in one shot. If the user lowers the cap
  below ``|D_C|``, each pass instead resamples a fresh subset.
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
import scipy.stats as _stats
import torch
from core import build_cnp
from schemas.data_models import InputMode, StandardBatch
from sklearn.metrics import roc_curve

from majorana_acp.analysis.metrics import analyze_sawtooth_suite
from majorana_acp.cut_acceptance.config import CutAcceptanceConfig, load_config
from majorana_acp.cut_acceptance.pipeline import (
    resolve_name,
    resolve_out_dir,
    wilson_interval,
)
from majorana_acp.cut_acceptance.positional_encoding import encode_phi

PEAK_MARKERS = [
    ("Tl-208 FE", 2614.0),
    ("Tl-208 SE", 2103.0),
    ("Tl-208 DEP", 1592.0),
    ("Bi-214 1620", 1620.0),
]

# Defaults exposed through the CLI / notebook for tuning.
DEFAULT_N_DENSE = 800
# Effectively unlimited — modest 5-figure cap that's safely above any
# realistic |D_C| in this project (~1.4 k events at the largest), so
# the cap never bites unless the user opts in explicitly. The earlier
# value (256) silently discarded ~80 % of available conditioning
# evidence and capped the model's representation bandwidth far below
# what the variable-N training paradigms can exploit. Keep the knob
# in the CLI for memory-constrained experiments only.
DEFAULT_MAX_CONTEXT_PER_PASS = 99999


@dataclass(frozen=True)
class PeakRegionMetrics:
    """Localized goodness-of-fit at a single γ-peak window.

    For each peak ``E₀`` we evaluate over the bins whose centers sit
    within ``[E₀ − half_window_kev, E₀ + half_window_kev]``. The same
    quantities are computed **twice** — once against the Context-set
    empirical rate (``_DC``, probes whether the CNP's representation
    matches the data it was conditioned on) and once against the
    Target set (``_DT``, probes generalization to held-out events).

    Fields
    ------
    chi2_<X>
        Reduced χ²:  ``(1/N) Σ (y_i − β_i)² / (σ_stat_i² + σ_CNP_i²)``.
        A value ≈ 1 indicates the model + uncertainty budget cleanly
        pierce the empirical averages. > 1: over-smoothing; < 1:
        over-confident σ.
    z_<X>
        Local Z-score:
        ``(ȳ_I − β̄_I) / √(Var(y)_I + ⟨σ_CNP²⟩_I)``.
        Sign indicates direction of systematic bias.
    p_<X>
        Two-tailed p-value from |Z| under N(0,1). ``p < 0.05`` flags a
        statistically significant feature mismatch; ``p > 0.5`` means
        the model is indistinguishable from the sharp physics truth.
    n_valid_<X>
        How many bins inside the window had a finite empirical rate
        (others are excluded from the χ² and Z averages).
    """

    peak_name: str
    peak_energy_kev: float
    half_window_kev: float
    n_bins_in_window: int
    # Context-side
    chi2_DC: float
    z_DC: float
    p_DC: float
    n_valid_DC: int
    # Target-side
    chi2_DT: float
    z_DT: float
    p_DT: float
    n_valid_DT: int


def _peak_chi2_and_z(
    rate: np.ndarray,
    sigma_emp: np.ndarray,
    beta: np.ndarray,
    beta_std: np.ndarray,
    in_window: np.ndarray,
) -> tuple[float, float, float, int]:
    """Return ``(chi2, z, p, n_valid)`` for one (peak, side) combination."""
    valid_mask = ~np.isnan(rate[in_window])
    valid = in_window[valid_mask]
    if valid.size == 0:
        return float("nan"), float("nan"), float("nan"), 0
    y = rate[valid]
    b = beta[valid]
    se = np.maximum(sigma_emp[valid], 1e-9)
    sc = np.maximum(beta_std[valid], 1e-9)
    denom = se**2 + sc**2
    chi2 = float(np.mean((y - b) ** 2 / denom))
    y_bar = float(np.mean(y))
    b_bar = float(np.mean(b))
    var_y = float(np.var(y, ddof=1)) if valid.size > 1 else 0.0
    mean_sc2 = float(np.mean(sc**2))
    denom_z = max(var_y + mean_sc2, 1e-18)
    z = (y_bar - b_bar) / float(np.sqrt(denom_z))
    p = float(2.0 * _stats.norm.sf(abs(z)))
    return chi2, z, p, int(valid.size)


def compute_peak_metrics(
    bin_centers: np.ndarray,
    beta: np.ndarray,
    beta_std: np.ndarray,
    rate_DC: np.ndarray,
    sigma_emp_DC: np.ndarray,
    rate_DT: np.ndarray,
    sigma_emp_DT: np.ndarray,
    *,
    peak_markers: list[tuple[str, float]] = None,  # type: ignore[assignment]
    half_window_kev: float = 5.0,
) -> list[PeakRegionMetrics]:
    """Localized peak-region χ²/Z/p for D_C and D_T, per peak."""
    if peak_markers is None:
        peak_markers = PEAK_MARKERS
    out: list[PeakRegionMetrics] = []
    for name, e0 in peak_markers:
        in_window = np.flatnonzero(np.abs(bin_centers - e0) <= half_window_kev)
        chi2_dc, z_dc, p_dc, n_dc = _peak_chi2_and_z(
            rate_DC,
            sigma_emp_DC,
            beta,
            beta_std,
            in_window,
        )
        chi2_dt, z_dt, p_dt, n_dt = _peak_chi2_and_z(
            rate_DT,
            sigma_emp_DT,
            beta,
            beta_std,
            in_window,
        )
        out.append(
            PeakRegionMetrics(
                peak_name=name,
                peak_energy_kev=float(e0),
                half_window_kev=float(half_window_kev),
                n_bins_in_window=int(in_window.size),
                chi2_DC=chi2_dc,
                z_DC=z_dc,
                p_DC=p_dc,
                n_valid_DC=n_dc,
                chi2_DT=chi2_dt,
                z_DT=z_dt,
                p_DT=p_dt,
                n_valid_DT=n_dt,
            )
        )
    return out


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
    n_context_per_pass: int  # actual context size used per MC pass.
    # Equals |D_C| under the default unlimited cap;
    # equals max_context_per_pass if the user
    # lowered it below |D_C| (then each pass
    # resamples a fresh subset).
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
    # Localized peak-region goodness-of-fit (per peak in PEAK_MARKERS).
    peak_metrics: list[PeakRegionMetrics]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _bin_edges_from_centers(bin_centers: np.ndarray, bin_width: float) -> np.ndarray:
    """Pipeline writes bin *centers*; reconstruct the edge array."""
    half = 0.5 * bin_width
    return np.concatenate([bin_centers - half, [bin_centers[-1] + half]])


def _load_cnp(cfg: CutAcceptanceConfig, ckpt_path: Path) -> torch.nn.Module:
    """Reconstruct the EVENT_ONLY CNP architecture and load weights.

    Dispatches by ``cfg.aggregator.type`` so the test path mirrors the
    training path exactly. ``dim_phi`` matches what the training
    pipeline used: ``phi_dim(cfg.positional_encoding)`` — ``2`` when PE
    is off (every pre-PE checkpoint) and ``2 * num_bands + 1`` when on.

    Backward compat: pre-aggregator YAMLs (no ``aggregator`` block) get
    ``cfg.aggregator.type == 'mean'`` from the pydantic default factory,
    which routes through upstream ``build_cnp`` — byte-identical to
    every legacy checkpoint.
    """
    from majorana_acp.cut_acceptance.positional_encoding import phi_dim
    from majorana_acp.models.attentive_cnp import build_attentive_cnp

    dim_phi = phi_dim(cfg.positional_encoding)
    if cfg.aggregator.type == "mean":
        cnp = build_cnp(
            cfg.encoder,
            dim_theta=None,
            dim_phi=dim_phi,
            decoder_hidden_dims=list(cfg.decoder_hidden_dims),
        )
    elif cfg.aggregator.type == "cross_attention":
        dm = cfg.aggregator.density_modulation
        bb = cfg.aggregator.bounded_bandwidth
        sfn = cfg.aggregator.sfn_modulation
        pdsfn = cfg.aggregator.pool_density_sfn
        needs_range = dm.enabled or bb.enabled or sfn.enabled or pdsfn.enabled
        # Pool-Density SFN: reconstruct the same pool buffer the training
        # pipeline used by re-running ``load_events`` against the SAME
        # train predictions file + class + energy filter. Deterministic →
        # byte-for-byte identical buffer at train and inference.
        pool_energies_kev = None
        if pdsfn.enabled:
            from majorana_acp.cut_acceptance.binned_sampler import load_events

            pool_e, _ = load_events(
                cfg.train_predictions_path,
                target_class=cfg.target_class,
                energy_range=cfg.energy_range,
            )
            pool_energies_kev = pool_e
        cnp = build_attentive_cnp(
            cfg.encoder,
            dim_theta=None,
            dim_phi=dim_phi,
            num_heads=cfg.aggregator.num_heads,
            attention_dim=cfg.aggregator.attention_dim,
            decoder_coordinate_gating=cfg.aggregator.decoder_coordinate_gating,
            gaussian_attention_bias=cfg.aggregator.gaussian_attention_bias,
            density_modulation_enabled=dm.enabled,
            density_sigma_local_kev=dm.sigma_local_kev if dm.enabled else None,
            density_sigma_global_kev=dm.sigma_global_kev if dm.enabled else None,
            density_epsilon=dm.epsilon,
            bounded_bandwidth_enabled=bb.enabled,
            bounded_sigma_max_kev=bb.sigma_max_kev if bb.enabled else None,
            bounded_sigma_min_kev=bb.sigma_min_kev if bb.enabled else None,
            bounded_alpha_max=bb.alpha_max if bb.enabled else None,
            bounded_sensitivity_hidden_dim=bb.sensitivity_hidden_dim,
            bounded_sigma_local_kev=bb.sigma_local_kev if bb.enabled else None,
            bounded_sigma_global_kev=bb.sigma_global_kev if bb.enabled else None,
            bounded_epsilon=bb.epsilon,
            sfn_modulation_enabled=sfn.enabled,
            sfn_sigma_max_kev=sfn.sigma_max_kev if sfn.enabled else None,
            sfn_sigma_min_kev=sfn.sigma_min_kev if sfn.enabled else None,
            sfn_hidden_dim=sfn.hidden_dim,
            sfn_sigma_local_kev=sfn.sigma_local_kev if sfn.enabled else None,
            sfn_sigma_global_kev=sfn.sigma_global_kev if sfn.enabled else None,
            sfn_epsilon=sfn.epsilon,
            pool_density_sfn_enabled=pdsfn.enabled,
            pool_density_sfn_sigma_max_kev=pdsfn.sigma_max_kev if pdsfn.enabled else None,
            pool_density_sfn_sigma_min_kev=pdsfn.sigma_min_kev if pdsfn.enabled else None,
            pool_density_sfn_hidden_dim=pdsfn.hidden_dim,
            pool_density_sfn_sigma_local_kev=pdsfn.sigma_local_kev if pdsfn.enabled else None,
            pool_density_sfn_sigma_global_kev=pdsfn.sigma_global_kev if pdsfn.enabled else None,
            pool_density_sfn_epsilon=pdsfn.epsilon,
            pool_density_sfn_temperature_gating=pdsfn.temperature_gating,
            pool_density_sfn_tau_min=pdsfn.tau_min_value,
            pool_density_sfn_tau_max=pdsfn.tau_max_value,
            pool_energies_kev=pool_energies_kev,
            energy_range_kev=cfg.energy_range if needs_range else None,
            decoder_hidden_dims=list(cfg.decoder_hidden_dims),
        )
    else:
        raise ValueError(f"unknown aggregator.type: {cfg.aggregator.type!r}")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cnp.load_state_dict(state["model_state"])
    cnp.eval()
    # Honor the config's device preference at inference too — running
    # MC Dropout × n_mc passes × N_T queries × (possibly chunked pool)
    # on GPU pulls a 4 min CPU job down to ~10 s. The training-time
    # pool buffer is non-persistent, so it was rebuilt CPU-side inside
    # the factory above; ``.to(device)`` moves it onto GPU along with
    # the model parameters in one shot.
    from majorana_acp.cut_acceptance.pipeline import _resolve_device

    device = _resolve_device(getattr(cfg, "device", "auto"))
    cnp.to(device)
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
    energy: np.ndarray,
    score: np.ndarray,
    *,
    test_fraction: float,
    context_fraction: float,
    seed: int,
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
    energies: np.ndarray,
    scores: np.ndarray,
    T: float,
    edges: np.ndarray,
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
    # encode_phi is a no-op when positional_encoding.enabled is False
    # (the default for every legacy checkpoint), so this branch is
    # bitwise-identical to the pre-PE inference path.
    tgt_phi = np.stack([q_norm, np.full_like(q_norm, t_norm)], axis=-1)[None, :, :]
    tgt_phi = encode_phi(
        tgt_phi,
        cfg.positional_encoding,
        energy_range_kev=cfg.energy_range,
    )
    tgt_labels = np.zeros((1, n_q), dtype=np.int8)
    tgt_batch = StandardBatch(
        mode=InputMode.EVENT_ONLY,
        theta=None,
        phi=tgt_phi,
        labels=tgt_labels,
    )

    n_ctx = int(min(ctx_energies.size, max_context_per_pass))
    use_all_context = n_ctx >= ctx_energies.size
    ctx_e_norm_all = (ctx_energies - e_lo) / (e_hi - e_lo)
    ctx_x_all = (ctx_scores >= T).astype(np.int8)

    # NOTE: context sampling is **natural density**, not bin-stratified
    # (despite training using bin-stratified). The asymmetry is intentional:
    # training balance forces the model to learn β(E) at sparse energies as
    # well as dense ones; at inference, an energy region with more events
    # genuinely carries more information about β there, and we let the
    # encoder's representation reflect that. Do not "fix" this to match
    # training without thinking through the information-flow implications.
    #
    # When ``use_all_context`` (the default — see DEFAULT_MAX_CONTEXT_PER_PASS)
    # we build the context tensor *once* outside the MC loop and reuse it for
    # every forward pass. Then the only stochasticity across the n_mc passes
    # is the dropout mask — exactly the model's epistemic uncertainty,
    # uncontaminated by per-pass context-resampling variance. The earlier
    # behavior (resample a 256-event subset every pass) inflated σ_CNP with
    # data-starvation noise that masked the real model bias.
    samples = np.empty((n_mc, n_q), dtype=np.float64)
    cnp.train()  # enable dropout

    def _make_ctx_batch(idx: np.ndarray) -> StandardBatch:
        ctx_e_norm = ctx_e_norm_all[idx]
        ctx_phi = np.stack([ctx_e_norm, np.full(idx.size, t_norm)], axis=-1)[None, :, :]
        # Apply the same Fourier expansion the training sampler used.
        # No-op (bitwise identity) when PE is disabled.
        ctx_phi = encode_phi(
            ctx_phi,
            cfg.positional_encoding,
            energy_range_kev=cfg.energy_range,
        )
        ctx_labels = ctx_x_all[idx][None, :]
        return StandardBatch(
            mode=InputMode.EVENT_ONLY,
            theta=None,
            phi=ctx_phi,
            labels=ctx_labels,
        )

    fixed_ctx_batch = _make_ctx_batch(np.arange(ctx_energies.size)) if use_all_context else None

    try:
        with torch.no_grad():
            for m in range(n_mc):
                if use_all_context:
                    ctx_batch = fixed_ctx_batch
                else:
                    picks = rng.choice(ctx_energies.size, size=n_ctx, replace=False)
                    ctx_batch = _make_ctx_batch(picks)
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
    rate: np.ndarray,
    rate_lo: np.ndarray,
    rate_hi: np.ndarray,
    beta: np.ndarray,
    beta_std: np.ndarray,
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
        energy_all,
        score_all,
        test_fraction=test_fraction,
        context_fraction=context_fraction,
        seed=seed,
    )

    # The pipeline's saved bin grid is still the canonical D_T binning
    # for the blue Wilson points + coverage. The CNP itself doesn't
    # know about it.
    cell_dir = resolve_out_dir(cfg)
    pool = np.load(cell_dir / "training_pool.npz")
    bin_centers = pool["bin_centers"]
    edges = _bin_edges_from_centers(bin_centers, cfg.energy_bin_width)
    e_lo, e_hi = cfg.energy_range
    dense_energies = np.linspace(e_lo, e_hi, int(n_dense))

    # Bin D_T (target) and D_C (context) on the same grid. D_T's
    # Wilson errorbars feed the blue points + coverage; D_C's also
    # binned now so peak-region χ² can be computed for the context
    # side ("does the CNP match the data it was conditioned on?").
    rate, rate_lo, rate_hi, counts_T = _empirical_with_wilson(e_T, s_T, T_star, edges)
    rate_DC, rate_DC_lo, rate_DC_hi, counts_C = _empirical_with_wilson(e_C, s_C, T_star, edges)
    sparse_T = counts_T < cfg.min_events_per_bin
    rate[sparse_T] = np.nan
    rate_lo[sparse_T] = np.nan
    rate_hi[sparse_T] = np.nan
    # D_C sparsity threshold: at least 1 event for a binomial rate to exist.
    sparse_C = counts_C < 1
    rate_DC[sparse_C] = np.nan
    rate_DC_lo[sparse_C] = np.nan
    rate_DC_hi[sparse_C] = np.nan

    cnp = _load_cnp(cfg, cell_dir / "cnp.ckpt")
    n_bin = bin_centers.size
    query_e = np.concatenate([bin_centers, dense_energies])
    beta_all, std_all, n_ctx_used = _cnp_infer_global(
        cnp,
        cfg,
        query_e,
        e_C,
        s_C,
        T_star,
        n_mc=n_mc,
        seed=seed,
        max_context_per_pass=max_context_per_pass,
    )
    beta_bin, std_bin = beta_all[:n_bin], std_all[:n_bin]
    beta_dense, std_dense = beta_all[n_bin:], std_all[n_bin:]

    cov_cnp, cov_comb, mean_off, pearson_r = _coverage_two_ways(
        rate, rate_lo, rate_hi, beta_bin, std_bin
    )

    # Localized peak-region metrics — one row per peak, evaluated
    # separately against D_C (representation) and D_T (generalization).
    sigma_emp_DT = 0.5 * (rate_hi - rate_lo)
    sigma_emp_DC = 0.5 * (rate_DC_hi - rate_DC_lo)
    peak_metrics = compute_peak_metrics(
        bin_centers,
        beta_bin,
        std_bin,
        rate_DC,
        sigma_emp_DC,
        rate,
        sigma_emp_DT,
    )

    return InferenceResult(
        bin_centers=bin_centers,
        rate=rate,
        rate_lo=rate_lo,
        rate_hi=rate_hi,
        n_target_per_bin=counts_T,
        beta=beta_bin,
        beta_std=std_bin,
        dense_energies=dense_energies,
        dense_beta=beta_dense,
        dense_beta_std=std_dense,
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
        peak_metrics=peak_metrics,
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
        res.bin_centers,
        res.rate,
        yerr=[yerr_lo, yerr_hi],
        fmt="o",
        ms=4,
        capsize=2,
        color="steelblue",
        label=f"D_T binned (Wilson 1σ)  N_target={res.n_target_total}",
    )

    mu = np.clip(res.dense_beta, 0.0, 1.0)
    if show_band:
        sigma_cnp = np.where(np.isnan(res.dense_beta_std), 0.0, res.dense_beta_std)
        ax.fill_between(
            res.dense_energies,
            np.clip(mu - sigma_cnp, 0.0, 1.0),
            np.clip(mu + sigma_cnp, 0.0, 1.0),
            color="firebrick",
            alpha=0.20,
            label="CNP ±σ_CNP  (MC Dropout)",
        )
    ax.plot(
        res.dense_energies,
        mu,
        color="firebrick",
        lw=1.6,
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
        f"{resolve_name(cfg)}   (bin = {cfg.energy_bin_width:.0f} keV, "
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
        ax.text(0.5, 0.5, "no valid bins", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title, fontsize=10)
        return
    counts, edges = np.histogram(z, bins=n_bins, range=x_range)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_width = edges[1] - edges[0]
    ax.bar(
        centers,
        counts,
        width=bin_width,
        color="steelblue",
        alpha=0.65,
        edgecolor="white",
        linewidth=0.5,
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
        ax,
        z_comb,
        title=f"pulls / σ_combined   (cov₁σ = {res.coverage_combined['1sigma']:.2f})",
    )
    fig.suptitle(
        f"{resolve_name(cfg)}   ·   bin = {cfg.energy_bin_width:.0f} keV   ·   "
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
    test_fraction: float,
    context_fraction: float,
    n_mc: int,
    seed: int,
) -> dict:
    """Persist text + JSON reports + the diagnostic figure."""
    out_dir.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        "=" * 78,
        f"Test-set inference  ·  {resolve_name(cfg)}",
        "=" * 78,
        f"  config:                       {resolve_out_dir(cfg).name}",
        f"  trained CNP:                  {resolve_out_dir(cfg) / 'cnp.ckpt'}",
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
        "[3] Localized peak-region goodness-of-fit  "
        f"(window = ±{res.peak_metrics[0].half_window_kev:.1f} keV around each peak)",
        "    target χ² ≈ 1.0  ·  p > 0.5 = indistinguishable from sharp truth  "
        "·  p < 0.05 = significant local miss",
        "    " + "─" * 72,
        ("    peak              n_bins      D_C: χ²    Z      p          D_T: χ²    Z      p"),
        "    " + "─" * 72,
        *[
            (
                f"    {pm.peak_name:<16s}  {pm.n_bins_in_window:>3d} "
                f"        {pm.chi2_DC:>5.2f}  {pm.z_DC:>+5.2f}  {pm.p_DC:>5.3f}    "
                f"   {pm.chi2_DT:>5.2f}  {pm.z_DT:>+5.2f}  {pm.p_DT:>5.3f}"
            )
            for pm in res.peak_metrics
        ],
        "    " + "─" * 72,
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

    # Sawtooth-diagnostic suite — three roughness metrics evaluated
    # in two pristine Compton-continuum windows clear of any γ-peaks.
    # MASD = amplitude axis, ED = frequency axis, ACF1 = pattern axis.
    # Computed on the same dense β̂(E) curve that drives the red trace
    # in the audit PNG so the numbers correspond directly to what we
    # see visually.
    sawtooth_metrics = {
        f"region_{int(lo)}_{int(hi)}": analyze_sawtooth_suite(
            res.dense_energies, res.dense_beta, (lo, hi)
        )
        for (lo, hi) in [(1700.0, 2000.0), (2200.0, 2400.0)]
    }

    metrics = dict(
        config=resolve_name(cfg),
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
    (out_dir / "test_set_audit.json").write_text(json.dumps(metrics, indent=2))
    print(text)
    return metrics


def run(
    cfg_path: Path,
    out_dir: Path,
    *,
    test_fraction: float = 1.0,
    context_fraction: float = 0.20,
    n_mc: int = 50,
    seed: int = 0,
    n_dense: int = DEFAULT_N_DENSE,
    max_context_per_pass: int = DEFAULT_MAX_CONTEXT_PER_PASS,
) -> dict:
    cfg = load_config(cfg_path)
    res = infer_and_evaluate(
        cfg,
        test_fraction=test_fraction,
        context_fraction=context_fraction,
        n_mc=n_mc,
        seed=seed,
        n_dense=n_dense,
        max_context_per_pass=max_context_per_pass,
    )
    return write_report(
        cfg,
        res,
        out_dir,
        test_fraction=test_fraction,
        context_fraction=context_fraction,
        n_mc=n_mc,
        seed=seed,
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
        "--n-dense",
        type=int,
        default=DEFAULT_N_DENSE,
        help="Number of points in the smooth red β(E) curve.",
    )
    ap.add_argument(
        "--max-context-per-pass",
        type=int,
        default=DEFAULT_MAX_CONTEXT_PER_PASS,
        help="Cap on D_C events fed to the CNP per MC pass (without replacement).",
    )
    args = ap.parse_args()
    out_dir = args.out_dir
    if out_dir is None:
        rel = args.config.relative_to(Path("configs/cut_acceptance"))
        out_dir = Path("analysis/cnp_audit") / rel.parent / rel.stem
    run(
        args.config,
        out_dir,
        test_fraction=args.test_fraction,
        context_fraction=args.context_fraction,
        n_mc=args.n_mc,
        seed=args.seed,
        n_dense=args.n_dense,
        max_context_per_pass=args.max_context_per_pass,
    )


if __name__ == "__main__":
    main()
