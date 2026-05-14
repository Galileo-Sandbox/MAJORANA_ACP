"""Binned-CNP cut-acceptance pipeline (no MFGP).

One ``run_pipeline(cfg)`` call performs:

1. Load the train-split predictions, filter by ``target_class`` +
   ``energy_range``, and build a :class:`BinnedSampler`.
2. Train a RESUM_FLEX CNP on this sampler.
3. Load the test-split predictions and compute the empirical binned
   pass rate at the Youden-J best T (the validation curve).
4. Evaluate the CNP on a ``(E_bin_center × T_grid)`` mesh and persist
   the prediction surface plus a 1-D slice at the same Youden-J best T.

Outputs (all under ``cfg.out_dir``):

* ``cnp.ckpt``              — RESUM_FLEX CNP checkpoint.
* ``training_pool.npz``     — bin centers + event counts used in training.
* ``validation_binned.npz`` — binned empirical A(E) at T*, binomial 1σ.
* ``cnp_predictions.npz``   — CNP β(E_grid, T_grid) + 1-D slice at T*.
* ``run_summary.json``      — one-page summary (paths + scalar metrics).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from core import build_cnp, save_checkpoint, train_cnp
from schemas.data_models import InputMode, StandardBatch
from sklearn.metrics import roc_curve

from majorana_acp.cut_acceptance.binned_sampler import BinnedSampler, load_events
from majorana_acp.cut_acceptance.config import CutAcceptanceConfig


@dataclass(frozen=True)
class PipelineSummary:
    name: str
    target_class: int | str
    energy_bin_width: float
    out_dir: str
    cnp_ckpt: str
    n_train_events: int
    n_validation_events: int
    n_bins_used: int
    cnp_final_train_loss: float
    youden_T_star: float
    mean_offset_at_T_star: float
    pearson_r_at_T_star: float

    def to_json(self, path: Path | str) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def _build_eval_grid(
    cfg: CutAcceptanceConfig, bin_centers: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(theta_query_norm, energy_grid, threshold_grid)``.

    ``energy_grid`` is the bin-center array (no continuous E sweep —
    we only evaluate on the bins themselves so the CNP output lines up
    with the empirical points exactly).
    """
    t_lo, t_hi = cfg.threshold_range
    t_grid = np.linspace(t_lo, t_hi, 51)
    e_grid = bin_centers
    ee, tt = np.meshgrid(e_grid, t_grid, indexing="ij")
    e_lo, e_hi = cfg.energy_range
    ee_norm = (ee - e_lo) / (e_hi - e_lo)
    tt_norm = (tt - t_lo) / (t_hi - t_lo)
    theta_query = np.stack([ee_norm.ravel(), tt_norm.ravel()], axis=-1).astype(np.float64)
    return theta_query, e_grid, t_grid


def _cnp_predict_grid(
    cnp,
    sampler: BinnedSampler,
    cfg: CutAcceptanceConfig,
    theta_query: np.ndarray,
    e_grid: np.ndarray,
    t_grid: np.ndarray,
    seed: int = 0,
) -> np.ndarray:
    """Run the CNP across the (E_bin × T) mesh.

    Each query point gets a context drawn from the *same* bin's pool
    (so the CNP is being asked the same kind of question it saw during
    training). β at a fixed E is then T-dependent purely through the
    CNP — exactly the property we want to inspect.
    """
    rng = np.random.default_rng(int(seed))
    n_e = e_grid.size
    n_t = t_grid.size
    n_ctx = cfg.n_per_trial
    beta = np.full((n_e, n_t), np.nan, dtype=np.float64)

    for i, _e_center in enumerate(e_grid):
        ev = sampler._index.bin_events[i]
        if ev.size == 0:  # shouldn't happen because BinnedSampler filtered
            continue
        ctx_scores = sampler._index.score[ev]
        # Build context labels per threshold (vectorised over T).
        for j, T in enumerate(t_grid):
            picks = rng.integers(0, ctx_scores.size, size=n_ctx)
            ctx_labels = (ctx_scores[picks] >= T).astype(np.int8)[None, :]
            row_query = theta_query[i * n_t + j][None, :]
            ctx_batch = StandardBatch(
                mode=InputMode.DESIGN_ONLY, theta=row_query, phi=None, labels=ctx_labels
            )
            tgt_batch = StandardBatch(
                mode=InputMode.DESIGN_ONLY,
                theta=row_query,
                phi=None,
                labels=ctx_labels.copy(),
            )
            with torch.no_grad():
                beta[i, j] = float(cnp.predict_beta(ctx_batch, tgt_batch).cpu().numpy().mean())
    return beta


def _empirical_binned(
    energy: np.ndarray,
    score: np.ndarray,
    T: float,
    bin_centers: np.ndarray,
    bin_width: float,
    min_events: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validation: per-bin pass rate at threshold T, with binomial 1σ.

    ``bin_centers`` is the *training-pool* bin grid; we apply the same
    grid to the validation data so the empirical and CNP curves are
    directly comparable. Bins with fewer than ``min_events`` validation
    events are returned as NaN.
    """
    half = 0.5 * bin_width
    edges = np.concatenate([bin_centers - half, [bin_centers[-1] + half]])
    total, _ = np.histogram(energy, bins=edges)
    pcnt, _ = np.histogram(energy[score >= T], bins=edges)
    rate = np.divide(pcnt, total, out=np.full(total.shape, np.nan), where=total > 0)
    err = np.where(total > 0, np.sqrt(rate * (1 - rate) / np.maximum(total, 1)), 0.0)
    # NaN-mask bins below the minimum count.
    sparse = total < min_events
    rate[sparse] = np.nan
    err[sparse] = np.nan
    return rate, err, total.astype(np.int64)


def run_pipeline(cfg: CutAcceptanceConfig, *, seed: int = 0) -> PipelineSummary:
    """End-to-end binned-CNP run; see module docstring for outputs."""
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Build the training sampler.
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
        t_sampling="boundary_mix",
    )
    np.savez(
        out_dir / "training_pool.npz",
        bin_centers=sampler.bin_centers,
        bin_event_counts=sampler.bin_event_counts,
        n_events_total=np.int64(train_e.size),
    )

    # 2. Train the CNP.
    torch.manual_seed(cfg.training.seed)
    cnp = build_cnp(
        cfg.encoder, dim_theta=2, dim_phi=None,
        decoder_hidden_dims=list(cfg.decoder_hidden_dims),
    )
    history = train_cnp(cnp, sampler, cnp_config=cfg.cnp, training_config=cfg.training)
    save_checkpoint(
        out_dir / "cnp.ckpt",
        cnp,
        encoder_config=cfg.encoder,
        dim_theta=2,
        dim_phi=None,
        history=history,
        metadata={
            "name": cfg.name,
            "target_class": cfg.target_class,
            "energy_bin_width": cfg.energy_bin_width,
            "train_predictions_path": str(cfg.train_predictions_path),
            "validation_predictions_path": str(cfg.validation_predictions_path),
            "decoder_hidden_dims": list(cfg.decoder_hidden_dims),
        },
    )
    final_train_loss = float(history["loss"][-1]) if history.get("loss") else float("nan")

    # 3. Validation: load test-split predictions and compute Youden-J best T
    #    from the FULL test ROC (signal vs background labels), then bin the
    #    class-filtered events at that T.
    with h5py.File(cfg.validation_predictions_path, "r") as f:
        val_energy_full = f["energy"][:].astype(np.float64)
        val_score_full = f["score"][:].astype(np.float64)
        val_label_full = f["label"][:].astype(np.int64)
    fpr, tpr, thr = roc_curve(val_label_full, val_score_full)
    T_star = float(thr[int(np.argmax(tpr - fpr))])

    # Class filter for the empirical curve (matches the CNP's training target).
    if cfg.target_class == "all":
        cls_mask = np.ones_like(val_label_full, dtype=bool)
    else:
        cls_mask = val_label_full == int(cfg.target_class)
    e_lo, e_hi = cfg.energy_range
    e_mask = (val_energy_full >= e_lo) & (val_energy_full <= e_hi)
    keep = cls_mask & e_mask
    val_e = val_energy_full[keep]
    val_s = val_score_full[keep]

    rate, err, counts = _empirical_binned(
        val_e, val_s, T_star,
        bin_centers=sampler.bin_centers,
        bin_width=cfg.energy_bin_width,
        min_events=cfg.min_events_per_bin,
    )
    np.savez(
        out_dir / "validation_binned.npz",
        bin_centers=sampler.bin_centers,
        rate=rate,
        rate_err=err,
        counts=counts,
        T_star=T_star,
        n_events=np.int64(val_e.size),
    )

    # 4. CNP prediction grid + 1-D slice at T*.
    theta_query, e_grid, t_grid = _build_eval_grid(cfg, sampler.bin_centers)
    beta_grid = _cnp_predict_grid(cnp, sampler, cfg, theta_query, e_grid, t_grid, seed=seed)
    j_star = int(np.argmin(np.abs(t_grid - T_star)))
    beta_at_T_star = beta_grid[:, j_star]

    np.savez(
        out_dir / "cnp_predictions.npz",
        energy_grid=e_grid,
        threshold_grid=t_grid,
        beta_grid=beta_grid,
        beta_at_T_star=beta_at_T_star,
        T_star=T_star,
    )

    # Headline metrics (mean offset + Pearson r at T*).
    valid = ~np.isnan(rate) & ~np.isnan(beta_at_T_star)
    if valid.any():
        mean_off = float(np.mean(rate[valid] - beta_at_T_star[valid]))
        pearson_r = (
            float(np.corrcoef(rate[valid], beta_at_T_star[valid])[0, 1])
            if valid.sum() > 1
            else float("nan")
        )
    else:
        mean_off = float("nan")
        pearson_r = float("nan")

    summary = PipelineSummary(
        name=cfg.name,
        target_class=cfg.target_class,
        energy_bin_width=cfg.energy_bin_width,
        out_dir=str(out_dir),
        cnp_ckpt=str(out_dir / "cnp.ckpt"),
        n_train_events=int(train_e.size),
        n_validation_events=int(val_e.size),
        n_bins_used=int(sampler.n_bins),
        cnp_final_train_loss=final_train_loss,
        youden_T_star=T_star,
        mean_offset_at_T_star=mean_off,
        pearson_r_at_T_star=pearson_r,
    )
    summary.to_json(out_dir / "run_summary.json")
    return summary
