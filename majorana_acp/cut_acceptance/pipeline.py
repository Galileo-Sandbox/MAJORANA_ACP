"""True-CNP cut-acceptance pipeline — *training only*.

The pipeline trains a 1D-regression CNP in ``InputMode.EVENT_ONLY``:
each context event carries its own coordinates ``(E_i_norm, T_norm)``
in ``phi``, and there is no broadcasted trial-level theta. The bin
grid is used **only** as a sampling-stratification mechanism inside
:class:`majorana_acp.cut_acceptance.event_sampler.EventSampler` — the
CNP itself never sees bin boundaries.

For one ``run_pipeline(cfg)`` call we:

1. Load the train-split predictions, filter by ``target_class`` +
   ``energy_range``, and build an :class:`EventSampler`.
2. Train a RESUM_FLEX CNP (``dim_theta=None``, ``dim_phi=2``) on it.
3. Compute the Youden-J best T* once from the *test* labels so
   downstream diagnostics share a fixed reference threshold.
4. Save the checkpoint + a small ``run_summary.json``.

Outputs (all under ``cfg.out_dir``):

* ``cnp.ckpt``           — RESUM_FLEX CNP checkpoint.
* ``training_pool.npz``  — bin centers + per-bin event counts on the
  train split. Kept *only* so the diagnostics script can bin D_T on the
  same grid for the blue Wilson errorbars; not load-bearing for the CNP.
* ``run_summary.json``   — scalars (paths, counts, final loss, T*).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from core import build_cnp, save_checkpoint, train_cnp
from sklearn.metrics import roc_curve

from majorana_acp.cut_acceptance.config import CutAcceptanceConfig
from majorana_acp.cut_acceptance.event_sampler import EventSampler, load_events


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

    def to_json(self, path: Path | str) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))


# ---------------------------------------------------------------------------
# Wilson score interval (1σ) for a binomial proportion.
# ---------------------------------------------------------------------------


def wilson_interval(
    k: np.ndarray, n: np.ndarray, *, z: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Return (lower, upper) Wilson score interval at level ``z``.

    Handles ``k=0``, ``k=n`` and small ``n`` correctly — the naive
    ``√(p(1-p)/n)`` errorbar collapses to 0 in those cases and is
    symmetric where it shouldn't be. Returns NaN bounds where ``n=0``.
    Inputs broadcast; outputs follow numpy broadcasting rules.
    """
    k = np.asarray(k, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(n > 0, k / n, np.nan)
        denom = 1.0 + (z * z) / n
        center = (p + (z * z) / (2.0 * n)) / denom
        half = (z * np.sqrt(p * (1.0 - p) / n + (z * z) / (4.0 * n * n))) / denom
    lo = np.clip(center - half, 0.0, 1.0)
    hi = np.clip(center + half, 0.0, 1.0)
    # FP guarantees: lo ≤ p ≤ hi (rounding can otherwise produce
    # ``hi = 0.9999…8`` against ``p = 1.0``, which breaks matplotlib
    # errorbars with negative half-widths).
    lo = np.minimum(lo, p)
    hi = np.maximum(hi, p)
    lo = np.where(n > 0, lo, np.nan)
    hi = np.where(n > 0, hi, np.nan)
    return lo, hi


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_pipeline(cfg: CutAcceptanceConfig, *, seed: int = 0) -> PipelineSummary:
    """Train the CNP and save the checkpoint. No scoring / coverage."""
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Build the training sampler.
    train_e, train_s = load_events(
        cfg.train_predictions_path,
        target_class=cfg.target_class,
        energy_range=cfg.energy_range,
    )
    sampler = EventSampler(
        train_e, train_s,
        energy_range=cfg.energy_range,
        energy_bin_width=cfg.energy_bin_width,
        threshold_range=cfg.threshold_range,
        min_events_per_bin=cfg.min_events_per_bin,
        t_sampling="boundary_mix",
    )
    np.savez(
        out_dir / "training_pool.npz",
        bin_centers=sampler.bin_centers,
        bin_event_counts=sampler.bin_event_counts,
        n_events_total=np.int64(train_e.size),
    )

    # 2. Train the CNP — EVENT_ONLY with per-event (E_i, T) in phi.
    torch.manual_seed(cfg.training.seed)
    cnp = build_cnp(
        cfg.encoder, dim_theta=None, dim_phi=2,
        decoder_hidden_dims=list(cfg.decoder_hidden_dims),
    )
    history = train_cnp(cnp, sampler, cnp_config=cfg.cnp, training_config=cfg.training)
    save_checkpoint(
        out_dir / "cnp.ckpt",
        cnp,
        encoder_config=cfg.encoder,
        dim_theta=None,
        dim_phi=2,
        history=history,
        metadata={
            "name": cfg.name,
            "target_class": cfg.target_class,
            "energy_bin_width": cfg.energy_bin_width,
            "train_predictions_path": str(cfg.train_predictions_path),
            "validation_predictions_path": str(cfg.validation_predictions_path),
            "decoder_hidden_dims": list(cfg.decoder_hidden_dims),
            "input_mode": "event_only",
            "sampling_strategy": "bin_stratified",
        },
    )
    final_train_loss = float(history["loss"][-1]) if history.get("loss") else float("nan")

    # 3. Youden-J best T* from the test labels — recorded once here so
    #    downstream diagnostics on the same (model, class) share a
    #    reference threshold without having to recompute it.
    with h5py.File(cfg.validation_predictions_path, "r") as f:
        val_score = f["score"][:].astype(np.float64)
        val_label = f["label"][:].astype(np.int64)
        val_energy = f["energy"][:].astype(np.float64)
    fpr, tpr, thr = roc_curve(val_label, val_score)
    T_star = float(thr[int(np.argmax(tpr - fpr))])

    # Count validation events that pass the (class, energy_range) filter.
    if cfg.target_class == "all":
        cls_mask = np.ones_like(val_label, dtype=bool)
    else:
        cls_mask = val_label == int(cfg.target_class)
    e_lo, e_hi = cfg.energy_range
    keep = cls_mask & (val_energy >= e_lo) & (val_energy <= e_hi)
    n_validation_events = int(keep.sum())

    summary = PipelineSummary(
        name=cfg.name,
        target_class=cfg.target_class,
        energy_bin_width=cfg.energy_bin_width,
        out_dir=str(out_dir),
        cnp_ckpt=str(out_dir / "cnp.ckpt"),
        n_train_events=int(train_e.size),
        n_validation_events=n_validation_events,
        n_bins_used=int(sampler.n_bins),
        cnp_final_train_loss=final_train_loss,
        youden_T_star=T_star,
    )
    summary.to_json(out_dir / "run_summary.json")
    return summary
