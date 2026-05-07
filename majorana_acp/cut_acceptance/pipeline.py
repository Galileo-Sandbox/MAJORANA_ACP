"""End-to-end cut-acceptance pipeline: partition → CNP → MFGP → coverage → heatmap.

One ``run_pipeline(cfg)`` call performs all of:

1. Load ``predictions.h5`` and partition events into LF / HF train / HF
   holdout (see :mod:`majorana_acp.cut_acceptance.partition`).
2. Train a RESUM_FLEX CNP on the LF pool — LF has broad energy
   coverage so the CNP learns ``β(E, T)`` over the full plane. The CNP
   is fidelity-invariant by design and is later applied to HF events
   too without retraining.
3. Build LF and HF ``StandardBatch`` objects via the nearest-neighbor
   ``DesignOnlySampler`` (theta sampled continuously over the configured
   boxes). The HF sampler is restricted to the bounding-box of the peak
   windows so HF trials don't waste budget in regions where HF has no
   events.
4. ``prepare_mfgp_datasets_from_batches`` produces the three fidelity
   tiers (``Y_lf_cnp``, ``Y_hf_cnp``, ``Y_hf_raw``) and
   ``fit_mfgp_three_fidelity`` builds the co-kriging MFGP.
5. Run ``evaluate_mfgp_coverage_from_batch`` on the held-out HF events.
6. Predict on a fine ``(E, T)`` grid for the final heatmap and save all
   artifacts.

Outputs go to ``cfg.out_dir``:

* ``cnp.ckpt`` — RESUM_FLEX CNP checkpoint (model + history + metadata).
* ``mfgp.pkl`` — pickled fitted MFGP.
* ``coverage.json`` — coverage table at ±1σ / ±2σ / ±3σ.
* ``heatmap.npz`` — predicted ``μ``, ``σ`` on the eval grid (for plots).
* ``mfgp_data.npz`` — the per-trial datasets that fed the MFGP, for
  inspection / re-running fits without regenerating CNP context sets.
* ``run_summary.json`` — a single-page summary of the run (paths +
  scalar metrics) so a notebook can ingest it without re-importing
  the pipeline.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from core import (
    build_cnp,
    evaluate_mfgp_coverage_from_batch,
    fit_mfgp_three_fidelity,
    prepare_mfgp_datasets_from_batches,
    save_checkpoint,
    save_mfgp,
    train_cnp,
)

from majorana_acp.cut_acceptance.config import CutAcceptanceConfig
from majorana_acp.cut_acceptance.partition import PartitionedIndices, partition_events
from majorana_acp.cut_acceptance.sampler import DesignOnlySampler, load_event_pool


@dataclass(frozen=True)
class PipelineSummary:
    """Lightweight summary returned (and saved) by ``run_pipeline``."""

    name: str
    target_class: int
    out_dir: str
    cnp_ckpt: str
    mfgp_path: str
    n_lf: int
    n_hf_train: int
    n_hf_holdout: int
    n_mfgp_lf_trials: int
    n_mfgp_hf_trials: int
    cnp_final_train_loss: float
    coverage_1sigma: float
    coverage_2sigma: float
    coverage_3sigma: float
    holdout_n_test: int

    def to_json(self, path: Path | str) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))


def _load_predictions(path: Path | str) -> tuple[np.ndarray, np.ndarray]:
    """Read (energy, label) from a classifier's predictions.h5."""
    with h5py.File(Path(path), "r") as f:
        energy = f["energy"][:]
        label = f["label"][:].astype(np.int64)
    return energy, label


def _hf_energy_box(cfg: CutAcceptanceConfig) -> tuple[float, float]:
    """Bounding box of the peak windows — used as the HF sampler's energy_box.

    HF has no events outside peak regions, so sampling theta_E uniformly
    over the bounding box at least concentrates trials near the actual
    HF support. (Future improvement: sample from the *union* of peak
    windows directly to avoid the inter-peak gaps.)
    """
    return (
        float(min(w.lo for w in cfg.peak_windows)),
        float(max(w.hi for w in cfg.peak_windows)),
    )


def _build_eval_grid(cfg: CutAcceptanceConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(theta_query, energy_grid, threshold_grid)`` for the heatmap."""
    e_grid = np.arange(
        cfg.energy_range[0], cfg.energy_range[1] + 0.5 * cfg.energy_grid_step, cfg.energy_grid_step
    )
    t_grid = np.arange(
        cfg.threshold_range[0],
        cfg.threshold_range[1] + 0.5 * cfg.threshold_grid_step,
        cfg.threshold_grid_step,
    )
    ee, tt = np.meshgrid(e_grid, t_grid, indexing="ij")
    theta_query = np.stack([ee.ravel(), tt.ravel()], axis=-1).astype(np.float64)
    return theta_query, e_grid, t_grid


def _save_partitioned(out_dir: Path, p: PartitionedIndices, cfg: CutAcceptanceConfig) -> None:
    """Persist the LF/HF/holdout indices so a notebook can reconstruct the split."""
    np.savez(
        out_dir / "partition.npz",
        base_mask=p.base_mask,
        lf=p.lf,
        hf_train=p.hf_train,
        hf_holdout=p.hf_holdout,
        partition_seed=cfg.partition_seed,
        target_class=cfg.target_class,
    )


def _save_mfgp_data(out_dir: Path, data: dict[str, np.ndarray]) -> None:
    np.savez(out_dir / "mfgp_data.npz", **data)


def run_pipeline(
    cfg: CutAcceptanceConfig,
    *,
    n_mfgp_lf_trials: int = 200,
    n_mfgp_hf_trials: int = 100,
    seed: int = 0,
) -> PipelineSummary:
    """End-to-end run; see module docstring for outputs.

    ``n_mfgp_lf_trials`` / ``n_mfgp_hf_trials`` control how many ``θ_k``
    samples feed the MFGP. They are independent of CNP training, which
    iterates on freshly resampled batches each step per
    ``cfg.training``.
    """
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load + partition.
    energy, label = _load_predictions(cfg.predictions_path)
    p = partition_events(energy, label, cfg)
    if p.lf.size < cfg.min_pool_size:
        raise ValueError(f"LF pool has {p.lf.size} events, below min_pool_size={cfg.min_pool_size}")
    if p.hf_train.size < cfg.min_pool_size:
        raise ValueError(
            f"HF train pool has {p.hf_train.size} events, below min_pool_size={cfg.min_pool_size}"
        )
    _save_partitioned(out_dir, p, cfg)

    # 2. Pools.
    lf_pool = load_event_pool(cfg.predictions_path, p.lf, base_mask=p.base_mask)
    hf_train_pool = load_event_pool(cfg.predictions_path, p.hf_train, base_mask=p.base_mask)
    hf_holdout_pool = load_event_pool(cfg.predictions_path, p.hf_holdout, base_mask=p.base_mask)

    # 3. Train the CNP on the LF pool (broadest energy coverage).
    torch.manual_seed(cfg.training.seed)
    lf_sampler_train = DesignOnlySampler(
        lf_pool, cfg.energy_range, cfg.threshold_range, cfg.n_per_trial_lf
    )
    cnp = build_cnp(cfg.encoder, dim_theta=2, dim_phi=None)
    history = train_cnp(cnp, lf_sampler_train, cnp_config=cfg.cnp, training_config=cfg.training)
    cnp_ckpt_path = out_dir / "cnp.ckpt"
    save_checkpoint(
        cnp_ckpt_path,
        cnp,
        encoder_config=cfg.encoder,
        dim_theta=2,
        dim_phi=None,
        history=history,
        metadata={
            "name": cfg.name,
            "target_class": cfg.target_class,
            "predictions_path": str(cfg.predictions_path),
            "partition_seed": cfg.partition_seed,
            "n_lf": int(p.lf.size),
        },
    )
    # RESUM_FLEX history dict uses key "loss" for the running CNP NLL.
    final_train_loss = float(history["loss"][-1]) if history.get("loss") else float("nan")

    # 4. Build the LF + HF batches that feed prepare_mfgp_datasets_from_batches.
    hf_energy_box = _hf_energy_box(cfg)
    lf_sampler_mfgp = DesignOnlySampler(
        lf_pool, cfg.energy_range, cfg.threshold_range, cfg.n_per_trial_lf
    )
    hf_sampler_mfgp = DesignOnlySampler(
        hf_train_pool, hf_energy_box, cfg.threshold_range, cfg.n_per_trial_hf
    )
    lf_batch = lf_sampler_mfgp.generate(
        n_trials=n_mfgp_lf_trials, n_events=cfg.n_per_trial_lf, seed=seed * 1000 + 1
    )
    hf_batch = hf_sampler_mfgp.generate(
        n_trials=n_mfgp_hf_trials, n_events=cfg.n_per_trial_hf, seed=seed * 1000 + 2
    )

    # 5. MFGP prep + fit.
    data = prepare_mfgp_datasets_from_batches(
        cnp, lf_batch, hf_batch, n_mc_samples=50, seed=seed * 1000 + 3
    )
    _save_mfgp_data(out_dir, data)
    mfgp = fit_mfgp_three_fidelity(data, kernel=cfg.mfgp.kernel, n_restarts=5)
    mfgp_path = out_dir / "mfgp.pkl"
    save_mfgp(mfgp_path, mfgp)

    # 6. Coverage on the held-out HF events.
    holdout_sampler = DesignOnlySampler(
        hf_holdout_pool, hf_energy_box, cfg.threshold_range, cfg.n_per_trial_hf
    )
    holdout_batch = holdout_sampler.generate(
        n_trials=n_mfgp_hf_trials, n_events=cfg.n_per_trial_hf, seed=seed * 1000 + 4
    )
    coverage = evaluate_mfgp_coverage_from_batch(mfgp, cnp, holdout_batch, seed=seed * 1000 + 5)
    cov_payload = {
        "1sigma": float(coverage["1sigma"]),
        "2sigma": float(coverage["2sigma"]),
        "3sigma": float(coverage["3sigma"]),
        "n_test": int(coverage["y_obs"].shape[0]),
    }
    (out_dir / "coverage.json").write_text(json.dumps(cov_payload, indent=2))
    # Also persist the per-point arrays so the notebook can plot without
    # rerunning evaluate_mfgp_coverage_from_batch.
    np.savez(
        out_dir / "coverage.npz",
        theta=coverage["theta"],
        y_obs=coverage["y_obs"],
        mu=coverage["mu"],
        sigma=coverage["sigma"],
    )

    # 7. Predict on the eval grid.
    theta_query, e_grid, t_grid = _build_eval_grid(cfg)
    mu, var = mfgp.predict(theta_query)
    mu_grid = mu.reshape((e_grid.size, t_grid.size))
    sigma_grid = np.sqrt(np.maximum(var, 0.0)).reshape((e_grid.size, t_grid.size))
    np.savez(
        out_dir / "heatmap.npz",
        energy_grid=e_grid,
        threshold_grid=t_grid,
        mu=mu_grid,
        sigma=sigma_grid,
    )

    summary = PipelineSummary(
        name=cfg.name,
        target_class=cfg.target_class,
        out_dir=str(out_dir),
        cnp_ckpt=str(cnp_ckpt_path),
        mfgp_path=str(mfgp_path),
        n_lf=int(p.lf.size),
        n_hf_train=int(p.hf_train.size),
        n_hf_holdout=int(p.hf_holdout.size),
        n_mfgp_lf_trials=int(n_mfgp_lf_trials),
        n_mfgp_hf_trials=int(n_mfgp_hf_trials),
        cnp_final_train_loss=final_train_loss,
        coverage_1sigma=cov_payload["1sigma"],
        coverage_2sigma=cov_payload["2sigma"],
        coverage_3sigma=cov_payload["3sigma"],
        holdout_n_test=cov_payload["n_test"],
    )
    summary.to_json(out_dir / "run_summary.json")
    return summary
