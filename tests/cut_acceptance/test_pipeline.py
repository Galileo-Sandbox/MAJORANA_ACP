"""End-to-end smoke test for the binned-CNP ``run_pipeline``.

Writes synthetic train / test predictions.h5 files, runs the full
pipeline with shrunken hyperparameters (~10 s on CPU), and verifies
every promised artifact exists, has the right shape, and round-trips.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest
from schemas.config import CNPConfig, EncoderConfig, TrainingConfig

from majorana_acp.cut_acceptance.config import CutAcceptanceConfig
from majorana_acp.cut_acceptance.pipeline import run_pipeline


def _write_synthetic_predictions(path: Path, *, n_total: int = 1500, seed: int = 0) -> None:
    """Mimic majorana_acp.cli.evaluate output: HDF5 with energy/score/label/etc."""
    rng = np.random.default_rng(seed)
    energy = rng.uniform(500.0, 3000.0, size=n_total)
    # Bimodal scores: ~half near 0, ~half near 1 (mirrors real classifier output).
    label = (rng.random(size=n_total) < 0.5).astype(np.int8)
    score = np.where(
        label == 1,
        rng.beta(8, 1, size=n_total),
        rng.beta(1, 8, size=n_total),
    )

    with h5py.File(path, "w") as f:
        f.create_dataset("energy", data=energy.astype(np.float32))
        f.create_dataset("score", data=score.astype(np.float32))
        f.create_dataset("label", data=label.astype(bool))
        f.create_dataset(
            "logit",
            data=np.log(np.clip(score, 1e-6, 1 - 1e-6) / (1 - np.clip(score, 1e-6, 1 - 1e-6))),
        )
        f.create_dataset("id", data=np.arange(n_total, dtype=np.int64))
        f.create_dataset("detector", data=np.zeros(n_total, dtype=np.int64))
        f.create_dataset("run_number", data=np.zeros(n_total, dtype=np.int64))
        f.create_dataset("tp0", data=np.zeros(n_total, dtype=np.int64))


def _fast_cfg(
    train_path: Path,
    val_path: Path,
    out_dir: Path,
    target_class: int | str = 1,
) -> CutAcceptanceConfig:
    """Tiny config so the smoke test completes in a few seconds."""
    return CutAcceptanceConfig(
        name="pipeline_smoke",
        train_predictions_path=train_path,
        validation_predictions_path=val_path,
        out_dir=out_dir,
        target_class=target_class,
        energy_bin_width=100.0,  # 25 bins across [500, 3000]
        n_per_trial=8,
        min_events_per_bin=2,
        encoder=EncoderConfig(type="mlp", latent_dim=8, hidden_dims=[16], dropout=0.0),
        cnp=CNPConfig(
            n_context_min=2, n_context_max=4, output_activation="sigmoid", mixup_alpha=0.01
        ),
        training=TrainingConfig(
            n_steps=20,
            learning_rate=1e-3,
            batch_size=4,
            n_events_per_trial=8,
            n_mc_samples=2,
            grad_clip=1.0,
            eval_every=0,
            seed=0,
        ),
        decoder_hidden_dims=[16, 16],
    )


@pytest.mark.slow
def test_run_pipeline_end_to_end(tmp_path: Path) -> None:
    train = tmp_path / "train.h5"
    val = tmp_path / "test.h5"
    out = tmp_path / "out"
    _write_synthetic_predictions(train, seed=0)
    _write_synthetic_predictions(val, seed=1)
    cfg = _fast_cfg(train, val, out, target_class=1)

    summary = run_pipeline(cfg, seed=0)

    assert summary.name == "pipeline_smoke"
    assert summary.target_class == 1
    assert summary.n_train_events > 0
    assert summary.n_validation_events > 0
    assert summary.n_bins_used >= 5
    assert 0.0 <= summary.youden_T_star <= 1.0

    # Every promised artifact exists.
    for name in (
        "cnp.ckpt",
        "training_pool.npz",
        "validation_binned.npz",
        "cnp_predictions.npz",
        "run_summary.json",
    ):
        assert (out / name).is_file(), f"missing artifact: {name}"

    # run_summary.json round-trips cleanly.
    parsed = json.loads((out / "run_summary.json").read_text())
    assert parsed["name"] == summary.name
    assert parsed["n_bins_used"] == summary.n_bins_used

    # cnp_predictions.npz has matching grid + slice + uncertainty.
    preds = np.load(out / "cnp_predictions.npz")
    grid_shape = (preds["energy_grid"].size, preds["threshold_grid"].size)
    assert preds["beta_grid"].shape == grid_shape
    assert preds["beta_std_grid"].shape == grid_shape
    assert preds["beta_at_T_star"].shape == preds["energy_grid"].shape
    assert preds["beta_std_at_T_star"].shape == preds["energy_grid"].shape
    assert np.all(np.isfinite(preds["beta_grid"]))
    # CNP std is non-negative and bounded by 0.5 (β ∈ [0,1] sigma can't exceed that).
    assert np.all(preds["beta_std_grid"] >= 0.0)
    assert np.all(preds["beta_std_grid"] <= 0.5)

    # Coverage values are fractions in [0, 1].
    for v in (summary.coverage_1sigma, summary.coverage_2sigma, summary.coverage_3sigma):
        assert 0.0 <= v <= 1.0

    # validation_binned.npz has same E grid; bins below min_events_per_bin are NaN.
    val_arr = np.load(out / "validation_binned.npz")
    assert val_arr["bin_centers"].shape == preds["energy_grid"].shape


@pytest.mark.slow
def test_pipeline_handles_inclusive_class(tmp_path: Path) -> None:
    train = tmp_path / "train.h5"
    val = tmp_path / "test.h5"
    _write_synthetic_predictions(train, seed=2)
    _write_synthetic_predictions(val, seed=3)
    cfg = _fast_cfg(train, val, tmp_path / "inc", target_class="all")
    summary = run_pipeline(cfg, seed=0)
    assert summary.target_class == "all"
