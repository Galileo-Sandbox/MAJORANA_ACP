"""End-to-end smoke test for ``run_pipeline`` on synthetic predictions.h5.

Slow-ish (CNP training + MFGP fit), but kept short (~10 s on CPU) by
shrinking n_steps, batch_size, n_per_trial, and the MFGP-prep budget.
We do not verify accuracy here — only that the pipeline executes,
saves every artifact, and the saved files round-trip cleanly.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest
from schemas.config import CNPConfig, EncoderConfig, TrainingConfig

from majorana_acp.cut_acceptance.config import (
    CutAcceptanceConfig,
    PeakSplit,
    PeakWindow,
)
from majorana_acp.cut_acceptance.pipeline import run_pipeline


def _write_synthetic_predictions(path: Path, *, n_total: int = 600, seed: int = 0) -> None:
    """Mimic majorana_acp.cli.evaluate output: HDF5 with energy/score/label/etc."""
    rng = np.random.default_rng(seed)
    # 70% continuum, 30% in two peak windows so HF tiers have enough events.
    n_cont = int(0.7 * n_total)
    n_peak = n_total - n_cont
    energy = np.concatenate(
        [
            rng.uniform(700.0, 2500.0, size=n_cont),
            rng.uniform(2605.0, 2620.0, size=n_peak // 2),
            rng.uniform(1587.0, 1597.0, size=n_peak - n_peak // 2),
        ]
    )
    # Scores trend higher inside peak windows so the CNP has a real signal to learn.
    score = rng.beta(2, 5, size=n_total)
    in_peak = ((energy >= 2605) & (energy <= 2620)) | ((energy >= 1587) & (energy <= 1597))
    score = np.where(in_peak, np.clip(score + 0.4, 0, 1), score)
    label = (rng.random(size=n_total) < 0.5).astype(np.int8)
    perm = rng.permutation(n_total)

    with h5py.File(path, "w") as f:
        f.create_dataset("energy", data=energy[perm])
        f.create_dataset("score", data=score[perm])
        f.create_dataset("label", data=label[perm])
        # Other fields evaluator emits — included so tests reflect real schema.
        f.create_dataset("logit", data=np.log(score[perm] / (1 - score[perm] + 1e-9)))
        f.create_dataset("id", data=np.arange(n_total))
        f.create_dataset("detector", data=np.zeros(n_total, dtype=np.int32))
        f.create_dataset("run_number", data=np.zeros(n_total, dtype=np.int32))
        f.create_dataset("tp0", data=np.zeros(n_total, dtype=np.int32))


def _fast_cfg(predictions_path: Path, out_dir: Path, target_class: int = 1) -> CutAcceptanceConfig:
    """A CutAcceptanceConfig small enough to run end-to-end in a few seconds."""
    return CutAcceptanceConfig(
        name="pipeline_smoke",
        predictions_path=predictions_path,
        out_dir=out_dir,
        target_class=target_class,
        peak_windows=[
            PeakWindow(lo=2605.0, hi=2620.0),
            PeakWindow(lo=1587.0, hi=1597.0),
        ],
        peak_split=PeakSplit(lf=0.10, hf_train=0.30, hf_holdout=0.60),
        n_per_trial_lf=16,
        n_per_trial_hf=8,
        min_pool_size=4,
        encoder=EncoderConfig(type="mlp", latent_dim=8, hidden_dims=[16], dropout=0.0),
        cnp=CNPConfig(
            n_context_min=4, n_context_max=8, output_activation="sigmoid", mixup_alpha=0.1
        ),
        training=TrainingConfig(
            n_steps=20,
            learning_rate=1e-3,
            batch_size=4,
            n_events_per_trial=16,
            n_mc_samples=2,
            grad_clip=1.0,
            eval_every=0,
            seed=0,
        ),
    )


@pytest.mark.slow
def test_run_pipeline_end_to_end(tmp_path: Path) -> None:
    pred = tmp_path / "predictions.h5"
    out = tmp_path / "out"
    _write_synthetic_predictions(pred)
    cfg = _fast_cfg(pred, out, target_class=1)

    summary = run_pipeline(cfg, n_mfgp_lf_trials=10, n_mfgp_hf_trials=6, seed=0)

    assert summary.name == "pipeline_smoke"
    assert summary.target_class == 1
    assert summary.n_lf > 0
    assert summary.n_hf_train >= 4
    assert summary.n_hf_holdout >= 4
    # Coverage values are fractions in [0, 1].
    for v in (summary.coverage_1sigma, summary.coverage_2sigma, summary.coverage_3sigma):
        assert 0.0 <= v <= 1.0

    # Every promised artifact exists.
    for name in (
        "cnp.ckpt",
        "mfgp.pkl",
        "coverage.json",
        "heatmap.npz",
        "mfgp_data.npz",
        "partition.npz",
        "run_summary.json",
    ):
        assert (out / name).is_file(), f"missing artifact: {name}"

    # run_summary.json is valid JSON and matches the returned dataclass.
    parsed = json.loads((out / "run_summary.json").read_text())
    assert parsed["name"] == summary.name
    assert parsed["target_class"] == summary.target_class

    # Heatmap grid has the expected shape and finite μ.
    grid = np.load(out / "heatmap.npz")
    assert grid["mu"].ndim == 2
    assert grid["sigma"].ndim == 2
    assert grid["mu"].shape == grid["sigma"].shape
    assert np.all(np.isfinite(grid["mu"]))


@pytest.mark.slow
def test_run_pipeline_signal_and_background_have_disjoint_pools(tmp_path: Path) -> None:
    pred = tmp_path / "predictions.h5"
    _write_synthetic_predictions(pred, n_total=800, seed=11)
    out_sig = tmp_path / "sig"
    out_bkg = tmp_path / "bkg"
    cfg_sig = _fast_cfg(pred, out_sig, target_class=1)
    cfg_bkg = _fast_cfg(pred, out_bkg, target_class=0)

    run_pipeline(cfg_sig, n_mfgp_lf_trials=10, n_mfgp_hf_trials=6, seed=0)
    run_pipeline(cfg_bkg, n_mfgp_lf_trials=10, n_mfgp_hf_trials=6, seed=0)

    sig_part = np.load(out_sig / "partition.npz")
    bkg_part = np.load(out_bkg / "partition.npz")
    assert not np.any(sig_part["base_mask"] & bkg_part["base_mask"])
