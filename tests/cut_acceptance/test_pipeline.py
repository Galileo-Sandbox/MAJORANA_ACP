"""Smoke test for the binned-CNP ``run_pipeline``.

Writes synthetic train / test predictions.h5 files, runs the pipeline
with shrunken hyperparameters (~5 s on CPU), and verifies the
training artifacts exist and round-trip. Coverage / scoring lives in
the diagnostic script now (tested separately).
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
    upstream_path: Path | None = None,
) -> CutAcceptanceConfig:
    return CutAcceptanceConfig(
        name="pipeline_smoke",
        train_predictions_path=train_path,
        validation_predictions_path=val_path,
        out_dir=out_dir,
        upstream_classifier_config=upstream_path or Path("dummy.yaml"),
        target_class=target_class,
        energy_bin_width=100.0,
        min_events_per_bin=2,
        encoder=EncoderConfig(type="mlp", latent_dim=8, hidden_dims=[16], dropout=0.1),
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
        mc_dropout_samples=4,
    )


@pytest.mark.slow
def test_run_pipeline_end_to_end(tmp_path: Path) -> None:
    train = tmp_path / "train.h5"
    val = tmp_path / "test.h5"
    out = tmp_path / "out"
    upstream = tmp_path / "stub_upstream.yaml"
    upstream.write_text("name: stub_upstream_classifier\n")
    _write_synthetic_predictions(train, seed=0)
    _write_synthetic_predictions(val, seed=1)
    cfg = _fast_cfg(train, val, out, target_class=1, upstream_path=upstream)

    summary = run_pipeline(cfg, seed=0)

    assert summary.name == "pipeline_smoke"
    assert summary.target_class == 1
    assert summary.n_train_events > 0
    assert summary.n_validation_events > 0
    assert summary.n_bins_used >= 5
    assert 0.0 <= summary.youden_T_star <= 1.0

    # Only the training-side artifacts exist now (no scoring outputs).
    for name in ("cnp.ckpt", "training_pool.npz", "run_summary.json"):
        assert (out / name).is_file(), f"missing artifact: {name}"
    for absent in ("validation_binned.npz", "cnp_predictions.npz"):
        assert not (out / absent).is_file(), (
            f"{absent} should no longer be produced by the pipeline"
        )

    parsed = json.loads((out / "run_summary.json").read_text())
    assert parsed["name"] == summary.name
    assert parsed["n_bins_used"] == summary.n_bins_used
    assert "coverage_cnp_1sigma" not in parsed  # legacy field removed
    # Lineage fields present + sha256 matches the stub file's content.
    import hashlib

    expected_sha = hashlib.sha256(upstream.read_bytes()).hexdigest()
    assert parsed["upstream_classifier_config"] == str(upstream)
    assert parsed["upstream_classifier_sha256"] == expected_sha

    pool = np.load(out / "training_pool.npz")
    assert pool["bin_centers"].ndim == 1
    assert pool["bin_event_counts"].shape == pool["bin_centers"].shape

    # Checkpoint must record the new EVENT_ONLY mode (dim_phi=2, dim_theta=None)
    # and the hybrid-scale fingerprint (default = baseline True-CNP).
    import torch

    state = torch.load(out / "cnp.ckpt", map_location="cpu", weights_only=False)
    assert state["dim_theta"] is None
    assert state["dim_phi"] == 2
    assert state["metadata"]["input_mode"] == "event_only"
    assert state["metadata"]["sampling_pattern"] == "flat_stratified"
    assert state["metadata"]["trial_size_strategy"] == "fixed"
    assert state["metadata"]["paradigm_path_suffix"] == "true_cnp"


@pytest.mark.slow
def test_pipeline_variable_n_and_auto_derived_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``variable_uniform`` trains via the wrapper and out_dir is auto-derived.

    Chdir into ``tmp_path`` so the auto-derived ``results/...`` path
    lands inside the temp dir rather than polluting the real
    ``results/`` tree under the repo.
    """
    monkeypatch.chdir(tmp_path)
    train = tmp_path / "train.h5"
    val = tmp_path / "test.h5"
    upstream = tmp_path / "my_classifier.yaml"
    upstream.write_text("name: stub_upstream\n")
    _write_synthetic_predictions(train, seed=4)
    _write_synthetic_predictions(val, seed=5)

    cfg = CutAcceptanceConfig(
        train_predictions_path=train,
        validation_predictions_path=val,
        upstream_classifier_config=upstream,
        target_class=1,
        energy_bin_width=100.0,
        min_events_per_bin=2,
        encoder=EncoderConfig(type="mlp", latent_dim=8, hidden_dims=[16], dropout=0.1),
        cnp=CNPConfig(
            n_context_min=2,
            n_context_max=6,
            output_activation="sigmoid",
            mixup_alpha=0.01,
        ),
        training=TrainingConfig(
            n_steps=15,
            learning_rate=1e-3,
            batch_size=4,
            n_events_per_trial=8,  # ignored under variable_uniform
            n_mc_samples=2,
            grad_clip=1.0,
            eval_every=0,
            seed=0,
        ),
        decoder_hidden_dims=[16, 16],
        mc_dropout_samples=4,
        # Hybrid-scale knobs.
        sampling_pattern="mixed_density",
        zoom_window_width_kev=300.0,
        local_event_fraction=0.7,
        trial_size_strategy="variable_uniform",
        n_trial_events_min=8,
        n_trial_events_max=12,
        # out_dir / name left blank — pipeline auto-derives.
    )
    summary = run_pipeline(cfg, seed=0)

    # Path derivation lands at the canonical hybrid_scale tree, with
    # the variable-N suffix appended.
    expected_suffix = "hybrid_scale/mixed_density_f0_70_w300_varN8-12"
    assert expected_suffix in summary.out_dir.replace("\\", "/")
    assert "__" + expected_suffix.replace("/", "__") in summary.name
    parsed = json.loads((Path(summary.out_dir) / "run_summary.json").read_text())
    assert parsed["sampling_pattern"] == "mixed_density"
    assert parsed["trial_size_strategy"] == "variable_uniform"
    assert parsed["paradigm_path_suffix"] == expected_suffix
    assert parsed["upstream_classifier_sha256"]


@pytest.mark.slow
def test_pipeline_handles_inclusive_class(tmp_path: Path) -> None:
    train = tmp_path / "train.h5"
    val = tmp_path / "test.h5"
    upstream = tmp_path / "stub_upstream.yaml"
    upstream.write_text("name: stub_upstream_classifier\n")
    _write_synthetic_predictions(train, seed=2)
    _write_synthetic_predictions(val, seed=3)
    cfg = _fast_cfg(
        train,
        val,
        tmp_path / "inc",
        target_class="all",
        upstream_path=upstream,
    )
    summary = run_pipeline(cfg, seed=0)
    assert summary.target_class == "all"
