"""Tests for ``majorana_acp.eval.evaluator``."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from majorana_acp.eval.evaluator import (
    compute_metrics,
    evaluate,
    load_checkpoint,
    run_inference,
)
from majorana_acp.training.config import (
    DataConfig,
    ExperimentConfig,
    LossConfig,
    ModelConfig,
    OptimConfig,
    TrainConfig,
)
from majorana_acp.training.trainer import train


def _trained_run(data_dir: Path, out_dir: Path) -> Path:
    """Train a tiny model on the synthetic data and return the run dir."""
    cfg = ExperimentConfig(
        name="eval_test",
        data=DataConfig(
            data_dir=data_dir,
            train_file_indices=[0, 1, 2],
            test_file_indices=[0],
            target_label="psd_label_low_avse",
            batch_size=4,
            num_workers=0,
            baseline_samples=500,
        ),
        model=ModelConfig(
            name="simple_cnn",
            params={"channels": [4, 8], "kernel_size": 3, "dropout": 0.0},
        ),
        optim=OptimConfig(),
        loss=LossConfig(type="bce"),
        train=TrainConfig(
            out_dir=out_dir,
            epochs=1,
            device="cpu",
            amp=False,
            log_every_n_steps=1,
        ),
    )
    return train(cfg)


# --- compute_metrics --------------------------------------------------


def test_compute_metrics_basic_shape() -> None:
    rng = np.random.default_rng(0)
    n = 100
    labels = (rng.random(n) > 0.5).astype(bool)
    scores = rng.random(n).astype(np.float32)
    logits = np.log(scores / (1 - scores)).astype(np.float32)

    m = compute_metrics(
        {
            "logit": logits,
            "score": scores,
            "label": labels,
            "energy": rng.random(n),
            "tp0": rng.integers(0, 100, n),
            "detector": rng.integers(0, 5, n),
            "run_number": np.zeros(n, dtype=np.int64),
            "id": np.arange(n),
        }
    )
    assert m["n_events"] == n
    assert m["n_positive"] + m["n_negative"] == n
    assert 0.0 <= m["roc_auc"] <= 1.0
    assert 0.0 <= m["accuracy_at_0.5"] <= 1.0


def test_compute_metrics_perfect_classifier() -> None:
    """If scores match labels exactly, ROC-AUC should be 1.0."""
    labels = np.array([0, 0, 1, 1], dtype=bool)
    scores = np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float32)
    m = compute_metrics(
        {
            "logit": np.zeros_like(scores),
            "score": scores,
            "label": labels,
            "energy": np.zeros(4),
            "tp0": np.zeros(4, dtype=np.int64),
            "detector": np.zeros(4, dtype=np.int64),
            "run_number": np.zeros(4, dtype=np.int64),
            "id": np.arange(4),
        }
    )
    assert m["roc_auc"] == pytest.approx(1.0)


def test_compute_metrics_single_class_returns_none_auc() -> None:
    labels = np.ones(10, dtype=bool)
    m = compute_metrics(
        {
            "logit": np.zeros(10, dtype=np.float32),
            "score": np.full(10, 0.7, dtype=np.float32),
            "label": labels,
            "energy": np.zeros(10),
            "tp0": np.zeros(10, dtype=np.int64),
            "detector": np.zeros(10, dtype=np.int64),
            "run_number": np.zeros(10, dtype=np.int64),
            "id": np.arange(10),
        }
    )
    assert m["roc_auc"] is None


# --- load_checkpoint --------------------------------------------------


def test_load_checkpoint_from_file(tiny_train_dir: Path, tmp_path: Path) -> None:
    run_dir = _trained_run(tiny_train_dir, tmp_path / "run")
    ckpt_file = run_dir / "epoch_001.pt"

    model, cfg, meta = load_checkpoint(ckpt_file)
    assert isinstance(model, torch.nn.Module)
    assert cfg.model.name == "simple_cnn"
    assert meta["epoch"] == 1


def test_load_checkpoint_from_dir_picks_latest(tiny_train_dir: Path, tmp_path: Path) -> None:
    """Pointing at a run dir should auto-select the latest epoch."""
    cfg = ExperimentConfig(
        name="multi",
        data=DataConfig(
            data_dir=tiny_train_dir,
            train_file_indices=[0, 1, 2],
            test_file_indices=[0],
            batch_size=4,
            num_workers=0,
        ),
        model=ModelConfig(
            name="simple_cnn",
            params={"channels": [4], "kernel_size": 3, "dropout": 0.0},
        ),
        optim=OptimConfig(),
        loss=LossConfig(type="bce"),
        train=TrainConfig(
            out_dir=tmp_path / "run",
            epochs=2,
            device="cpu",
            amp=False,
            log_every_n_steps=1,
        ),
    )
    run_dir = train(cfg)

    _, _, meta = load_checkpoint(run_dir)
    assert meta["epoch"] == 2  # latest


def test_load_checkpoint_missing_path_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "no_such_file.pt")


def test_load_checkpoint_empty_dir_raises(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        load_checkpoint(empty)


# --- run_inference ---------------------------------------------------


def test_run_inference_returns_aligned_arrays(tiny_train_dir: Path, tmp_path: Path) -> None:
    run_dir = _trained_run(tiny_train_dir, tmp_path / "run")
    model, cfg, _ = load_checkpoint(run_dir)

    from torch.utils.data import DataLoader

    from majorana_acp.data import DatasetConfig as DSCfg
    from majorana_acp.data import MajoranaWaveformDataset, resolve_files

    test_files = resolve_files(cfg.data.data_dir, "test", cfg.data.test_file_indices)
    ds = MajoranaWaveformDataset(DSCfg(files=test_files, target_label=cfg.data.target_label))
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)

    preds = run_inference(model, loader, torch.device("cpu"))

    n = len(preds["logit"])
    assert n == len(ds)
    # All output arrays should be the same length.
    for k in ("score", "label", "energy", "tp0", "detector", "run_number", "id"):
        assert len(preds[k]) == n, f"length mismatch for {k}"
    # Scores should be sigmoid of logits.
    expected_scores = 1.0 / (1.0 + np.exp(-preds["logit"]))
    np.testing.assert_allclose(preds["score"], expected_scores, atol=1e-6)


# --- evaluate (end-to-end) ------------------------------------------


def test_evaluate_writes_expected_artifacts(tiny_train_dir: Path, tmp_path: Path) -> None:
    run_dir = _trained_run(tiny_train_dir, tmp_path / "run")
    eval_dir = evaluate(run_dir)

    assert eval_dir == run_dir / "eval"
    assert (eval_dir / "metrics.json").is_file()
    assert (eval_dir / "predictions.h5").is_file()
    assert (eval_dir / "eval.log").is_file()


def test_evaluate_metrics_json_is_valid(tiny_train_dir: Path, tmp_path: Path) -> None:
    run_dir = _trained_run(tiny_train_dir, tmp_path / "run")
    eval_dir = evaluate(run_dir)

    metrics = json.loads((eval_dir / "metrics.json").read_text())
    assert metrics["n_events"] == 16  # test file has 16 events
    assert metrics["epoch"] == 1
    assert metrics["model_name"] == "simple_cnn"
    # roc_auc may be None if the test fixture happens to give a single class,
    # but with seed=99 and pass_rate=0.6 we have both classes.
    assert metrics["roc_auc"] is not None
    assert 0.0 <= metrics["roc_auc"] <= 1.0


def test_evaluate_predictions_h5_schema(tiny_train_dir: Path, tmp_path: Path) -> None:
    run_dir = _trained_run(tiny_train_dir, tmp_path / "run")
    eval_dir = evaluate(run_dir)

    with h5py.File(eval_dir / "predictions.h5", "r") as f:
        keys = set(f.keys())
        assert keys >= {
            "logit",
            "score",
            "label",
            "energy",
            "tp0",
            "detector",
            "run_number",
            "id",
        }
        n = f["logit"].shape[0]
        assert n == 16
        for k in keys:
            assert f[k].shape[0] == n
        # score should always be in [0, 1].
        scores = f["score"][:]
        assert scores.min() >= 0.0
        assert scores.max() <= 1.0


def test_evaluate_custom_out_dir_override(tiny_train_dir: Path, tmp_path: Path) -> None:
    run_dir = _trained_run(tiny_train_dir, tmp_path / "run")
    custom = tmp_path / "elsewhere"

    eval_dir = evaluate(run_dir, out_dir=custom)
    assert eval_dir == custom
    assert (custom / "metrics.json").is_file()


def test_evaluate_accepts_explicit_file(tiny_train_dir: Path, tmp_path: Path) -> None:
    """Pointing evaluate() at a specific .pt should work too."""
    run_dir = _trained_run(tiny_train_dir, tmp_path / "run")
    ckpt_file = run_dir / "epoch_001.pt"
    eval_dir = evaluate(ckpt_file)

    assert (eval_dir / "metrics.json").is_file()
