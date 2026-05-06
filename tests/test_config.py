"""Tests for ``majorana_acp.training.config``."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from majorana_acp.training.config import (
    DataConfig,
    ExperimentConfig,
    LossConfig,
    ModelConfig,
    OptimConfig,
    TrainConfig,
    load_config,
)


def _minimal_yaml(data_dir: Path, out_dir: Path) -> str:
    """A complete, valid experiment YAML that exercises defaults elsewhere."""
    return f"""
name: smoke_test
data:
  data_dir: {data_dir}
  train_file_indices: [0, 1]
  test_file_indices: all
model:
  name: simple_cnn
  params:
    channels: [16, 32, 64]
    kernel_size: 5
optim:
  optimizer: adamw
  lr: 1.0e-3
loss:
  type: weighted_bce
  pos_weight: auto
train:
  epochs: 3
  out_dir: {out_dir}
"""


# --- DataConfig ------------------------------------------------------


def test_data_config_defaults(tmp_path: Path) -> None:
    cfg = DataConfig(data_dir=tmp_path)
    assert cfg.train_file_indices == "all"
    assert cfg.test_file_indices == "all"
    assert cfg.target_label == "psd_label_low_avse"
    assert cfg.batch_size == 256
    assert cfg.baseline_samples == 500


def test_data_config_rejects_missing_dir(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        DataConfig(data_dir=tmp_path / "nope")


def test_data_config_rejects_negative_indices(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        DataConfig(data_dir=tmp_path, train_file_indices=[-1])


def test_data_config_rejects_duplicate_indices(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        DataConfig(data_dir=tmp_path, train_file_indices=[0, 0, 1])


def test_data_config_rejects_zero_batch_size(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        DataConfig(data_dir=tmp_path, batch_size=0)


def test_data_config_rejects_unknown_target_label(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        DataConfig(data_dir=tmp_path, target_label="not_a_label")


def test_data_config_train_portion_default(tmp_path: Path) -> None:
    cfg = DataConfig(data_dir=tmp_path)
    assert cfg.train_portion == 1.0


def test_data_config_rejects_zero_train_portion(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        DataConfig(data_dir=tmp_path, train_portion=0.0)


def test_data_config_rejects_train_portion_above_one(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        DataConfig(data_dir=tmp_path, train_portion=1.5)


def test_data_config_default_sampler_strategies_empty(tmp_path: Path) -> None:
    cfg = DataConfig(data_dir=tmp_path)
    assert cfg.sampler_strategies == []
    assert cfg.energy_range is None


def test_data_config_accepts_sampler_strategies(tmp_path: Path) -> None:
    cfg = DataConfig(
        data_dir=tmp_path,
        sampler_strategies=["class_balanced", "energy_balanced"],
    )
    assert cfg.sampler_strategies == ["class_balanced", "energy_balanced"]


def test_data_config_rejects_duplicate_sampler_strategies(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        DataConfig(
            data_dir=tmp_path,
            sampler_strategies=["class_balanced", "class_balanced"],
        )


def test_data_config_rejects_unknown_sampler_strategy(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        DataConfig(data_dir=tmp_path, sampler_strategies=["random_walk"])


def test_data_config_subset_portion_default(tmp_path: Path) -> None:
    cfg = DataConfig(data_dir=tmp_path)
    assert cfg.subset_portion == 1.0
    assert cfg.subset_seed == 0


def test_data_config_rejects_zero_subset_portion(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        DataConfig(data_dir=tmp_path, subset_portion=0.0)


def test_data_config_rejects_subset_portion_above_one(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        DataConfig(data_dir=tmp_path, subset_portion=1.5)


def test_data_config_energy_range_validation(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        DataConfig(data_dir=tmp_path, energy_range=(1000.0, 500.0))
    with pytest.raises(ValidationError):
        DataConfig(data_dir=tmp_path, energy_range=(-10.0, 100.0))
    # Valid case
    cfg = DataConfig(data_dir=tmp_path, energy_range=(500.0, 3000.0))
    assert cfg.energy_range == (500.0, 3000.0)


# --- ModelConfig -----------------------------------------------------


def test_model_config_minimal() -> None:
    cfg = ModelConfig(name="simple_cnn")
    assert cfg.name == "simple_cnn"
    assert cfg.params == {}


def test_model_config_with_params() -> None:
    cfg = ModelConfig(name="simple_cnn", params={"channels": [8, 16], "dropout": 0.1})
    assert cfg.params["channels"] == [8, 16]
    assert cfg.params["dropout"] == 0.1


def test_model_config_rejects_empty_name() -> None:
    with pytest.raises(ValidationError):
        ModelConfig(name="")


# --- OptimConfig -----------------------------------------------------


def test_optim_config_defaults() -> None:
    cfg = OptimConfig()
    assert cfg.optimizer == "adamw"
    assert cfg.lr == pytest.approx(1e-3)


def test_optim_config_rejects_non_positive_lr() -> None:
    with pytest.raises(ValidationError):
        OptimConfig(lr=0.0)


def test_optim_config_rejects_unknown_optimizer() -> None:
    with pytest.raises(ValidationError):
        OptimConfig(optimizer="nadam")


# --- LossConfig ------------------------------------------------------


def test_loss_config_defaults() -> None:
    cfg = LossConfig()
    assert cfg.type == "weighted_bce"
    assert cfg.pos_weight == "auto"
    assert cfg.balanced_sampler is False


def test_loss_config_pos_weight_float() -> None:
    cfg = LossConfig(pos_weight=2.5)
    assert cfg.pos_weight == pytest.approx(2.5)


def test_loss_config_rejects_non_positive_pos_weight() -> None:
    with pytest.raises(ValidationError):
        LossConfig(pos_weight=0.0)


def test_loss_config_rejects_unknown_type() -> None:
    with pytest.raises(ValidationError):
        LossConfig(type="hinge")


# --- TrainConfig -----------------------------------------------------


def test_train_config_minimal(tmp_path: Path) -> None:
    cfg = TrainConfig(out_dir=tmp_path / "run")
    assert cfg.epochs == 10
    assert cfg.device == "auto"
    assert cfg.amp is True


def test_train_config_rejects_zero_epochs(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        TrainConfig(out_dir=tmp_path, epochs=0)


# --- Top-level + extra-field rejection ------------------------------


def test_extra_fields_are_rejected(tmp_path: Path) -> None:
    """``extra="forbid"`` catches typos in YAML."""
    with pytest.raises(ValidationError):
        DataConfig(data_dir=tmp_path, batch_sze=256)  # typo


def test_experiment_config_uses_defaults(tmp_path: Path) -> None:
    """optim / loss can fall back to defaults; data / model / train cannot."""
    cfg = ExperimentConfig(
        name="exp",
        data=DataConfig(data_dir=tmp_path),
        model=ModelConfig(name="simple_cnn"),
        train=TrainConfig(out_dir=tmp_path / "run"),
    )
    assert cfg.optim.optimizer == "adamw"
    assert cfg.loss.type == "weighted_bce"


# --- YAML loading ----------------------------------------------------


def test_load_config_round_trip(tmp_path: Path) -> None:
    yaml_path = tmp_path / "exp.yaml"
    yaml_path.write_text(_minimal_yaml(tmp_path, tmp_path / "run"))

    cfg = load_config(yaml_path)

    assert cfg.name == "smoke_test"
    assert cfg.data.train_file_indices == [0, 1]
    assert cfg.data.test_file_indices == "all"
    assert cfg.model.name == "simple_cnn"
    assert cfg.model.params["channels"] == [16, 32, 64]
    assert cfg.optim.lr == pytest.approx(1e-3)
    assert cfg.loss.pos_weight == "auto"
    assert cfg.train.epochs == 3


def test_load_config_rejects_bad_yaml(tmp_path: Path) -> None:
    yaml_path = tmp_path / "bad.yaml"
    yaml_path.write_text(
        f"""
name: bad
data:
  data_dir: {tmp_path}
  batch_size: -1     # invalid
model:
  name: simple_cnn
train:
  out_dir: {tmp_path / "run"}
"""
    )
    with pytest.raises(ValidationError):
        load_config(yaml_path)
