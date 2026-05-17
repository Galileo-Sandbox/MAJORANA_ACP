"""Tests for ``majorana_acp.cut_acceptance.config``."""

from __future__ import annotations

import pytest

from majorana_acp.cut_acceptance.config import CutAcceptanceConfig, load_config


def _kwargs(**overrides):
    base = dict(
        name="test",
        train_predictions_path="dummy_train.h5",
        validation_predictions_path="dummy_test.h5",
        out_dir="dummy",
        target_class=1,
        upstream_classifier_config="dummy_upstream.yaml",
    )
    base.update(overrides)
    return base


def test_defaults_load() -> None:
    cfg = CutAcceptanceConfig(**_kwargs())
    assert cfg.energy_range == (500.0, 3000.0)
    assert cfg.energy_bin_width == 10.0
    assert cfg.threshold_range == (0.0, 1.0)
    assert cfg.n_per_trial is None  # deprecated; legacy YAMLs may still set it
    assert cfg.min_events_per_bin == 4
    assert cfg.decoder_hidden_dims == [128, 128, 128]


def test_target_class_only_zero_one_or_all() -> None:
    CutAcceptanceConfig(**_kwargs(target_class=0))
    CutAcceptanceConfig(**_kwargs(target_class=1))
    CutAcceptanceConfig(**_kwargs(target_class="all"))
    with pytest.raises(ValueError):
        CutAcceptanceConfig(**_kwargs(target_class=2))
    with pytest.raises(ValueError):
        CutAcceptanceConfig(**_kwargs(target_class="signal"))


def test_energy_range_must_be_ordered() -> None:
    with pytest.raises(ValueError, match="hi > lo"):
        CutAcceptanceConfig(**_kwargs(energy_range=(3000.0, 500.0)))


def test_threshold_range_must_be_ordered() -> None:
    with pytest.raises(ValueError, match="hi > lo"):
        CutAcceptanceConfig(**_kwargs(threshold_range=(1.0, 0.0)))


def test_energy_bin_width_positive() -> None:
    with pytest.raises(ValueError):
        CutAcceptanceConfig(**_kwargs(energy_bin_width=0.0))
    with pytest.raises(ValueError):
        CutAcceptanceConfig(**_kwargs(energy_bin_width=-5.0))


def test_extra_field_is_rejected() -> None:
    """Frozen + extra='forbid' catches typos / leftover MFGP fields."""
    with pytest.raises(ValueError):
        CutAcceptanceConfig(**_kwargs(unknown_field=42))
    with pytest.raises(ValueError):
        CutAcceptanceConfig(**_kwargs(peak_windows=[]))
    with pytest.raises(ValueError):
        CutAcceptanceConfig(**_kwargs(n_mfgp_hf_trials=400))


def test_yaml_roundtrip(tmp_path) -> None:
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(
        "name: smoke\n"
        "train_predictions_path: /tmp/train.h5\n"
        "validation_predictions_path: /tmp/test.h5\n"
        "out_dir: /tmp/out\n"
        "target_class: 0\n"
        "energy_bin_width: 5.0\n"
        "upstream_classifier_config: /tmp/upstream.yaml\n"
    )
    cfg = load_config(yaml_path)
    assert cfg.name == "smoke"
    assert cfg.target_class == 0
    assert cfg.energy_bin_width == 5.0
    assert str(cfg.upstream_classifier_config) == "/tmp/upstream.yaml"


def test_upstream_classifier_config_is_required() -> None:
    """Lineage hash needs an upstream to point at — no silent default."""
    incomplete = dict(
        name="missing-upstream",
        train_predictions_path="t.h5",
        validation_predictions_path="v.h5",
        out_dir="out",
        target_class=1,
    )
    with pytest.raises(ValueError, match="upstream_classifier_config"):
        CutAcceptanceConfig(**incomplete)
