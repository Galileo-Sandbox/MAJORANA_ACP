"""Tests for ``majorana_acp.cut_acceptance.config``."""

from __future__ import annotations

import pytest

from majorana_acp.cut_acceptance.config import (
    CutAcceptanceConfig,
    PeakSplit,
    PeakWindow,
    load_config,
)


def _kwargs(**overrides):
    base = dict(
        name="test",
        predictions_path="dummy.h5",
        out_dir="dummy",
        target_class=1,
    )
    base.update(overrides)
    return base


def test_defaults_load() -> None:
    cfg = CutAcceptanceConfig(**_kwargs())
    assert cfg.energy_range == (500.0, 3000.0)
    assert len(cfg.peak_windows) == 4
    assert cfg.peak_split.lf + cfg.peak_split.hf_train + cfg.peak_split.hf_holdout == 1.0


def test_peak_split_sums_to_one() -> None:
    PeakSplit(lf=0.10, hf_train=0.30, hf_holdout=0.60)  # OK
    with pytest.raises(ValueError, match="sum to 1"):
        PeakSplit(lf=0.20, hf_train=0.30, hf_holdout=0.60)


def test_peak_window_orders_lo_hi() -> None:
    with pytest.raises(ValueError, match="hi > lo"):
        PeakWindow(lo=2620.0, hi=2605.0)


def test_energy_range_must_be_ordered() -> None:
    with pytest.raises(ValueError, match="hi > lo"):
        CutAcceptanceConfig(**_kwargs(energy_range=(3000.0, 500.0)))


def test_threshold_range_must_be_ordered() -> None:
    with pytest.raises(ValueError, match="hi > lo"):
        CutAcceptanceConfig(**_kwargs(threshold_range=(1.0, 0.0)))


def test_target_class_only_zero_or_one() -> None:
    CutAcceptanceConfig(**_kwargs(target_class=0))
    CutAcceptanceConfig(**_kwargs(target_class=1))
    with pytest.raises(ValueError):
        CutAcceptanceConfig(**_kwargs(target_class=2))


def test_yaml_roundtrip(tmp_path) -> None:
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(
        "name: smoke\n"
        "predictions_path: /tmp/dummy.h5\n"
        "out_dir: /tmp/out\n"
        "target_class: 0\n"
        "partition_seed: 7\n"
    )
    cfg = load_config(yaml_path)
    assert cfg.name == "smoke"
    assert cfg.target_class == 0
    assert cfg.partition_seed == 7


def test_extra_field_is_rejected() -> None:
    """Frozen + extra='forbid' catches typos in config files."""
    with pytest.raises(ValueError):
        CutAcceptanceConfig(**_kwargs(unknown_field=42))


def test_real_example_yaml_loads() -> None:
    """Sanity-check the YAML committed to configs/cut_acceptance/."""
    cfg = load_config("configs/cut_acceptance/resnet_single_ultra_small_signal.yaml")
    assert cfg.target_class == 1
    assert cfg.name == "resnet_single_ultra_small_signal"
    assert len(cfg.peak_windows) == 4
