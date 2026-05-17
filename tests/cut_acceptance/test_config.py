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


def test_hybrid_defaults_reproduce_true_cnp() -> None:
    """Bare config (no hybrid knobs touched) must equal the True-CNP baseline."""
    cfg = CutAcceptanceConfig(**_kwargs())
    assert cfg.trial_size_strategy == "fixed"
    assert cfg.sampling_pattern == "flat_stratified"
    assert cfg.zoom_window_width_kev == 50.0
    assert cfg.local_event_fraction == 0.70
    assert cfg.n_clusters == 2
    assert cfg.physics_peaks_kev == [1460.0, 2614.0]


def test_n_trial_events_max_must_exceed_min() -> None:
    with pytest.raises(ValueError, match="n_trial_events_max"):
        CutAcceptanceConfig(
            **_kwargs(n_trial_events_min=32, n_trial_events_max=16)
        )


def test_physics_peaks_outside_energy_range_are_dropped() -> None:
    """Peaks past energy_range get silently filtered to keep focus selection valid."""
    cfg = CutAcceptanceConfig(
        **_kwargs(
            energy_range=(500.0, 2000.0),
            physics_peaks_kev=[1460.0, 2614.0],
        )
    )
    assert cfg.physics_peaks_kev == [1460.0]  # 2614 is outside


def test_out_dir_and_name_are_optional() -> None:
    """Both can be left blank; pipeline auto-derives from paradigm + bin + class."""
    cfg = CutAcceptanceConfig(
        train_predictions_path="t.h5",
        validation_predictions_path="v.h5",
        target_class=1,
        upstream_classifier_config="up.yaml",
    )
    assert cfg.out_dir is None
    assert cfg.name is None
