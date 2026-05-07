"""Tests for ``majorana_acp.cut_acceptance.partition``."""

from __future__ import annotations

import numpy as np
import pytest

from majorana_acp.cut_acceptance.config import (
    CutAcceptanceConfig,
    PeakSplit,
    PeakWindow,
)
from majorana_acp.cut_acceptance.partition import partition_events


def _cfg(**overrides) -> CutAcceptanceConfig:
    base = dict(
        name="test",
        predictions_path="dummy.h5",
        out_dir="dummy",
        target_class=1,
        energy_range=(500.0, 3000.0),
        peak_windows=[
            PeakWindow(lo=2605.0, hi=2620.0),
            PeakWindow(lo=1587.0, hi=1597.0),
        ],
        peak_split=PeakSplit(lf=0.10, hf_train=0.30, hf_holdout=0.60),
        partition_seed=0,
    )
    base.update(overrides)
    return CutAcceptanceConfig(**base)


def _synthetic_events(n_continuum: int = 200, n_peak: int = 100, seed: int = 7):
    """Mix of continuum events + events distributed inside two peak windows."""
    rng = np.random.default_rng(seed)
    energy_cont = rng.uniform(700.0, 1500.0, size=n_continuum)
    energy_peak1 = rng.uniform(2605.0, 2620.0, size=n_peak // 2)
    energy_peak2 = rng.uniform(1587.0, 1597.0, size=n_peak // 2)
    energy = np.concatenate([energy_cont, energy_peak1, energy_peak2])
    label = np.ones_like(energy, dtype=np.int64)  # all signal
    # Shuffle so peak/non-peak are interleaved (catches index-vs-mask bugs).
    perm = rng.permutation(energy.size)
    return energy[perm], label[perm]


def test_disjoint_and_total_count_preserved() -> None:
    energy, label = _synthetic_events()
    cfg = _cfg()
    out = partition_events(energy, label, cfg)
    assert out.disjoint()
    n_total = out.lf.size + out.hf_train.size + out.hf_holdout.size
    assert n_total == int(out.base_mask.sum())


def test_lf_contains_all_continuum_events() -> None:
    """Continuum (non-peak) events must all land in LF."""
    energy, label = _synthetic_events(n_continuum=150, n_peak=80)
    cfg = _cfg()
    out = partition_events(energy, label, cfg)
    energy_filt = energy[out.base_mask]
    in_peak = np.zeros_like(energy_filt, dtype=bool)
    for w in cfg.peak_windows:
        in_peak |= (energy_filt >= w.lo) & (energy_filt <= w.hi)
    continuum_idx = np.flatnonzero(~in_peak)
    # Every continuum index should be in LF.
    assert set(continuum_idx.tolist()).issubset(set(out.lf.tolist()))


def test_peak_split_fractions_match_config() -> None:
    energy, label = _synthetic_events(n_continuum=0, n_peak=1000)
    cfg = _cfg(peak_split=PeakSplit(lf=0.10, hf_train=0.30, hf_holdout=0.60))
    out = partition_events(energy, label, cfg)
    energy_filt = energy[out.base_mask]
    in_peak = np.zeros_like(energy_filt, dtype=bool)
    for w in cfg.peak_windows:
        in_peak |= (energy_filt >= w.lo) & (energy_filt <= w.hi)
    n_peak = int(in_peak.sum())
    # All LF here is peak (continuum count = 0), so |lf| ≈ 10% of peak.
    assert abs(out.lf.size - 0.10 * n_peak) <= 1
    assert abs(out.hf_train.size - 0.30 * n_peak) <= 1
    assert abs(out.hf_holdout.size - 0.60 * n_peak) <= 1


def test_filters_by_target_class() -> None:
    """target_class=0 should drop signal-labelled events from the partition."""
    energy_sig, label_sig = _synthetic_events(n_continuum=120, n_peak=80, seed=11)
    energy_bkg, label_bkg = _synthetic_events(n_continuum=80, n_peak=60, seed=22)
    label_sig[:] = 1
    label_bkg[:] = 0
    energy = np.concatenate([energy_sig, energy_bkg])
    label = np.concatenate([label_sig, label_bkg])
    cfg = _cfg(target_class=0)
    out = partition_events(energy, label, cfg)
    # Every selected event should have label == 0.
    assert np.all(label[out.base_mask] == 0)


def test_partition_is_seed_reproducible() -> None:
    energy, label = _synthetic_events()
    out1 = partition_events(energy, label, _cfg(partition_seed=42))
    out2 = partition_events(energy, label, _cfg(partition_seed=42))
    np.testing.assert_array_equal(out1.lf, out2.lf)
    np.testing.assert_array_equal(out1.hf_train, out2.hf_train)
    np.testing.assert_array_equal(out1.hf_holdout, out2.hf_holdout)


def test_partition_changes_with_seed() -> None:
    energy, label = _synthetic_events()
    out1 = partition_events(energy, label, _cfg(partition_seed=0))
    out2 = partition_events(energy, label, _cfg(partition_seed=1))
    # The peak split should differ between seeds (LF continuum part is identical).
    assert not np.array_equal(out1.hf_train, out2.hf_train)


def test_signal_and_background_runs_use_disjoint_pools() -> None:
    """The two pipelines must consume non-overlapping events.

    Same partition_seed across signal/background runs is the spec, but
    they're filtered by different target_class so the underlying base
    masks must not share any index.
    """
    energy_sig, label_sig = _synthetic_events(n_continuum=120, n_peak=80, seed=11)
    energy_bkg, label_bkg = _synthetic_events(n_continuum=80, n_peak=60, seed=22)
    label_sig[:] = 1
    label_bkg[:] = 0
    energy = np.concatenate([energy_sig, energy_bkg])
    label = np.concatenate([label_sig, label_bkg])
    sig = partition_events(energy, label, _cfg(target_class=1, partition_seed=42))
    bkg = partition_events(energy, label, _cfg(target_class=0, partition_seed=42))
    assert not np.any(sig.base_mask & bkg.base_mask)


def test_raises_when_not_enough_peak_events() -> None:
    rng = np.random.default_rng(0)
    energy = rng.uniform(700.0, 1400.0, size=50)  # all continuum, no peak events
    label = np.ones_like(energy, dtype=np.int64)
    with pytest.raises(ValueError, match="peak event"):
        partition_events(energy, label, _cfg())


def test_raises_when_filter_drops_everything() -> None:
    energy = np.array([100.0, 200.0])  # below energy_range
    label = np.array([1, 1])
    with pytest.raises(ValueError, match="no events survive"):
        partition_events(energy, label, _cfg())
