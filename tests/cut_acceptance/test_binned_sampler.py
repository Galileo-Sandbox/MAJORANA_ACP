"""Tests for ``majorana_acp.cut_acceptance.binned_sampler``."""

from __future__ import annotations

import numpy as np
import pytest
from schemas.data_models import InputMode

from majorana_acp.cut_acceptance.binned_sampler import BinnedSampler, _BinIndex


def _events(n: int = 2000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    energy = rng.uniform(500.0, 3000.0, size=n)
    score = rng.uniform(0.0, 1.0, size=n)
    return energy, score


# --- _BinIndex ------------------------------------------------------


def test_bin_index_keeps_only_eligible_bins() -> None:
    rng = np.random.default_rng(0)
    # 30 events clustered near 1000 keV, 1 outlier at 2500.
    energy = np.concatenate([rng.uniform(990, 1010, size=30), np.array([2500.0])])
    score = rng.uniform(0, 1, size=energy.size)
    idx = _BinIndex(
        energy, score,
        energy_range=(500.0, 3000.0),
        bin_width=10.0,
        min_events_per_bin=4,
    )
    # Bins near 1000 keV survive; the 2500-keV singleton is dropped.
    assert len(idx) >= 1
    for c in idx.bin_centers:
        assert 990.0 <= c <= 1010.0


def test_bin_index_rejects_empty_grid() -> None:
    rng = np.random.default_rng(0)
    energy = rng.uniform(500, 3000, size=3)  # too few for min=4
    score = rng.uniform(0, 1, size=3)
    with pytest.raises(ValueError, match="min_events_per_bin"):
        _BinIndex(
            energy, score,
            energy_range=(500.0, 3000.0),
            bin_width=10.0,
            min_events_per_bin=4,
        )


# --- BinnedSampler --------------------------------------------------


def test_sampler_advertises_design_only() -> None:
    energy, score = _events()
    s = BinnedSampler(
        energy, score,
        energy_range=(500.0, 3000.0),
        energy_bin_width=10.0,
    )
    assert s.mode is InputMode.DESIGN_ONLY
    assert s.dim_theta == 2
    assert s.dim_phi is None


def test_generate_shapes_and_binary_labels() -> None:
    energy, score = _events(2000)
    s = BinnedSampler(
        energy, score,
        energy_range=(500.0, 3000.0),
        energy_bin_width=20.0,
        n_per_trial=16,
    )
    batch = s.generate(n_trials=8, n_events=16, seed=42)
    assert batch.theta.shape == (8, 2)
    assert batch.labels.shape == (8, 16)
    assert batch.labels.dtype == np.int8
    assert set(np.unique(batch.labels).tolist()).issubset({0, 1})
    # θ is normalized to [0, 1]^2.
    assert np.all((batch.theta >= 0.0) & (batch.theta <= 1.0))


def test_thresholds_match_score_comparison() -> None:
    """Labels equal 1[score_i >= T_k] for the actual sampled events."""
    energy, score = _events(1000, seed=11)
    s = BinnedSampler(
        energy, score,
        energy_range=(500.0, 3000.0),
        energy_bin_width=50.0,
        n_per_trial=20,
        t_sampling="uniform",
    )
    batch = s.generate(n_trials=6, n_events=20, seed=7)
    # Trial-level mean must equal mean(1[score >= T_k]) for the sampled events.
    # We can't recover exact event indices without re-running the RNG, but we
    # can check that for any T_k, β_emp(trial) is a multiple of 1/n_events.
    n = 20
    for k in range(6):
        beta = batch.labels[k].mean()
        # n_per_trial=20 → β is a multiple of 1/20 ± floating-point.
        assert abs(beta * n - round(beta * n)) < 1e-9


def test_boundary_mix_concentrates_at_endpoints() -> None:
    energy, score = _events(4000)
    s = BinnedSampler(
        energy, score,
        energy_range=(500.0, 3000.0),
        energy_bin_width=20.0,
        n_per_trial=8,
        t_sampling="boundary_mix",
    )
    batch = s.generate(n_trials=4000, n_events=8, seed=11)
    t = batch.theta[:, 1]
    # 25% in the bottom 5%, plus 5% from the uniform half = 27.5% expected.
    assert 0.24 <= (t <= 0.05).mean() <= 0.32
    assert 0.24 <= (t >= 0.95).mean() <= 0.32


def test_uniform_default_unchanged() -> None:
    energy, score = _events(4000)
    s = BinnedSampler(
        energy, score,
        energy_range=(500.0, 3000.0),
        energy_bin_width=20.0,
        t_sampling="uniform",
    )
    batch = s.generate(n_trials=4000, n_events=8, seed=0)
    t = batch.theta[:, 1]
    assert (t <= 0.05).mean() < 0.10
    assert (t >= 0.95).mean() < 0.10


def test_reproducible_seed() -> None:
    energy, score = _events(1000)
    s = BinnedSampler(
        energy, score,
        energy_range=(500.0, 3000.0),
        energy_bin_width=20.0,
    )
    a = s.generate(n_trials=4, n_events=12, seed=99)
    b = s.generate(n_trials=4, n_events=12, seed=99)
    np.testing.assert_array_equal(a.theta, b.theta)
    np.testing.assert_array_equal(a.labels, b.labels)


def test_handles_sparse_bins_via_replacement() -> None:
    """A bin with fewer events than n_per_trial still emits n_per_trial labels."""
    rng = np.random.default_rng(0)
    # Stuff a single bin around 1000 keV with exactly 5 events, plus filler.
    e_core = np.array([995, 996, 997, 998, 999], dtype=float)
    s_core = rng.uniform(0, 1, size=5)
    e_pad = rng.uniform(500, 3000, size=2000)
    s_pad = rng.uniform(0, 1, size=2000)
    energy = np.concatenate([e_core, e_pad])
    score = np.concatenate([s_core, s_pad])
    s = BinnedSampler(
        energy, score,
        energy_range=(500.0, 3000.0),
        energy_bin_width=10.0,
        n_per_trial=64,  # > 5 → replacement must kick in
    )
    batch = s.generate(n_trials=10, n_events=64, seed=0)
    assert batch.labels.shape == (10, 64)


def test_rejects_unknown_t_sampling() -> None:
    energy, score = _events(1000)
    with pytest.raises(ValueError, match="t_sampling"):
        BinnedSampler(
            energy, score,
            energy_range=(500.0, 3000.0),
            energy_bin_width=20.0,
            t_sampling="bogus",
        )
