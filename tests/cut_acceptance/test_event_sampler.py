"""Tests for ``majorana_acp.cut_acceptance.event_sampler``."""

from __future__ import annotations

import numpy as np
import pytest
from schemas.data_models import InputMode

from majorana_acp.cut_acceptance.event_sampler import EventSampler


def _pool(n: int = 2000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic (energy, score) pool spread across [500, 3000] keV."""
    rng = np.random.default_rng(seed)
    energy = rng.uniform(500.0, 3000.0, size=n)
    score = rng.uniform(0.0, 1.0, size=n)
    return energy, score


def _sampler(
    *,
    min_events_per_bin: int = 4,
    t_sampling: str = "boundary_mix",
    pool_seed: int = 0,
    pool_n: int = 2000,
    bin_width: float = 100.0,
) -> EventSampler:
    e, s = _pool(pool_n, seed=pool_seed)
    return EventSampler(
        e, s,
        energy_range=(500.0, 3000.0),
        energy_bin_width=bin_width,
        threshold_range=(0.0, 1.0),
        min_events_per_bin=min_events_per_bin,
        t_sampling=t_sampling,
    )


# --- mode / shape contract ------------------------------------------


def test_sampler_advertises_event_only() -> None:
    s = _sampler()
    assert s.mode is InputMode.EVENT_ONLY
    assert s.dim_theta is None
    assert s.dim_phi == 2


def test_generate_shapes_and_binary_labels() -> None:
    s = _sampler()
    batch = s.generate(n_trials=8, n_events=16, seed=0)
    assert batch.mode is InputMode.EVENT_ONLY
    assert batch.theta is None
    assert batch.phi.shape == (8, 16, 2)
    assert batch.labels.shape == (8, 16)
    assert batch.labels.dtype == np.int8
    assert set(np.unique(batch.labels).tolist()).issubset({0, 1})


def test_phi_in_unit_interval() -> None:
    s = _sampler()
    batch = s.generate(n_trials=8, n_events=32, seed=1)
    assert batch.phi.min() >= 0.0
    assert batch.phi.max() <= 1.0


# --- T-sampling -----------------------------------------------------


def test_t_constant_within_trial_varies_across_trials() -> None:
    """All events in trial k share T_k; different trials get different T_k."""
    s = _sampler()
    batch = s.generate(n_trials=20, n_events=8, seed=3)
    t_per_event_per_trial = batch.phi[:, :, 1]   # [B, N]
    for k in range(batch.phi.shape[0]):
        assert np.allclose(t_per_event_per_trial[k], t_per_event_per_trial[k, 0])
    t_per_trial = t_per_event_per_trial[:, 0]
    assert np.unique(t_per_trial).size > 1


def test_boundary_mix_concentrates_at_endpoints() -> None:
    s = _sampler(t_sampling="boundary_mix")
    batch = s.generate(n_trials=4000, n_events=4, seed=11)
    t = batch.phi[:, 0, 1]
    # Same expectations as BinnedSampler: ~27.5% in each 5% tail.
    assert 0.24 <= (t <= 0.05).mean() <= 0.32
    assert 0.24 <= (t >= 0.95).mean() <= 0.32


def test_uniform_t_does_not_concentrate() -> None:
    s = _sampler(t_sampling="uniform")
    batch = s.generate(n_trials=4000, n_events=4, seed=0)
    t = batch.phi[:, 0, 1]
    assert (t <= 0.05).mean() < 0.10
    assert (t >= 0.95).mean() < 0.10


# --- labels round-trip ----------------------------------------------


def test_labels_consistent_with_score_threshold() -> None:
    """β_emp per trial must be a multiple of 1/n_events (integer kept-event count)."""
    s = _sampler(t_sampling="uniform")
    batch = s.generate(n_trials=6, n_events=20, seed=7)
    n = 20
    for k in range(6):
        beta = batch.labels[k].mean()
        assert abs(beta * n - round(beta * n)) < 1e-9


# --- reproducibility ------------------------------------------------


def test_reproducible_with_same_seed() -> None:
    s = _sampler()
    a = s.generate(n_trials=4, n_events=16, seed=42)
    b = s.generate(n_trials=4, n_events=16, seed=42)
    np.testing.assert_array_equal(a.phi, b.phi)
    np.testing.assert_array_equal(a.labels, b.labels)


def test_different_seeds_diverge() -> None:
    s = _sampler()
    a = s.generate(n_trials=4, n_events=16, seed=0)
    b = s.generate(n_trials=4, n_events=16, seed=1)
    assert not (np.array_equal(a.phi, b.phi) and np.array_equal(a.labels, b.labels))


# --- the core design property: bin-stratification + real-energy phi --


def test_bin_stratification_is_approximately_flat() -> None:
    """Picks should be roughly uniform across the bin grid, not natural-density.

    A natural-density sampler would over-represent dense regions of the
    spectrum. With a uniform pool this is trivial; the more telling
    check is with a deliberately non-uniform pool — see
    :func:`test_bin_stratification_beats_natural_density`.
    """
    s = _sampler(pool_n=10_000, bin_width=100.0)
    batch = s.generate(n_trials=1, n_events=20_000, seed=7)
    e_norm = batch.phi[0, :, 0]
    counts, _ = np.histogram(e_norm, bins=25)
    cv = counts.std() / counts.mean()
    assert cv < 0.15, f"CV={cv:.3f} suggests sampling is not flat across bins"


def test_bin_stratification_beats_natural_density() -> None:
    """With a pool that's 5× denser at low E, stratified picks stay flat."""
    rng = np.random.default_rng(0)
    # 5000 events in [500, 1500] keV (dense low-E) + 1000 events in
    # [1500, 3000] keV (sparse high-E). A natural-density sampler would
    # put ~83% of its picks in the low-E half; bin-stratified should not.
    e_low = rng.uniform(500.0, 1500.0, size=5000)
    e_high = rng.uniform(1500.0, 3000.0, size=1000)
    energy = np.concatenate([e_low, e_high])
    score = rng.uniform(0.0, 1.0, size=energy.size)
    s = EventSampler(
        energy, score,
        energy_range=(500.0, 3000.0),
        energy_bin_width=100.0,
        min_events_per_bin=4,
    )
    batch = s.generate(n_trials=1, n_events=20_000, seed=0)
    e_picked = batch.phi[0, :, 0] * (3000.0 - 500.0) + 500.0
    frac_low = (e_picked < 1500.0).mean()
    # Low-E half is 10 bins out of 25 in this range → expect ~40% picks.
    # Natural density would give ~83%; stratified should be ~40% ± few %.
    assert 0.35 <= frac_low <= 0.45, (
        f"frac_low={frac_low:.3f} — bin-stratification not working as expected"
    )


def test_phi_carries_real_event_energies_not_bin_centers() -> None:
    """phi_i must record the event's actual energy, not its bin center."""
    s = _sampler(pool_n=2000, bin_width=100.0)
    batch = s.generate(n_trials=1, n_events=500, seed=11)
    e_norm = batch.phi[0, :, 0]
    # 500 events across ~25 bins should give >> 25 unique values if the
    # real energies are preserved; ≤ 25 unique values would indicate
    # bin-center quantization.
    n_unique = np.unique(e_norm).size
    assert n_unique > 100, (
        f"phi has only {n_unique} unique E values across 500 events — "
        f"looks like bin centers, not per-event energies"
    )


# --- validation -----------------------------------------------------


def test_rejects_unknown_t_sampling() -> None:
    e, s = _pool()
    with pytest.raises(ValueError, match="t_sampling"):
        EventSampler(
            e, s,
            energy_range=(500.0, 3000.0),
            energy_bin_width=20.0,
            t_sampling="bogus",
        )


def test_rejects_bad_threshold_range() -> None:
    e, s = _pool()
    with pytest.raises(ValueError, match="threshold_range"):
        EventSampler(
            e, s,
            energy_range=(500.0, 3000.0),
            energy_bin_width=20.0,
            threshold_range=(1.0, 0.0),
        )


def test_rejects_zero_n_trials_or_events() -> None:
    s = _sampler()
    with pytest.raises(ValueError, match="n_trials"):
        s.generate(n_trials=0, n_events=4, seed=0)
    with pytest.raises(ValueError, match="n_events"):
        s.generate(n_trials=2, n_events=0, seed=0)
