"""Tests for ``majorana_acp.cut_acceptance.sampler``."""

from __future__ import annotations

import numpy as np
import pytest
from schemas.data_models import InputMode

from majorana_acp.cut_acceptance.sampler import DesignOnlySampler, EventPool


def _pool(n: int = 200, seed: int = 0) -> EventPool:
    rng = np.random.default_rng(seed)
    energy = rng.uniform(500.0, 3000.0, size=n)
    score = rng.uniform(0.0, 1.0, size=n)
    return EventPool(energy=energy, score=score)


# --- EventPool ------------------------------------------------------


def test_event_pool_sorts_by_energy() -> None:
    pool = _pool(50)
    diffs = np.diff(pool.energy)
    assert np.all(diffs >= 0.0)


def test_event_pool_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        EventPool(energy=np.arange(5), score=np.arange(7))


def test_event_pool_rejects_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        EventPool(energy=np.array([]), score=np.array([]))


def test_event_pool_rejects_non_1d() -> None:
    with pytest.raises(ValueError, match="1-D"):
        EventPool(energy=np.zeros((3, 2)), score=np.zeros((3, 2)))


def test_nearest_indices_returns_n_when_pool_large() -> None:
    pool = _pool(200)
    idx = pool.nearest_indices(E_k=1500.0, n=10)
    assert idx.size == 10
    # All returned indices should be in [0, len(pool)).
    assert idx.min() >= 0
    assert idx.max() < len(pool)


def test_nearest_indices_are_actually_nearest() -> None:
    energy = np.array([100.0, 500.0, 1000.0, 1500.0, 2000.0])
    score = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    pool = EventPool(energy=energy, score=score)
    # Closest 3 to 1100: energies 1000, 1500, 500 -> sorted-by-energy idx 2, 3, 1.
    idx = pool.nearest_indices(E_k=1100.0, n=3)
    assert sorted(idx.tolist()) == [1, 2, 3]


def test_nearest_indices_caps_at_pool_size() -> None:
    pool = _pool(5)
    idx = pool.nearest_indices(E_k=1500.0, n=20)
    assert idx.size == 5  # capped


def test_nearest_indices_rejects_zero_n() -> None:
    pool = _pool(20)
    with pytest.raises(ValueError, match=">= 1"):
        pool.nearest_indices(E_k=1000.0, n=0)


# --- DesignOnlySampler ----------------------------------------------


def test_sampler_advertises_design_only() -> None:
    sampler = DesignOnlySampler(_pool(), (500.0, 3000.0), (0.0, 1.0), n_per_trial=8)
    assert sampler.mode is InputMode.DESIGN_ONLY
    assert sampler.dim_theta == 2
    assert sampler.dim_phi is None


def test_sampler_generates_correct_shapes() -> None:
    sampler = DesignOnlySampler(_pool(200), (500.0, 3000.0), (0.0, 1.0), n_per_trial=16)
    batch = sampler.generate(n_trials=5, n_events=16, seed=0)
    assert batch.theta.shape == (5, 2)
    assert batch.phi is None
    assert batch.labels.shape == (5, 16)
    assert batch.labels.dtype == np.int8


def test_sampler_labels_are_binary() -> None:
    sampler = DesignOnlySampler(_pool(200), (500.0, 3000.0), (0.0, 1.0), n_per_trial=16)
    batch = sampler.generate(n_trials=10, n_events=16, seed=0)
    unique = np.unique(batch.labels)
    assert set(unique.tolist()).issubset({0, 1})


def test_theta_lies_inside_box() -> None:
    sampler = DesignOnlySampler(_pool(200), (700.0, 2500.0), (0.1, 0.9), n_per_trial=8)
    batch = sampler.generate(n_trials=20, n_events=8, seed=0)
    e_k = batch.theta[:, 0]
    t_k = batch.theta[:, 1]
    assert np.all((e_k >= 700.0) & (e_k <= 2500.0))
    assert np.all((t_k >= 0.1) & (t_k <= 0.9))


def test_threshold_rule_matches_score_comparison() -> None:
    """For each trial, X_ki should equal 1[score_i >= T_k] for the chosen events."""
    pool = _pool(100, seed=3)
    sampler = DesignOnlySampler(pool, (500.0, 3000.0), (0.0, 1.0), n_per_trial=12)
    batch = sampler.generate(n_trials=8, n_events=12, seed=42)

    # Reproduce the sampler's RNG to recover (E_k, T_k) and check labels manually.
    rng = np.random.default_rng(42)
    e_k = rng.uniform(500.0, 3000.0, size=8)
    t_k = rng.uniform(0.0, 1.0, size=8)
    np.testing.assert_allclose(batch.theta[:, 0], e_k)
    np.testing.assert_allclose(batch.theta[:, 1], t_k)
    for k in range(8):
        near = pool.nearest_indices(e_k[k], 12)
        # The pool has >= 12 events, so no padding needed in this fixture.
        assert near.size == 12
        expected = (pool.score[near] >= t_k[k]).astype(np.int8)
        np.testing.assert_array_equal(batch.labels[k], expected)


def test_sampler_is_seed_reproducible() -> None:
    sampler = DesignOnlySampler(_pool(200), (500.0, 3000.0), (0.0, 1.0), n_per_trial=16)
    a = sampler.generate(n_trials=4, n_events=16, seed=7)
    b = sampler.generate(n_trials=4, n_events=16, seed=7)
    np.testing.assert_array_equal(a.theta, b.theta)
    np.testing.assert_array_equal(a.labels, b.labels)


def test_replacement_padding_when_pool_smaller_than_n() -> None:
    """With allow_replacement=True a small pool still produces n_events labels."""
    pool = _pool(5)
    sampler = DesignOnlySampler(
        pool, (500.0, 3000.0), (0.0, 1.0), n_per_trial=16, allow_replacement=True
    )
    batch = sampler.generate(n_trials=3, n_events=16, seed=0)
    assert batch.labels.shape == (3, 16)


def test_constructor_rejects_no_replacement_with_small_pool() -> None:
    with pytest.raises(ValueError, match="replacement"):
        DesignOnlySampler(
            _pool(5),
            (500.0, 3000.0),
            (0.0, 1.0),
            n_per_trial=16,
            allow_replacement=False,
        )


def test_constructor_rejects_bad_boxes() -> None:
    pool = _pool(50)
    with pytest.raises(ValueError, match="energy_box"):
        DesignOnlySampler(pool, (3000.0, 500.0), (0.0, 1.0), n_per_trial=8)
    with pytest.raises(ValueError, match="threshold_box"):
        DesignOnlySampler(pool, (500.0, 3000.0), (1.0, 0.0), n_per_trial=8)


def test_constructor_rejects_zero_n_per_trial() -> None:
    with pytest.raises(ValueError, match="n_per_trial"):
        DesignOnlySampler(_pool(20), (500.0, 3000.0), (0.0, 1.0), n_per_trial=0)
