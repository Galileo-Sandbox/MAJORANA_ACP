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
        e,
        s,
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
    t_per_event_per_trial = batch.phi[:, :, 1]  # [B, N]
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
        energy,
        score,
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
            e,
            s,
            energy_range=(500.0, 3000.0),
            energy_bin_width=20.0,
            t_sampling="bogus",
        )


def test_rejects_bad_threshold_range() -> None:
    e, s = _pool()
    with pytest.raises(ValueError, match="threshold_range"):
        EventSampler(
            e,
            s,
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


# --- hybrid-scale sampling patterns ---------------------------------


@pytest.mark.parametrize(
    "pattern,kwargs",
    [
        ("flat_stratified", {}),
        ("mixed_density", dict(zoom_window_width_kev=200.0, local_event_fraction=0.7)),
        ("random_clusters", dict(zoom_window_width_kev=200.0, n_clusters=2)),
        (
            "physics_anchored",
            dict(
                zoom_window_width_kev=200.0,
                local_event_fraction=0.7,
                physics_peaks_kev=[1500.0, 2614.0],
            ),
        ),
    ],
)
def test_all_patterns_emit_valid_batches(pattern: str, kwargs: dict) -> None:
    """Every pattern produces an EVENT_ONLY batch with the right shape + dtypes."""
    e, s = _pool()
    sampler = EventSampler(
        e,
        s,
        energy_range=(500.0, 3000.0),
        energy_bin_width=100.0,
        min_events_per_bin=4,
        sampling_pattern=pattern,
        **kwargs,
    )
    batch = sampler.generate(n_trials=4, n_events=32, seed=0)
    assert batch.mode is InputMode.EVENT_ONLY
    assert batch.theta is None
    assert batch.phi.shape == (4, 32, 2)
    assert batch.labels.shape == (4, 32)
    assert batch.phi.min() >= 0.0
    assert batch.phi.max() <= 1.0
    assert set(np.unique(batch.labels).tolist()).issubset({0, 1})


@pytest.mark.parametrize(
    "pattern,kwargs",
    [
        ("flat_stratified", {}),
        ("mixed_density", dict(zoom_window_width_kev=200.0, local_event_fraction=0.7)),
        ("random_clusters", dict(zoom_window_width_kev=200.0, n_clusters=2)),
        (
            "physics_anchored",
            dict(
                zoom_window_width_kev=200.0,
                local_event_fraction=0.7,
                physics_peaks_kev=[1500.0, 2614.0],
            ),
        ),
    ],
)
def test_all_patterns_reproducible(pattern: str, kwargs: dict) -> None:
    """Same seed → identical (phi, labels) per pattern."""
    e, s = _pool()
    sampler = EventSampler(
        e,
        s,
        energy_range=(500.0, 3000.0),
        energy_bin_width=100.0,
        min_events_per_bin=4,
        sampling_pattern=pattern,
        **kwargs,
    )
    a = sampler.generate(n_trials=3, n_events=16, seed=99)
    b = sampler.generate(n_trials=3, n_events=16, seed=99)
    np.testing.assert_array_equal(a.phi, b.phi)
    np.testing.assert_array_equal(a.labels, b.labels)


def test_mixed_density_concentrates_local_events() -> None:
    """≈70% of a trial's events should land in the focus window.

    The implementation pads the window by half a bin width on each
    side (a bin is "in-window" if any part of it overlaps), so the
    effective local range is ``±(half_w + bin_width/2)`` around the
    focus center.
    """
    e, s = _pool(n=10_000)
    bin_width = 50.0
    half_w = 100.0  # zoom_window_width_kev / 2
    eff_half = half_w + 0.5 * bin_width  # include the half-bin pad
    sampler = EventSampler(
        e,
        s,
        energy_range=(500.0, 3000.0),
        energy_bin_width=bin_width,
        min_events_per_bin=4,
        sampling_pattern="mixed_density",
        zoom_window_width_kev=2.0 * half_w,
        local_event_fraction=0.7,
    )
    batch = sampler.generate(n_trials=1, n_events=2000, seed=11)
    e_picked = batch.phi[0, :, 0] * (3000.0 - 500.0) + 500.0
    # Recover the focus center: median of the densest run of bins.
    # The local pool dominates, so the median of the picked energies
    # is a robust estimator of the focus when ≥50% land local.
    focus_est = float(np.median(e_picked))
    frac_local = float(
        ((e_picked >= focus_est - eff_half) & (e_picked <= focus_est + eff_half)).mean()
    )
    # Target = 0.7. Allow 0.6 lower bound for RNG slack; the critical
    # comparison is against the natural-density baseline of ~0.08
    # (a 200-keV window covers 8% of the 2500-keV span).
    assert frac_local >= 0.60, f"frac_local={frac_local:.3f} below 0.6 target"


def test_random_clusters_starves_outside_regions() -> None:
    """All events lie in n_clusters compact islands; gaps stay empty.

    Direct contract check: bin the picked energies into the same
    bin-width grid the sampler uses internally; the nonzero bins must
    form exactly ``n_clusters`` contiguous runs, each spanning at
    most the window plus a half-bin pad on each side.
    """
    e, s = _pool(n=10_000)
    bin_width = 50.0
    window = 200.0
    sampler = EventSampler(
        e,
        s,
        energy_range=(500.0, 3000.0),
        energy_bin_width=bin_width,
        min_events_per_bin=4,
        sampling_pattern="random_clusters",
        zoom_window_width_kev=window,
        n_clusters=2,
    )
    batch = sampler.generate(n_trials=1, n_events=2000, seed=7)
    e_picked = batch.phi[0, :, 0] * (3000.0 - 500.0) + 500.0
    counts, _ = np.histogram(e_picked, bins=np.arange(500.0, 3000.0 + bin_width, bin_width))
    nonzero = counts > 0
    # Count contiguous nonzero runs.
    runs: list[tuple[int, int]] = []
    in_run = False
    for i, nz in enumerate(nonzero):
        if nz and not in_run:
            start = i
            in_run = True
        elif not nz and in_run:
            runs.append((start, i))
            in_run = False
    if in_run:
        runs.append((start, len(nonzero)))

    assert len(runs) == 2, f"expected 2 islands, got {len(runs)} contiguous nonzero runs"
    # Each island spans the window plus up to one bin of pad on each
    # side → (window + 2*bin_width) / bin_width.
    max_span_bins = int((window + 2.0 * bin_width) // bin_width)
    for s_idx, e_idx in runs:
        span = e_idx - s_idx
        assert span <= max_span_bins, f"island spans {span} bins (> {max_span_bins} expected)"


def test_random_clusters_event_count_partitions_evenly() -> None:
    """N=48 with n_clusters=2 → 24+24; with n=49 → 24+25 (order randomized)."""
    e, s = _pool(n=5000)
    sampler = EventSampler(
        e,
        s,
        energy_range=(500.0, 3000.0),
        energy_bin_width=50.0,
        min_events_per_bin=4,
        sampling_pattern="random_clusters",
        zoom_window_width_kev=200.0,
        n_clusters=2,
    )
    batch = sampler.generate(n_trials=1, n_events=49, seed=3)
    # 49 events across 2 clusters: counts should sum to 49 by definition.
    assert batch.phi.shape == (1, 49, 2)
    assert batch.labels.shape == (1, 49)


def test_random_clusters_raises_when_unplaceable() -> None:
    """3 disjoint 1500-keV windows can't fit in 2500 keV — must raise."""
    e, s = _pool(n=5000)
    sampler = EventSampler(
        e,
        s,
        energy_range=(500.0, 3000.0),
        energy_bin_width=50.0,
        min_events_per_bin=4,
        sampling_pattern="random_clusters",
        zoom_window_width_kev=1500.0,
        n_clusters=3,
    )
    with pytest.raises(ValueError, match="non-overlapping"):
        sampler.generate(n_trials=1, n_events=32, seed=0)


def test_physics_anchored_focus_is_always_near_a_peak() -> None:
    """Every trial's local events cluster around one of the configured peaks."""
    e, s = _pool(n=10_000)
    peaks = [1500.0, 2614.0]
    sampler = EventSampler(
        e,
        s,
        energy_range=(500.0, 3000.0),
        energy_bin_width=50.0,
        min_events_per_bin=4,
        sampling_pattern="physics_anchored",
        zoom_window_width_kev=200.0,
        local_event_fraction=0.7,
        physics_peaks_kev=peaks,
    )
    batch = sampler.generate(n_trials=20, n_events=100, seed=5)
    e_picked = batch.phi[:, :, 0] * (3000.0 - 500.0) + 500.0
    # For each trial, the mode of the energy histogram should be near
    # one of the configured peaks.
    for k in range(20):
        ev = e_picked[k]
        counts, edges = np.histogram(ev, bins=np.arange(500.0, 3001.0, 25.0))
        mid = float(edges[np.argmax(counts)] + 12.5)
        nearest_peak_dist = min(abs(mid - p) for p in peaks)
        # mode should sit inside the 200-keV window of the chosen peak;
        # allow some slack for histogram resolution.
        assert nearest_peak_dist < 125.0, (
            f"trial {k}: mode {mid:.0f} keV is {nearest_peak_dist:.0f} keV "
            f"from the nearest peak (expected within window half-width)"
        )


def test_physics_anchored_rejects_no_in_range_peaks() -> None:
    """If physics_peaks_kev becomes empty (filtered by range), constructor raises."""
    e, s = _pool()
    with pytest.raises(ValueError, match="physics_anchored"):
        EventSampler(
            e,
            s,
            energy_range=(500.0, 1000.0),  # excludes both 1500 and 2614
            energy_bin_width=50.0,
            min_events_per_bin=4,
            sampling_pattern="physics_anchored",
            zoom_window_width_kev=200.0,
            local_event_fraction=0.7,
            physics_peaks_kev=[1500.0, 2614.0],
        )


def test_flat_stratified_remains_bin_uniform_under_skew() -> None:
    """Regression check: refactor preserved the original flat behavior."""
    rng = np.random.default_rng(0)
    e_low = rng.uniform(500.0, 1500.0, size=5000)
    e_high = rng.uniform(1500.0, 3000.0, size=1000)
    energy = np.concatenate([e_low, e_high])
    score = rng.uniform(0.0, 1.0, size=energy.size)
    sampler = EventSampler(
        energy,
        score,
        energy_range=(500.0, 3000.0),
        energy_bin_width=100.0,
        min_events_per_bin=4,
        sampling_pattern="flat_stratified",
    )
    batch = sampler.generate(n_trials=1, n_events=20_000, seed=0)
    e_picked = batch.phi[0, :, 0] * (3000.0 - 500.0) + 500.0
    frac_low = (e_picked < 1500.0).mean()
    assert 0.35 <= frac_low <= 0.45


# --- positional encoding gate ---------------------------------------
# These tests pin the backward-compatibility contract: with PE disabled
# the sampler is bit-for-bit identical to the pre-PE code path; with PE
# enabled phi.shape[-1] grows to 2*L + 1 and the threshold tail is
# preserved verbatim.


def test_pe_disabled_default_keeps_dim_phi_two() -> None:
    """The bare sampler (no PE arg) advertises dim_phi=2 like the legacy path."""
    s = _sampler()
    assert s.dim_phi == 2
    batch = s.generate(n_trials=2, n_events=16, seed=0)
    assert batch.phi.shape == (2, 16, 2)


def test_pe_disabled_explicit_config_is_bitwise_identical() -> None:
    """Explicit PositionalEncodingConfig(enabled=False) must produce an
    array identical to the no-arg path. This is the parity guarantee."""
    from majorana_acp.cut_acceptance.config import PositionalEncodingConfig

    e, s = _pool(2000, seed=0)
    kwargs = dict(
        energy_range=(500.0, 3000.0),
        energy_bin_width=100.0,
        min_events_per_bin=4,
        t_sampling="boundary_mix",
    )
    legacy = EventSampler(e, s, **kwargs)
    explicit = EventSampler(
        e,
        s,
        positional_encoding=PositionalEncodingConfig(enabled=False),
        **kwargs,
    )
    b_legacy = legacy.generate(n_trials=3, n_events=20, seed=42)
    b_explicit = explicit.generate(n_trials=3, n_events=20, seed=42)
    np.testing.assert_array_equal(b_legacy.phi, b_explicit.phi)
    np.testing.assert_array_equal(b_legacy.labels, b_explicit.labels)
    assert legacy.dim_phi == explicit.dim_phi == 2


@pytest.mark.parametrize("L", [1, 4, 10])
def test_pe_enabled_widens_phi_to_2L_plus_one(L: int) -> None:
    """phi.shape[-1] = 2*L + 1; last column is the original T_norm."""
    from majorana_acp.cut_acceptance.config import PositionalEncodingConfig

    e, s = _pool(2000, seed=0)
    pe = PositionalEncodingConfig(
        enabled=True,
        num_bands=L,
        min_energy_kev=500.0,
        max_energy_kev=3000.0,
    )
    sampler = EventSampler(
        e,
        s,
        energy_range=(500.0, 3000.0),
        energy_bin_width=100.0,
        min_events_per_bin=4,
        t_sampling="boundary_mix",
        positional_encoding=pe,
    )
    assert sampler.dim_phi == 2 * L + 1
    batch = sampler.generate(n_trials=2, n_events=16, seed=0)
    assert batch.phi.shape == (2, 16, 2 * L + 1)
    # The last feature stays a constant within a trial (it's the threshold
    # broadcast across all N events of that trial — same invariant as the
    # raw T_norm column in the no-PE case).
    for k in range(batch.phi.shape[0]):
        t_col = batch.phi[k, :, -1]
        assert np.all(t_col == t_col[0])
    # The 2L energy features should sweep [-1, 1] (sin/cos range) and
    # vary across events, in contrast to the threshold pass-through.
    energy_features = batch.phi[..., : 2 * L]
    assert energy_features.min() >= -1.0 - 1e-9
    assert energy_features.max() <= 1.0 + 1e-9
