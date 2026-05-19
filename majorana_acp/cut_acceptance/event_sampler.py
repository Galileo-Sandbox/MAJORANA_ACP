"""Event-level sampler for the True-CNP cut-acceptance pipeline.

The CNP trained by this pipeline is a textbook 1D-regression CNP: it
ingests per-event coordinates ``phi_i = (E_i_norm, T_norm)`` and binary
labels ``X_i``. There is no broadcasted trial-level theta — the encoder
sees each context event's own energy. This is the key difference from
``BinnedSampler``, which broadcast a single ``(E_bin_center, T)`` across
all events in a trial.

Spatial distribution of context energies is controlled by
``sampling_pattern`` (Axis B in the hybrid-scale matrix):

* ``flat_stratified`` — bin-uniform draw across the entire kept-bin
  spectrum. Flattens the training distribution so the loss is
  balanced; recovers the previous default exactly.
* ``mixed_density`` — pick a random focus bin; place
  ``local_event_fraction`` of events in a ``zoom_window_width_kev``-wide
  window around it; the remainder is drawn globally.
* ``random_clusters`` — pick ``n_clusters`` disjoint windows uniformly
  at random across the spectrum; events are split evenly between
  them. Events outside the clusters are *not* drawn. Forces the CNP
  to extrapolate β(E) into data-blind regions and widen σ_CNP honestly.
* ``physics_anchored`` — identical to ``mixed_density`` except the
  focus is sampled from ``physics_peaks_kev``. Concentrates training
  on physics-known acceptance-transition regions.

Per-trial composition:

* One ``T_k`` drawn via boundary-mix.
* ``n_events`` events, each carrying its own ``(E_i, T_k, X_i)``.

Emitted batches use ``InputMode.EVENT_ONLY`` (``theta=None``, per-event
``phi``).
"""

from __future__ import annotations

from typing import Literal

import numpy as np

# RESUM_FLEX schemas — installed as a library, imported directly.
from schemas.data_models import InputMode, StandardBatch

from majorana_acp.cut_acceptance.binned_sampler import (
    TSamplingStrategy,
    _BinIndex,
    load_events,
)
from majorana_acp.cut_acceptance.config import PositionalEncodingConfig
from majorana_acp.cut_acceptance.positional_encoding import encode_phi, phi_dim

__all__ = ["EventSampler", "load_events"]


SamplingPattern = Literal[
    "flat_stratified",
    "mixed_density",
    "random_clusters",
    "physics_anchored",
]

DensitySampling = Literal["bin_stratified", "continuous"]

_MAX_CLUSTER_REJECTIONS = 50


class EventSampler:
    """Bin-stratified event sampler emitting ``EVENT_ONLY`` batches.

    Duck-types the RESUM_FLEX ``PseudoDataGenerator`` (``mode``,
    ``dim_theta``, ``dim_phi``, ``generate``), so it drops directly into
    the existing ``train_cnp`` loop.

    Spatial pattern is decoupled from trial size — ``generate`` always
    uses the ``n_events`` value supplied by the caller (the training
    loop). The hybrid-scale "variable-N per step" wrapper lives in the
    pipeline, not here.
    """

    mode = InputMode.EVENT_ONLY
    dim_theta: None = None

    def __init__(
        self,
        energy: np.ndarray,
        score: np.ndarray,
        *,
        energy_range: tuple[float, float],
        energy_bin_width: float,
        threshold_range: tuple[float, float] = (0.0, 1.0),
        min_events_per_bin: int = 4,
        t_sampling: TSamplingStrategy = "boundary_mix",
        sampling_pattern: SamplingPattern = "flat_stratified",
        zoom_window_width_kev: float = 50.0,
        local_event_fraction: float = 0.70,
        n_clusters: int = 2,
        physics_peaks_kev: list[float] | None = None,
        positional_encoding: PositionalEncodingConfig | None = None,
        density_sampling: DensitySampling = "bin_stratified",
        density_kde_radius_kev: float = 10.0,
    ) -> None:
        if not threshold_range[1] > threshold_range[0]:
            raise ValueError(f"threshold_range must satisfy hi > lo, got {threshold_range}")
        if t_sampling not in ("uniform", "boundary_mix"):
            raise ValueError(f"t_sampling must be 'uniform' or 'boundary_mix', got {t_sampling!r}")
        if sampling_pattern not in (
            "flat_stratified",
            "mixed_density",
            "random_clusters",
            "physics_anchored",
        ):
            raise ValueError(f"unknown sampling_pattern: {sampling_pattern!r}")
        if zoom_window_width_kev <= 0.0:
            raise ValueError(f"zoom_window_width_kev must be > 0, got {zoom_window_width_kev}")
        if not 0.0 < local_event_fraction < 1.0:
            raise ValueError(f"local_event_fraction must be in (0, 1), got {local_event_fraction}")
        if n_clusters < 1:
            raise ValueError(f"n_clusters must be >= 1, got {n_clusters}")
        if density_sampling not in ("bin_stratified", "continuous"):
            raise ValueError(
                f"density_sampling must be 'bin_stratified' or 'continuous', "
                f"got {density_sampling!r}"
            )
        if density_kde_radius_kev <= 0.0:
            raise ValueError(f"density_kde_radius_kev must be > 0, got {density_kde_radius_kev}")

        self._index = _BinIndex(
            energy,
            score,
            energy_range=energy_range,
            bin_width=energy_bin_width,
            min_events_per_bin=min_events_per_bin,
        )
        self.energy_range = (float(energy_range[0]), float(energy_range[1]))
        self.energy_bin_width = float(energy_bin_width)
        self.threshold_range = (float(threshold_range[0]), float(threshold_range[1]))
        self.t_sampling: TSamplingStrategy = t_sampling
        self.sampling_pattern: SamplingPattern = sampling_pattern
        self.zoom_window_width_kev = float(zoom_window_width_kev)
        self.local_event_fraction = float(local_event_fraction)
        self.n_clusters = int(n_clusters)
        # Filter peaks to kept-bin range; assert non-empty at pipeline use time.
        peaks = physics_peaks_kev or []
        self.physics_peaks_kev = [
            float(p)
            for p in peaks
            if self.bin_centers[0] - 0.5 * self.energy_bin_width
            <= p
            <= self.bin_centers[-1] + 0.5 * self.energy_bin_width
        ]
        if sampling_pattern == "physics_anchored" and not self.physics_peaks_kev:
            raise ValueError(
                "sampling_pattern='physics_anchored' requires at least one peak "
                "in physics_peaks_kev inside the kept-bin energy range"
            )

        # Positional encoding (1D Fourier expansion of energy). Default
        # to disabled when caller passes None so the sampler stays
        # backward-compatible with the old 2-arg phi contract.
        self.positional_encoding: PositionalEncodingConfig = (
            positional_encoding or PositionalEncodingConfig()
        )
        # dim_phi is dynamic: 2*L + 1 when PE is on, 2 otherwise. The
        # CNP encoder/decoder are built from this same number.
        self.dim_phi: int = phi_dim(self.positional_encoding)

        # Continuous-density book-keeping. Always populate the kept-
        # event id list so the continuous draw shares the same kept-
        # event set as the bin-stratified path; only precompute the
        # inverse-density weights when actually needed.
        self.density_sampling: DensitySampling = density_sampling
        self.density_kde_radius_kev = float(density_kde_radius_kev)
        kept_event_ids = np.concatenate(self._index.bin_events).astype(np.int64)
        self._kept_event_ids = kept_event_ids
        self._kept_event_energies = self._index.energy[kept_event_ids]
        if self.density_sampling == "continuous":
            self._continuous_global_weights = self._compute_inverse_density_weights(
                self._kept_event_energies, self.density_kde_radius_kev
            )
        else:
            self._continuous_global_weights = None

    @staticmethod
    def _compute_inverse_density_weights(energies: np.ndarray, kde_radius_kev: float) -> np.ndarray:
        """Per-event inverse-density weight via a box kernel of half-
        width ``kde_radius_kev``. ``O(N log N)`` via sorted searchsorted.

        Returned weights are normalised to sum to 1 across the input
        pool, ready to be passed straight to ``rng.choice(p=...)``.
        """
        sort_idx = np.argsort(energies)
        e_sorted = energies[sort_idx]
        lo = np.searchsorted(e_sorted, e_sorted - kde_radius_kev, side="left")
        hi = np.searchsorted(e_sorted, e_sorted + kde_radius_kev, side="right")
        counts_sorted = (hi - lo).astype(np.float64)
        # Map back to the original (unsorted) event order so the weight
        # at slot i is for event i in ``energies``.
        counts = np.empty_like(counts_sorted)
        counts[sort_idx] = counts_sorted
        inv = 1.0 / counts
        return inv / inv.sum()

    # ----- introspection (used by pipeline + diagnostics) ---------------

    @property
    def n_bins(self) -> int:
        return len(self._index)

    @property
    def bin_centers(self) -> np.ndarray:
        return self._index.bin_centers

    @property
    def bin_event_counts(self) -> np.ndarray:
        return np.asarray([ev.size for ev in self._index.bin_events], dtype=np.int64)

    # ----- T sampler (shared scheme with BinnedSampler) -----------------

    def _sample_t(self, rng: np.random.Generator, n_trials: int) -> np.ndarray:
        t_lo, t_hi = self.threshold_range
        if self.t_sampling == "uniform":
            return rng.uniform(t_lo, t_hi, size=n_trials)
        band = 0.05 * (t_hi - t_lo)
        which = rng.choice(3, size=n_trials, p=[0.5, 0.25, 0.25])
        t_k = np.empty(n_trials, dtype=np.float64)
        m0, m1, m2 = which == 0, which == 1, which == 2
        if m0.any():
            t_k[m0] = rng.uniform(t_lo, t_hi, size=int(m0.sum()))
        if m1.any():
            t_k[m1] = rng.uniform(t_lo, t_lo + band, size=int(m1.sum()))
        if m2.any():
            t_k[m2] = rng.uniform(t_hi - band, t_hi, size=int(m2.sum()))
        return t_k

    # ----- Shared primitive: bin-stratified draw over a bin subset ------

    def _bin_stratified_draw(
        self,
        rng: np.random.Generator,
        n: int,
        bin_indices: np.ndarray,
    ) -> np.ndarray:
        """Draw ``n`` pool-event indices via uniform-over-bins, then
        uniform-within-bin. ``bin_indices`` is an array of kept-bin
        indices (into ``self._index.bin_events``) defining the active
        subset. Empty subsets are a configuration error — raise.
        """
        if n == 0:
            return np.empty(0, dtype=np.int64)
        if bin_indices.size == 0:
            raise ValueError("_bin_stratified_draw: empty bin_indices subset")
        # Step 1: each of n slots picks a bin uniformly from bin_indices.
        picked_bin = bin_indices[rng.integers(0, bin_indices.size, size=n)]
        # Step 2: within each picked bin, draw uniformly from its event pool.
        out = np.empty(n, dtype=np.int64)
        for b in np.unique(picked_bin):
            mask = picked_bin == b
            pool = self._index.bin_events[int(b)]
            out[mask] = pool[rng.integers(0, pool.size, size=int(mask.sum()))]
        return out

    # ----- Continuous draws (continuous density sampling mode) ---------

    def _continuous_global_draw(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """Inverse-density-weighted draw from the kept-event pool. Returns
        pool-event indices (into ``self._index.energy``)."""
        if n == 0:
            return np.empty(0, dtype=np.int64)
        kept_idx = rng.choice(
            self._kept_event_ids.size,
            size=n,
            replace=True,
            p=self._continuous_global_weights,
        )
        return self._kept_event_ids[kept_idx]

    def _continuous_local_draw(
        self,
        rng: np.random.Generator,
        n: int,
        focus_e: float,
        half_w: float,
    ) -> np.ndarray:
        """Uniform draw (with replacement) from kept events satisfying
        ``|E_i − focus_e| ≤ half_w``. Returns pool-event indices.

        Raises if the local pool is empty — same error semantics as the
        bin-stratified ``_bin_stratified_draw`` on an empty subset.
        """
        if n == 0:
            return np.empty(0, dtype=np.int64)
        mask = np.abs(self._kept_event_energies - focus_e) <= half_w
        local_pool_size = int(mask.sum())
        if local_pool_size == 0:
            raise ValueError(
                f"continuous local draw: focus {focus_e:.1f} keV has no kept "
                f"events within ±{half_w:.1f} keV"
            )
        local_event_ids = self._kept_event_ids[mask]
        # Uniform within the local set, with replacement so n_local can
        # exceed the local-pool size (consistent with the bin-stratified
        # path, which also samples within-bin with replacement).
        idx = rng.integers(0, local_event_ids.size, size=n)
        return local_event_ids[idx]

    def _bins_in_window(self, e_center: float, half_w: float) -> np.ndarray:
        """Kept-bin indices whose bin overlaps the window
        ``[e_center − half_w, e_center + half_w]``. A bin is included
        if any part of it lies inside (half-bin pad on each side)."""
        pad = 0.5 * self.energy_bin_width
        return np.flatnonzero(np.abs(self.bin_centers - e_center) <= half_w + pad)

    # ----- Pattern dispatch + 4 per-trial draw routines ----------------

    def _draw_flat_stratified(self, rng: np.random.Generator, n: int) -> np.ndarray:
        if self.density_sampling == "continuous":
            return self._continuous_global_draw(rng, n)
        return self._bin_stratified_draw(rng, n, np.arange(self.n_bins))

    def _draw_mixed_density(self, rng: np.random.Generator, n: int) -> np.ndarray:
        focus_idx = int(rng.integers(0, self.n_bins))
        return self._foveated_draw(rng, n, focus_e=float(self.bin_centers[focus_idx]))

    def _draw_physics_anchored(self, rng: np.random.Generator, n: int) -> np.ndarray:
        focus_e = float(rng.choice(self.physics_peaks_kev))
        return self._foveated_draw(rng, n, focus_e=focus_e)

    def _foveated_draw(self, rng: np.random.Generator, n: int, *, focus_e: float) -> np.ndarray:
        """Common engine for ``mixed_density`` and ``physics_anchored``:
        a fraction of events in a window around ``focus_e``, the rest
        drawn globally. Branches on ``density_sampling`` — both branches
        respect the same ``zoom_window_width_kev`` half-window for the
        local fraction."""
        half_w = 0.5 * self.zoom_window_width_kev
        n_local = int(round(n * self.local_event_fraction))
        n_global = n - n_local
        if self.density_sampling == "continuous":
            local = self._continuous_local_draw(rng, n_local, focus_e, half_w)
            global_ = self._continuous_global_draw(rng, n_global)
        else:
            local_bins = self._bins_in_window(focus_e, half_w)
            if local_bins.size == 0:
                raise ValueError(
                    f"focus energy {focus_e:.1f} keV has no kept bins within "
                    f"±{half_w:.1f} keV — check energy_range / min_events_per_bin"
                )
            local = self._bin_stratified_draw(rng, n_local, local_bins)
            global_ = self._bin_stratified_draw(rng, n_global, np.arange(self.n_bins))
        out = np.concatenate([local, global_])
        rng.shuffle(out)
        return out

    def _draw_random_clusters(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """Pick ``n_clusters`` disjoint window centers via rejection
        sampling, then split ``n`` events evenly among them. Raise if
        the requested cluster count won't fit non-overlapping after
        :data:`_MAX_CLUSTER_REJECTIONS` attempts."""
        w = self.zoom_window_width_kev
        half_w = 0.5 * w
        centers: list[float] = []
        attempts = 0
        while len(centers) < self.n_clusters:
            if attempts >= _MAX_CLUSTER_REJECTIONS:
                raise ValueError(
                    f"random_clusters: could not place {self.n_clusters} "
                    f"non-overlapping windows of width {w} keV across "
                    f"{self.n_bins} kept bins in "
                    f"{_MAX_CLUSTER_REJECTIONS} rejection attempts. "
                    "Reduce n_clusters or zoom_window_width_kev."
                )
            attempts += 1
            idx = int(rng.integers(0, self.n_bins))
            e_cand = float(self.bin_centers[idx])
            # Strict non-overlap: window centers must be > w apart.
            if all(abs(e_cand - c) > w for c in centers):
                centers.append(e_cand)

        # Distribute n event slots across clusters: floor split + 1 to
        # the first `remainder` clusters; shuffle so the extras land
        # randomly rather than always on cluster 0.
        base, remainder = divmod(n, self.n_clusters)
        per_cluster = np.array(
            [base + (1 if i < remainder else 0) for i in range(self.n_clusters)],
            dtype=np.int64,
        )
        rng.shuffle(per_cluster)

        pieces: list[np.ndarray] = []
        for c, k in zip(centers, per_cluster, strict=True):
            if k == 0:
                continue
            if self.density_sampling == "continuous":
                pieces.append(self._continuous_local_draw(rng, int(k), c, half_w))
            else:
                bins_here = self._bins_in_window(c, half_w)
                pieces.append(self._bin_stratified_draw(rng, int(k), bins_here))
        out = np.concatenate(pieces) if pieces else np.empty(0, dtype=np.int64)
        rng.shuffle(out)
        return out

    def _draw_one_trial(self, rng: np.random.Generator, n: int) -> np.ndarray:
        if self.sampling_pattern == "flat_stratified":
            return self._draw_flat_stratified(rng, n)
        if self.sampling_pattern == "mixed_density":
            return self._draw_mixed_density(rng, n)
        if self.sampling_pattern == "random_clusters":
            return self._draw_random_clusters(rng, n)
        if self.sampling_pattern == "physics_anchored":
            return self._draw_physics_anchored(rng, n)
        raise AssertionError(f"unhandled sampling_pattern: {self.sampling_pattern!r}")

    # ----- StandardBatch generator (RESUM_FLEX duck-type) ---------------

    def generate(self, n_trials: int, n_events: int, seed: int) -> StandardBatch:
        if n_trials < 1:
            raise ValueError(f"n_trials must be >= 1, got {n_trials}")
        if n_events < 1:
            raise ValueError(f"n_events must be >= 1, got {n_events}")

        rng = np.random.default_rng(int(seed))
        t_k = self._sample_t(rng, n_trials)

        event_indices = np.empty((n_trials, n_events), dtype=np.int64)
        for k in range(n_trials):
            event_indices[k] = self._draw_one_trial(rng, n_events)

        # Per-event features carry the event's **real** energy, not the
        # bin center — bins are only a stratification grid.
        energy_picked = self._index.energy[event_indices]  # [B, N]
        score_picked = self._index.score[event_indices]  # [B, N]
        labels = (score_picked >= t_k[:, None]).astype(np.int8)

        e_lo, e_hi = self.energy_range
        t_lo, t_hi = self.threshold_range
        e_norm = (energy_picked - e_lo) / (e_hi - e_lo)  # [B, N]
        t_norm = np.broadcast_to(((t_k - t_lo) / (t_hi - t_lo))[:, None], e_norm.shape)  # [B, N]
        phi = np.stack([e_norm, t_norm], axis=-1).astype(np.float64)  # [B, N, 2]
        # 1D Fourier positional encoding of E (no-op when disabled →
        # bitwise identity on the (..., 2) array).
        phi = encode_phi(phi, self.positional_encoding, energy_range_kev=self.energy_range)

        return StandardBatch(
            mode=InputMode.EVENT_ONLY,
            theta=None,
            phi=phi,
            labels=labels,
        )
