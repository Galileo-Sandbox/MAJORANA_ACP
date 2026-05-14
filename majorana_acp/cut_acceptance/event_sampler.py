"""Event-level sampler for the True-CNP cut-acceptance pipeline.

The CNP trained by this pipeline is a textbook 1D-regression CNP: it
ingests per-event coordinates ``phi_i = (E_i_norm, T_norm)`` and binary
labels ``X_i``. There is no broadcasted trial-level theta — the encoder
sees each context event's own energy. This is the key difference from
``BinnedSampler``, which broadcast a single ``(E_bin_center, T)`` across
all events in a trial.

To make the training distribution flat across the spectrum (so the
model learns sparse high-energy regions as well as the dense low-energy
ones), the sampler uses a **bin-stratified** draw: each event slot
picks a kept bin uniformly at random, then picks one event uniformly
from that bin's pool. The bin grid is used **only** as a sampling-
stratification mechanism — the CNP itself never sees bin boundaries.

Per-trial composition:

* One ``T_k`` drawn via boundary-mix (same scheme as ``BinnedSampler``).
* ``n_events`` events, each carrying its own ``(E_i, T_k, X_i)``.

Emitted batches use ``InputMode.EVENT_ONLY`` (``theta=None``, per-event
``phi``).
"""

from __future__ import annotations

import numpy as np

# RESUM_FLEX schemas — installed as a library, imported directly.
from schemas.data_models import InputMode, StandardBatch

from majorana_acp.cut_acceptance.binned_sampler import (
    TSamplingStrategy,
    _BinIndex,
    load_events,
)

__all__ = ["EventSampler", "load_events"]


class EventSampler:
    """Bin-stratified event sampler emitting ``EVENT_ONLY`` batches.

    Duck-types the RESUM_FLEX ``PseudoDataGenerator`` (``mode``,
    ``dim_theta``, ``dim_phi``, ``generate``), so it drops directly into
    the existing ``train_cnp`` loop.

    Parameters
    ----------
    energy, score
        Pool arrays (one entry per event), already filtered by class +
        energy range upstream (typically via :func:`load_events`).
    energy_range
        Inclusive ``(lo, hi)`` energy window in keV. Same role as in
        ``BinnedSampler``: defines the normalization ``E_norm = (E - lo)
        / (hi - lo)`` used in ``phi``.
    energy_bin_width
        Width (keV) of the **stratification** grid. The CNP never sees
        bin centers; this only controls how aggressively we flatten the
        training distribution.
    threshold_range
        Inclusive ``(lo, hi)`` interval the per-trial ``T_k`` is drawn
        from. Defaults to ``(0.0, 1.0)`` (the CNN-score domain).
    min_events_per_bin
        Bins with fewer events than this are excluded from the
        stratification pool — events in those bins do not contribute to
        training. Default 4.
    t_sampling
        ``"boundary_mix"`` (default) or ``"uniform"``. Boundary-mix puts
        50% of trials at uniform-T, 25% near ``t_lo``, 25% near
        ``t_hi`` — the CNN-score distribution is bimodal and the
        empirical β(T) plateau makes uniform-T trials under-cover the
        cliffs.
    """

    mode = InputMode.EVENT_ONLY
    dim_theta: None = None
    dim_phi: int = 2

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
    ) -> None:
        if not threshold_range[1] > threshold_range[0]:
            raise ValueError(
                f"threshold_range must satisfy hi > lo, got {threshold_range}"
            )
        if t_sampling not in ("uniform", "boundary_mix"):
            raise ValueError(
                f"t_sampling must be 'uniform' or 'boundary_mix', got {t_sampling!r}"
            )

        self._index = _BinIndex(
            energy, score,
            energy_range=energy_range,
            bin_width=energy_bin_width,
            min_events_per_bin=min_events_per_bin,
        )
        self.energy_range = (float(energy_range[0]), float(energy_range[1]))
        self.energy_bin_width = float(energy_bin_width)
        self.threshold_range = (float(threshold_range[0]), float(threshold_range[1]))
        self.t_sampling: TSamplingStrategy = t_sampling

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

    # ----- StandardBatch generator (RESUM_FLEX duck-type) ---------------

    def generate(self, n_trials: int, n_events: int, seed: int) -> StandardBatch:
        if n_trials < 1:
            raise ValueError(f"n_trials must be >= 1, got {n_trials}")
        if n_events < 1:
            raise ValueError(f"n_events must be >= 1, got {n_events}")

        rng = np.random.default_rng(int(seed))
        t_k = self._sample_t(rng, n_trials)

        n_kept = self.n_bins
        # Bin-stratified per-event draw:
        #   step 1: each event slot picks a kept bin uniformly at random.
        #   step 2: each event slot picks one pool index uniformly from
        #           that bin's event list.
        # The two-step scheme flattens the training distribution across
        # the bin grid regardless of natural energy density.
        bin_choices = rng.integers(0, n_kept, size=(n_trials, n_events))
        event_indices = np.empty((n_trials, n_events), dtype=np.int64)
        for b in range(n_kept):
            mask = bin_choices == b
            if not mask.any():
                continue
            pool = self._index.bin_events[b]
            picks_in_bin = rng.integers(0, pool.size, size=int(mask.sum()))
            event_indices[mask] = pool[picks_in_bin]

        # Per-event features carry the event's **real** energy, not the
        # bin center — bins are only a stratification grid.
        energy_picked = self._index.energy[event_indices]   # [B, N]
        score_picked = self._index.score[event_indices]     # [B, N]
        labels = (score_picked >= t_k[:, None]).astype(np.int8)

        e_lo, e_hi = self.energy_range
        t_lo, t_hi = self.threshold_range
        e_norm = (energy_picked - e_lo) / (e_hi - e_lo)                  # [B, N]
        t_norm = np.broadcast_to(
            ((t_k - t_lo) / (t_hi - t_lo))[:, None], e_norm.shape
        )                                                                # [B, N]
        phi = np.stack([e_norm, t_norm], axis=-1).astype(np.float64)     # [B, N, 2]

        return StandardBatch(
            mode=InputMode.EVENT_ONLY,
            theta=None,
            phi=phi,
            labels=labels,
        )
