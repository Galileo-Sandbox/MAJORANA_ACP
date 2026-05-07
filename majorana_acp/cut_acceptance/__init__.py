"""Cut-acceptance estimation via RESUM_FLEX (CNP + MFGP) on a 2D (energy, threshold) plane.

This subpackage adapts a trained PSD-classifier's per-event scores into a
multi-fidelity rare-event-design problem so that we can replace the
binned-with-binomial-error-bars acceptance plot in
``notebooks/data_visualization.ipynb`` §8.4 with a smooth, calibrated
β(E, T) heatmap that has principled uncertainty.

Mapping to the RESuM framework (S8 in their validation matrix):

- ``θ = (E, T) ∈ ℝ²`` — design parameters (energy bin centre, threshold).
- ``φ = None`` — DESIGN_ONLY mode; events have no per-event covariates
  beyond their identity.
- ``X_ki = 1[score_i ≥ T_k]`` — per-event binary pass/fail at threshold
  ``T_k`` for events near energy ``E_k``.

Two independent pipelines run on disjoint data slices: one with
``target_class=1`` (signal acceptance) and one with ``target_class=0``
(background rejection).
"""

from majorana_acp.cut_acceptance.config import CutAcceptanceConfig, load_config
from majorana_acp.cut_acceptance.partition import (
    PartitionedIndices,
    partition_events,
)
from majorana_acp.cut_acceptance.sampler import (
    DesignOnlySampler,
    EventPool,
    load_event_pool,
)

__all__ = [
    "CutAcceptanceConfig",
    "DesignOnlySampler",
    "EventPool",
    "PartitionedIndices",
    "load_config",
    "load_event_pool",
    "partition_events",
]
