"""Cut-acceptance estimation via a binned CNP on the (E_bin, T) plane.

The CNP learns β(E_bin_center, T) = P(score ≥ T | event in that bin),
trained on classifier predictions over the **train** Majorana split and
validated against an empirical binned pass rate computed on the **test**
split. No MFGP, no peak/continuum partition — see ``pipeline.py``.

Mapping to the RESuM framework (S8 in their validation matrix):

- ``θ = (E_bin_center, T) ∈ ℝ²`` — design parameters in normalized
  ``[0, 1]^2``.
- ``φ = None`` — DESIGN_ONLY mode.
- ``X_ki = 1[score_i ≥ T_k]`` — per-event binary pass/fail.

Three runs per (group, model): ``target_class=1`` (signal acceptance),
``target_class=0`` (background rejection), and ``target_class="all"``
(inclusive pass rate marginalised over the natural class composition).
"""

from majorana_acp.cut_acceptance.binned_sampler import BinnedSampler, load_events
from majorana_acp.cut_acceptance.config import CutAcceptanceConfig, load_config

__all__ = [
    "BinnedSampler",
    "CutAcceptanceConfig",
    "load_config",
    "load_events",
]
