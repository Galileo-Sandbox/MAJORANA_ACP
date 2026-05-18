"""1D Fourier-feature positional encoding for the energy coordinate.

When ``PositionalEncodingConfig.enabled``, the per-event energy scalar
``E`` is lifted into a ``2 * num_bands``-dimensional sinusoidal basis
before it reaches the CNP's encoder / decoder MLPs. The threshold
dimension ``T`` is kept as a single scalar and appended at the end of
the phi vector, giving:

    dim_phi  =  2 * num_bands + 1       (enabled)
    dim_phi  =  2                       (disabled, pass-through)

The transformation is intentionally implemented as pure NumPy and
applied at *phi-construction time* — in the ``EventSampler`` for
training and in the inference helper. The CNP module itself never
sees the config flag; it just gets a ``StandardBatch`` with the
already-encoded ``dim_phi``. That keeps RESUM_FLEX upstream untouched
and makes the gate trivially testable: ``enabled=False`` is a bitwise
identity on the input ``phi`` array.

Formula
-------
With ``E_min``, ``E_max`` from the config:

    E_norm  = (E - E_min) / (E_max - E_min)         # ∈ [0, 1] for in-range E
    γ(E_norm) = [
        sin(2^k π E_norm), cos(2^k π E_norm)
        for k in 0 .. num_bands - 1
    ]                                                # shape (..., 2 * num_bands)

The k-th band has period ``2 / 2^k`` in the normalized coordinate; at
``num_bands = 10`` over a 2500 keV range, the highest band's period is
``2500 / 2^9 ≈ 4.9 keV`` (`2π` per cycle → quarter-cycle in ~2.5 keV).
"""

from __future__ import annotations

import numpy as np

from majorana_acp.cut_acceptance.config import PositionalEncodingConfig


def phi_dim(pe_cfg: PositionalEncodingConfig) -> int:
    """Final dim_phi after applying ``encode_phi`` with this config.

    ``2 * num_bands + 1`` when enabled (encoded energy + scalar T);
    ``2`` when disabled (raw (E_norm, T_norm) pass-through).
    """
    if pe_cfg.enabled:
        return 2 * pe_cfg.num_bands + 1
    return 2


def encode_energy(e_norm: np.ndarray, num_bands: int) -> np.ndarray:
    """Lift normalized energy ``E_norm`` ∈ [0, 1] into a ``2L``-dim sin/cos basis.

    Parameters
    ----------
    e_norm
        Input of arbitrary leading shape ``(...)`` with values in
        ``[0, 1]`` (no clamping is applied; out-of-range inputs still
        evaluate, but the periodicity may alias).
    num_bands
        Number of frequency bands ``L``. Must be ≥ 1.

    Returns
    -------
    np.ndarray
        Shape ``(..., 2 * num_bands)``. Layout is interleaved
        ``[sin_0, cos_0, sin_1, cos_1, ..., sin_{L-1}, cos_{L-1}]``
        so that paired sin/cos for the same band stay adjacent in
        memory — that's the layout the user signed off on as
        "contiguous γ(E)".
    """
    if num_bands < 1:
        raise ValueError(f"num_bands must be >= 1, got {num_bands}")
    e = np.asarray(e_norm, dtype=np.float64)
    # Band indices 0 .. L-1; frequencies 2^k π in the normalized coord.
    k = np.arange(num_bands, dtype=np.float64)
    freqs = (2.0**k) * np.pi  # shape (L,)
    # angles[..., k] = freqs[k] * e[...]
    angles = e[..., None] * freqs  # shape (..., L)
    s = np.sin(angles)
    c = np.cos(angles)
    # Interleave (sin_k, cos_k) along last axis to preserve band locality.
    out = np.empty(angles.shape[:-1] + (2 * num_bands,), dtype=np.float64)
    out[..., 0::2] = s
    out[..., 1::2] = c
    return out


def encode_phi(
    phi: np.ndarray,
    pe_cfg: PositionalEncodingConfig,
    *,
    energy_range_kev: tuple[float, float] | None = None,
) -> np.ndarray:
    """Apply PE to a ``(..., 2)`` phi array of (E_norm, T_norm).

    Parameters
    ----------
    phi
        Shape ``(..., 2)``. The first feature is treated as
        ``E_norm``, the second as ``T_norm``. ``E_norm`` is the
        already-normalized coordinate against the sampler's
        ``energy_range`` (i.e. in ``[0, 1]``), *not* the raw E in keV.
    pe_cfg
        Positional-encoding config. When ``enabled=False`` the function
        is a bitwise identity on ``phi`` — same array returned (caller
        may rely on this for parity testing).
    energy_range_kev
        Optional ``(E_min, E_max)`` that produced ``E_norm`` in ``phi``.
        Provided so the PE config's own ``(min_energy_kev,
        max_energy_kev)`` can re-normalize when it disagrees with the
        sampler's range. In practice the two should match and the
        re-normalization is a no-op; we just guard against the user
        setting them to different ranges in YAML.

    Returns
    -------
    np.ndarray
        ``phi`` unchanged when disabled (``dim_phi = 2``).
        ``(..., 2L + 1)`` when enabled, with layout
        ``[γ(E_norm), T_norm]``: 2L encoded energy features followed
        by the original scalar threshold.
    """
    if not pe_cfg.enabled:
        return phi
    phi = np.asarray(phi)
    if phi.shape[-1] != 2:
        raise ValueError(f"encode_phi expects shape (..., 2), got {phi.shape}")
    e_norm = phi[..., 0]
    t_norm = phi[..., 1]
    # Re-normalize from the sampler's energy_range to the PE's window
    # only if they differ. The PE's (min, max) take effect here so the
    # encoded coordinate uses the config-declared window even if the
    # sampler trained on a wider energy_range.
    if energy_range_kev is not None:
        e_lo_src, e_hi_src = energy_range_kev
        e_lo_pe, e_hi_pe = pe_cfg.min_energy_kev, pe_cfg.max_energy_kev
        if (e_lo_src, e_hi_src) != (e_lo_pe, e_hi_pe):
            # Convert E_norm (sampler) → keV → E_norm (PE)
            e_kev = e_norm * (e_hi_src - e_lo_src) + e_lo_src
            e_norm = (e_kev - e_lo_pe) / (e_hi_pe - e_lo_pe)
    gamma = encode_energy(e_norm, pe_cfg.num_bands)  # (..., 2L)
    return np.concatenate([gamma, t_norm[..., None]], axis=-1)
