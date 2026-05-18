"""Tests for the 1D Fourier positional encoding helper."""

from __future__ import annotations

import numpy as np
import pytest

from majorana_acp.cut_acceptance.config import PositionalEncodingConfig
from majorana_acp.cut_acceptance.positional_encoding import (
    encode_energy,
    encode_phi,
    phi_dim,
)

# ---------------------------------------------------------------------- #
# Pure-shape / dim_phi contract
# ---------------------------------------------------------------------- #


def test_phi_dim_disabled_is_two() -> None:
    pe = PositionalEncodingConfig(enabled=False)
    assert phi_dim(pe) == 2


@pytest.mark.parametrize("L", [1, 2, 6, 10, 14])
def test_phi_dim_enabled_is_2L_plus_one(L: int) -> None:
    pe = PositionalEncodingConfig(enabled=True, num_bands=L)
    assert phi_dim(pe) == 2 * L + 1


def test_encode_energy_shape() -> None:
    e = np.linspace(0.0, 1.0, 7)
    gamma = encode_energy(e, num_bands=4)
    assert gamma.shape == (7, 8)


def test_encode_energy_batch_shape() -> None:
    e = np.zeros((3, 5))
    gamma = encode_energy(e, num_bands=3)
    assert gamma.shape == (3, 5, 6)


# ---------------------------------------------------------------------- #
# Numerical correctness of γ(E_norm)
# ---------------------------------------------------------------------- #


def test_encode_energy_at_zero() -> None:
    """E_norm = 0 → every sin(2^k π · 0) = 0, every cos(...) = 1."""
    gamma = encode_energy(np.array([0.0]), num_bands=10)
    sins = gamma[..., 0::2]
    coss = gamma[..., 1::2]
    np.testing.assert_allclose(sins, 0.0, atol=1e-12)
    np.testing.assert_allclose(coss, 1.0, atol=1e-12)


def test_encode_energy_at_one() -> None:
    """E_norm = 1 → sin(2^k π) = 0 (integer multiples of π);
    cos(2^k π) = +1 for k>=1 (even multiples), and = -1 for k=0 (just π).
    """
    gamma = encode_energy(np.array([1.0]), num_bands=4)
    sins = gamma[..., 0::2].ravel()
    coss = gamma[..., 1::2].ravel()
    np.testing.assert_allclose(sins, 0.0, atol=1e-9)
    expected_cos = np.array([np.cos(np.pi), 1.0, 1.0, 1.0])  # k = 0,1,2,3
    np.testing.assert_allclose(coss, expected_cos, atol=1e-12)


def test_encode_energy_known_half_point() -> None:
    """At E_norm = 0.5 the k=0 band gives (sin(π/2), cos(π/2)) = (1, 0)."""
    gamma = encode_energy(np.array([0.5]), num_bands=2)
    # Layout is [sin_0, cos_0, sin_1, cos_1]
    np.testing.assert_allclose(gamma[0], [1.0, 0.0, 0.0, -1.0], atol=1e-12)
    # k=1 band: sin(π) = 0, cos(π) = -1


def test_encode_energy_rejects_zero_bands() -> None:
    with pytest.raises(ValueError, match="num_bands"):
        encode_energy(np.array([0.5]), num_bands=0)


# ---------------------------------------------------------------------- #
# encode_phi: backward-compat contract (the parity test the user
# explicitly required).
# ---------------------------------------------------------------------- #


def test_encode_phi_disabled_is_bitwise_identity() -> None:
    """The key gated-branch contract: enabled=False ⇒ output IS the input.

    The CNP pipeline depends on this for backward compatibility — all
    pre-PE checkpoints expect ``dim_phi=2`` and any deviation when the
    feature is OFF would break loading.
    """
    rng = np.random.default_rng(7)
    phi = rng.uniform(0, 1, size=(4, 16, 2)).astype(np.float64)
    pe = PositionalEncodingConfig(enabled=False)
    out = encode_phi(phi, pe)
    assert out is phi or np.array_equal(out, phi)
    assert out.shape == phi.shape  # dim_phi unchanged at 2


def test_encode_phi_disabled_preserves_arbitrary_dtype() -> None:
    """The pass-through must not silently up-cast or copy."""
    phi = np.array([[[0.1, 0.5], [0.7, 0.3]]], dtype=np.float32)
    pe = PositionalEncodingConfig(enabled=False)
    out = encode_phi(phi, pe)
    np.testing.assert_array_equal(out, phi)


def test_encode_phi_enabled_shape_and_layout() -> None:
    rng = np.random.default_rng(0)
    phi = rng.uniform(0, 1, size=(2, 5, 2))
    pe = PositionalEncodingConfig(enabled=True, num_bands=3)
    out = encode_phi(phi, pe)
    # 2 * 3 + 1 = 7
    assert out.shape == (2, 5, 7)
    # Last column is the original threshold pass-through.
    np.testing.assert_array_equal(out[..., -1], phi[..., 1])
    # First 6 columns are γ(E_norm). Check that the first sin/cos pair
    # matches sin(π·E)/cos(π·E) for k=0.
    expected_sin0 = np.sin(np.pi * phi[..., 0])
    expected_cos0 = np.cos(np.pi * phi[..., 0])
    np.testing.assert_allclose(out[..., 0], expected_sin0, atol=1e-12)
    np.testing.assert_allclose(out[..., 1], expected_cos0, atol=1e-12)


def test_encode_phi_rejects_non_2d_feature() -> None:
    phi = np.zeros((3, 5, 4))  # 4-dim features — wrong
    pe = PositionalEncodingConfig(enabled=True, num_bands=2)
    with pytest.raises(ValueError, match=r"shape \(\.\.\., 2\)"):
        encode_phi(phi, pe)


def test_encode_phi_renormalizes_when_pe_window_differs() -> None:
    """If the sampler's energy_range differs from the PE's window,
    encode_phi re-normalizes E_norm into the PE's window before sin/cos.
    """
    # Sampler emits E_norm against [500, 3000] keV; PE config declares
    # a narrower window [1500, 2500] keV. An event at the midpoint of the
    # sampler range (1750 keV → E_norm_src = 0.5) should land at
    # (1750 - 1500) / (2500 - 1500) = 0.25 in the PE window.
    phi = np.array([[[0.5, 0.2]]])  # one event, E_norm_src = 0.5
    pe = PositionalEncodingConfig(
        enabled=True,
        num_bands=1,
        min_energy_kev=1500.0,
        max_energy_kev=2500.0,
    )
    out = encode_phi(phi, pe, energy_range_kev=(500.0, 3000.0))
    # k=0 band at E_norm_pe = 0.25 → sin(π·0.25), cos(π·0.25)
    np.testing.assert_allclose(
        out[0, 0, :2],
        [np.sin(np.pi * 0.25), np.cos(np.pi * 0.25)],
        atol=1e-12,
    )
    # Threshold passes through unchanged.
    assert out[0, 0, -1] == 0.2
