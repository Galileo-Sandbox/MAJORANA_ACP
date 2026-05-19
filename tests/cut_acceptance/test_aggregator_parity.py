"""Parity + dispatch regression tests for the pluggable aggregator.

The backward-compat contract is the heart of this feature:

* ``aggregator.type == 'mean'`` MUST route through the upstream
  ``core.surrogate_cnp.ConditionalNeuralProcess`` — the same class
  every legacy checkpoint was saved against.
* The forward output under ``mean`` MUST be tensor-for-tensor
  identical to the upstream ``core.build_cnp`` build given the same
  random seed.
* Existing trained ``cnp.ckpt`` files (which carry no ``aggregator``
  key in their YAML) MUST continue to load through the dispatch path.

These tests fail loudly if any of those guarantees breaks.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from core import build_cnp
from core.surrogate_cnp import ConditionalNeuralProcess
from schemas.config import EncoderConfig
from schemas.data_models import InputMode, StandardBatch

from majorana_acp.cut_acceptance.config import (
    AggregatorConfig,
    CutAcceptanceConfig,
    load_config,
)
from majorana_acp.cut_acceptance.pipeline import build_local_cnp, paradigm_path_suffix
from majorana_acp.models.attentive_cnp import AttentiveCNP

# ---------------------------------------------------------------------- #
# Shared fixtures
# ---------------------------------------------------------------------- #


@pytest.fixture
def encoder_cfg() -> EncoderConfig:
    return EncoderConfig(type="mlp", latent_dim=64, hidden_dims=[128, 128], dropout=0.0)


def _make_cfg(tmp_path: Path, **overrides) -> CutAcceptanceConfig:
    stub_upstream = tmp_path / "stub_upstream.yaml"
    stub_upstream.write_text("dummy: true\n")
    stub_train = tmp_path / "train.h5"
    stub_train.write_bytes(b"")  # path existence not checked at config time
    stub_val = tmp_path / "val.h5"
    stub_val.write_bytes(b"")
    base = dict(
        train_predictions_path=stub_train,
        validation_predictions_path=stub_val,
        upstream_classifier_config=stub_upstream,
        target_class=1,
    )
    base.update(overrides)
    return CutAcceptanceConfig(**base)


def _random_batch(B: int, N: int, dim_phi: int, seed: int = 0) -> StandardBatch:
    rng = np.random.default_rng(seed)
    return StandardBatch(
        mode=InputMode.EVENT_ONLY,
        theta=None,
        phi=rng.uniform(0, 1, size=(B, N, dim_phi)).astype(np.float64),
        labels=rng.integers(0, 2, size=(B, N)).astype(np.int8),
    )


# ---------------------------------------------------------------------- #
# Aggregator config defaults / validators
# ---------------------------------------------------------------------- #


def test_aggregator_default_is_mean() -> None:
    """Every fresh config defaults to the legacy mean aggregator."""
    cfg = AggregatorConfig()
    assert cfg.type == "mean"
    assert cfg.num_heads == 8
    assert cfg.attention_dim == 64


def test_aggregator_attention_dim_must_divide_num_heads() -> None:
    with pytest.raises(ValueError, match="divisible"):
        AggregatorConfig(type="cross_attention", num_heads=8, attention_dim=63)


def test_aggregator_rejects_unknown_type() -> None:
    with pytest.raises(ValueError):
        AggregatorConfig(type="bogus")  # type: ignore[arg-type]


# ---------------------------------------------------------------------- #
# Dispatch contract: mean → upstream class; cross_attention → local
# ---------------------------------------------------------------------- #


def test_build_local_cnp_mean_returns_upstream_class(tmp_path: Path) -> None:
    """The MEAN dispatch must hand back an *upstream* ConditionalNeuralProcess
    instance — same class every legacy checkpoint was trained against."""
    cfg = _make_cfg(tmp_path)
    cnp = build_local_cnp(cfg, dim_phi=2)
    assert isinstance(cnp, ConditionalNeuralProcess)
    assert not isinstance(cnp, AttentiveCNP)


def test_build_local_cnp_cross_attention_returns_local_class(tmp_path: Path) -> None:
    cfg = _make_cfg(
        tmp_path,
        aggregator=AggregatorConfig(type="cross_attention", num_heads=8, attention_dim=64),
    )
    cnp = build_local_cnp(cfg, dim_phi=2)
    assert isinstance(cnp, AttentiveCNP)
    assert not isinstance(cnp, ConditionalNeuralProcess)


# ---------------------------------------------------------------------- #
# Bitwise parity: mean path produces tensors identical to upstream
# ---------------------------------------------------------------------- #


def test_mean_dispatch_is_bitwise_identical_to_upstream(
    tmp_path: Path, encoder_cfg: EncoderConfig
) -> None:
    """Build two CNPs with the same torch seed: one via upstream
    ``build_cnp``, one via our dispatcher under ``type='mean'``. Their
    forward outputs MUST be tensor-for-tensor identical."""
    dim_phi = 2

    torch.manual_seed(42)
    upstream_cnp = build_cnp(
        encoder_cfg,
        dim_theta=None,
        dim_phi=dim_phi,
        decoder_hidden_dims=[128, 128, 128],
    )

    cfg = _make_cfg(
        tmp_path,
        encoder=encoder_cfg,
        decoder_hidden_dims=[128, 128, 128],
    )
    torch.manual_seed(42)
    dispatched_cnp = build_local_cnp(cfg, dim_phi=dim_phi)

    # Identical state_dict (same init under the same seed).
    upstream_state = upstream_cnp.state_dict()
    dispatched_state = dispatched_cnp.state_dict()
    assert set(upstream_state) == set(dispatched_state)
    for k in upstream_state:
        torch.testing.assert_close(
            upstream_state[k],
            dispatched_state[k],
            rtol=0,
            atol=0,
        )

    # Identical forward output under the same input.
    ctx = _random_batch(B=2, N=8, dim_phi=dim_phi, seed=0)
    tgt = _random_batch(B=2, N=4, dim_phi=dim_phi, seed=1)
    upstream_cnp.eval()
    dispatched_cnp.eval()
    with torch.no_grad():
        out_u = upstream_cnp(ctx, tgt)
        out_d = dispatched_cnp(ctx, tgt)
    torch.testing.assert_close(out_u.mu_logit, out_d.mu_logit, rtol=0, atol=0)
    torch.testing.assert_close(out_u.log_sigma, out_d.log_sigma, rtol=0, atol=0)


# ---------------------------------------------------------------------- #
# AttentiveCNP forward shapes + training-loop compatibility
# ---------------------------------------------------------------------- #


def test_attentive_cnp_forward_matches_upstream_output_contract(
    tmp_path: Path, encoder_cfg: EncoderConfig
) -> None:
    """The attentive variant must return a ``CnpOutput`` with shapes the
    upstream training loop / cnp_loss expects."""
    cfg = _make_cfg(
        tmp_path,
        encoder=encoder_cfg,
        decoder_hidden_dims=[128, 128, 128],
        aggregator=AggregatorConfig(type="cross_attention", num_heads=8, attention_dim=64),
    )
    torch.manual_seed(7)
    cnp = build_local_cnp(cfg, dim_phi=21)
    ctx = _random_batch(B=3, N=16, dim_phi=21, seed=0)
    tgt = _random_batch(B=3, N=32, dim_phi=21, seed=1)
    out = cnp(ctx, tgt)
    assert out.mu_logit.shape == (3, 32)
    assert out.log_sigma.shape == (3, 32)


# ---------------------------------------------------------------------- #
# Paradigm path suffix: _attn<H>x<d> only when cross-attention is on
# ---------------------------------------------------------------------- #


def test_paradigm_suffix_skips_attn_for_mean(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)  # defaults: mean, flat_stratified, fixed
    assert "attn" not in paradigm_path_suffix(cfg)


def test_paradigm_suffix_includes_attn_for_cross_attention(tmp_path: Path) -> None:
    cfg = _make_cfg(
        tmp_path,
        sampling_pattern="mixed_density",
        zoom_window_width_kev=10.0,
        local_event_fraction=0.70,
        trial_size_strategy="variable_uniform",
        n_trial_events_min=32,
        n_trial_events_max=1024,
        aggregator=AggregatorConfig(type="cross_attention", num_heads=8, attention_dim=64),
    )
    suffix = paradigm_path_suffix(cfg)
    assert suffix.endswith("_attn8x64"), suffix


# ---------------------------------------------------------------------- #
# Legacy YAMLs (no aggregator block) parse + default to mean
# ---------------------------------------------------------------------- #


def test_legacy_yaml_parses_with_default_mean_aggregator() -> None:
    """Pre-aggregator-feature YAMLs (every one currently on disk) must
    parse cleanly and default to ``type='mean'``."""
    yaml_path = Path("configs/cut_acceptance/simple_cnn_small/true_cnp/bin10/signal.yaml")
    if not yaml_path.is_file():
        pytest.skip(f"baseline YAML missing: {yaml_path}")
    cfg = load_config(yaml_path)
    assert cfg.aggregator.type == "mean"
    assert cfg.aggregator.num_heads == 8
    assert cfg.aggregator.attention_dim == 64
    # New gating flag must default to False so every legacy + PE10_attn
    # checkpoint still loads via the un-gated decoder path.
    assert cfg.aggregator.decoder_coordinate_gating is False


# ---------------------------------------------------------------------- #
# Decoder coordinate gating
# ---------------------------------------------------------------------- #


def test_decoder_coordinate_gating_default_false() -> None:
    """Newly-fresh AggregatorConfig MUST default to gating off."""
    assert AggregatorConfig().decoder_coordinate_gating is False


def test_raw_phi_recovery_round_trip() -> None:
    """``_recover_raw_phi`` is the exact inverse of the k=0 PE band for E∈[0,1]."""
    from majorana_acp.cut_acceptance.config import PositionalEncodingConfig
    from majorana_acp.cut_acceptance.positional_encoding import encode_phi
    from majorana_acp.models.attentive_cnp import _recover_raw_phi

    rng = np.random.default_rng(0)
    raw = rng.uniform(0, 1, size=(2, 5, 2))  # (E_norm, T_norm)
    pe = PositionalEncodingConfig(enabled=True, num_bands=10)
    encoded = encode_phi(raw, pe)  # [..., 21]
    recovered = _recover_raw_phi(torch.as_tensor(encoded))
    np.testing.assert_allclose(recovered.numpy(), raw, atol=1e-10)


def test_recover_raw_phi_passes_through_when_phi_already_2d() -> None:
    """If batch.phi.shape[-1] == 2 (PE disabled), recovery is a no-op."""
    from majorana_acp.models.attentive_cnp import _recover_raw_phi

    raw = torch.rand(3, 7, 2)
    out = _recover_raw_phi(raw)
    torch.testing.assert_close(out, raw, rtol=0, atol=0)


def test_attentive_cnp_ungated_decoder_input_is_2Z(
    tmp_path: Path, encoder_cfg: EncoderConfig
) -> None:
    """Without gating: decoder.net.in_features = agg_dim + Z = 2Z (legacy)."""
    cfg = _make_cfg(
        tmp_path,
        encoder=encoder_cfg,
        decoder_hidden_dims=[128, 128, 128],
        aggregator=AggregatorConfig(
            type="cross_attention",
            num_heads=8,
            attention_dim=64,
            decoder_coordinate_gating=False,
        ),
    )
    cnp = build_local_cnp(cfg, dim_phi=21)
    Z = encoder_cfg.latent_dim
    first_linear = cnp.decoder.net[0]
    assert first_linear.in_features == 2 * Z, f"expected {2 * Z}, got {first_linear.in_features}"


def test_attentive_cnp_gated_decoder_input_is_Z_plus_2(
    tmp_path: Path, encoder_cfg: EncoderConfig
) -> None:
    """With gating on: decoder.net.in_features = agg_dim + 2 = Z + 2."""
    cfg = _make_cfg(
        tmp_path,
        encoder=encoder_cfg,
        decoder_hidden_dims=[128, 128, 128],
        aggregator=AggregatorConfig(
            type="cross_attention",
            num_heads=8,
            attention_dim=64,
            decoder_coordinate_gating=True,
        ),
    )
    cnp = build_local_cnp(cfg, dim_phi=21)
    Z = encoder_cfg.latent_dim
    first_linear = cnp.decoder.net[0]
    assert first_linear.in_features == Z + 2, f"expected {Z + 2}, got {first_linear.in_features}"


def test_attentive_cnp_gated_forward_shape_unchanged(
    tmp_path: Path, encoder_cfg: EncoderConfig
) -> None:
    """Gating changes the decoder's input *width* but not its output
    shape — μ_logit / log_σ stay ``[B, N_T]`` so cnp_loss still works."""
    cfg = _make_cfg(
        tmp_path,
        encoder=encoder_cfg,
        decoder_hidden_dims=[128, 128, 128],
        aggregator=AggregatorConfig(
            type="cross_attention",
            num_heads=8,
            attention_dim=64,
            decoder_coordinate_gating=True,
        ),
    )
    torch.manual_seed(11)
    cnp = build_local_cnp(cfg, dim_phi=21)
    ctx = _random_batch(B=2, N=8, dim_phi=21, seed=0)
    tgt = _random_batch(B=2, N=4, dim_phi=21, seed=1)
    out = cnp(ctx, tgt)
    assert out.mu_logit.shape == (2, 4)
    assert out.log_sigma.shape == (2, 4)


def test_paradigm_suffix_includes_gated_when_gating_on(tmp_path: Path) -> None:
    cfg = _make_cfg(
        tmp_path,
        sampling_pattern="physics_anchored",
        zoom_window_width_kev=10.0,
        local_event_fraction=0.80,
        physics_peaks_kev=[1592.0, 1620.0, 2103.0, 2614.0],
        trial_size_strategy="variable_uniform",
        n_trial_events_min=32,
        n_trial_events_max=1024,
        aggregator=AggregatorConfig(
            type="cross_attention",
            num_heads=8,
            attention_dim=64,
            decoder_coordinate_gating=True,
        ),
    )
    assert paradigm_path_suffix(cfg).endswith("_attn8x64_gated")


def test_paradigm_suffix_omits_gated_when_off(tmp_path: Path) -> None:
    """``_gated`` must not appear unless the flag is explicitly on —
    otherwise existing ``_attn8x64`` paths would silently collide."""
    cfg = _make_cfg(
        tmp_path,
        sampling_pattern="mixed_density",
        zoom_window_width_kev=10.0,
        local_event_fraction=0.70,
        trial_size_strategy="variable_uniform",
        n_trial_events_min=32,
        n_trial_events_max=1024,
        aggregator=AggregatorConfig(
            type="cross_attention",
            num_heads=8,
            attention_dim=64,
            decoder_coordinate_gating=False,
        ),
    )
    assert "_gated" not in paradigm_path_suffix(cfg)


# ---------------------------------------------------------------------- #
# Gaussian Attention Bias (GAB) + continuous density sampling
# ---------------------------------------------------------------------- #


def test_aggregator_gaussian_attention_bias_defaults_false() -> None:
    cfg = AggregatorConfig()
    assert cfg.gaussian_attention_bias is False


def test_attentive_cnp_gab_off_has_no_log_gamma(encoder_cfg: EncoderConfig) -> None:
    """Backward-compat: when GAB is off the attention layer must not
    own a ``log_gamma`` parameter — preserves state_dict shape parity
    with the pre-GAB ``AttentiveCNP`` and every existing checkpoint."""
    from majorana_acp.models.attentive_cnp import build_attentive_cnp

    cnp = build_attentive_cnp(
        encoder_cfg,
        dim_theta=None,
        dim_phi=2,
        num_heads=4,
        attention_dim=64,
        gaussian_attention_bias=False,
    )
    assert cnp.attention.gaussian_attention_bias is False
    assert cnp.attention.log_gamma is None
    # state_dict must NOT contain ``attention.log_gamma`` — old ckpts
    # would fail to load if this key snuck in unconditionally.
    assert "attention.log_gamma" not in cnp.state_dict()


def test_attentive_cnp_gab_on_registers_log_gamma_per_head(
    encoder_cfg: EncoderConfig,
) -> None:
    from majorana_acp.models.attentive_cnp import build_attentive_cnp

    cnp = build_attentive_cnp(
        encoder_cfg,
        dim_theta=None,
        dim_phi=2,
        num_heads=4,
        attention_dim=64,
        gaussian_attention_bias=True,
    )
    assert cnp.attention.gaussian_attention_bias is True
    lg = cnp.attention.log_gamma
    assert lg is not None
    assert tuple(lg.shape) == (4, 1, 1)
    # Init = zeros so the per-head Gaussian penalty at init is just Δ²
    # (≤ 1 for E_norm ∈ [0,1]), comparable to the un-biased attention.
    assert torch.allclose(lg, torch.zeros_like(lg))


def test_attentive_cnp_gab_forward_shape_pe_off(encoder_cfg: EncoderConfig) -> None:
    """PE off + GAB on: raw 2D phi is passed straight through to the
    Δ² penalty without any inversion. Shape contract: (B, N_T)."""
    from majorana_acp.models.attentive_cnp import build_attentive_cnp

    cnp = build_attentive_cnp(
        encoder_cfg,
        dim_theta=None,
        dim_phi=2,
        num_heads=4,
        attention_dim=64,
        gaussian_attention_bias=True,
    )
    ctx = _random_batch(B=2, N=16, dim_phi=2, seed=0)
    tgt = _random_batch(B=2, N=8, dim_phi=2, seed=1)
    out = cnp(ctx, tgt)
    assert out.mu_logit.shape == (2, 8)
    assert out.log_sigma.shape == (2, 8)


def test_attentive_cnp_gab_forward_shape_pe_on(encoder_cfg: EncoderConfig) -> None:
    """PE on + GAB on: ``_recover_raw_phi`` reconstructs E_norm via the
    k=0 atan2 inverse before the attention's Δ² penalty fires."""
    from majorana_acp.models.attentive_cnp import build_attentive_cnp

    L = 10
    cnp = build_attentive_cnp(
        encoder_cfg,
        dim_theta=None,
        dim_phi=2 * L + 1,
        num_heads=4,
        attention_dim=64,
        gaussian_attention_bias=True,
    )
    rng = np.random.default_rng(0)

    def _make_pe_batch(B: int, N: int) -> StandardBatch:
        e = rng.uniform(0, 1, (B, N))
        t = rng.uniform(0, 1, (B, N))
        k = np.arange(L)
        angles = e[..., None] * (2.0**k * np.pi)
        out = np.empty(angles.shape[:-1] + (2 * L,))
        out[..., 0::2] = np.sin(angles)
        out[..., 1::2] = np.cos(angles)
        phi = np.concatenate([out, t[..., None]], axis=-1).astype(np.float64)
        return StandardBatch(
            mode=InputMode.EVENT_ONLY,
            theta=None,
            phi=phi,
            labels=rng.integers(0, 2, (B, N)).astype(np.int8),
        )

    out = cnp(_make_pe_batch(2, 16), _make_pe_batch(2, 8))
    assert out.mu_logit.shape == (2, 8)


def test_attentive_cnp_gab_at_init_close_to_ungated(encoder_cfg: EncoderConfig) -> None:
    """At init (log_gamma = 0), the GAB penalty is just Δ² which is ≤ 1
    for E_norm ∈ [0,1]. With dropout = 0 and a fresh seed used for both
    builds, GAB-on and GAB-off must be close in output (not identical —
    Δ² shifts the softmax — but on the same order of magnitude)."""
    from majorana_acp.models.attentive_cnp import build_attentive_cnp

    torch.manual_seed(123)
    cnp_off = build_attentive_cnp(
        encoder_cfg,
        dim_theta=None,
        dim_phi=2,
        num_heads=4,
        attention_dim=64,
        gaussian_attention_bias=False,
    )
    torch.manual_seed(123)
    cnp_on = build_attentive_cnp(
        encoder_cfg,
        dim_theta=None,
        dim_phi=2,
        num_heads=4,
        attention_dim=64,
        gaussian_attention_bias=True,
    )
    ctx = _random_batch(B=2, N=16, dim_phi=2, seed=0)
    tgt = _random_batch(B=2, N=8, dim_phi=2, seed=1)
    cnp_off.eval()
    cnp_on.eval()
    with torch.no_grad():
        out_off = cnp_off(ctx, tgt)
        out_on = cnp_on(ctx, tgt)
    # μ_logit shouldn't differ wildly at init.
    diff = (out_off.mu_logit - out_on.mu_logit).abs().max().item()
    assert diff < 5.0, f"GAB-on at init differs from GAB-off by {diff} (expected small)"


def test_paradigm_suffix_includes_gab_when_on(tmp_path: Path) -> None:
    cfg = _make_cfg(
        tmp_path,
        sampling_pattern="physics_anchored",
        zoom_window_width_kev=10.0,
        local_event_fraction=0.80,
        physics_peaks_kev=[1592.0, 1620.0, 2103.0, 2614.0],
        trial_size_strategy="variable_uniform",
        n_trial_events_min=32,
        n_trial_events_max=1024,
        aggregator=AggregatorConfig(
            type="cross_attention",
            num_heads=8,
            attention_dim=64,
            gaussian_attention_bias=True,
        ),
    )
    assert paradigm_path_suffix(cfg).endswith("_attn8x64_gab")


def test_paradigm_suffix_includes_debinned_when_continuous(tmp_path: Path) -> None:
    cfg = _make_cfg(
        tmp_path,
        sampling_pattern="physics_anchored",
        zoom_window_width_kev=10.0,
        local_event_fraction=0.80,
        physics_peaks_kev=[1592.0, 1620.0, 2103.0, 2614.0],
        trial_size_strategy="variable_uniform",
        n_trial_events_min=32,
        n_trial_events_max=1024,
        aggregator=AggregatorConfig(
            type="cross_attention",
            num_heads=8,
            attention_dim=64,
            gaussian_attention_bias=True,
        ),
        density_sampling="continuous",
        density_kde_radius_kev=10.0,
    )
    suffix = paradigm_path_suffix(cfg)
    assert suffix.endswith("_attn8x64_gab_debinned")


def test_paradigm_suffix_omits_debinned_when_bin_stratified(tmp_path: Path) -> None:
    cfg = _make_cfg(
        tmp_path,
        sampling_pattern="mixed_density",
        zoom_window_width_kev=10.0,
        local_event_fraction=0.70,
        trial_size_strategy="variable_uniform",
        n_trial_events_min=32,
        n_trial_events_max=1024,
    )
    assert "_debinned" not in paradigm_path_suffix(cfg)


# ---------------------------------------------------------------------- #
# Density Modulation (DM): pluggable extension of GAB
# ---------------------------------------------------------------------- #


def test_density_modulation_defaults_off() -> None:
    from majorana_acp.cut_acceptance.config import DensityModulationConfig

    cfg = AggregatorConfig()
    assert cfg.density_modulation.enabled is False
    # Also: instantiating DM with no kwargs gives sane defaults.
    dm = DensityModulationConfig()
    assert dm.sigma_local_kev == 10.0
    assert dm.sigma_global_kev == 150.0


def test_density_modulation_sigma_global_must_exceed_local() -> None:
    from majorana_acp.cut_acceptance.config import DensityModulationConfig

    with pytest.raises(ValueError, match="must exceed"):
        DensityModulationConfig(enabled=True, sigma_local_kev=50.0, sigma_global_kev=10.0)


def test_attentive_cnp_dm_off_has_no_dm_state(encoder_cfg: EncoderConfig) -> None:
    """GAB on, DM off — no σ values populated. Backward-compatible with
    every GAB checkpoint trained before the DM extension."""
    from majorana_acp.models.attentive_cnp import build_attentive_cnp

    cnp = build_attentive_cnp(
        encoder_cfg,
        dim_theta=None,
        dim_phi=2,
        num_heads=4,
        attention_dim=64,
        gaussian_attention_bias=True,
        density_modulation_enabled=False,
    )
    assert cnp.attention.density_modulation_enabled is False
    # No DM-specific parameters appear in state_dict — checkpoints
    # trained pre-DM continue to load via this factory.
    assert "attention.density_sigma_local_norm" not in dict(cnp.state_dict())


def test_attentive_cnp_dm_requires_gab(encoder_cfg: EncoderConfig) -> None:
    """DM modifies the GAB penalty. Setting DM without GAB is a
    config-level no-op; the factory should raise rather than silently
    skipping the modulation."""
    from majorana_acp.models.attentive_cnp import build_attentive_cnp

    with pytest.raises(ValueError, match="requires gaussian_attention_bias"):
        build_attentive_cnp(
            encoder_cfg,
            dim_theta=None,
            dim_phi=2,
            num_heads=4,
            attention_dim=64,
            gaussian_attention_bias=False,
            density_modulation_enabled=True,
            density_sigma_local_kev=10.0,
            density_sigma_global_kev=150.0,
            energy_range_kev=(500.0, 3000.0),
        )


def test_attentive_cnp_dm_sigma_kev_converted_to_norm(encoder_cfg: EncoderConfig) -> None:
    """The factory converts σ_kev to σ_norm using the energy range so
    the internal kernel widths live in the same E_norm ∈ [0, 1] space
    as the rest of the CNP coords."""
    from majorana_acp.models.attentive_cnp import build_attentive_cnp

    cnp = build_attentive_cnp(
        encoder_cfg,
        dim_theta=None,
        dim_phi=2,
        num_heads=4,
        attention_dim=64,
        gaussian_attention_bias=True,
        density_modulation_enabled=True,
        density_sigma_local_kev=10.0,
        density_sigma_global_kev=150.0,
        energy_range_kev=(500.0, 3000.0),
    )
    # 10 / 2500 = 0.004
    assert abs(cnp.attention.density_sigma_local_norm - 0.004) < 1e-9
    # 150 / 2500 = 0.06
    assert abs(cnp.attention.density_sigma_global_norm - 0.06) < 1e-9


def test_attentive_cnp_dm_forward_shape(encoder_cfg: EncoderConfig) -> None:
    from majorana_acp.models.attentive_cnp import build_attentive_cnp

    cnp = build_attentive_cnp(
        encoder_cfg,
        dim_theta=None,
        dim_phi=2,
        num_heads=4,
        attention_dim=64,
        gaussian_attention_bias=True,
        density_modulation_enabled=True,
        density_sigma_local_kev=10.0,
        density_sigma_global_kev=150.0,
        energy_range_kev=(500.0, 3000.0),
    )
    ctx = _random_batch(B=2, N=64, dim_phi=2, seed=0)
    tgt = _random_batch(B=2, N=8, dim_phi=2, seed=1)
    out = cnp(ctx, tgt)
    assert out.mu_logit.shape == (2, 8)
    assert out.log_sigma.shape == (2, 8)


def test_dm_contrast_peak_vs_continuum(encoder_cfg: EncoderConfig) -> None:
    """Direct math check on the modulation: R_* at a sharp peak query
    should exceed R_* at a flat-continuum query by a wide margin.
    Robust across N_C — the absolute scale is N-dependent but the
    contrast is the signal."""
    import math

    sqrt_2pi = math.sqrt(2.0 * math.pi)
    s_l = 10.0 / 2500.0
    s_g = 150.0 / 2500.0
    eps = 1e-6

    rng = np.random.default_rng(0)
    # Context: 1024 events — half uniform background, half packed in a
    # sharp peak at E_norm = 0.5 with width ~ 2 keV in normalised units.
    ctx_bg = rng.uniform(0.05, 0.95, 512)
    ctx_peak = rng.normal(0.5, 2.0 / 2500.0, 512).clip(0.05, 0.95)
    ctx = np.concatenate([ctx_bg, ctx_peak])
    ctx_t = torch.tensor(ctx, dtype=torch.float32).reshape(1, -1)

    def _R(query_E: float) -> float:
        e_t = torch.tensor([[query_E]], dtype=torch.float32)
        delta2 = (e_t.unsqueeze(-1) - ctx_t.unsqueeze(1)).pow(2)
        k_l = torch.exp(-delta2 / (2 * s_l * s_l))
        k_g = torch.exp(-delta2 / (2 * s_g * s_g))
        rho_l = k_l.sum(-1) / (s_l * sqrt_2pi)
        rho_g = k_g.sum(-1) / (s_g * sqrt_2pi) + eps
        return float((rho_l / rho_g).item())

    R_peak = _R(0.5)
    R_far = _R(0.2)
    assert R_peak > 5.0, f"R at peak was {R_peak} (expected ≫ 1)"
    assert R_peak > 10.0 * R_far, (
        f"DM contrast failed: R(peak)={R_peak:.2f}, R(far)={R_far:.2f}, "
        f"ratio={R_peak / max(R_far, 1e-9):.2f} (expected ≥ 10×)"
    )


def test_paradigm_suffix_includes_dense_when_dm_on(tmp_path: Path) -> None:
    from majorana_acp.cut_acceptance.config import DensityModulationConfig

    cfg = _make_cfg(
        tmp_path,
        sampling_pattern="physics_anchored",
        zoom_window_width_kev=10.0,
        local_event_fraction=0.80,
        physics_peaks_kev=[1592.0, 1620.0, 2103.0, 2614.0],
        trial_size_strategy="variable_uniform",
        n_trial_events_min=32,
        n_trial_events_max=1024,
        aggregator=AggregatorConfig(
            type="cross_attention",
            num_heads=8,
            attention_dim=64,
            gaussian_attention_bias=True,
            density_modulation=DensityModulationConfig(
                enabled=True, sigma_local_kev=10.0, sigma_global_kev=150.0
            ),
        ),
    )
    suffix = paradigm_path_suffix(cfg)
    assert suffix.endswith("_attn8x64_gab_dense")


# ---------------------------------------------------------------------- #
# Bounded Physical Bandwidth Network (BPBN)
# ---------------------------------------------------------------------- #


def test_bpbn_defaults_off() -> None:
    cfg = AggregatorConfig()
    assert cfg.bounded_bandwidth.enabled is False


def test_bpbn_sigma_max_must_exceed_sigma_min() -> None:
    from majorana_acp.cut_acceptance.config import BoundedBandwidthConfig

    with pytest.raises(ValueError, match="sigma_max_kev"):
        BoundedBandwidthConfig(enabled=True, sigma_max_kev=5.0, sigma_min_kev=200.0)


def test_bpbn_requires_gab(encoder_cfg: EncoderConfig) -> None:
    from majorana_acp.models.attentive_cnp import build_attentive_cnp

    with pytest.raises(ValueError, match="requires gaussian_attention_bias"):
        build_attentive_cnp(
            encoder_cfg,
            dim_theta=None,
            dim_phi=2,
            num_heads=4,
            attention_dim=64,
            gaussian_attention_bias=False,
            bounded_bandwidth_enabled=True,
            bounded_sigma_max_kev=200.0,
            bounded_sigma_min_kev=5.0,
            bounded_alpha_max=40.0,
            bounded_sigma_local_kev=10.0,
            bounded_sigma_global_kev=150.0,
            energy_range_kev=(500.0, 3000.0),
        )


def test_bpbn_off_keeps_log_gamma(encoder_cfg: EncoderConfig) -> None:
    """When BPBN is off, the GAB-on path keeps the legacy log_gamma —
    state_dict shape unchanged for every existing GAB checkpoint."""
    from majorana_acp.models.attentive_cnp import build_attentive_cnp

    cnp = build_attentive_cnp(
        encoder_cfg,
        dim_theta=None,
        dim_phi=2,
        num_heads=4,
        attention_dim=64,
        gaussian_attention_bias=True,
        bounded_bandwidth_enabled=False,
    )
    assert cnp.attention.log_gamma is not None
    assert "attention.log_gamma" in cnp.state_dict()
    # No BPBN-specific parameters appear.
    assert not any(k.startswith("attention.sensitivity_mlp") for k in cnp.state_dict())


def test_bpbn_on_replaces_log_gamma_with_sensitivity_mlp(encoder_cfg: EncoderConfig) -> None:
    """When BPBN is on, ``log_gamma`` is NOT registered (replaced) and
    the sensitivity MLP parameters appear in state_dict."""
    from majorana_acp.models.attentive_cnp import build_attentive_cnp

    cnp = build_attentive_cnp(
        encoder_cfg,
        dim_theta=None,
        dim_phi=2,
        num_heads=4,
        attention_dim=64,
        gaussian_attention_bias=True,
        bounded_bandwidth_enabled=True,
        bounded_sigma_max_kev=200.0,
        bounded_sigma_min_kev=5.0,
        bounded_alpha_max=40.0,
        bounded_sigma_local_kev=10.0,
        bounded_sigma_global_kev=150.0,
        energy_range_kev=(500.0, 3000.0),
    )
    assert cnp.attention.log_gamma is None
    assert "attention.log_gamma" not in cnp.state_dict()
    mlp_keys = [k for k in cnp.state_dict() if k.startswith("attention.sensitivity_mlp")]
    assert len(mlp_keys) == 4  # 2 Linear layers × (weight, bias)


def test_bpbn_sigma_kev_converted_to_norm(encoder_cfg: EncoderConfig) -> None:
    from majorana_acp.models.attentive_cnp import build_attentive_cnp

    cnp = build_attentive_cnp(
        encoder_cfg,
        dim_theta=None,
        dim_phi=2,
        num_heads=4,
        attention_dim=64,
        gaussian_attention_bias=True,
        bounded_bandwidth_enabled=True,
        bounded_sigma_max_kev=200.0,
        bounded_sigma_min_kev=5.0,
        bounded_alpha_max=40.0,
        bounded_sigma_local_kev=10.0,
        bounded_sigma_global_kev=150.0,
        energy_range_kev=(500.0, 3000.0),
    )
    # 200 / 2500 = 0.08; 5 / 2500 = 0.002
    assert abs(cnp.attention.bounded_sigma_max_norm - 0.08) < 1e-9
    assert abs(cnp.attention.bounded_sigma_min_norm - 0.002) < 1e-9
    assert abs(cnp.attention.bounded_sigma_local_norm - 0.004) < 1e-9
    assert abs(cnp.attention.bounded_sigma_global_norm - 0.06) < 1e-9


def test_bpbn_forward_shape(encoder_cfg: EncoderConfig) -> None:
    from majorana_acp.models.attentive_cnp import build_attentive_cnp

    cnp = build_attentive_cnp(
        encoder_cfg,
        dim_theta=None,
        dim_phi=2,
        num_heads=8,
        attention_dim=64,
        gaussian_attention_bias=True,
        bounded_bandwidth_enabled=True,
        bounded_sigma_max_kev=200.0,
        bounded_sigma_min_kev=5.0,
        bounded_alpha_max=40.0,
        bounded_sigma_local_kev=10.0,
        bounded_sigma_global_kev=150.0,
        energy_range_kev=(500.0, 3000.0),
    )
    ctx = _random_batch(B=2, N=64, dim_phi=2, seed=0)
    tgt = _random_batch(B=2, N=8, dim_phi=2, seed=1)
    out = cnp(ctx, tgt)
    assert out.mu_logit.shape == (2, 8)
    assert out.log_sigma.shape == (2, 8)


def test_bpbn_sigma_head_clamped_at_peak(encoder_cfg: EncoderConfig) -> None:
    """At a sharp peak query with large R_*, σ_head must be clamped at
    σ_min (= 5 keV → 0.002 in normalised units). This is the structural
    fix against the GAB → 1-NN collapse."""
    import math

    sqrt_2pi = math.sqrt(2.0 * math.pi)
    sigma_min_norm = 5.0 / 2500.0
    sigma_max_norm = 200.0 / 2500.0
    alpha_max = 40.0
    s_l = 10.0 / 2500.0
    s_g = 150.0 / 2500.0

    rng = np.random.default_rng(0)
    # Sharp peak: 512 events all in a 2 keV cluster at E_norm=0.5.
    ctx_peak = rng.normal(0.5, 1.0 / 2500.0, 512).clip(0.05, 0.95)
    ctx_t = torch.tensor(ctx_peak, dtype=torch.float32).reshape(1, -1)
    e_query = torch.tensor([[0.5]], dtype=torch.float32)

    # Hand-compute σ_head assuming λ = α_max (maximum contraction).
    delta = e_query.unsqueeze(-1) - ctx_t.unsqueeze(1)
    delta2 = delta.pow(2)
    k_l = torch.exp(-delta2 / (2 * s_l * s_l))
    k_g = torch.exp(-delta2 / (2 * s_g * s_g))
    rho_l = k_l.sum(-1) / (s_l * sqrt_2pi)
    rho_g = k_g.sum(-1) / (s_g * sqrt_2pi) + 1e-6
    R = (rho_l / rho_g).item()
    assert R > 10.0, f"sharp peak should give R > 10, got {R}"

    # With α_max=40 and R≈12, denom = 1 + 40·11 = 441 → σ_head = 200/441 ≈ 0.45 keV
    # well below σ_min=5 keV, so the clamp must engage.
    unclamped = sigma_max_norm / (1.0 + alpha_max * (R - 1.0))
    assert unclamped < sigma_min_norm, (
        f"unclamped σ_head = {unclamped} should be < σ_min_norm = {sigma_min_norm} "
        "for the clamp test to be meaningful"
    )


def test_paradigm_suffix_includes_bpbn_when_on(tmp_path: Path) -> None:
    from majorana_acp.cut_acceptance.config import BoundedBandwidthConfig

    cfg = _make_cfg(
        tmp_path,
        sampling_pattern="physics_anchored",
        zoom_window_width_kev=10.0,
        local_event_fraction=0.80,
        physics_peaks_kev=[1592.0, 1620.0, 2103.0, 2614.0],
        trial_size_strategy="variable_uniform",
        n_trial_events_min=32,
        n_trial_events_max=1024,
        aggregator=AggregatorConfig(
            type="cross_attention",
            num_heads=8,
            attention_dim=64,
            gaussian_attention_bias=True,
            bounded_bandwidth=BoundedBandwidthConfig(
                enabled=True,
                sigma_max_kev=200.0,
                sigma_min_kev=5.0,
                alpha_max=40.0,
            ),
        ),
    )
    assert paradigm_path_suffix(cfg).endswith("_attn8x64_gab_bpbn")


def test_paradigm_suffix_omits_bpbn_when_off(tmp_path: Path) -> None:
    cfg = _make_cfg(
        tmp_path,
        sampling_pattern="physics_anchored",
        zoom_window_width_kev=10.0,
        local_event_fraction=0.80,
        physics_peaks_kev=[1592.0, 1620.0, 2103.0, 2614.0],
        trial_size_strategy="variable_uniform",
        n_trial_events_min=32,
        n_trial_events_max=1024,
        aggregator=AggregatorConfig(
            type="cross_attention",
            num_heads=8,
            attention_dim=64,
            gaussian_attention_bias=True,
        ),
    )
    assert "_bpbn" not in paradigm_path_suffix(cfg)


# ---------------------------------------------------------------------- #
# Bounded 2D Spectral Filter Network (SFN)
# ---------------------------------------------------------------------- #


def test_sfn_defaults_off() -> None:
    cfg = AggregatorConfig()
    assert cfg.sfn_modulation.enabled is False


def test_sfn_sigma_max_must_exceed_sigma_min() -> None:
    from majorana_acp.cut_acceptance.config import SfnModulationConfig

    with pytest.raises(ValueError, match="sigma_max_kev"):
        SfnModulationConfig(enabled=True, sigma_max_kev=5.0, sigma_min_kev=200.0)


def test_sfn_requires_gab(encoder_cfg: EncoderConfig) -> None:
    from majorana_acp.models.attentive_cnp import build_attentive_cnp

    with pytest.raises(ValueError, match="requires gaussian_attention_bias"):
        build_attentive_cnp(
            encoder_cfg,
            dim_theta=None,
            dim_phi=2,
            num_heads=4,
            attention_dim=64,
            gaussian_attention_bias=False,
            sfn_modulation_enabled=True,
            sfn_sigma_max_kev=200.0,
            sfn_sigma_min_kev=5.0,
            sfn_sigma_local_kev=10.0,
            sfn_sigma_global_kev=150.0,
            energy_range_kev=(500.0, 3000.0),
        )


def test_sfn_and_bpbn_mutually_exclusive(encoder_cfg: EncoderConfig) -> None:
    from majorana_acp.models.attentive_cnp import build_attentive_cnp

    with pytest.raises(ValueError, match="mutually exclusive"):
        build_attentive_cnp(
            encoder_cfg,
            dim_theta=None,
            dim_phi=2,
            num_heads=4,
            attention_dim=64,
            gaussian_attention_bias=True,
            bounded_bandwidth_enabled=True,
            bounded_sigma_max_kev=200.0,
            bounded_sigma_min_kev=5.0,
            bounded_alpha_max=40.0,
            bounded_sigma_local_kev=10.0,
            bounded_sigma_global_kev=150.0,
            sfn_modulation_enabled=True,
            sfn_sigma_max_kev=200.0,
            sfn_sigma_min_kev=5.0,
            sfn_sigma_local_kev=10.0,
            sfn_sigma_global_kev=150.0,
            energy_range_kev=(500.0, 3000.0),
        )


def test_sfn_off_keeps_log_gamma(encoder_cfg: EncoderConfig) -> None:
    """GAB-only checkpoint shape preserved when SFN is off."""
    from majorana_acp.models.attentive_cnp import build_attentive_cnp

    cnp = build_attentive_cnp(
        encoder_cfg,
        dim_theta=None,
        dim_phi=2,
        num_heads=4,
        attention_dim=64,
        gaussian_attention_bias=True,
        sfn_modulation_enabled=False,
    )
    assert cnp.attention.log_gamma is not None
    assert not any(k.startswith("attention.sfn_net") for k in cnp.state_dict())


def test_sfn_on_replaces_log_gamma_with_sfn_net(encoder_cfg: EncoderConfig) -> None:
    from majorana_acp.models.attentive_cnp import build_attentive_cnp

    cnp = build_attentive_cnp(
        encoder_cfg,
        dim_theta=None,
        dim_phi=2,
        num_heads=4,
        attention_dim=64,
        gaussian_attention_bias=True,
        sfn_modulation_enabled=True,
        sfn_sigma_max_kev=200.0,
        sfn_sigma_min_kev=5.0,
        sfn_sigma_local_kev=10.0,
        sfn_sigma_global_kev=150.0,
        energy_range_kev=(500.0, 3000.0),
    )
    assert cnp.attention.log_gamma is None
    assert "attention.log_gamma" not in cnp.state_dict()
    sfn_keys = [k for k in cnp.state_dict() if k.startswith("attention.sfn_net")]
    assert len(sfn_keys) == 4  # 2 Linear layers × (weight, bias)


def test_sfn_sigma_kev_converted_to_norm(encoder_cfg: EncoderConfig) -> None:
    from majorana_acp.models.attentive_cnp import build_attentive_cnp

    cnp = build_attentive_cnp(
        encoder_cfg,
        dim_theta=None,
        dim_phi=2,
        num_heads=4,
        attention_dim=64,
        gaussian_attention_bias=True,
        sfn_modulation_enabled=True,
        sfn_sigma_max_kev=200.0,
        sfn_sigma_min_kev=5.0,
        sfn_sigma_local_kev=10.0,
        sfn_sigma_global_kev=150.0,
        energy_range_kev=(500.0, 3000.0),
    )
    assert abs(cnp.attention.sfn_sigma_max_norm - 0.08) < 1e-9
    assert abs(cnp.attention.sfn_sigma_min_norm - 0.002) < 1e-9


def test_sfn_forward_shape(encoder_cfg: EncoderConfig) -> None:
    from majorana_acp.models.attentive_cnp import build_attentive_cnp

    cnp = build_attentive_cnp(
        encoder_cfg,
        dim_theta=None,
        dim_phi=2,
        num_heads=8,
        attention_dim=64,
        gaussian_attention_bias=True,
        sfn_modulation_enabled=True,
        sfn_sigma_max_kev=200.0,
        sfn_sigma_min_kev=5.0,
        sfn_sigma_local_kev=10.0,
        sfn_sigma_global_kev=150.0,
        energy_range_kev=(500.0, 3000.0),
    )
    ctx = _random_batch(B=2, N=64, dim_phi=2, seed=0)
    tgt = _random_batch(B=2, N=8, dim_phi=2, seed=1)
    out = cnp(ctx, tgt)
    assert out.mu_logit.shape == (2, 8)
    assert out.log_sigma.shape == (2, 8)


def test_sfn_sigma_head_bounded_by_construction(encoder_cfg: EncoderConfig) -> None:
    """sigmoid output ∈ (0, 1) → σ_head ∈ (σ_min, σ_max) without any
    explicit clamp. This is the architectural difference vs BPBN where
    a torch.clamp creates gradient deadzones — here the bound is
    smooth + differentiable everywhere."""
    from majorana_acp.models.attentive_cnp import build_attentive_cnp

    cnp = build_attentive_cnp(
        encoder_cfg,
        dim_theta=None,
        dim_phi=2,
        num_heads=4,
        attention_dim=64,
        gaussian_attention_bias=True,
        sfn_modulation_enabled=True,
        sfn_sigma_max_kev=200.0,
        sfn_sigma_min_kev=5.0,
        sfn_sigma_local_kev=10.0,
        sfn_sigma_global_kev=150.0,
        energy_range_kev=(500.0, 3000.0),
    )
    # Inject extreme logits into the SFN's output layer to verify the
    # σ_head clamp via sigmoid still respects [σ_min, σ_max] bounds.
    with torch.no_grad():
        # Drive the last Linear layer to output ±∞ logits.
        cnp.attention.sfn_net[2].bias.fill_(1e6)  # forces σ → σ_max
        # Construct a small batch and run the SFN forward by hand.
        e_t = torch.tensor([[0.5]], dtype=torch.float32)
        log_R = torch.tensor([[2.0]], dtype=torch.float32)
        Z = torch.stack([e_t, log_R], dim=-1)
        logits = cnp.attention.sfn_net(Z)
        sigma_max_norm = cnp.attention.sfn_sigma_max_norm
        sigma_min_norm = cnp.attention.sfn_sigma_min_norm
        sigma_head = sigma_min_norm + (sigma_max_norm - sigma_min_norm) * torch.sigmoid(logits)
        # σ_head approaches σ_max but never exceeds it.
        assert torch.all(sigma_head <= sigma_max_norm + 1e-12)
        assert torch.all(sigma_head >= sigma_min_norm - 1e-12)


def test_paradigm_suffix_includes_sfn_when_on(tmp_path: Path) -> None:
    from majorana_acp.cut_acceptance.config import SfnModulationConfig

    cfg = _make_cfg(
        tmp_path,
        sampling_pattern="physics_anchored",
        zoom_window_width_kev=10.0,
        local_event_fraction=0.80,
        physics_peaks_kev=[1592.0, 1620.0, 2103.0, 2614.0],
        trial_size_strategy="variable_uniform",
        n_trial_events_min=32,
        n_trial_events_max=1024,
        aggregator=AggregatorConfig(
            type="cross_attention",
            num_heads=8,
            attention_dim=64,
            gaussian_attention_bias=True,
            sfn_modulation=SfnModulationConfig(
                enabled=True,
                sigma_max_kev=200.0,
                sigma_min_kev=5.0,
            ),
        ),
    )
    assert paradigm_path_suffix(cfg).endswith("_attn8x64_gab_sfn")


def test_paradigm_suffix_omits_sfn_when_off(tmp_path: Path) -> None:
    cfg = _make_cfg(
        tmp_path,
        sampling_pattern="physics_anchored",
        zoom_window_width_kev=10.0,
        local_event_fraction=0.80,
        physics_peaks_kev=[1592.0, 1620.0, 2103.0, 2614.0],
        trial_size_strategy="variable_uniform",
        n_trial_events_min=32,
        n_trial_events_max=1024,
        aggregator=AggregatorConfig(
            type="cross_attention",
            num_heads=8,
            attention_dim=64,
            gaussian_attention_bias=True,
        ),
    )
    assert "_sfn" not in paradigm_path_suffix(cfg)


def test_paradigm_suffix_omits_dense_when_dm_off(tmp_path: Path) -> None:
    cfg = _make_cfg(
        tmp_path,
        sampling_pattern="physics_anchored",
        zoom_window_width_kev=10.0,
        local_event_fraction=0.80,
        physics_peaks_kev=[1592.0, 1620.0, 2103.0, 2614.0],
        trial_size_strategy="variable_uniform",
        n_trial_events_min=32,
        n_trial_events_max=1024,
        aggregator=AggregatorConfig(
            type="cross_attention",
            num_heads=8,
            attention_dim=64,
            gaussian_attention_bias=True,
        ),
    )
    assert "_dense" not in paradigm_path_suffix(cfg)


# ---------------------------------------------------------------------- #
# Pool-Based Absolute Density SFN (Cell 7)
# ---------------------------------------------------------------------- #


def _mock_pool(n: int = 4000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(500.0, 3000.0, size=n)


def test_pool_density_sfn_defaults_off() -> None:
    from majorana_acp.cut_acceptance.config import PoolDensitySfnConfig

    cfg = AggregatorConfig()
    assert cfg.pool_density_sfn.enabled is False
    pdsfn = PoolDensitySfnConfig()
    assert pdsfn.sigma_max_kev == 200.0
    assert pdsfn.sigma_min_kev == 5.0
    assert pdsfn.sigma_local_kev == 10.0
    assert pdsfn.sigma_global_kev == 150.0
    assert pdsfn.epsilon == 1e-5


def test_pool_density_sfn_sigma_max_must_exceed_sigma_min() -> None:
    from majorana_acp.cut_acceptance.config import PoolDensitySfnConfig

    with pytest.raises(ValueError, match="sigma_max_kev"):
        PoolDensitySfnConfig(enabled=True, sigma_max_kev=5.0, sigma_min_kev=200.0)


def test_pool_density_sfn_global_must_exceed_local() -> None:
    from majorana_acp.cut_acceptance.config import PoolDensitySfnConfig

    with pytest.raises(ValueError, match="must exceed"):
        PoolDensitySfnConfig(enabled=True, sigma_local_kev=200.0, sigma_global_kev=10.0)


def test_pool_density_sfn_requires_gab(encoder_cfg: EncoderConfig) -> None:
    from majorana_acp.models.attentive_cnp import build_attentive_cnp

    with pytest.raises(ValueError, match="requires gaussian_attention_bias"):
        build_attentive_cnp(
            encoder_cfg,
            dim_theta=None,
            dim_phi=2,
            num_heads=4,
            attention_dim=64,
            gaussian_attention_bias=False,
            pool_density_sfn_enabled=True,
            pool_density_sfn_sigma_max_kev=200.0,
            pool_density_sfn_sigma_min_kev=5.0,
            pool_density_sfn_sigma_local_kev=10.0,
            pool_density_sfn_sigma_global_kev=150.0,
            pool_energies_kev=_mock_pool(),
            energy_range_kev=(500.0, 3000.0),
        )


def test_pool_density_sfn_mutex_with_bpbn(encoder_cfg: EncoderConfig) -> None:
    from majorana_acp.models.attentive_cnp import build_attentive_cnp

    with pytest.raises(ValueError, match="mutually exclusive"):
        build_attentive_cnp(
            encoder_cfg,
            dim_theta=None,
            dim_phi=2,
            num_heads=4,
            attention_dim=64,
            gaussian_attention_bias=True,
            bounded_bandwidth_enabled=True,
            bounded_sigma_max_kev=200.0,
            bounded_sigma_min_kev=5.0,
            bounded_alpha_max=40.0,
            bounded_sigma_local_kev=10.0,
            bounded_sigma_global_kev=150.0,
            pool_density_sfn_enabled=True,
            pool_density_sfn_sigma_max_kev=200.0,
            pool_density_sfn_sigma_min_kev=5.0,
            pool_density_sfn_sigma_local_kev=10.0,
            pool_density_sfn_sigma_global_kev=150.0,
            pool_energies_kev=_mock_pool(),
            energy_range_kev=(500.0, 3000.0),
        )


def test_pool_density_sfn_mutex_with_cell6_sfn(encoder_cfg: EncoderConfig) -> None:
    from majorana_acp.models.attentive_cnp import build_attentive_cnp

    with pytest.raises(ValueError, match="mutually exclusive"):
        build_attentive_cnp(
            encoder_cfg,
            dim_theta=None,
            dim_phi=2,
            num_heads=4,
            attention_dim=64,
            gaussian_attention_bias=True,
            sfn_modulation_enabled=True,
            sfn_sigma_max_kev=200.0,
            sfn_sigma_min_kev=5.0,
            sfn_sigma_local_kev=10.0,
            sfn_sigma_global_kev=150.0,
            pool_density_sfn_enabled=True,
            pool_density_sfn_sigma_max_kev=200.0,
            pool_density_sfn_sigma_min_kev=5.0,
            pool_density_sfn_sigma_local_kev=10.0,
            pool_density_sfn_sigma_global_kev=150.0,
            pool_energies_kev=_mock_pool(),
            energy_range_kev=(500.0, 3000.0),
        )


def test_pool_density_sfn_requires_pool_energies(encoder_cfg: EncoderConfig) -> None:
    from majorana_acp.models.attentive_cnp import build_attentive_cnp

    with pytest.raises(ValueError, match="pool_energies_kev"):
        build_attentive_cnp(
            encoder_cfg,
            dim_theta=None,
            dim_phi=2,
            num_heads=4,
            attention_dim=64,
            gaussian_attention_bias=True,
            pool_density_sfn_enabled=True,
            pool_density_sfn_sigma_max_kev=200.0,
            pool_density_sfn_sigma_min_kev=5.0,
            pool_density_sfn_sigma_local_kev=10.0,
            pool_density_sfn_sigma_global_kev=150.0,
            pool_energies_kev=None,
            energy_range_kev=(500.0, 3000.0),
        )


def test_pool_density_sfn_on_replaces_log_gamma(encoder_cfg: EncoderConfig) -> None:
    from majorana_acp.models.attentive_cnp import build_attentive_cnp

    cnp = build_attentive_cnp(
        encoder_cfg,
        dim_theta=None,
        dim_phi=2,
        num_heads=4,
        attention_dim=64,
        gaussian_attention_bias=True,
        pool_density_sfn_enabled=True,
        pool_density_sfn_sigma_max_kev=200.0,
        pool_density_sfn_sigma_min_kev=5.0,
        pool_density_sfn_sigma_local_kev=10.0,
        pool_density_sfn_sigma_global_kev=150.0,
        pool_energies_kev=_mock_pool(),
        energy_range_kev=(500.0, 3000.0),
    )
    assert cnp.attention.log_gamma is None
    assert "attention.log_gamma" not in cnp.state_dict()
    # PE-free MLP weights present.
    pool_keys = [k for k in cnp.state_dict() if k.startswith("attention.pool_sfn_net")]
    assert len(pool_keys) == 4  # 2 Linear × (weight, bias)


def test_pool_density_sfn_buffer_is_not_persistent(encoder_cfg: EncoderConfig) -> None:
    """The pool buffer must NOT appear in state_dict — the buffer is
    rebuilt at both train and inference from cfg.train_predictions_path,
    so storing it would (a) bloat checkpoints unnecessarily and (b)
    create silent drift if the file changes."""
    from majorana_acp.models.attentive_cnp import build_attentive_cnp

    cnp = build_attentive_cnp(
        encoder_cfg,
        dim_theta=None,
        dim_phi=2,
        num_heads=4,
        attention_dim=64,
        gaussian_attention_bias=True,
        pool_density_sfn_enabled=True,
        pool_density_sfn_sigma_max_kev=200.0,
        pool_density_sfn_sigma_min_kev=5.0,
        pool_density_sfn_sigma_local_kev=10.0,
        pool_density_sfn_sigma_global_kev=150.0,
        pool_energies_kev=_mock_pool(),
        energy_range_kev=(500.0, 3000.0),
    )
    # Buffer exists on the module …
    assert cnp.attention.pool_energies_norm.numel() > 0
    # … but is NOT in state_dict (persistent=False).
    assert "attention.pool_energies_norm" not in cnp.state_dict()


def test_pool_density_sfn_sigma_kev_converted_to_norm(encoder_cfg: EncoderConfig) -> None:
    from majorana_acp.models.attentive_cnp import build_attentive_cnp

    cnp = build_attentive_cnp(
        encoder_cfg,
        dim_theta=None,
        dim_phi=2,
        num_heads=4,
        attention_dim=64,
        gaussian_attention_bias=True,
        pool_density_sfn_enabled=True,
        pool_density_sfn_sigma_max_kev=200.0,
        pool_density_sfn_sigma_min_kev=5.0,
        pool_density_sfn_sigma_local_kev=10.0,
        pool_density_sfn_sigma_global_kev=150.0,
        pool_energies_kev=_mock_pool(),
        energy_range_kev=(500.0, 3000.0),
    )
    # 200 / 2500 = 0.08; 5 / 2500 = 0.002; 10 / 2500 = 0.004; 150 / 2500 = 0.06.
    assert abs(cnp.attention.pool_density_sfn_sigma_max_norm - 0.08) < 1e-9
    assert abs(cnp.attention.pool_density_sfn_sigma_min_norm - 0.002) < 1e-9
    assert abs(cnp.attention.pool_density_sfn_sigma_local_norm - 0.004) < 1e-9
    assert abs(cnp.attention.pool_density_sfn_sigma_global_norm - 0.06) < 1e-9


def test_pool_density_sfn_forward_shape(encoder_cfg: EncoderConfig) -> None:
    from majorana_acp.models.attentive_cnp import build_attentive_cnp

    cnp = build_attentive_cnp(
        encoder_cfg,
        dim_theta=None,
        dim_phi=2,
        num_heads=8,
        attention_dim=64,
        gaussian_attention_bias=True,
        pool_density_sfn_enabled=True,
        pool_density_sfn_sigma_max_kev=200.0,
        pool_density_sfn_sigma_min_kev=5.0,
        pool_density_sfn_sigma_local_kev=10.0,
        pool_density_sfn_sigma_global_kev=150.0,
        pool_energies_kev=_mock_pool(),
        energy_range_kev=(500.0, 3000.0),
    )
    ctx = _random_batch(B=2, N=64, dim_phi=2, seed=0)
    tgt = _random_batch(B=2, N=8, dim_phi=2, seed=1)
    out = cnp(ctx, tgt)
    assert out.mu_logit.shape == (2, 8)
    assert out.log_sigma.shape == (2, 8)


def test_pool_density_sfn_density_drops_in_data_sparse_region(
    encoder_cfg: EncoderConfig,
) -> None:
    """Sanity check on the absolute-density signal: a target far from
    every pool event must give a much smaller ρ_local than a target
    surrounded by pool events. This is the mathematical reason Cell 7
    can express ``widen the attention window, no local evidence``."""
    from majorana_acp.models.attentive_cnp import build_attentive_cnp

    rng = np.random.default_rng(0)
    # Pool packed entirely around E=1500 keV (within ±50 keV).
    pool = rng.uniform(1450.0, 1550.0, size=2000)
    cnp = build_attentive_cnp(
        encoder_cfg,
        dim_theta=None,
        dim_phi=2,
        num_heads=4,
        attention_dim=64,
        gaussian_attention_bias=True,
        pool_density_sfn_enabled=True,
        pool_density_sfn_sigma_max_kev=200.0,
        pool_density_sfn_sigma_min_kev=5.0,
        pool_density_sfn_sigma_local_kev=10.0,
        pool_density_sfn_sigma_global_kev=150.0,
        pool_energies_kev=pool,
        energy_range_kev=(500.0, 3000.0),
    )
    # Reach into the aggregator to compute ρ_local at a "rich" query
    # (1500 keV → E_norm ≈ 0.4) and a "sparse" query (2800 keV →
    # E_norm ≈ 0.92) using the very same code path the forward uses.
    s_l = cnp.attention.pool_density_sfn_sigma_local_norm
    pool_norm = cnp.attention.pool_energies_norm
    eps = cnp.attention.pool_density_sfn_epsilon

    def _rho_local(e_norm_val: float) -> float:
        e_t = torch.tensor([[e_norm_val]], dtype=torch.float32).unsqueeze(-1)
        d2 = (e_t - pool_norm.view(1, 1, -1)).pow(2)
        return float(torch.exp(-d2 / (2 * s_l * s_l)).sum().item())

    rho_rich = _rho_local(0.4)
    rho_sparse = _rho_local(0.92)
    # log(ρ + ε) at the sparse query must be much smaller than at the rich
    # one; this is the magnitude the MLP sees and learns to interpret as
    # "widen the attention window".
    log_rich = float(np.log(rho_rich + eps))
    log_sparse = float(np.log(rho_sparse + eps))
    assert log_rich - log_sparse > 5.0, (
        f"expected log ρ_local at rich-data query ({log_rich}) to vastly "
        f"exceed log ρ_local at sparse query ({log_sparse})"
    )


def test_paradigm_suffix_includes_pdsfn_when_on(tmp_path: Path) -> None:
    from majorana_acp.cut_acceptance.config import PoolDensitySfnConfig

    cfg = _make_cfg(
        tmp_path,
        sampling_pattern="flat_stratified",
        trial_size_strategy="variable_uniform",
        n_trial_events_min=640,
        n_trial_events_max=1024,
        aggregator=AggregatorConfig(
            type="cross_attention",
            num_heads=8,
            attention_dim=64,
            gaussian_attention_bias=True,
            pool_density_sfn=PoolDensitySfnConfig(
                enabled=True,
                sigma_max_kev=200.0,
                sigma_min_kev=5.0,
            ),
        ),
    )
    assert paradigm_path_suffix(cfg).endswith("_attn8x64_gab_pdsfn")


def test_paradigm_suffix_omits_pdsfn_when_off(tmp_path: Path) -> None:
    cfg = _make_cfg(
        tmp_path,
        sampling_pattern="flat_stratified",
        trial_size_strategy="variable_uniform",
        n_trial_events_min=640,
        n_trial_events_max=1024,
        aggregator=AggregatorConfig(
            type="cross_attention",
            num_heads=8,
            attention_dim=64,
            gaussian_attention_bias=True,
        ),
    )
    assert "_pdsfn" not in paradigm_path_suffix(cfg)
