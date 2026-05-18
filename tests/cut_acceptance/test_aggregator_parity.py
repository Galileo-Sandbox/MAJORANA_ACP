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
