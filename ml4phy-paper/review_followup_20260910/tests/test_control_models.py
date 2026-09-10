"""Correctness tests for the additive mechanism controls."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch
from torch import nn

MODULE_PATH = Path(__file__).parents[1] / "control_models.py"
SPEC = importlib.util.spec_from_file_location("review_controls", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
CONTROLS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONTROLS)


class FakeAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.gaussian_attention_bias = True
        self.pe_detach_qk = True
        self.kappa_raw = nn.Parameter(torch.tensor(0.4))
        self.pool_density_sfn_num_bands = 10
        self.pool_density_sfn_band_filter_alpha = 5.0
        self.pool_sfn_net = nn.Sequential(nn.Linear(2, 16), nn.GELU(), nn.Linear(16, 1))
        self.pool_tau_net = nn.Sequential(nn.Linear(2, 16), nn.GELU(), nn.Linear(16, 1))

    def forward(self, target, *_args, **_kwargs):
        z = target[..., :2]
        side = {
            "band_weights": target.new_zeros((*target.shape[:2], 10)),
            "contrast_ratio": target[..., 0],
            "sigma": self.pool_sfn_net(z),
            "tau": self.pool_tau_net(z),
        }
        return target, side


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = FakeAttention()
        self.decoder = nn.Module()
        self.decoder.net = nn.Sequential(nn.Linear(5, 3))


def test_global_gate_initializes_to_three_and_is_query_invariant():
    wrapper = CONTROLS.ControlledAttention(FakeAttention(), "global_gate")
    target = torch.randn(2, 7, 4)
    _, side = wrapper(target)
    expected = torch.sigmoid(5 * (torch.tensor(3.0) - torch.arange(10)))
    torch.testing.assert_close(side["band_weights"][0, 0], expected)
    assert torch.equal(side["band_weights"][0, 0], side["band_weights"][1, 6])


def test_global_attention_uses_constant_density_features():
    wrapper = CONTROLS.ControlledAttention(FakeAttention(), "global_attention")
    first = torch.randn(1, 5, 4)
    second = first.clone()
    second[..., :2] += 100
    _, first_side = wrapper(first)
    _, second_side = wrapper(second)
    assert torch.equal(first_side["sigma"], second_side["sigma"])
    assert torch.equal(first_side["tau"], second_side["tau"])


def test_density_free_zeroes_only_direct_side_feature():
    wrapper = CONTROLS.ControlledAttention(FakeAttention(), "density_free_global")
    target = torch.randn(1, 3, 4)
    representation, side = wrapper(target)
    assert torch.equal(representation, target)
    assert torch.count_nonzero(side["contrast_ratio"]) == 0
    assert side["band_weights"].shape == (1, 3, 10)


def test_global_gate_phi_has_finite_nonzero_gradient():
    wrapper = CONTROLS.ControlledAttention(FakeAttention(), "global_gate")
    _, side = wrapper(torch.randn(1, 3, 4))
    side["band_weights"].sum().backward()
    assert torch.isfinite(wrapper.base.kappa_raw.grad)
    assert wrapper.base.kappa_raw.grad.abs() > 0


def test_task_schedule_is_deterministic_and_seed_specific():
    assert CONTROLS.task_schedule(0, 20) == CONTROLS.task_schedule(0, 20)
    assert CONTROLS.task_schedule(0, 20) != CONTROLS.task_schedule(1, 20)


def test_global_gate_checkpoint_restores_trained_phi(tmp_path):
    model = FakeModel()
    CONTROLS.apply_control_mode(model, "global_gate")
    with torch.no_grad():
        CONTROLS.base_attention(model).kappa_raw.fill_(0.75)
    checkpoint = tmp_path / "trained-global-gate.ckpt"
    CONTROLS.save_control_checkpoint(
        checkpoint,
        model,
        mode="global_gate",
        history={},
        metadata={},
    )
    restored, _ = CONTROLS.load_control_checkpoint(
        checkpoint,
        FakeModel(),
        expected_mode="global_gate",
    )
    assert float(CONTROLS.base_attention(restored).kappa_raw.detach()) == pytest.approx(0.75)
