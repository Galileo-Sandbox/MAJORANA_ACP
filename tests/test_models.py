"""Tests for ``majorana_acp.models``."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from majorana_acp.models import build_model, list_models, register_model
from majorana_acp.models.simple_cnn import SimpleCNN

# --- Registry --------------------------------------------------------


def test_simple_cnn_is_registered() -> None:
    assert "simple_cnn" in list_models()


def test_build_model_unknown_name_raises() -> None:
    with pytest.raises(KeyError):
        build_model("does_not_exist")


def test_build_model_returns_nn_module() -> None:
    model = build_model("simple_cnn")
    assert isinstance(model, nn.Module)
    assert isinstance(model, SimpleCNN)


def test_build_model_passes_params_through() -> None:
    model = build_model("simple_cnn", channels=[8, 16], kernel_size=3, dropout=0.0)
    assert isinstance(model, SimpleCNN)
    # The last conv block has 16 out_channels.
    last_conv = next(m for m in reversed(list(model.features)) if isinstance(m, nn.Conv1d))
    assert last_conv.out_channels == 16
    assert last_conv.kernel_size == (3,)


def test_register_model_rejects_duplicate_name() -> None:
    @register_model("dup_test_unique_name")
    class _A(nn.Module):
        pass

    with pytest.raises(ValueError):

        @register_model("dup_test_unique_name")
        class _B(nn.Module):
            pass


def test_list_models_is_sorted() -> None:
    names = list_models()
    assert names == sorted(names)


# --- SimpleCNN -------------------------------------------------------


def test_simple_cnn_forward_shape_2d_input() -> None:
    model = SimpleCNN()
    x = torch.randn(4, 3800)
    out = model(x)
    assert out.shape == (4,)
    assert out.dtype == torch.float32


def test_simple_cnn_forward_shape_3d_input() -> None:
    """Model accepts an explicit channel dim too: (B, 1, L)."""
    model = SimpleCNN()
    x = torch.randn(4, 1, 3800)
    out = model(x)
    assert out.shape == (4,)


def test_simple_cnn_outputs_logits_not_probabilities() -> None:
    """Logits can be negative; a sigmoid output never can."""
    torch.manual_seed(0)
    model = SimpleCNN()
    x = torch.randn(64, 3800) * 5  # crank inputs to provoke extreme logits
    out = model(x)
    assert (out < 0).any() or (out > 1).any(), (
        "model output looks like a probability, expected raw logits"
    )


def test_simple_cnn_gradient_flows() -> None:
    model = SimpleCNN()
    x = torch.randn(2, 3800, requires_grad=False)
    target = torch.tensor([0.0, 1.0])
    logits = model(x)
    loss = nn.functional.binary_cross_entropy_with_logits(logits, target)
    loss.backward()
    # Every parameter should have a gradient after backward.
    grads_present = [p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()]
    assert all(grads_present)


def test_simple_cnn_works_on_short_input() -> None:
    """Adaptive pooling handles a shorter waveform than the 3800 default."""
    model = SimpleCNN(channels=[8, 16], kernel_size=3)
    out = model(torch.randn(2, 1024))
    assert out.shape == (2,)


def test_simple_cnn_rejects_bad_kernel_size() -> None:
    with pytest.raises(ValueError):
        SimpleCNN(kernel_size=4)  # even
    with pytest.raises(ValueError):
        SimpleCNN(kernel_size=0)


def test_simple_cnn_rejects_bad_dropout() -> None:
    with pytest.raises(ValueError):
        SimpleCNN(dropout=1.0)
    with pytest.raises(ValueError):
        SimpleCNN(dropout=-0.1)


def test_simple_cnn_rejects_empty_channels() -> None:
    with pytest.raises(ValueError):
        SimpleCNN(channels=[])
