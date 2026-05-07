"""Tests for ``majorana_acp.models``."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from majorana_acp.models import build_model, list_models, register_model
from majorana_acp.models.inception_time import InceptionTime
from majorana_acp.models.mlp import MLP
from majorana_acp.models.resnet1d import ResNet1D
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


def test_simple_cnn_supports_multi_channel_input() -> None:
    """With in_channels=2 the model accepts (B, 2, L)."""
    model = SimpleCNN(in_channels=2, channels=[8, 16], kernel_size=3)
    out = model(torch.randn(4, 2, 1024))
    assert out.shape == (4,)


def test_simple_cnn_default_in_channels_is_one() -> None:
    """Default in_channels=1 keeps the (B, L) input form working."""
    model = SimpleCNN()
    assert model.in_channels == 1
    out = model(torch.randn(4, 3800))
    assert out.shape == (4,)


def test_simple_cnn_rejects_zero_in_channels() -> None:
    with pytest.raises(ValueError):
        SimpleCNN(in_channels=0)


def test_simple_cnn_uses_batchnorm_by_default() -> None:
    model = SimpleCNN(channels=[16, 32], kernel_size=3)
    assert any(isinstance(m, nn.BatchNorm1d) for m in model.modules())
    assert not any(isinstance(m, nn.GroupNorm) for m in model.modules())


def test_simple_cnn_groupnorm_swap_replaces_all_batchnorms() -> None:
    """norm='group' should replace every BatchNorm1d with GroupNorm."""
    model = SimpleCNN(channels=[16, 32], kernel_size=3, norm="group", num_groups=8)
    assert not any(isinstance(m, nn.BatchNorm1d) for m in model.modules())
    gns = [m for m in model.modules() if isinstance(m, nn.GroupNorm)]
    assert len(gns) == 2
    assert all(gn.num_groups == 8 for gn in gns)


def test_simple_cnn_groupnorm_forward_shape() -> None:
    model = SimpleCNN(channels=[16, 32], kernel_size=3, norm="group", num_groups=8)
    out = model(torch.randn(4, 1024))
    assert out.shape == (4,)


def test_simple_cnn_groupnorm_rejects_indivisible_channels() -> None:
    """num_groups=8 cannot evenly divide 10 channels."""
    with pytest.raises(ValueError):
        SimpleCNN(channels=[10, 20], kernel_size=3, norm="group", num_groups=8)


def test_simple_cnn_layernorm_rejected() -> None:
    """LayerNorm is reserved for flat (B, F) activations, not (B, C, L)."""
    with pytest.raises(ValueError, match="not applicable to 1D conv"):
        SimpleCNN(channels=[16, 32], kernel_size=3, norm="layer")


def test_simple_cnn_unknown_norm_raises() -> None:
    with pytest.raises(ValueError):
        SimpleCNN(channels=[16, 32], kernel_size=3, norm="instance")  # type: ignore[arg-type]


# --- MLP -------------------------------------------------------------


def test_mlp_is_registered() -> None:
    assert "mlp" in list_models()


def test_build_model_mlp_returns_nn_module() -> None:
    model = build_model("mlp", input_dim=128, hidden_dims=[16, 8])
    assert isinstance(model, MLP)


def test_mlp_forward_shape_2d_input() -> None:
    model = MLP()
    x = torch.randn(4, 3800)
    out = model(x)
    assert out.shape == (4,)
    assert out.dtype == torch.float32


def test_mlp_forward_shape_3d_input() -> None:
    """(B, 1, L) input is flattened to (B, L)."""
    model = MLP()
    x = torch.randn(4, 1, 3800)
    out = model(x)
    assert out.shape == (4,)


def test_mlp_outputs_logits_not_probabilities() -> None:
    torch.manual_seed(0)
    model = MLP()
    x = torch.randn(64, 3800) * 5
    out = model(x)
    assert (out < 0).any() or (out > 1).any(), (
        "MLP output looks like a probability, expected raw logits"
    )


def test_mlp_gradient_flows() -> None:
    model = MLP(input_dim=64, hidden_dims=[16, 8], dropout=0.0)
    x = torch.randn(2, 64)
    target = torch.tensor([0.0, 1.0])
    loss = nn.functional.binary_cross_entropy_with_logits(model(x), target)
    loss.backward()
    grads = [p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()]
    assert all(grads)


def test_mlp_input_size_mismatch_raises() -> None:
    model = MLP(input_dim=3800)
    with pytest.raises(ValueError):
        model(torch.randn(2, 100))  # wrong input length


def test_mlp_rejects_bad_input_dim() -> None:
    with pytest.raises(ValueError):
        MLP(input_dim=0)


def test_mlp_rejects_empty_hidden_dims() -> None:
    with pytest.raises(ValueError):
        MLP(hidden_dims=[])


def test_mlp_rejects_bad_dropout() -> None:
    with pytest.raises(ValueError):
        MLP(dropout=1.0)
    with pytest.raises(ValueError):
        MLP(dropout=-0.1)


def test_mlp_handles_multi_channel_via_flatten() -> None:
    """Multi-channel input flattens cleanly when input_dim = C * L."""
    L, C = 100, 2
    model = MLP(input_dim=C * L, hidden_dims=[16, 8], dropout=0.0)
    out = model(torch.randn(4, C, L))
    assert out.shape == (4,)


def test_mlp_uses_batchnorm_by_default() -> None:
    model = MLP(input_dim=128, hidden_dims=[16, 8])
    assert any(isinstance(m, nn.BatchNorm1d) for m in model.modules())
    assert not any(isinstance(m, nn.LayerNorm) for m in model.modules())


def test_mlp_layernorm_swap_replaces_all_batchnorms() -> None:
    """norm='layer' should replace every BatchNorm1d with LayerNorm."""
    model = MLP(input_dim=128, hidden_dims=[16, 8], norm="layer")
    assert not any(isinstance(m, nn.BatchNorm1d) for m in model.modules())
    lns = [m for m in model.modules() if isinstance(m, nn.LayerNorm)]
    assert len(lns) == 2
    assert lns[0].normalized_shape == (16,)
    assert lns[1].normalized_shape == (8,)


def test_mlp_layernorm_forward_shape() -> None:
    model = MLP(input_dim=128, hidden_dims=[16, 8], norm="layer", dropout=0.0)
    out = model(torch.randn(4, 128))
    assert out.shape == (4,)


def test_mlp_groupnorm_rejected() -> None:
    """GroupNorm is meaningless on flat (B, F) activations — must error."""
    with pytest.raises(ValueError, match="not applicable to flat"):
        MLP(input_dim=128, hidden_dims=[16, 8], norm="group")


def test_mlp_unknown_norm_raises() -> None:
    with pytest.raises(ValueError):
        MLP(input_dim=128, hidden_dims=[16, 8], norm="instance")  # type: ignore[arg-type]


# --- ResNet1D --------------------------------------------------------


def test_resnet1d_is_registered() -> None:
    assert "resnet1d" in list_models()


def test_build_model_resnet1d_returns_nn_module() -> None:
    model = build_model("resnet1d", base_channels=8, blocks_per_stage=[1, 1])
    assert isinstance(model, ResNet1D)


def test_resnet1d_forward_shape_2d_input() -> None:
    model = ResNet1D(base_channels=8, blocks_per_stage=[1, 1])
    out = model(torch.randn(4, 1024))
    assert out.shape == (4,)
    assert out.dtype == torch.float32


def test_resnet1d_forward_shape_3d_input() -> None:
    model = ResNet1D(base_channels=8, blocks_per_stage=[1, 1])
    out = model(torch.randn(4, 1, 1024))
    assert out.shape == (4,)


def test_resnet1d_supports_multi_channel_input() -> None:
    model = ResNet1D(in_channels=2, base_channels=8, blocks_per_stage=[1, 1])
    out = model(torch.randn(4, 2, 1024))
    assert out.shape == (4,)


def test_resnet1d_outputs_logits_not_probabilities() -> None:
    torch.manual_seed(0)
    model = ResNet1D(base_channels=8, blocks_per_stage=[1, 1])
    x = torch.randn(64, 1024) * 5
    out = model(x)
    assert (out < 0).any() or (out > 1).any()


def test_resnet1d_gradient_flows() -> None:
    model = ResNet1D(base_channels=8, blocks_per_stage=[1, 1], dropout=0.0)
    x = torch.randn(2, 1024)
    target = torch.tensor([0.0, 1.0])
    loss = nn.functional.binary_cross_entropy_with_logits(model(x), target)
    loss.backward()
    grads = [p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()]
    assert all(grads)


def test_resnet1d_rejects_zero_in_channels() -> None:
    with pytest.raises(ValueError):
        ResNet1D(in_channels=0)


def test_resnet1d_rejects_empty_blocks_per_stage() -> None:
    with pytest.raises(ValueError):
        ResNet1D(blocks_per_stage=[])


def test_resnet1d_rejects_zero_blocks() -> None:
    with pytest.raises(ValueError):
        ResNet1D(blocks_per_stage=[2, 0, 2])


def test_resnet1d_rejects_even_kernel() -> None:
    with pytest.raises(ValueError):
        ResNet1D(kernel_size=4)


def test_resnet1d_rejects_bad_dropout() -> None:
    with pytest.raises(ValueError):
        ResNet1D(dropout=1.0)
    with pytest.raises(ValueError):
        ResNet1D(dropout=-0.1)


def test_resnet1d_uses_batchnorm_by_default() -> None:
    model = ResNet1D(base_channels=8, blocks_per_stage=[1, 1])
    assert any(isinstance(m, nn.BatchNorm1d) for m in model.modules())
    assert not any(isinstance(m, nn.GroupNorm) for m in model.modules())


def test_resnet1d_groupnorm_swap_replaces_all_batchnorms() -> None:
    """norm='group' replaces every BatchNorm — stem + each block + each downsample."""
    model = ResNet1D(base_channels=16, blocks_per_stage=[1, 1], norm="group", num_groups=8)
    assert not any(isinstance(m, nn.BatchNorm1d) for m in model.modules())
    gns = [m for m in model.modules() if isinstance(m, nn.GroupNorm)]
    # stem(1) + 2 stages × (2 block-norms + 1 downsample for the strided 2nd stage) = 6
    assert len(gns) == 6
    assert all(gn.num_groups == 8 for gn in gns)


def test_resnet1d_groupnorm_forward_shape() -> None:
    model = ResNet1D(base_channels=8, blocks_per_stage=[1, 1], norm="group", num_groups=4)
    out = model(torch.randn(4, 1024))
    assert out.shape == (4,)


def test_resnet1d_groupnorm_rejects_indivisible_channels() -> None:
    with pytest.raises(ValueError):
        ResNet1D(base_channels=10, blocks_per_stage=[1, 1], norm="group", num_groups=8)


def test_resnet1d_layernorm_rejected() -> None:
    with pytest.raises(ValueError, match="not applicable to 1D conv"):
        ResNet1D(base_channels=8, blocks_per_stage=[1, 1], norm="layer")


def test_resnet1d_groupnorm_init_weights_set() -> None:
    """GroupNorm weights should be 1, biases 0 after _init_weights."""
    model = ResNet1D(base_channels=8, blocks_per_stage=[1, 1], norm="group", num_groups=4)
    for m in model.modules():
        if isinstance(m, nn.GroupNorm):
            assert torch.all(m.weight == 1.0)
            assert torch.all(m.bias == 0.0)


# --- InceptionTime ---------------------------------------------------


def test_inception_time_is_registered() -> None:
    assert "inception_time" in list_models()


def test_build_model_inception_time_returns_nn_module() -> None:
    model = build_model("inception_time", n_filters=8, n_blocks=1, modules_per_block=2)
    assert isinstance(model, InceptionTime)


def test_inception_time_forward_shape_2d_input() -> None:
    model = InceptionTime(n_filters=8, n_blocks=1, modules_per_block=2)
    out = model(torch.randn(4, 1024))
    assert out.shape == (4,)
    assert out.dtype == torch.float32


def test_inception_time_forward_shape_3d_input() -> None:
    model = InceptionTime(n_filters=8, n_blocks=1, modules_per_block=2)
    out = model(torch.randn(4, 1, 1024))
    assert out.shape == (4,)


def test_inception_time_supports_multi_channel_input() -> None:
    model = InceptionTime(in_channels=2, n_filters=8, n_blocks=1, modules_per_block=2)
    out = model(torch.randn(4, 2, 1024))
    assert out.shape == (4,)


def test_inception_time_preserves_temporal_length_via_same_padding() -> None:
    """``padding='same'`` (and matching maxpool padding) keeps the
    temporal axis the same length through the backbone."""
    model = InceptionTime(n_filters=8, n_blocks=1, modules_per_block=2)
    x = torch.randn(2, 1, 333)
    feats = model.backbone(x)  # before GAP
    assert feats.shape[-1] == x.shape[-1]


def test_inception_time_outputs_logits_not_probabilities() -> None:
    torch.manual_seed(0)
    model = InceptionTime(n_filters=8, n_blocks=1, modules_per_block=2)
    out = model(torch.randn(64, 1024) * 5)
    assert (out < 0).any() or (out > 1).any()


def test_inception_time_gradient_flows() -> None:
    model = InceptionTime(n_filters=8, n_blocks=1, modules_per_block=2, dropout=0.0)
    x = torch.randn(2, 1024)
    target = torch.tensor([0.0, 1.0])
    loss = nn.functional.binary_cross_entropy_with_logits(model(x), target)
    loss.backward()
    grads = [p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()]
    assert all(grads)


def test_inception_time_rejects_zero_in_channels() -> None:
    with pytest.raises(ValueError):
        InceptionTime(in_channels=0)


def test_inception_time_rejects_zero_n_filters() -> None:
    with pytest.raises(ValueError):
        InceptionTime(n_filters=0)


def test_inception_time_rejects_empty_kernel_sizes() -> None:
    with pytest.raises(ValueError):
        InceptionTime(kernel_sizes=[])


def test_inception_time_rejects_zero_n_blocks() -> None:
    with pytest.raises(ValueError):
        InceptionTime(n_blocks=0)


def test_inception_time_rejects_zero_modules_per_block() -> None:
    with pytest.raises(ValueError):
        InceptionTime(modules_per_block=0)


def test_inception_time_rejects_bad_dropout() -> None:
    with pytest.raises(ValueError):
        InceptionTime(dropout=1.0)
    with pytest.raises(ValueError):
        InceptionTime(dropout=-0.1)


def test_inception_time_uses_batchnorm_by_default() -> None:
    model = InceptionTime(n_filters=8, n_blocks=1, modules_per_block=2)
    assert any(isinstance(m, nn.BatchNorm1d) for m in model.modules())
    assert not any(isinstance(m, nn.GroupNorm) for m in model.modules())


def test_inception_time_groupnorm_swap_replaces_all_batchnorms() -> None:
    """norm='group' should replace every BatchNorm — module BN + shortcut BN."""
    model = InceptionTime(n_filters=8, n_blocks=1, modules_per_block=2, norm="group", num_groups=4)
    assert not any(isinstance(m, nn.BatchNorm1d) for m in model.modules())
    gns = [m for m in model.modules() if isinstance(m, nn.GroupNorm)]
    assert all(gn.num_groups == 4 for gn in gns)
    # 2 modules × 1 BN each = 2, plus 1 shortcut BN (in_channels != out) = 3.
    assert len(gns) == 3


def test_inception_time_groupnorm_forward_shape() -> None:
    model = InceptionTime(n_filters=8, n_blocks=1, modules_per_block=2, norm="group", num_groups=4)
    out = model(torch.randn(4, 1024))
    assert out.shape == (4,)


def test_inception_time_layernorm_rejected() -> None:
    with pytest.raises(ValueError, match="not applicable to 1D conv"):
        InceptionTime(n_filters=8, n_blocks=1, modules_per_block=2, norm="layer")


def test_inception_time_groupnorm_rejects_indivisible_channels() -> None:
    """4 kernels × 7 filters = 28 channels — not divisible by 8."""
    with pytest.raises(ValueError):
        InceptionTime(
            n_filters=7,
            kernel_sizes=[3, 5, 7],
            n_blocks=1,
            modules_per_block=1,
            norm="group",
            num_groups=8,
        )
