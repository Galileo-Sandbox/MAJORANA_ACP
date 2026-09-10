"""Additive mechanism controls for the frozen density-guided CNP."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn

MODES = (
    "full_density",
    "global_gate",
    "global_attention",
    "global_both",
    "density_free_global",
)
GLOBAL_GATE_MODES = {"global_gate", "global_both", "density_free_global"}
GLOBAL_ATTENTION_MODES = {
    "global_attention",
    "global_both",
    "density_free_global",
}
MODE_CODES = {name: index for index, name in enumerate(MODES)}


class ConstantDensityInput(nn.Module):
    """Call an existing density network on Z0=(0,0) at every query."""

    def __init__(self, network: nn.Module) -> None:
        super().__init__()
        self.network = network

    def forward(self, density_features: torch.Tensor) -> torch.Tensor:
        return self.network(torch.zeros_like(density_features))


class ControlledAttention(nn.Module):
    """Modify only the decoder-gate and direct-density side outputs."""

    def __init__(self, base: nn.Module, mode: str) -> None:
        super().__init__()
        if mode not in MODES or mode == "full_density":
            raise ValueError(f"ControlledAttention requires a non-full mode, got {mode!r}")
        self.base = base
        self.mode = mode
        self.register_buffer("control_mode_code", torch.tensor(MODE_CODES[mode]))
        if mode in GLOBAL_GATE_MODES:
            if not hasattr(base, "kappa_raw"):
                raise ValueError("Global gate requires the original trainable kappa_raw tensor")
            with torch.no_grad():
                base.kappa_raw.fill_(math.log(2.0 / 7.0))
        if mode in GLOBAL_ATTENTION_MODES:
            base.pool_sfn_net = ConstantDensityInput(base.pool_sfn_net)
            base.pool_tau_net = ConstantDensityInput(base.pool_tau_net)

    @property
    def gaussian_attention_bias(self) -> bool:
        return bool(self.base.gaussian_attention_bias)

    @property
    def pe_detach_qk(self) -> bool:
        return bool(self.base.pe_detach_qk)

    def forward(self, z_phi_target: torch.Tensor, *args: Any, **kwargs: Any):
        representation, side_info = self.base(z_phi_target, *args, **kwargs)
        side_info = dict(side_info)
        if self.mode in GLOBAL_GATE_MODES:
            phi = self.base.kappa_raw
            lambda_global = 1.0 + 9.0 * torch.sigmoid(phi)
            band_index = torch.arange(
                self.base.pool_density_sfn_num_bands,
                dtype=z_phi_target.dtype,
                device=z_phi_target.device,
            ).view(1, 1, -1)
            weights = torch.sigmoid(
                self.base.pool_density_sfn_band_filter_alpha * (lambda_global - band_index)
            )
            side_info["band_weights"] = weights.expand(
                z_phi_target.shape[0], z_phi_target.shape[1], -1
            )
        if self.mode == "density_free_global":
            side_info["contrast_ratio"] = torch.zeros_like(side_info["contrast_ratio"])
        return representation, side_info


def apply_control_mode(model: nn.Module, mode: str) -> nn.Module:
    """Apply a frozen control mode after constructing the original model."""
    if mode not in MODES:
        raise ValueError(f"Unknown control mode {mode!r}")
    if mode == "full_density":
        model.control_mode = mode
        return model
    model.attention = ControlledAttention(model.attention, mode)
    model.control_mode = mode
    return model


def base_attention(model: nn.Module) -> nn.Module:
    return (
        model.attention.base
        if isinstance(model.attention, ControlledAttention)
        else model.attention
    )


def canonical_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return original-model key names while retaining control-trained tensors."""
    if not isinstance(model.attention, ControlledAttention):
        return dict(model.state_dict())
    state: dict[str, torch.Tensor] = {}
    for key, value in model.state_dict().items():
        if key == "attention.control_mode_code":
            continue
        canonical = key.replace("attention.base.", "attention.")
        canonical = canonical.replace("pool_sfn_net.network.", "pool_sfn_net.")
        canonical = canonical.replace("pool_tau_net.network.", "pool_tau_net.")
        state[canonical] = value
    return state


def save_control_checkpoint(
    path: Path,
    model: nn.Module,
    *,
    mode: str,
    history: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> None:
    payload = {
        "schema_version": 1,
        "control_mode": mode,
        "model_state": canonical_state_dict(model),
        "history": dict(history),
        "metadata": dict(metadata),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_control_checkpoint(
    path: Path, base_model: nn.Module, *, expected_mode: str
) -> tuple[nn.Module, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    observed_mode = payload.get("control_mode")
    if observed_mode != expected_mode:
        raise ValueError(
            f"Checkpoint mode {observed_mode!r} does not match requested {expected_mode!r}"
        )
    base_model.load_state_dict(payload["model_state"], strict=True)
    apply_control_mode(base_model, expected_mode)
    return base_model, payload


def parameter_inventory(model: nn.Module, mode: str) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    inactive = 0
    attention = base_attention(model)
    if mode in GLOBAL_ATTENTION_MODES:
        inactive += attention.pool_sfn_net.network[0].weight.numel()
        inactive += attention.pool_tau_net.network[0].weight.numel()
    if mode == "density_free_global":
        inactive += model.decoder.net[0].weight.shape[0]
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "effectively_participating_parameters": trainable - inactive,
        "structurally_inactive_parameters": inactive,
    }


def task_schedule(
    seed: int, steps: int, n_min: int = 640, n_max: int = 1024
) -> list[tuple[int, int, int, int]]:
    """Reproduce the executed variable-N trainer's RNG call order."""
    import numpy as np

    rng = np.random.default_rng(seed)
    result = []
    for _ in range(steps):
        n_events = int(rng.integers(n_min, n_max + 1))
        n_context = int(rng.integers(128, min(512, n_events - 1) + 1))
        sampler_seed = int(rng.integers(0, 2**31 - 1))
        split_seed = int(rng.integers(0, 2**31 - 1))
        result.append((n_events, n_context, sampler_seed, split_seed))
    return result
