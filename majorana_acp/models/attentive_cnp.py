"""Local Attentive Conditional Neural Process for cut-acceptance.

Replaces the hard-coded global mean aggregator in upstream
``core.surrogate_cnp.ConditionalNeuralProcess`` with a target-conditioned
multi-head cross-attention. Used when ``CutAcceptanceConfig.aggregator``
is set to ``type: "cross_attention"``; the ``mean`` path stays on
upstream and is byte-for-byte unchanged.

Math (math contract identical to the spec in CLAUDE.md):

* Encode the context and target through the upstream ``UniversalEncoder``
  to obtain per-event latents ``z_φ_C: [B, N_C, Z]`` and
  ``z_φ_T: [B, N_T, Z]``.
* Run the upstream ``ContextPointEncoder`` to obtain per-event values
  ``h_C: [B, N_C, agg]`` (we set ``agg = Z`` so all downstream shapes
  stay symmetric).
* Multi-head cross-attention:
      Q  =  z_φ_T · W_Q                  ∈ [B, N_T, attention_dim]
      K  =  z_φ_C · W_K                  ∈ [B, N_C, attention_dim]
      V  =  h_C   · W_V                  ∈ [B, N_C, attention_dim]
  Split into ``num_heads`` heads of width ``d = attention_dim / num_heads``,
  apply scaled dot-product attention, concat heads, project back via
  ``W_O`` to ``[B, N_T, Z]`` (locked d_v = Z so the upstream decoder's
  input dim stays 2Z and the existing ``CnpDecoder.net`` MLP works
  unmodified).
* Decode by concatenating ``[r(φ_T), z_φ_T]`` and running it through
  the upstream ``CnpDecoder.net`` MLP directly (we bypass its
  ``forward()`` because that method assumes mean-aggregator shape
  ``[B, agg]`` and broadcasts via ``unsqueeze(1).expand``).

Output: the same ``core.surrogate_cnp.CnpOutput`` dataclass the
upstream training loop and loss function consume, so this class is a
drop-in replacement for ``ConditionalNeuralProcess`` from the caller's
perspective.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from core.networks import UniversalEncoder, build_encoder  # noqa: E402
from core.surrogate_cnp import (
    CnpDecoder,
    CnpOutput,
    ContextPointEncoder,
)
from schemas.config import EncoderConfig
from schemas.data_models import StandardBatch

__all__ = ["AttentiveCNP", "CrossAttentionAggregator", "build_attentive_cnp"]

_SQRT_2PI = math.sqrt(2.0 * math.pi)

# Per-chunk size when accumulating pool-density kernel sums under the
# Pool-Density-SFN branch. Each chunk produces a [B, N_T, chunk] kernel
# tensor; at the configured chunk = 4096 with the largest realistic
# training-batch shape (B=16, N_T≈1000) this peaks at ~250 MB which
# fits comfortably on CPU and is invisible on the 32 GB GPU.
_POOL_DENSITY_KERNEL_CHUNK = 4096


class CrossAttentionAggregator(nn.Module):
    """Multi-head cross-attention from target queries onto context (K, V).

    Q is derived from ``z_φ_T`` (post-phi-encoder target latents).
    K is derived from ``z_φ_C`` (post-phi-encoder context latents).
    V is derived from ``h_C`` (the per-event context representation
    produced by ``ContextPointEncoder``, which already absorbed the
    binary label ``X_C``).

    Output dim is locked to the encoder's ``latent_dim = Z`` so the
    downstream decoder's input dim stays ``[B, N_T, 2Z]`` — the same
    shape produced by the legacy mean path. This lets us reuse the
    upstream ``CnpDecoder.net`` MLP without architecture changes.

    Optional ``gaussian_attention_bias`` adds a continuous learnable
    relative-energy penalty to the attention logits:

        scores = (Q · K^T) / √d_k − exp(log_gamma_h) · Δ²

    where ``Δ² = (E_target_norm − E_ctx_norm)²`` (broadcast to per-head)
    and ``log_gamma`` is a learnable ``[num_heads, 1, 1]`` parameter
    initialised at zero. Each head learns its own effective receptive-
    field width on the continuous energy axis. With ``log_gamma = 0``
    the penalty is ``Δ²`` itself, which is ≤ 1 for ``E_norm ∈ [0, 1]``
    — small compared to typical pre-softmax logits, so init is close
    to the un-biased attention layer.
    """

    def __init__(
        self,
        *,
        latent_dim: int,
        agg_dim: int,
        num_heads: int,
        attention_dim: int,
        dropout: float = 0.0,
        gaussian_attention_bias: bool = False,
        density_modulation_enabled: bool = False,
        density_sigma_local_norm: float | None = None,
        density_sigma_global_norm: float | None = None,
        density_epsilon: float = 1e-6,
        bounded_bandwidth_enabled: bool = False,
        bounded_sigma_max_norm: float | None = None,
        bounded_sigma_min_norm: float | None = None,
        bounded_alpha_max: float | None = None,
        bounded_sensitivity_hidden_dim: int = 16,
        bounded_sigma_local_norm: float | None = None,
        bounded_sigma_global_norm: float | None = None,
        bounded_epsilon: float = 1e-6,
        sfn_modulation_enabled: bool = False,
        sfn_sigma_max_norm: float | None = None,
        sfn_sigma_min_norm: float | None = None,
        sfn_hidden_dim: int = 16,
        sfn_sigma_local_norm: float | None = None,
        sfn_sigma_global_norm: float | None = None,
        sfn_epsilon: float = 1e-6,
        pool_density_sfn_enabled: bool = False,
        pool_density_sfn_sigma_max_norm: float | None = None,
        pool_density_sfn_sigma_min_norm: float | None = None,
        pool_density_sfn_hidden_dim: int = 16,
        pool_density_sfn_sigma_local_norm: float | None = None,
        pool_density_sfn_sigma_global_norm: float | None = None,
        pool_density_sfn_epsilon: float = 1e-5,
        pool_energies_norm: torch.Tensor | None = None,
        pool_density_sfn_temperature_gating: bool = False,
        pool_density_sfn_tau_min: float = 1.0,
        pool_density_sfn_tau_max: float = 10.0,
    ) -> None:
        super().__init__()
        if attention_dim % num_heads != 0:
            raise ValueError(
                f"attention_dim ({attention_dim}) must be divisible by num_heads ({num_heads})"
            )
        self.num_heads = int(num_heads)
        self.attention_dim = int(attention_dim)
        self.head_dim = self.attention_dim // self.num_heads
        self.scale = self.head_dim**-0.5

        # Q / K source: z_φ (latent_dim wide). V source: h_C (agg_dim
        # wide). All three project into ``attention_dim`` total.
        self.w_q = nn.Linear(latent_dim, attention_dim, bias=False)
        self.w_k = nn.Linear(latent_dim, attention_dim, bias=False)
        self.w_v = nn.Linear(agg_dim, attention_dim, bias=False)
        # Final projection back to ``latent_dim`` (= d_v = Z), so the
        # decoder input dim stays 2Z whatever the aggregator.
        self.w_o = nn.Linear(attention_dim, latent_dim, bias=False)

        # Reuse ``encoder.dropout`` for the attention weights AND the
        # output projection — user-locked decision.
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

        # Per-head learnable log-scale for the continuous relative-
        # distance penalty. Registered only when the bias is requested
        # AND neither BPBN nor SFN is in use — both alternatives replace
        # log_gamma with their own bounded-σ parametrisation. Existing
        # un-BPBN/un-SFN checkpoints keep their state_dict shape.
        self.gaussian_attention_bias = bool(gaussian_attention_bias)
        self.bounded_bandwidth_enabled = bool(bounded_bandwidth_enabled)
        self.sfn_modulation_enabled = bool(sfn_modulation_enabled)
        self.pool_density_sfn_enabled = bool(pool_density_sfn_enabled)
        if self.bounded_bandwidth_enabled and not self.gaussian_attention_bias:
            raise ValueError(
                "bounded_bandwidth_enabled=True requires "
                "gaussian_attention_bias=True; BPBN replaces the GAB "
                "penalty parametrisation."
            )
        if self.sfn_modulation_enabled and not self.gaussian_attention_bias:
            raise ValueError(
                "sfn_modulation_enabled=True requires "
                "gaussian_attention_bias=True; SFN replaces the GAB "
                "penalty parametrisation."
            )
        if self.pool_density_sfn_enabled and not self.gaussian_attention_bias:
            raise ValueError(
                "pool_density_sfn_enabled=True requires "
                "gaussian_attention_bias=True; pool-density SFN replaces "
                "the GAB penalty parametrisation."
            )
        # Three-way mutex — BPBN (Cell 5), context-SFN (Cell 6) and
        # pool-density SFN (Cell 7) all replace the same GAB log_gamma
        # parametrisation. Enabling more than one would route through
        # multiple penalties, contradicting itself.
        _bandwidth_branches = (
            self.bounded_bandwidth_enabled,
            self.sfn_modulation_enabled,
            self.pool_density_sfn_enabled,
        )
        if sum(_bandwidth_branches) > 1:
            raise ValueError(
                "bounded_bandwidth, sfn_modulation, and pool_density_sfn "
                "are mutually exclusive — all three replace the GAB "
                "log_gamma parametrisation. Enable at most one."
            )
        if self.gaussian_attention_bias and not any(_bandwidth_branches):
            self.log_gamma = nn.Parameter(torch.zeros(self.num_heads, 1, 1))
        else:
            self.register_parameter("log_gamma", None)

        # Density modulation — kicks in only when both GAB and DM are on
        # (DM modifies the GAB penalty; no GAB → no penalty to modify).
        self.density_modulation_enabled = bool(density_modulation_enabled)
        if self.density_modulation_enabled:
            if not self.gaussian_attention_bias:
                # DM without GAB is a config-level no-op, surfaced here
                # as an explicit raise so silent miswiring can't hide.
                raise ValueError(
                    "density_modulation_enabled=True requires "
                    "gaussian_attention_bias=True; DM modifies the GAB "
                    "penalty, so without GAB it has nothing to modify."
                )
            if density_sigma_local_norm is None or density_sigma_global_norm is None:
                raise ValueError(
                    "density_modulation_enabled requires both "
                    "density_sigma_local_norm and density_sigma_global_norm "
                    "(precomputed from keV via the factory)."
                )
            if density_sigma_global_norm <= density_sigma_local_norm:
                raise ValueError(
                    f"density_sigma_global_norm ({density_sigma_global_norm}) "
                    f"must exceed density_sigma_local_norm ({density_sigma_local_norm})"
                )
            # Stash as plain floats — neither σ is learnable for v1.
            self.density_sigma_local_norm = float(density_sigma_local_norm)
            self.density_sigma_global_norm = float(density_sigma_global_norm)
            self.density_epsilon = float(density_epsilon)
        else:
            self.density_sigma_local_norm = 0.0  # unused
            self.density_sigma_global_norm = 0.0
            self.density_epsilon = float(density_epsilon)

        # Bounded Physical Bandwidth Network — replaces log_gamma·Δ²
        # with an explicit clamped σ_head(E_*). When enabled we build
        # the sensitivity MLP and store its three rigid hyperparameters
        # (σ_max, σ_min, α_max all in normalised E units).
        if self.bounded_bandwidth_enabled:
            if (
                bounded_sigma_max_norm is None
                or bounded_sigma_min_norm is None
                or bounded_alpha_max is None
            ):
                raise ValueError(
                    "bounded_bandwidth_enabled requires sigma_max_norm, "
                    "sigma_min_norm, and alpha_max."
                )
            if bounded_sigma_min_norm >= bounded_sigma_max_norm:
                raise ValueError(
                    f"bounded_sigma_min_norm ({bounded_sigma_min_norm}) must be < "
                    f"bounded_sigma_max_norm ({bounded_sigma_max_norm})"
                )
            if bounded_sigma_local_norm is None or bounded_sigma_global_norm is None:
                raise ValueError(
                    "bounded_bandwidth_enabled requires BOTH sigma_local_norm "
                    "and sigma_global_norm for the R_* kernel-density ratio."
                )
            if bounded_sigma_global_norm <= bounded_sigma_local_norm:
                raise ValueError(
                    f"bounded_sigma_global_norm ({bounded_sigma_global_norm}) "
                    f"must exceed bounded_sigma_local_norm ({bounded_sigma_local_norm})"
                )
            self.bounded_sigma_max_norm = float(bounded_sigma_max_norm)
            self.bounded_sigma_min_norm = float(bounded_sigma_min_norm)
            self.bounded_alpha_max = float(bounded_alpha_max)
            self.bounded_sigma_local_norm = float(bounded_sigma_local_norm)
            self.bounded_sigma_global_norm = float(bounded_sigma_global_norm)
            self.bounded_epsilon = float(bounded_epsilon)
            # Sensitivity MLP: smooth 1D → per-head λ. Input dim 1
            # because E_target_norm is a scalar per query.
            self.sensitivity_mlp = nn.Sequential(
                nn.Linear(1, int(bounded_sensitivity_hidden_dim)),
                nn.GELU(),
                nn.Linear(int(bounded_sensitivity_hidden_dim), self.num_heads),
            )
        else:
            self.bounded_sigma_max_norm = 0.0  # unused
            self.bounded_sigma_min_norm = 0.0
            self.bounded_alpha_max = 0.0
            self.bounded_sigma_local_norm = 0.0
            self.bounded_sigma_global_norm = 0.0
            self.bounded_epsilon = float(bounded_epsilon)

        # Bounded 2D Spectral Filter Network — replaces log_gamma·Δ²
        # with σ_head = σ_min + (σ_max − σ_min) · sigmoid(SFN(Z_*)),
        # where Z_* = [E_target_norm, log(R_* + ε)] ∈ R². The SFN is a
        # small Linear(2 → 16) → GELU → Linear(16 → H) MLP whose spectral
        # bias preferentially smooths R_*'s shot noise while still
        # tracking persistent macro signatures of real γ-peaks.
        if self.sfn_modulation_enabled:
            if sfn_sigma_max_norm is None or sfn_sigma_min_norm is None:
                raise ValueError(
                    "sfn_modulation_enabled requires sigma_max_norm and "
                    "sigma_min_norm (precomputed from keV via the factory)."
                )
            if sfn_sigma_min_norm >= sfn_sigma_max_norm:
                raise ValueError(
                    f"sfn_sigma_min_norm ({sfn_sigma_min_norm}) must be < "
                    f"sfn_sigma_max_norm ({sfn_sigma_max_norm})"
                )
            if sfn_sigma_local_norm is None or sfn_sigma_global_norm is None:
                raise ValueError(
                    "sfn_modulation_enabled requires both sigma_local_norm "
                    "and sigma_global_norm for the R_* kernel-density ratio."
                )
            if sfn_sigma_global_norm <= sfn_sigma_local_norm:
                raise ValueError(
                    f"sfn_sigma_global_norm ({sfn_sigma_global_norm}) "
                    f"must exceed sfn_sigma_local_norm ({sfn_sigma_local_norm})"
                )
            self.sfn_sigma_max_norm = float(sfn_sigma_max_norm)
            self.sfn_sigma_min_norm = float(sfn_sigma_min_norm)
            self.sfn_sigma_local_norm = float(sfn_sigma_local_norm)
            self.sfn_sigma_global_norm = float(sfn_sigma_global_norm)
            self.sfn_epsilon = float(sfn_epsilon)
            # Network input: 2D coordinate [E_target_norm, log R_*].
            self.sfn_net = nn.Sequential(
                nn.Linear(2, int(sfn_hidden_dim)),
                nn.GELU(),
                nn.Linear(int(sfn_hidden_dim), self.num_heads),
            )
        else:
            self.sfn_sigma_max_norm = 0.0  # unused
            self.sfn_sigma_min_norm = 0.0
            self.sfn_sigma_local_norm = 0.0
            self.sfn_sigma_global_norm = 0.0
            self.sfn_epsilon = float(sfn_epsilon)

        # Pool-Based Absolute Density SFN (Cell 7). Same physical bounds
        # and same sigmoid bandwidth gate as Cell 6, but:
        #   * R_* is replaced by [log ρ_local, log ρ_global] — absolute
        #     UN-normalised kernel sums (no σ √(2π) division). The MLP
        #     therefore sees the absolute count of events near E_*, not
        #     a self-normalised ratio that collapses to 1 in sparse-data
        #     regions.
        #   * Densities are evaluated against a *fixed pool buffer* of
        #     training-pool event energies — registered as a
        #     non-persistent buffer so the state_dict stays small and
        #     identical across train/inference rebuilds (the same call to
        #     load_events at both endpoints reconstructs the buffer
        #     byte-for-byte).
        if self.pool_density_sfn_enabled:
            if (
                pool_density_sfn_sigma_max_norm is None
                or pool_density_sfn_sigma_min_norm is None
            ):
                raise ValueError(
                    "pool_density_sfn_enabled requires sigma_max_norm and "
                    "sigma_min_norm (precomputed from keV via the factory)."
                )
            if pool_density_sfn_sigma_min_norm >= pool_density_sfn_sigma_max_norm:
                raise ValueError(
                    f"pool_density_sfn_sigma_min_norm ({pool_density_sfn_sigma_min_norm}) "
                    f"must be < pool_density_sfn_sigma_max_norm "
                    f"({pool_density_sfn_sigma_max_norm})"
                )
            if (
                pool_density_sfn_sigma_local_norm is None
                or pool_density_sfn_sigma_global_norm is None
            ):
                raise ValueError(
                    "pool_density_sfn_enabled requires both sigma_local_norm "
                    "and sigma_global_norm for the pool-density kernels."
                )
            if pool_density_sfn_sigma_global_norm <= pool_density_sfn_sigma_local_norm:
                raise ValueError(
                    f"pool_density_sfn_sigma_global_norm "
                    f"({pool_density_sfn_sigma_global_norm}) must exceed "
                    f"pool_density_sfn_sigma_local_norm "
                    f"({pool_density_sfn_sigma_local_norm})"
                )
            if pool_energies_norm is None:
                raise ValueError(
                    "pool_density_sfn_enabled=True requires a pool_energies_norm "
                    "tensor (the kept-event energies of the training pool, "
                    "normalised to [0, 1]). The factory caller "
                    "(pipeline.build_local_cnp / cnp_test_inference._load_cnp) "
                    "is responsible for loading and normalising it."
                )
            pool_t = pool_energies_norm.detach().to(torch.float32).reshape(-1)
            if pool_t.numel() == 0:
                raise ValueError(
                    "pool_energies_norm is empty — cannot compute pool-density "
                    "features against zero events."
                )
            # ``persistent=False`` keeps the pool out of ``state_dict``
            # so checkpoints stay small AND identical regardless of pool
            # size. The buffer is rebuilt by the factory at both
            # training and inference time.
            self.register_buffer("pool_energies_norm", pool_t, persistent=False)
            self.pool_density_sfn_sigma_max_norm = float(pool_density_sfn_sigma_max_norm)
            self.pool_density_sfn_sigma_min_norm = float(pool_density_sfn_sigma_min_norm)
            self.pool_density_sfn_sigma_local_norm = float(pool_density_sfn_sigma_local_norm)
            self.pool_density_sfn_sigma_global_norm = float(pool_density_sfn_sigma_global_norm)
            self.pool_density_sfn_epsilon = float(pool_density_sfn_epsilon)
            # PE-free MLP. Input: 2D ``[log(ρ_local + ε), log(ρ_global + ε)]``.
            self.pool_sfn_net = nn.Sequential(
                nn.Linear(2, int(pool_density_sfn_hidden_dim)),
                nn.GELU(),
                nn.Linear(int(pool_density_sfn_hidden_dim), self.num_heads),
            )
            # Cell 8 — Dual-Gated SFN. A second PE-free MLP head emits a
            # per-target temperature τ_head ∈ [τ_min, τ_max] that scales
            # the pre-softmax ``Q·K^T``. Closes the asymmetric loophole
            # in Cell 7 where σ_head → σ_max in vacuum kills the spatial
            # penalty and leaves the raw PE10-driven QK^T free to fit
            # shot noise. Only registered when ``temperature_gating`` is
            # on, so Cell 7 checkpoints stay state-dict-compatible.
            self.pool_density_sfn_temperature_gating = bool(
                pool_density_sfn_temperature_gating
            )
            if self.pool_density_sfn_temperature_gating:
                if pool_density_sfn_tau_max <= pool_density_sfn_tau_min:
                    raise ValueError(
                        f"pool_density_sfn_tau_max ({pool_density_sfn_tau_max}) "
                        f"must exceed pool_density_sfn_tau_min "
                        f"({pool_density_sfn_tau_min})"
                    )
                self.pool_density_sfn_tau_min = float(pool_density_sfn_tau_min)
                self.pool_density_sfn_tau_max = float(pool_density_sfn_tau_max)
                self.pool_tau_net = nn.Sequential(
                    nn.Linear(2, int(pool_density_sfn_hidden_dim)),
                    nn.GELU(),
                    nn.Linear(int(pool_density_sfn_hidden_dim), self.num_heads),
                )
            else:
                self.pool_density_sfn_tau_min = float(pool_density_sfn_tau_min)
                self.pool_density_sfn_tau_max = float(pool_density_sfn_tau_max)
        else:
            self.pool_density_sfn_sigma_max_norm = 0.0  # unused
            self.pool_density_sfn_sigma_min_norm = 0.0
            self.pool_density_sfn_sigma_local_norm = 0.0
            self.pool_density_sfn_sigma_global_norm = 0.0
            self.pool_density_sfn_epsilon = float(pool_density_sfn_epsilon)
            self.pool_density_sfn_temperature_gating = False
            self.pool_density_sfn_tau_min = float(pool_density_sfn_tau_min)
            self.pool_density_sfn_tau_max = float(pool_density_sfn_tau_max)

    def forward(
        self,
        z_phi_target: torch.Tensor,  # [B, N_T, Z]
        z_phi_ctx: torch.Tensor,  # [B, N_C, Z]
        h_ctx: torch.Tensor,  # [B, N_C, agg]
        *,
        e_target_norm: torch.Tensor | None = None,  # [B, N_T] (optional)
        e_ctx_norm: torch.Tensor | None = None,  # [B, N_C] (optional)
    ) -> torch.Tensor:
        B, N_T, _ = z_phi_target.shape
        N_C = z_phi_ctx.size(1)
        H, d = self.num_heads, self.head_dim

        q = self.w_q(z_phi_target).reshape(B, N_T, H, d).transpose(1, 2)
        k = self.w_k(z_phi_ctx).reshape(B, N_C, H, d).transpose(1, 2)
        v = self.w_v(h_ctx).reshape(B, N_C, H, d).transpose(1, 2)
        # q, k, v: [B, H, *, d]

        attn_logits = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, N_T, N_C]

        if self.gaussian_attention_bias:
            if e_target_norm is None or e_ctx_norm is None:
                raise ValueError(
                    "gaussian_attention_bias is enabled but e_target_norm / "
                    "e_ctx_norm were not passed to the attention forward."
                )
            # Δ² = (E_t − E_c)² ∈ [0, 1] when E_norm ∈ [0, 1].
            delta = e_target_norm.unsqueeze(-1) - e_ctx_norm.unsqueeze(1)  # [B, N_T, N_C]
            delta2_flat = delta.pow(2)  # [B, N_T, N_C]
            delta2 = delta2_flat.unsqueeze(1)  # [B, 1, N_T, N_C]

            if self.bounded_bandwidth_enabled:
                # BPBN — explicit bounded physical bandwidth σ_head(E_*).
                # 1. R_* via the same dual-σ self-normalised KDE as DM.
                s_l = self.bounded_sigma_local_norm
                s_g = self.bounded_sigma_global_norm
                k_local_b = torch.exp(-delta2_flat / (2.0 * s_l * s_l))
                k_global_b = torch.exp(-delta2_flat / (2.0 * s_g * s_g))
                rho_local_b = k_local_b.sum(dim=-1) / (s_l * _SQRT_2PI)  # [B, N_T]
                rho_global_b = k_global_b.sum(dim=-1) / (s_g * _SQRT_2PI) + self.bounded_epsilon
                R_b = rho_local_b / rho_global_b  # [B, N_T]
                # 2. Per-head sensitivity λ(E_*) ∈ [0, α_max].
                e_in = e_target_norm.unsqueeze(-1)  # [B, N_T, 1]
                lam_logits = self.sensitivity_mlp(e_in)  # [B, N_T, H]
                lam = self.bounded_alpha_max * torch.sigmoid(lam_logits)  # [B, N_T, H]
                lam = lam.transpose(1, 2).unsqueeze(-1)  # [B, H, N_T, 1]
                # 3. σ_head(E_*) = σ_max / (1 + λ · (R − 1)_+), then clamp.
                R_minus_1 = F.relu(R_b - 1.0).unsqueeze(1).unsqueeze(-1)  # [B, 1, N_T, 1]
                sigma_head = self.bounded_sigma_max_norm / (1.0 + lam * R_minus_1)
                sigma_head = torch.clamp(
                    sigma_head,
                    min=self.bounded_sigma_min_norm,
                    max=self.bounded_sigma_max_norm,
                )
                # 4. Penalty = Δ² / (2 σ_head²).
                penalty = delta2 / (2.0 * sigma_head.pow(2))  # [B, H, N_T, N_C]
                attn_logits = attn_logits - penalty
            elif self.pool_density_sfn_enabled:
                # Pool-Based Absolute Density SFN (Cell 7). Density
                # features are computed against the FIXED pool buffer,
                # not against the per-trial context — eliminating the
                # bimodal anchor noise + shot-noise variance that
                # corrupted the Cell-6 ratio R_*. Kernel sums are
                # UN-normalised so the MLP sees absolute statistical
                # bounds: log ρ → -∞ in data-sparse regions tells the
                # network "widen the attention window, you have no
                # local evidence", a cue Cell 6's self-normalised
                # R_* erases.
                #
                # The buffer is large (~10⁴–10⁵ events). Per-event
                # kernel sums against it create a [B, N_T, N_pool]
                # tensor whose autograd graph would dominate memory.
                # The two ρ values are inputs to the MLP — not
                # parameters — so we drop the graph entirely with
                # ``torch.no_grad`` and chunk over the pool dimension
                # to cap peak memory; only the MLP forward (which
                # *is* learnable) keeps a graph.
                e_t = e_target_norm.unsqueeze(-1)  # [B, N_T, 1]
                s_l = self.pool_density_sfn_sigma_local_norm
                s_g = self.pool_density_sfn_sigma_global_norm
                inv2_l = 1.0 / (2.0 * s_l * s_l)
                inv2_g = 1.0 / (2.0 * s_g * s_g)
                pool = self.pool_energies_norm  # [N_pool]
                B_q, N_T_q = e_target_norm.shape
                rho_l = torch.zeros(
                    (B_q, N_T_q),
                    dtype=e_target_norm.dtype,
                    device=e_target_norm.device,
                )
                rho_g = torch.zeros_like(rho_l)
                with torch.no_grad():
                    N_pool = int(pool.numel())
                    for start in range(0, N_pool, _POOL_DENSITY_KERNEL_CHUNK):
                        e_pool_chunk = pool[
                            start : start + _POOL_DENSITY_KERNEL_CHUNK
                        ].view(1, 1, -1)
                        d2_chunk = (e_t - e_pool_chunk).pow(2)
                        rho_l = rho_l + torch.exp(-d2_chunk * inv2_l).sum(dim=-1)
                        rho_g = rho_g + torch.exp(-d2_chunk * inv2_g).sum(dim=-1)
                    log_l = torch.log(rho_l + self.pool_density_sfn_epsilon)
                    log_g = torch.log(rho_g + self.pool_density_sfn_epsilon)
                    Z = torch.stack([log_l, log_g], dim=-1)  # [B, N_T, 2]
                # MLP forward keeps a graph — its weights are the
                # learnable parameters of this branch.
                logits = self.pool_sfn_net(Z)  # [B, N_T, H]
                logits = logits.transpose(1, 2).unsqueeze(-1)  # [B, H, N_T, 1]
                sigma_range = (
                    self.pool_density_sfn_sigma_max_norm
                    - self.pool_density_sfn_sigma_min_norm
                )
                sigma_head = self.pool_density_sfn_sigma_min_norm + sigma_range * torch.sigmoid(
                    logits
                )
                # Cell 8 — Dual-Gated SFN: apply per-target temperature
                # scaling to ``Q·K^T`` BEFORE the spatial penalty
                # subtraction. Implements
                #   Scores = (Q·K^T) / (√d · τ) − Δ² / (2 σ²)
                # When the σ-head saturates at σ_max (vacuum region), the
                # spatial penalty collapses → τ_head saturating at
                # τ_max simultaneously flattens the raw QK^T spikes so
                # PE10's bin-scale Fourier features cannot push the
                # post-softmax distribution into single-event attention.
                if self.pool_density_sfn_temperature_gating:
                    tau_logits = self.pool_tau_net(Z)  # [B, N_T, H]
                    tau_logits = tau_logits.transpose(1, 2).unsqueeze(-1)  # [B, H, N_T, 1]
                    tau_range = (
                        self.pool_density_sfn_tau_max - self.pool_density_sfn_tau_min
                    )
                    tau_head = self.pool_density_sfn_tau_min + tau_range * torch.sigmoid(
                        tau_logits
                    )
                    attn_logits = attn_logits / tau_head
                penalty = delta2 / (2.0 * sigma_head.pow(2))  # [B, H, N_T, N_C]
                attn_logits = attn_logits - penalty
            elif self.sfn_modulation_enabled:
                # SFN — bounded σ_head via a learned 2D coordinate MLP.
                # 1. R_* via the same dual-σ self-normalised KDE as DM/BPBN.
                s_l = self.sfn_sigma_local_norm
                s_g = self.sfn_sigma_global_norm
                k_local_s = torch.exp(-delta2_flat / (2.0 * s_l * s_l))
                k_global_s = torch.exp(-delta2_flat / (2.0 * s_g * s_g))
                rho_local_s = k_local_s.sum(dim=-1) / (s_l * _SQRT_2PI)  # [B, N_T]
                rho_global_s = k_global_s.sum(dim=-1) / (s_g * _SQRT_2PI) + self.sfn_epsilon
                R_s = rho_local_s / rho_global_s  # [B, N_T]
                # 2. Build the 2D coordinate Z_* = [E_target_norm, log R_*].
                #    log(R + ε) keeps the feature finite when R → 0 and
                #    compresses R's dynamic range (peaks have R ~ 10–15)
                #    so the SFN MLP sees both knobs on comparable scales.
                log_R = torch.log(R_s + self.sfn_epsilon)  # [B, N_T]
                Z = torch.stack([e_target_norm, log_R], dim=-1)  # [B, N_T, 2]
                # 3. Forward pass through the SFN — emits one logit per head.
                logits = self.sfn_net(Z)  # [B, N_T, H]
                logits = logits.transpose(1, 2).unsqueeze(-1)  # [B, H, N_T, 1]
                # 4. σ_head = σ_min + (σ_max − σ_min) · sigmoid(logits).
                #    sigmoid is naturally bounded in [0, 1] so σ_head is
                #    automatically clamped to [σ_min, σ_max] without an
                #    explicit ``torch.clamp`` — no gradient deadzone.
                sigma_range = self.sfn_sigma_max_norm - self.sfn_sigma_min_norm
                sigma_head = self.sfn_sigma_min_norm + sigma_range * torch.sigmoid(logits)
                # 5. Penalty = Δ² / (2 σ_head²).
                penalty = delta2 / (2.0 * sigma_head.pow(2))  # [B, H, N_T, N_C]
                attn_logits = attn_logits - penalty
            else:
                if self.density_modulation_enabled:
                    # Self-normalised Gaussian-kernel densities at each
                    # target query E_*, summed over context events:
                    #   ρ̂_σ(E_*) = (1/(σ √(2π))) · Σ_j exp(-Δ²_{*,j}/(2σ²))
                    # Dividing each kernel sum by ``σ √(2π)`` (the integral
                    # of the unnormalised Gaussian) means a uniform context
                    # density f produces ρ̂_local = ρ̂_global = f and the
                    # ratio ``R_* ≈ 1`` in flat regions. Sharp peak clusters
                    # of width << σ_local raise ρ̂_local relative to ρ̂_global
                    # by ~σ_global / σ_local, driving ``R_* ≫ 1`` and
                    # contracting the head's effective receptive field
                    # exactly where the physics signal is sharpest.
                    s_l = self.density_sigma_local_norm
                    s_g = self.density_sigma_global_norm
                    k_local = torch.exp(-delta2_flat / (2.0 * s_l * s_l))
                    k_global = torch.exp(-delta2_flat / (2.0 * s_g * s_g))
                    rho_local = k_local.sum(dim=-1) / (s_l * _SQRT_2PI)  # [B, N_T]
                    rho_global = k_global.sum(dim=-1) / (s_g * _SQRT_2PI) + self.density_epsilon
                    R = (rho_local / rho_global).unsqueeze(1).unsqueeze(-1)  # [B, 1, N_T, 1]
                    penalty_per_pair = R * delta2  # [B, 1, N_T, N_C]
                else:
                    penalty_per_pair = delta2  # [B, 1, N_T, N_C]

                # exp(log_gamma) is positive — a per-head learnable scale.
                gamma = torch.exp(self.log_gamma).unsqueeze(0)  # [1, H, 1, 1]
                attn_logits = attn_logits - gamma * penalty_per_pair

        attn = F.softmax(attn_logits, dim=-1)
        attn = self.attn_dropout(attn)

        out = attn @ v  # [B, H, N_T, d]
        out = out.transpose(1, 2).reshape(B, N_T, H * d)  # [B, N_T, attention_dim]
        out = self.w_o(out)  # [B, N_T, latent_dim]
        out = self.out_dropout(out)
        return out


def _recover_raw_phi(phi: torch.Tensor) -> torch.Tensor:
    """Return the raw 2D ``(E_norm, T_norm)`` array given a possibly
    PE-encoded ``phi``.

    * ``phi.shape[-1] == 2`` (PE disabled): pass-through.
    * ``phi.shape[-1] >= 3`` (PE enabled with our locked layout):
      the k=0 sin/cos band is at columns 0 and 1; recover
      ``E_norm = atan2(sin(πE), cos(πE)) / π``. The threshold scalar
      is at column −1.

    Exact inverse of the encoding for ``E_norm ∈ [0, 1]``: at those
    inputs the angle ``πE_norm`` stays in ``[0, π]`` where ``atan2``
    is single-valued.
    """
    if phi.shape[-1] == 2:
        return phi
    sin0 = phi[..., 0]
    cos0 = phi[..., 1]
    t = phi[..., -1]
    e_norm = torch.atan2(sin0, cos0) / torch.pi
    return torch.stack([e_norm, t], dim=-1)


class AttentiveCNP(nn.Module):
    """Drop-in replacement for ``ConditionalNeuralProcess`` with cross-attention.

    The forward signature ``(ctx_batch, target_batch) → CnpOutput`` is
    identical to upstream so ``core.training.train_cnp`` and
    ``core.surrogate_cnp.cnp_loss`` work without modification.

    Architecture: ``UniversalEncoder + ContextPointEncoder`` are
    reused from upstream; the mean aggregator + decoder-broadcast
    pair is replaced by ``CrossAttentionAggregator + decoder.net``.

    ``decoder_coordinate_gating`` (default ``False``) controls whether
    the decoder's coordinate input is the high-frequency latent
    ``z_φ_T`` (legacy attentive behavior — shape ``[B, N_T, 2Z]``) or
    the raw ``(E_norm, T_norm)`` 2D vector recovered from the batch
    (shape ``[B, N_T, Z + 2]``). Gating prevents the decoder MLP from
    using bin-scale Fourier features directly — the attention layer
    still navigates with the full PE bandwidth, but the decoder
    becomes blind to bin-scale noise.
    """

    def __init__(
        self,
        encoder: UniversalEncoder,
        context_encoder: ContextPointEncoder,
        attention: CrossAttentionAggregator,
        decoder: CnpDecoder,
        *,
        decoder_coordinate_gating: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.context_encoder = context_encoder
        self.attention = attention
        self.decoder = decoder
        self.decoder_coordinate_gating = bool(decoder_coordinate_gating)

    def _encode_per_event(self, batch: StandardBatch) -> tuple[torch.Tensor, torch.Tensor]:
        z_theta, z_phi = self.encoder(batch)  # [B, Z], [B, N, Z]
        N = z_phi.size(1)
        z_theta_per_event = z_theta.unsqueeze(1).expand(-1, N, -1)
        return z_theta_per_event, z_phi

    def forward(
        self,
        ctx_batch: StandardBatch,
        target_batch: StandardBatch,
    ) -> CnpOutput:
        if ctx_batch.batch_size != target_batch.batch_size:
            raise ValueError(f"ctx B={ctx_batch.batch_size} != target B={target_batch.batch_size}")
        if ctx_batch.mode is not target_batch.mode:
            raise ValueError(f"ctx mode {ctx_batch.mode} != target mode {target_batch.mode}")

        # Encode context (need both z_φ_C for K and h_C for V).
        z_theta_pe_C, z_phi_C = self._encode_per_event(ctx_batch)
        x_ctx = torch.as_tensor(ctx_batch.labels, dtype=z_phi_C.dtype, device=z_phi_C.device)
        h_C = self.context_encoder(z_theta_pe_C, z_phi_C, x_ctx)  # [B, N_C, agg]

        # Encode target (z_φ_T → Q AND decoder concat input).
        _, z_phi_T = self._encode_per_event(target_batch)  # [B, N_T, Z]

        # Recover raw E_norm for the GAB penalty when the attention
        # layer asked for it. Same atan2 inverse used by decoder
        # coordinate gating — exact for E_norm ∈ [0, 1] under our PE
        # layout, and a pass-through when PE is disabled (phi.shape[-1] == 2).
        if getattr(self.attention, "gaussian_attention_bias", False):
            phi_t_raw = _recover_raw_phi(
                torch.as_tensor(target_batch.phi, dtype=z_phi_T.dtype, device=z_phi_T.device)
            )
            phi_c_raw = _recover_raw_phi(
                torch.as_tensor(ctx_batch.phi, dtype=z_phi_C.dtype, device=z_phi_C.device)
            )
            e_target_norm = phi_t_raw[..., 0]  # [B, N_T]
            e_ctx_norm = phi_c_raw[..., 0]  # [B, N_C]
            r_target = self.attention(
                z_phi_T,
                z_phi_C,
                h_C,
                e_target_norm=e_target_norm,
                e_ctx_norm=e_ctx_norm,
            )
        else:
            # Cross-attention → target-conditioned representation.
            r_target = self.attention(z_phi_T, z_phi_C, h_C)  # [B, N_T, Z]

        # Decoder coordinate input. By default the decoder concatenates
        # r_target with the high-frequency latent z_phi_T (legacy
        # attentive behavior; preserves byte-for-byte parity with the
        # pre-gating AttentiveCNP). With coordinate gating on, swap to
        # the **raw** 2D (E_norm, T_norm) recovered from the batch:
        # the attention layer still benefits from PE bandwidth via Q/K,
        # but the decoder MLP no longer sees the bin-scale Fourier
        # features and therefore cannot fit bin-scale Bernoulli noise.
        if self.decoder_coordinate_gating:
            phi_tgt_t = torch.as_tensor(
                target_batch.phi,
                dtype=z_phi_T.dtype,
                device=z_phi_T.device,
            )
            coord_input = _recover_raw_phi(phi_tgt_t)  # [B, N_T, 2]
        else:
            coord_input = z_phi_T  # [B, N_T, Z]

        # Decode via the upstream CnpDecoder MLP directly. We skip
        # ``decoder.forward`` because it does
        # ``r_trial.unsqueeze(1).expand(-1, N_t, -1)`` which assumes
        # the mean-path shape [B, agg]; we already have per-target r.
        cat = torch.cat([r_target, coord_input], dim=-1)  # [B, N_T, Z + d_coord]
        out = self.decoder.net(cat)  # [B, N_T, 2]
        return CnpOutput(mu_logit=out[..., 0], log_sigma=out[..., 1])

    def predict_beta(
        self,
        ctx_batch: StandardBatch,
        target_batch: StandardBatch,
    ) -> torch.Tensor:
        """Deterministic ``β = sigmoid(μ_logit)``; mirrors upstream API."""
        out = self(ctx_batch, target_batch)
        return torch.sigmoid(out.mu_logit)


def build_attentive_cnp(
    encoder_config: EncoderConfig,
    dim_theta: int | None,
    dim_phi: int | None,
    *,
    num_heads: int,
    attention_dim: int,
    decoder_coordinate_gating: bool = False,
    gaussian_attention_bias: bool = False,
    density_modulation_enabled: bool = False,
    density_sigma_local_kev: float | None = None,
    density_sigma_global_kev: float | None = None,
    density_epsilon: float = 1e-6,
    bounded_bandwidth_enabled: bool = False,
    bounded_sigma_max_kev: float | None = None,
    bounded_sigma_min_kev: float | None = None,
    bounded_alpha_max: float | None = None,
    bounded_sensitivity_hidden_dim: int = 16,
    bounded_sigma_local_kev: float | None = None,
    bounded_sigma_global_kev: float | None = None,
    bounded_epsilon: float = 1e-6,
    sfn_modulation_enabled: bool = False,
    sfn_sigma_max_kev: float | None = None,
    sfn_sigma_min_kev: float | None = None,
    sfn_hidden_dim: int = 16,
    sfn_sigma_local_kev: float | None = None,
    sfn_sigma_global_kev: float | None = None,
    sfn_epsilon: float = 1e-6,
    pool_density_sfn_enabled: bool = False,
    pool_density_sfn_sigma_max_kev: float | None = None,
    pool_density_sfn_sigma_min_kev: float | None = None,
    pool_density_sfn_hidden_dim: int = 16,
    pool_density_sfn_sigma_local_kev: float | None = None,
    pool_density_sfn_sigma_global_kev: float | None = None,
    pool_density_sfn_epsilon: float = 1e-5,
    pool_density_sfn_temperature_gating: bool = False,
    pool_density_sfn_tau_min: float = 1.0,
    pool_density_sfn_tau_max: float = 10.0,
    pool_energies_kev: "np.ndarray | torch.Tensor | None" = None,
    energy_range_kev: tuple[float, float] | None = None,
    aggregator_dim: int | None = None,
    context_hidden_dims: list[int] | None = None,
    decoder_hidden_dims: list[int] | None = None,
) -> AttentiveCNP:
    """Mirror of upstream ``core.build_cnp`` that wires the attentive variant.

    Identical signature to ``core.build_cnp`` plus two attention knobs
    and the optional ``decoder_coordinate_gating`` flag. All MLPs are
    reused from upstream (``UniversalEncoder``, ``ContextPointEncoder``,
    ``CnpDecoder``) — only the aggregator and the decoder's coord-
    input wiring differ from the legacy CNP.

    When ``decoder_coordinate_gating`` is on, the decoder's
    ``CnpDecoder`` is built with ``latent_dim=2`` so its input layer
    has shape ``Z + 2`` (the size of the raw coordinate vector); when
    off, ``latent_dim=Z`` reproduces the legacy ``Z + Z = 2Z`` input.
    """
    encoder = build_encoder(encoder_config, dim_theta, dim_phi)
    Z = encoder_config.latent_dim
    agg_dim = aggregator_dim or Z
    ctx_hidden = list(context_hidden_dims or encoder_config.hidden_dims)
    dec_hidden = list(decoder_hidden_dims or encoder_config.hidden_dims)
    context_encoder = ContextPointEncoder(
        latent_dim=Z,
        hidden_dims=ctx_hidden,
        out_dim=agg_dim,
        dropout=encoder_config.dropout,
    )
    # Convert DM / BPBN keV sigmas to normalised units (the aggregator
    # operates on E_norm ∈ [0, 1], so its kernel widths must live in
    # the same normalised space). The conversion uses the energy range
    # that drives all CNP coordinate normalisation upstream — passed
    # in by the factory caller (pipeline.build_local_cnp / _load_cnp)
    # so this build site stays config-aware without re-reading YAML.
    sigma_local_norm: float | None = None
    sigma_global_norm: float | None = None
    width: float | None = None
    if energy_range_kev is not None:
        e_lo, e_hi = energy_range_kev
        width = float(e_hi) - float(e_lo)
        if width <= 0.0:
            raise ValueError(f"energy_range_kev must satisfy hi > lo, got {energy_range_kev}")

    if density_modulation_enabled:
        if width is None:
            raise ValueError(
                "density_modulation_enabled=True requires energy_range_kev "
                "so σ_kev can be converted to normalised units."
            )
        if density_sigma_local_kev is None or density_sigma_global_kev is None:
            raise ValueError(
                "density_modulation_enabled=True requires both "
                "density_sigma_local_kev and density_sigma_global_kev."
            )
        sigma_local_norm = density_sigma_local_kev / width
        sigma_global_norm = density_sigma_global_kev / width

    sfn_sigma_max_norm_v: float | None = None
    sfn_sigma_min_norm_v: float | None = None
    sfn_sigma_local_norm_v: float | None = None
    sfn_sigma_global_norm_v: float | None = None
    if sfn_modulation_enabled:
        if width is None:
            raise ValueError(
                "sfn_modulation_enabled=True requires energy_range_kev "
                "so σ_kev can be converted to normalised units."
            )
        if sfn_sigma_max_kev is None or sfn_sigma_min_kev is None:
            raise ValueError(
                "sfn_modulation_enabled=True requires sigma_max_kev and sigma_min_kev."
            )
        if sfn_sigma_local_kev is None or sfn_sigma_global_kev is None:
            raise ValueError(
                "sfn_modulation_enabled=True requires both sfn_sigma_local_kev "
                "and sfn_sigma_global_kev."
            )
        sfn_sigma_max_norm_v = sfn_sigma_max_kev / width
        sfn_sigma_min_norm_v = sfn_sigma_min_kev / width
        sfn_sigma_local_norm_v = sfn_sigma_local_kev / width
        sfn_sigma_global_norm_v = sfn_sigma_global_kev / width

    pool_sfn_sigma_max_norm_v: float | None = None
    pool_sfn_sigma_min_norm_v: float | None = None
    pool_sfn_sigma_local_norm_v: float | None = None
    pool_sfn_sigma_global_norm_v: float | None = None
    pool_energies_norm_t: torch.Tensor | None = None
    if pool_density_sfn_enabled:
        if width is None:
            raise ValueError(
                "pool_density_sfn_enabled=True requires energy_range_kev "
                "so σ_kev can be converted to normalised units."
            )
        if pool_density_sfn_sigma_max_kev is None or pool_density_sfn_sigma_min_kev is None:
            raise ValueError(
                "pool_density_sfn_enabled=True requires sigma_max_kev and sigma_min_kev."
            )
        if (
            pool_density_sfn_sigma_local_kev is None
            or pool_density_sfn_sigma_global_kev is None
        ):
            raise ValueError(
                "pool_density_sfn_enabled=True requires both "
                "pool_density_sfn_sigma_local_kev and pool_density_sfn_sigma_global_kev."
            )
        if pool_energies_kev is None:
            raise ValueError(
                "pool_density_sfn_enabled=True requires pool_energies_kev — "
                "the kept-event energies of the training pool, in keV. The "
                "factory caller (pipeline.build_local_cnp / "
                "cnp_test_inference._load_cnp) is responsible for loading "
                "them via majorana_acp.cut_acceptance.binned_sampler.load_events."
            )
        assert energy_range_kev is not None  # ruled out by `width is None` raise above
        e_lo_kev, _ = energy_range_kev
        pool_sfn_sigma_max_norm_v = pool_density_sfn_sigma_max_kev / width
        pool_sfn_sigma_min_norm_v = pool_density_sfn_sigma_min_kev / width
        pool_sfn_sigma_local_norm_v = pool_density_sfn_sigma_local_kev / width
        pool_sfn_sigma_global_norm_v = pool_density_sfn_sigma_global_kev / width
        # Normalise pool energies into [0, 1] using the same energy
        # range that drives all CNP coordinate normalisation. Keep on
        # CPU as float32 — the aggregator will move the buffer to the
        # model's device when the parent module is moved.
        pool_arr = pool_energies_kev
        if isinstance(pool_arr, torch.Tensor):
            pool_tensor = pool_arr.detach().to(torch.float32).reshape(-1)
        else:
            pool_tensor = torch.as_tensor(pool_arr, dtype=torch.float32).reshape(-1)
        pool_energies_norm_t = (pool_tensor - float(e_lo_kev)) / width

    bound_sigma_max_norm: float | None = None
    bound_sigma_min_norm: float | None = None
    bound_sigma_local_norm: float | None = None
    bound_sigma_global_norm: float | None = None
    if bounded_bandwidth_enabled:
        if width is None:
            raise ValueError(
                "bounded_bandwidth_enabled=True requires energy_range_kev "
                "so σ_kev can be converted to normalised units."
            )
        if (
            bounded_sigma_max_kev is None
            or bounded_sigma_min_kev is None
            or bounded_alpha_max is None
        ):
            raise ValueError(
                "bounded_bandwidth_enabled=True requires sigma_max_kev, "
                "sigma_min_kev, and alpha_max."
            )
        if bounded_sigma_local_kev is None or bounded_sigma_global_kev is None:
            raise ValueError(
                "bounded_bandwidth_enabled=True requires both "
                "bounded_sigma_local_kev and bounded_sigma_global_kev."
            )
        bound_sigma_max_norm = bounded_sigma_max_kev / width
        bound_sigma_min_norm = bounded_sigma_min_kev / width
        bound_sigma_local_norm = bounded_sigma_local_kev / width
        bound_sigma_global_norm = bounded_sigma_global_kev / width

    attention = CrossAttentionAggregator(
        latent_dim=Z,
        agg_dim=agg_dim,
        num_heads=num_heads,
        attention_dim=attention_dim,
        dropout=encoder_config.dropout,
        gaussian_attention_bias=gaussian_attention_bias,
        density_modulation_enabled=density_modulation_enabled,
        density_sigma_local_norm=sigma_local_norm,
        density_sigma_global_norm=sigma_global_norm,
        density_epsilon=density_epsilon,
        bounded_bandwidth_enabled=bounded_bandwidth_enabled,
        bounded_sigma_max_norm=bound_sigma_max_norm,
        bounded_sigma_min_norm=bound_sigma_min_norm,
        bounded_alpha_max=bounded_alpha_max,
        bounded_sensitivity_hidden_dim=bounded_sensitivity_hidden_dim,
        bounded_sigma_local_norm=bound_sigma_local_norm,
        bounded_sigma_global_norm=bound_sigma_global_norm,
        bounded_epsilon=bounded_epsilon,
        sfn_modulation_enabled=sfn_modulation_enabled,
        sfn_sigma_max_norm=sfn_sigma_max_norm_v,
        sfn_sigma_min_norm=sfn_sigma_min_norm_v,
        sfn_hidden_dim=sfn_hidden_dim,
        sfn_sigma_local_norm=sfn_sigma_local_norm_v,
        sfn_sigma_global_norm=sfn_sigma_global_norm_v,
        sfn_epsilon=sfn_epsilon,
        pool_density_sfn_enabled=pool_density_sfn_enabled,
        pool_density_sfn_sigma_max_norm=pool_sfn_sigma_max_norm_v,
        pool_density_sfn_sigma_min_norm=pool_sfn_sigma_min_norm_v,
        pool_density_sfn_hidden_dim=pool_density_sfn_hidden_dim,
        pool_density_sfn_sigma_local_norm=pool_sfn_sigma_local_norm_v,
        pool_density_sfn_sigma_global_norm=pool_sfn_sigma_global_norm_v,
        pool_density_sfn_epsilon=pool_density_sfn_epsilon,
        pool_energies_norm=pool_energies_norm_t,
        pool_density_sfn_temperature_gating=pool_density_sfn_temperature_gating,
        pool_density_sfn_tau_min=pool_density_sfn_tau_min,
        pool_density_sfn_tau_max=pool_density_sfn_tau_max,
    )
    # Decoder input dim is ``agg_dim + decoder_latent_dim`` where
    # decoder_latent_dim is the size of the second concat input. With
    # gating on, that input is raw (E_norm, T_norm) → 2; off, it's
    # z_phi_T → Z. Both code paths reuse the same upstream CnpDecoder
    # MLP architecture; only the first-layer width adapts.
    decoder_latent_dim = 2 if decoder_coordinate_gating else Z
    decoder = CnpDecoder(
        latent_dim=decoder_latent_dim,
        agg_dim=agg_dim,
        hidden_dims=dec_hidden,
        dropout=encoder_config.dropout,
    )
    return AttentiveCNP(
        encoder=encoder,
        context_encoder=context_encoder,
        attention=attention,
        decoder=decoder,
        decoder_coordinate_gating=decoder_coordinate_gating,
    )
