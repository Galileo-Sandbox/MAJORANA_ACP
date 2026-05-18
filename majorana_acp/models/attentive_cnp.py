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
    """

    def __init__(
        self,
        *,
        latent_dim: int,
        agg_dim: int,
        num_heads: int,
        attention_dim: int,
        dropout: float = 0.0,
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

    def forward(
        self,
        z_phi_target: torch.Tensor,  # [B, N_T, Z]
        z_phi_ctx: torch.Tensor,  # [B, N_C, Z]
        h_ctx: torch.Tensor,  # [B, N_C, agg]
    ) -> torch.Tensor:
        B, N_T, _ = z_phi_target.shape
        N_C = z_phi_ctx.size(1)
        H, d = self.num_heads, self.head_dim

        q = self.w_q(z_phi_target).reshape(B, N_T, H, d).transpose(1, 2)
        k = self.w_k(z_phi_ctx).reshape(B, N_C, H, d).transpose(1, 2)
        v = self.w_v(h_ctx).reshape(B, N_C, H, d).transpose(1, 2)
        # q, k, v: [B, H, *, d]

        attn_logits = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, N_T, N_C]
        attn = F.softmax(attn_logits, dim=-1)
        attn = self.attn_dropout(attn)

        out = attn @ v  # [B, H, N_T, d]
        out = out.transpose(1, 2).reshape(B, N_T, H * d)  # [B, N_T, attention_dim]
        out = self.w_o(out)  # [B, N_T, latent_dim]
        out = self.out_dropout(out)
        return out


class AttentiveCNP(nn.Module):
    """Drop-in replacement for ``ConditionalNeuralProcess`` with cross-attention.

    The forward signature ``(ctx_batch, target_batch) → CnpOutput`` is
    identical to upstream so ``core.training.train_cnp`` and
    ``core.surrogate_cnp.cnp_loss`` work without modification.

    Architecture: ``UniversalEncoder + ContextPointEncoder`` are
    reused from upstream; the mean aggregator + decoder-broadcast
    pair is replaced by ``CrossAttentionAggregator + decoder.net``.
    """

    def __init__(
        self,
        encoder: UniversalEncoder,
        context_encoder: ContextPointEncoder,
        attention: CrossAttentionAggregator,
        decoder: CnpDecoder,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.context_encoder = context_encoder
        self.attention = attention
        self.decoder = decoder

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

        # Cross-attention → target-conditioned representation.
        r_target = self.attention(z_phi_T, z_phi_C, h_C)  # [B, N_T, Z]

        # Decode via the upstream CnpDecoder MLP directly. We skip
        # ``decoder.forward`` because it does
        # ``r_trial.unsqueeze(1).expand(-1, N_t, -1)`` which assumes
        # the mean-path shape [B, agg]; we already have per-target r.
        cat = torch.cat([r_target, z_phi_T], dim=-1)  # [B, N_T, 2Z]
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
    aggregator_dim: int | None = None,
    context_hidden_dims: list[int] | None = None,
    decoder_hidden_dims: list[int] | None = None,
) -> AttentiveCNP:
    """Mirror of upstream ``core.build_cnp`` that wires the attentive variant.

    Identical signature to ``core.build_cnp`` plus two attention knobs.
    All MLPs are reused from upstream (``UniversalEncoder``,
    ``ContextPointEncoder``, ``CnpDecoder``) — only the aggregator
    differs.
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
    attention = CrossAttentionAggregator(
        latent_dim=Z,
        agg_dim=agg_dim,
        num_heads=num_heads,
        attention_dim=attention_dim,
        dropout=encoder_config.dropout,
    )
    decoder = CnpDecoder(
        latent_dim=Z,
        agg_dim=agg_dim,
        hidden_dims=dec_hidden,
        dropout=encoder_config.dropout,
    )
    return AttentiveCNP(
        encoder=encoder,
        context_encoder=context_encoder,
        attention=attention,
        decoder=decoder,
    )
