# Copyright (c) 2026 Darrell Thomas. MIT License. See LICENSE file.

"""Python bindings for Multi-Latent Attention (MLA) on sm_120."""

import torch


def mla_attn_forward(
    Q_nope: torch.Tensor,   # [B, H, T, D_NOPE] or [B, T, H, D_NOPE] BF16
    Q_rope: torch.Tensor,   # [B, H, T, D_ROPE] or [B, T, H, D_ROPE] BF16
    K_nope: torch.Tensor,   # [B, H, S, D_NOPE] or [B, S, H, D_NOPE] BF16
    K_rope: torch.Tensor,   # [B, H, S, D_ROPE] or [B, S, H, D_ROPE] BF16
    V: torch.Tensor,        # [B, H, S, D_V] or [B, S, H, D_V] BF16
    causal: bool = True,
    scale: float | None = None,
    layout: str = "bhtd",   # "bhtd" or "bthd"
) -> tuple[torch.Tensor, torch.Tensor]:
    """MLA forward pass optimized for RTX 5090 (sm_120).

    Args:
        layout: "bhtd" for [B,H,T,D] (default), "bthd" for [B,T,H,D]

    Returns:
        O: [B, H, T, D_V]  BF16 (always BHTD)
        L: [B, H, T]       FP32  (logsumexp for backward)
    """
    from blackwell_kernels._C import mla_attn_forward as _mla_fwd

    if scale is None:
        d_qk = Q_nope.shape[-1] + Q_rope.shape[-1]
        scale = d_qk ** -0.5

    bthd = layout.lower() == "bthd"
    return _mla_fwd(Q_nope, Q_rope, K_nope, K_rope, V, scale, causal, bthd)


class MLAAttentionFunc(torch.autograd.Function):
    """torch.autograd wrapper for MLA forward + backward."""

    @staticmethod
    def forward(ctx, Q_nope, Q_rope, K_nope, K_rope, V, causal, scale):
        O, L = mla_attn_forward(Q_nope, Q_rope, K_nope, K_rope, V, causal, scale)
        ctx.save_for_backward(Q_nope, Q_rope, K_nope, K_rope, V, O, L)
        ctx.causal = causal
        ctx.scale = scale
        return O

    @staticmethod
    def backward(ctx, dO):
        from blackwell_kernels._C import mla_attn_backward as _mla_bwd
        Q_nope, Q_rope, K_nope, K_rope, V, O, L = ctx.saved_tensors
        dO = dO.contiguous()
        dQ_nope, dQ_rope, dK_nope, dK_rope, dV = _mla_bwd(
            dO, Q_nope, Q_rope, K_nope, K_rope, V, O, L,
            ctx.scale, ctx.causal)
        return dQ_nope, dQ_rope, dK_nope, dK_rope, dV, None, None


def mla_attention(
    Q_nope: torch.Tensor,
    Q_rope: torch.Tensor,
    K_nope: torch.Tensor,
    K_rope: torch.Tensor,
    V: torch.Tensor,
    causal: bool = True,
    scale: float | None = None,
) -> torch.Tensor:
    """Drop-in replacement for the attention portion of MLAttention.forward()."""
    if scale is None:
        d_qk = Q_nope.shape[-1] + Q_rope.shape[-1]
        scale = d_qk ** -0.5
    return MLAAttentionFunc.apply(Q_nope, Q_rope, K_nope, K_rope, V, causal, scale)
