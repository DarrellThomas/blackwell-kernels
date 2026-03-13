# Copyright (c) 2026 Darrell Thomas. MIT License. See LICENSE file.

"""Python bindings for flash attention on sm_120."""

import torch


def flash_attn_sm120(
    Q: torch.Tensor,  # [B, H, N, D] BF16
    K: torch.Tensor,  # [B, H, N, D] BF16
    V: torch.Tensor,  # [B, H, N, D] BF16
    causal: bool = False,
    scale: float | None = None,
) -> torch.Tensor:
    """Flash attention forward pass optimized for RTX 5090 (sm_120).

    Args:
        Q: Query tensor [batch, heads, seq_len, head_dim] in BF16
        K: Key tensor [batch, heads, seq_len, head_dim] in BF16
        V: Value tensor [batch, heads, seq_len, head_dim] in BF16
        causal: Whether to apply causal masking
        scale: Attention scale factor (default: 1/sqrt(head_dim))

    Returns:
        Output tensor [batch, heads, seq_len, head_dim] in BF16
    """
    from blackwell_kernels._C import flash_attn_forward

    if scale is None:
        scale = Q.shape[-1] ** -0.5

    return flash_attn_forward(Q, K, V, scale, causal)


def flash_attn_v2_sm120(
    Q: torch.Tensor,  # [B, H, N, D] BF16
    K: torch.Tensor,  # [B, H, N, D] BF16
    V: torch.Tensor,  # [B, H, N, D] BF16
    causal: bool = False,
    scale: float | None = None,
) -> torch.Tensor:
    """Flash attention v2 forward pass using MMA tensor cores on sm_120.

    Args:
        Q: Query tensor [batch, heads, seq_len, head_dim] in BF16
        K: Key tensor [batch, heads, seq_len, head_dim] in BF16
        V: Value tensor [batch, heads, seq_len, head_dim] in BF16
        causal: Whether to apply causal masking
        scale: Attention scale factor (default: 1/sqrt(head_dim))

    Returns:
        Output tensor [batch, heads, seq_len, head_dim] in BF16
    """
    from blackwell_kernels._C import flash_attn_v2_forward

    if scale is None:
        scale = Q.shape[-1] ** -0.5

    return flash_attn_v2_forward(Q, K, V, scale, causal)


def flash_attn_v3_sm120(
    Q: torch.Tensor,  # [B, H, N, D] BF16
    K: torch.Tensor,  # [B, H, N, D] BF16
    V: torch.Tensor,  # [B, H, N, D] BF16
    causal: bool = False,
    scale: float | None = None,
) -> torch.Tensor:
    """Flash attention v3 forward pass — fused exp2f+PV scheduling on sm_120."""
    from blackwell_kernels._C import flash_attn_v3_forward

    if scale is None:
        scale = Q.shape[-1] ** -0.5

    return flash_attn_v3_forward(Q, K, V, scale, causal)


def flash_attn_fp8_sm120(
    Q: torch.Tensor,  # [B, H, N, D] BF16
    K: torch.Tensor,  # [B, H, N, D] BF16
    V: torch.Tensor,  # [B, H, N, D] BF16
    causal: bool = False,
    scale: float | None = None,
) -> torch.Tensor:
    """Flash attention FP8 forward pass — e4m3 m16n8k32 on sm_120.

    Inputs are BF16 (converted to FP8 inside kernel). Softmax in FP32.
    2x tensor core throughput vs BF16 at reduced precision.
    """
    from blackwell_kernels._C import flash_attn_fp8_forward

    if scale is None:
        scale = Q.shape[-1] ** -0.5

    return flash_attn_fp8_forward(Q, K, V, scale, causal)
