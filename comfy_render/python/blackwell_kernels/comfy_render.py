# Copyright (c) 2026 Darrell Thomas. MIT License. See LICENSE file.

"""ComfyUI render kernels for sm_120a: Flash Attention + Fused GroupNorm+Linear."""

import torch
from blackwell_kernels._C import flash_attn_forward, fused_group_norm_linear_forward


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
) -> torch.Tensor:
    """Scaled dot-product attention (Flash Attention, sm_120a).

    Args:
        q: [B, H, N, D] BF16 queries
        k: [B, H, N, D] BF16 keys
        v: [B, H, N, D] BF16 values
        causal: if True, apply causal mask

    Returns:
        [B, H, N, D] BF16 output
    """
    assert q.dtype == torch.bfloat16, f"Expected BF16, got {q.dtype}"
    assert q.dim() == 4, f"Expected 4D tensor, got {q.dim()}D"
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    return flash_attn_forward(q, k, v, causal)


def fused_group_norm_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    linear_bias: torch.Tensor,
    groups: int = 32,
) -> torch.Tensor:
    """Fused GroupNorm + Linear projection (sm_120a).

    Dispatches between a custom fused GEMM (for bandwidth-bound cases)
    and a cuBLAS path (for compute-bound cases).

    Args:
        x: [*, C_in] BF16 input (any number of leading dimensions)
        weight: [C_out, C_in] BF16 linear weight
        norm_weight: [C_in] GroupNorm scale (gamma), any float dtype
        norm_bias: [C_in] GroupNorm bias (beta), any float dtype
        linear_bias: [C_out] linear bias, any float dtype
        groups: number of groups for GroupNorm (default 32)

    Returns:
        [*, C_out] BF16 output
    """
    assert x.dtype == torch.bfloat16, f"Expected BF16 input, got {x.dtype}"
    assert weight.dtype == torch.bfloat16, f"Expected BF16 weight, got {weight.dtype}"

    orig_shape = x.shape
    C_in = x.size(-1)
    C_out = weight.size(0)
    x_2d = x.reshape(-1, C_in).contiguous()

    y = fused_group_norm_linear_forward(
        x_2d, weight.contiguous(),
        norm_weight.contiguous(), norm_bias.contiguous(),
        linear_bias.contiguous(), groups)
    return y.reshape(*orig_shape[:-1], C_out)
