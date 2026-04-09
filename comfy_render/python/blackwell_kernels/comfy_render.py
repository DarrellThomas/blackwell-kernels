# Copyright (c) 2026 Darrell Thomas. MIT License. See LICENSE file.

"""Flash Attention wrapper for sm_120a."""

import torch
from blackwell_kernels._C import flash_attn_forward


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
