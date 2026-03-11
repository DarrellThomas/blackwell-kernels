# Copyright (c) 2026 Darrell Thomas. MIT License. See LICENSE file.

"""Test flash attention kernel against PyTorch reference."""

import torch
import torch.nn.functional as F


def reference_attention(Q, K, V, scale, causal=False):
    """PyTorch reference implementation for correctness validation."""
    attn = torch.matmul(Q, K.transpose(-2, -1)) * scale
    if causal:
        N = Q.shape[-2]
        mask = torch.triu(torch.ones(N, N, device=Q.device, dtype=torch.bool), diagonal=1)
        attn.masked_fill_(mask, float("-inf"))
    attn = F.softmax(attn, dim=-1)
    return torch.matmul(attn, V)


def test_flash_attn_correctness():
    """Compare custom kernel output against PyTorch reference."""
    torch.manual_seed(42)
    B, H, N, D = 2, 8, 128, 64
    device = "cuda:0"

    Q = torch.randn(B, H, N, D, device=device, dtype=torch.bfloat16)
    K = torch.randn(B, H, N, D, device=device, dtype=torch.bfloat16)
    V = torch.randn(B, H, N, D, device=device, dtype=torch.bfloat16)
    scale = D**-0.5

    ref = reference_attention(Q.float(), K.float(), V.float(), scale).bfloat16()

    from blackwell_kernels import flash_attn_sm120

    out = flash_attn_sm120(Q, K, V, causal=False, scale=scale)

    # BF16 tolerance: rtol=1e-2, atol=1e-2 (as per plan: rtol=1e-3 is the target)
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
    print("PASS: flash_attn non-causal")


def test_flash_attn_causal():
    """Test causal masking."""
    torch.manual_seed(42)
    B, H, N, D = 2, 8, 128, 64
    device = "cuda:0"

    Q = torch.randn(B, H, N, D, device=device, dtype=torch.bfloat16)
    K = torch.randn(B, H, N, D, device=device, dtype=torch.bfloat16)
    V = torch.randn(B, H, N, D, device=device, dtype=torch.bfloat16)
    scale = D**-0.5

    ref = reference_attention(Q.float(), K.float(), V.float(), scale, causal=True).bfloat16()

    from blackwell_kernels import flash_attn_sm120

    out = flash_attn_sm120(Q, K, V, causal=True, scale=scale)

    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
    print("PASS: flash_attn causal")


def test_flash_attn_v2_correctness():
    """Compare v2 (MMA) kernel output against PyTorch reference."""
    torch.manual_seed(42)
    B, H, N, D = 2, 8, 128, 64
    device = "cuda:0"

    Q = torch.randn(B, H, N, D, device=device, dtype=torch.bfloat16)
    K = torch.randn(B, H, N, D, device=device, dtype=torch.bfloat16)
    V = torch.randn(B, H, N, D, device=device, dtype=torch.bfloat16)
    scale = D**-0.5

    ref = reference_attention(Q.float(), K.float(), V.float(), scale).bfloat16()

    from blackwell_kernels import flash_attn_v2_sm120

    out = flash_attn_v2_sm120(Q, K, V, causal=False, scale=scale)

    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
    print("PASS: flash_attn_v2 non-causal")


def test_flash_attn_v2_causal():
    """Test v2 (MMA) kernel with causal masking."""
    torch.manual_seed(42)
    B, H, N, D = 2, 8, 128, 64
    device = "cuda:0"

    Q = torch.randn(B, H, N, D, device=device, dtype=torch.bfloat16)
    K = torch.randn(B, H, N, D, device=device, dtype=torch.bfloat16)
    V = torch.randn(B, H, N, D, device=device, dtype=torch.bfloat16)
    scale = D**-0.5

    ref = reference_attention(Q.float(), K.float(), V.float(), scale, causal=True).bfloat16()

    from blackwell_kernels import flash_attn_v2_sm120

    out = flash_attn_v2_sm120(Q, K, V, causal=True, scale=scale)

    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
    print("PASS: flash_attn_v2 causal")


def test_flash_attn_v2_d128():
    """Test v2 kernel with HEAD_DIM=128."""
    torch.manual_seed(42)
    B, H, N, D = 2, 8, 128, 128
    device = "cuda:0"

    Q = torch.randn(B, H, N, D, device=device, dtype=torch.bfloat16)
    K = torch.randn(B, H, N, D, device=device, dtype=torch.bfloat16)
    V = torch.randn(B, H, N, D, device=device, dtype=torch.bfloat16)
    scale = D**-0.5

    ref = reference_attention(Q.float(), K.float(), V.float(), scale).bfloat16()

    from blackwell_kernels import flash_attn_v2_sm120

    out = flash_attn_v2_sm120(Q, K, V, causal=False, scale=scale)

    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
    print("PASS: flash_attn_v2 D=128 non-causal")


def test_flash_attn_v2_long_seq():
    """Test v2 kernel with longer sequences (multiple Q blocks)."""
    torch.manual_seed(42)
    B, H, N, D = 1, 4, 2048, 64
    device = "cuda:0"

    Q = torch.randn(B, H, N, D, device=device, dtype=torch.bfloat16)
    K = torch.randn(B, H, N, D, device=device, dtype=torch.bfloat16)
    V = torch.randn(B, H, N, D, device=device, dtype=torch.bfloat16)
    scale = D**-0.5

    ref = reference_attention(Q.float(), K.float(), V.float(), scale, causal=True).bfloat16()

    from blackwell_kernels import flash_attn_v2_sm120

    out = flash_attn_v2_sm120(Q, K, V, causal=True, scale=scale)

    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
    print("PASS: flash_attn_v2 N=2048 causal")


if __name__ == "__main__":
    # v1 tests
    test_flash_attn_correctness()
    test_flash_attn_causal()

    # v2 tests
    test_flash_attn_v2_correctness()
    test_flash_attn_v2_causal()
    test_flash_attn_v2_d128()
    test_flash_attn_v2_long_seq()
