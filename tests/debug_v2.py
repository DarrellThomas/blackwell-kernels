# Copyright (c) 2026 Darrell Thomas. MIT License. See LICENSE file.
"""Debug v2 kernel with tiny inputs to trace layout issues."""

import torch
import torch.nn.functional as F

torch.manual_seed(42)
device = "cuda:0"

from blackwell_kernels import flash_attn_sm120, flash_attn_v2_sm120


def reference_attention(Q, K, V, scale, causal=False):
    attn = torch.matmul(Q, K.transpose(-2, -1)) * scale
    if causal:
        N = Q.shape[-2]
        mask = torch.triu(torch.ones(N, N, device=Q.device, dtype=torch.bool), diagonal=1)
        attn.masked_fill_(mask, float("-inf"))
    attn = F.softmax(attn, dim=-1)
    return torch.matmul(attn, V)


# Test 1: Single MMA tile (N=64 = BLOCK_Q, D=64)
print("=" * 60)
print("Test 1: N=64 (single Q block), D=64, non-causal")
B, H, N, D = 1, 1, 64, 64
Q = torch.randn(B, H, N, D, device=device, dtype=torch.bfloat16)
K = torch.randn(B, H, N, D, device=device, dtype=torch.bfloat16)
V = torch.randn(B, H, N, D, device=device, dtype=torch.bfloat16)
scale = D**-0.5

ref = reference_attention(Q.float(), K.float(), V.float(), scale).bfloat16()
v1_out = flash_attn_sm120(Q, K, V, causal=False, scale=scale)
v2_out = flash_attn_v2_sm120(Q, K, V, causal=False, scale=scale)

# Check v1 vs ref
v1_diff = (v1_out.float() - ref.float()).abs()
print(f"  v1 vs ref: max_err={v1_diff.max():.6f}, mean_err={v1_diff.mean():.6f}")

# Check v2 vs ref
v2_diff = (v2_out.float() - ref.float()).abs()
v2_mismatch = (v2_diff > 0.01).sum().item()
v2_total = v2_diff.numel()
print(f"  v2 vs ref: max_err={v2_diff.max():.6f}, mean_err={v2_diff.mean():.6f}, "
      f"mismatch={v2_mismatch}/{v2_total} ({100*v2_mismatch/v2_total:.1f}%)")

# Check v2 vs v1
v2v1_diff = (v2_out.float() - v1_out.float()).abs()
print(f"  v2 vs v1:  max_err={v2v1_diff.max():.6f}, mean_err={v2v1_diff.mean():.6f}")

# Print first row of output for inspection
print(f"  ref[0,0,0,:8]  = {ref[0,0,0,:8].float().tolist()}")
print(f"  v1[0,0,0,:8]   = {v1_out[0,0,0,:8].float().tolist()}")
print(f"  v2[0,0,0,:8]   = {v2_out[0,0,0,:8].float().tolist()}")
print(f"  ref[0,0,0,8:16] = {ref[0,0,0,8:16].float().tolist()}")
print(f"  v2[0,0,0,8:16]  = {v2_out[0,0,0,8:16].float().tolist()}")

# Check if v2 output has any pattern (e.g., zeros, constant)
print(f"  v2 min={v2_out.min():.4f} max={v2_out.max():.4f} mean={v2_out.float().mean():.4f}")
print(f"  ref min={ref.min():.4f} max={ref.max():.4f} mean={ref.float().mean():.4f}")


# Test 2: Identity-like Q and K to verify S = Q*K^T
print("\n" + "=" * 60)
print("Test 2: Identity-like inputs to check Q*K^T")
B, H, N, D = 1, 1, 64, 64

# Q = identity-like (first row = [1,0,0,...], second = [0,1,0,...], etc.)
Q = torch.zeros(B, H, N, D, device=device, dtype=torch.bfloat16)
for i in range(min(N, D)):
    Q[0, 0, i, i] = 1.0

# K = small random
K = torch.randn(B, H, N, D, device=device, dtype=torch.bfloat16) * 0.1
V = torch.ones(B, H, N, D, device=device, dtype=torch.bfloat16)
scale = 1.0

ref = reference_attention(Q.float(), K.float(), V.float(), scale).bfloat16()
v2_out = flash_attn_v2_sm120(Q, K, V, causal=False, scale=scale)

v2_diff = (v2_out.float() - ref.float()).abs()
print(f"  v2 vs ref: max_err={v2_diff.max():.6f}, mean_err={v2_diff.mean():.6f}")
print(f"  ref[0,0,0,:8] = {ref[0,0,0,:8].float().tolist()}")
print(f"  v2[0,0,0,:8]  = {v2_out[0,0,0,:8].float().tolist()}")


# Test 3: Very small N to check boundary behavior
print("\n" + "=" * 60)
print("Test 3: N=128 (2 Q blocks), D=64")
B, H, N, D = 1, 1, 128, 64
Q = torch.randn(B, H, N, D, device=device, dtype=torch.bfloat16)
K = torch.randn(B, H, N, D, device=device, dtype=torch.bfloat16)
V = torch.randn(B, H, N, D, device=device, dtype=torch.bfloat16)
scale = D**-0.5

ref = reference_attention(Q.float(), K.float(), V.float(), scale).bfloat16()
v2_out = flash_attn_v2_sm120(Q, K, V, causal=False, scale=scale)

v2_diff = (v2_out.float() - ref.float()).abs()
v2_mismatch = (v2_diff > 0.01).sum().item()
v2_total = v2_diff.numel()
print(f"  v2 vs ref: max_err={v2_diff.max():.6f}, mean_err={v2_diff.mean():.6f}, "
      f"mismatch={v2_mismatch}/{v2_total} ({100*v2_mismatch/v2_total:.1f}%)")

# Check per-row error to see if certain rows are worse
per_row_err = v2_diff[0, 0].max(dim=-1).values
worst_rows = per_row_err.topk(5)
print(f"  Worst 5 rows: {worst_rows.indices.tolist()} with max_err {worst_rows.values.tolist()}")
best_rows = per_row_err.topk(5, largest=False)
print(f"  Best 5 rows:  {best_rows.indices.tolist()} with max_err {best_rows.values.tolist()}")
