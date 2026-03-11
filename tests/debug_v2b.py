# Copyright (c) 2026 Darrell Thomas. MIT License. See LICENSE file.
"""More targeted debug: test with structured inputs to isolate layout bugs."""

import torch
import torch.nn.functional as F

torch.manual_seed(42)
device = "cuda:0"

from blackwell_kernels import flash_attn_v2_sm120


def reference_attention(Q, K, V, scale, causal=False):
    attn = torch.matmul(Q, K.transpose(-2, -1)) * scale
    if causal:
        N = Q.shape[-2]
        mask = torch.triu(torch.ones(N, N, device=Q.device, dtype=torch.bool), diagonal=1)
        attn.masked_fill_(mask, float("-inf"))
    attn = F.softmax(attn, dim=-1)
    return torch.matmul(attn, V)


# Test A: All K rows identical → softmax is uniform → O = mean(V)
print("=" * 60)
print("Test A: Uniform K → O should equal mean(V)")
B, H, N, D = 1, 1, 64, 64
Q = torch.randn(B, H, N, D, device=device, dtype=torch.bfloat16)
# All K rows = same random vector
k_row = torch.randn(1, 1, 1, D, device=device, dtype=torch.bfloat16)
K = k_row.expand(B, H, N, D).contiguous()
V = torch.randn(B, H, N, D, device=device, dtype=torch.bfloat16)
scale = D**-0.5

ref = reference_attention(Q.float(), K.float(), V.float(), scale).bfloat16()
v2_out = flash_attn_v2_sm120(Q, K, V, causal=False, scale=scale)
v_mean = V.float().mean(dim=2, keepdim=True).bfloat16()

v2_diff = (v2_out.float() - ref.float()).abs()
print(f"  v2 vs ref:    max_err={v2_diff.max():.6f}")
print(f"  ref vs V_mean: max_err={(ref.float() - v_mean.float()).abs().max():.6f}")
print(f"  v2 vs V_mean:  max_err={(v2_out.float() - v_mean.float()).abs().max():.6f}")
print(f"  ref[0,0,0,:8] = {ref[0,0,0,:8].float().tolist()}")
print(f"  v2[0,0,0,:8]  = {v2_out[0,0,0,:8].float().tolist()}")
print(f"  V_mean[:8]    = {v_mean[0,0,0,:8].float().tolist()}")


# Test B: Q = single row repeated → all output rows should be identical
print("\n" + "=" * 60)
print("Test B: Q = single row repeated → all output rows equal")
B, H, N, D = 1, 1, 64, 64
q_row = torch.randn(1, 1, 1, D, device=device, dtype=torch.bfloat16)
Q = q_row.expand(B, H, N, D).contiguous()
K = torch.randn(B, H, N, D, device=device, dtype=torch.bfloat16)
V = torch.randn(B, H, N, D, device=device, dtype=torch.bfloat16)
scale = D**-0.5

ref = reference_attention(Q.float(), K.float(), V.float(), scale).bfloat16()
v2_out = flash_attn_v2_sm120(Q, K, V, causal=False, scale=scale)

v2_diff = (v2_out.float() - ref.float()).abs()
print(f"  v2 vs ref: max_err={v2_diff.max():.6f}")

# Check if all output rows are the same (they should be for repeated Q)
v2_row0 = v2_out[0, 0, 0:1, :].float()
v2_rows = v2_out[0, 0, :, :].float()
row_var = (v2_rows - v2_row0).abs().max()
ref_row0 = ref[0, 0, 0:1, :].float()
ref_rows = ref[0, 0, :, :].float()
ref_row_var = (ref_rows - ref_row0).abs().max()
print(f"  ref row variation: {ref_row_var:.6f} (should be ~0)")
print(f"  v2 row variation:  {row_var:.6f} (should be ~0)")

# Print per-warp first row to see if warps differ
for w in range(4):
    r = w * 16
    print(f"  v2 row {r:2d} (warp {w}): {v2_out[0,0,r,:8].float().tolist()}")
print(f"  ref row  0:          {ref[0,0,0,:8].float().tolist()}")


# Test C: Check if the issue is with Q*K^T specifically
# Use V=I and check that O = softmax(Q*K^T * scale)
print("\n" + "=" * 60)
print("Test C: V = identity → O = rows of softmax(S)")
B, H, N, D = 1, 1, 64, 64
Q = torch.randn(B, H, N, D, device=device, dtype=torch.bfloat16)
K = torch.randn(B, H, N, D, device=device, dtype=torch.bfloat16)
V = torch.eye(N, D, device=device, dtype=torch.bfloat16).unsqueeze(0).unsqueeze(0)
scale = D**-0.5

ref = reference_attention(Q.float(), K.float(), V.float(), scale).bfloat16()
v2_out = flash_attn_v2_sm120(Q, K, V, causal=False, scale=scale)

v2_diff = (v2_out.float() - ref.float()).abs()
print(f"  v2 vs ref: max_err={v2_diff.max():.6f}, mean_err={v2_diff.mean():.6f}")
print(f"  ref[0,0,0,:8] = {ref[0,0,0,:8].float().tolist()}")
print(f"  v2[0,0,0,:8]  = {v2_out[0,0,0,:8].float().tolist()}")
# Row sums should equal 1 (softmax)
ref_rowsum = ref[0,0,:,:].float().sum(dim=-1)
v2_rowsum = v2_out[0,0,:,:].float().sum(dim=-1)
print(f"  ref row sums (first 8): {ref_rowsum[:8].tolist()}")
print(f"  v2 row sums (first 8):  {v2_rowsum[:8].tolist()}")


# Test D: Scale = 0 → softmax is uniform → O = mean(V) for all rows
print("\n" + "=" * 60)
print("Test D: scale=0 → uniform attention → O = mean(V)")
B, H, N, D = 1, 1, 64, 64
Q = torch.randn(B, H, N, D, device=device, dtype=torch.bfloat16)
K = torch.randn(B, H, N, D, device=device, dtype=torch.bfloat16)
V = torch.randn(B, H, N, D, device=device, dtype=torch.bfloat16)
scale = 0.0  # All scores are 0 → softmax is uniform

ref = reference_attention(Q.float(), K.float(), V.float(), scale).bfloat16()
v2_out = flash_attn_v2_sm120(Q, K, V, causal=False, scale=scale)

v2_diff = (v2_out.float() - ref.float()).abs()
v_mean = V.float().mean(dim=2, keepdim=True).bfloat16()
print(f"  v2 vs ref:    max_err={v2_diff.max():.6f}")
print(f"  v2 vs V_mean: max_err={(v2_out.float() - v_mean.float()).abs().max():.6f}")
print(f"  ref[0,0,0,:8] = {ref[0,0,0,:8].float().tolist()}")
print(f"  v2[0,0,0,:8]  = {v2_out[0,0,0,:8].float().tolist()}")
print(f"  V_mean[:8]    = {v_mean[0,0,0,:8].float().tolist()}")
