# Copyright (c) 2026 Darrell Thomas. MIT License. See LICENSE file.
# Profiling script for v2 kernel — runs a single config for ncu analysis.

import torch
import sys
sys.path.insert(0, "python")
from blackwell_kernels import flash_attn_v2_sm120

B, H, N, D = 2, 8, 2048, 64
causal = True

Q = torch.randn(B, H, N, D, dtype=torch.bfloat16, device="cuda")
K = torch.randn(B, H, N, D, dtype=torch.bfloat16, device="cuda")
V = torch.randn(B, H, N, D, dtype=torch.bfloat16, device="cuda")

# Warmup
for _ in range(3):
    O = flash_attn_v2_sm120(Q, K, V, causal=causal)

torch.cuda.synchronize()

# Profiled run
O = flash_attn_v2_sm120(Q, K, V, causal=causal)
torch.cuda.synchronize()
