# Copyright (c) 2026 Darrell Thomas. MIT License. See LICENSE file.
# Profiling script for GEMM kernel — runs a single config for ncu analysis.

import torch
import sys
sys.path.insert(0, "python")
from blackwell_kernels import bf16_gemm

M, K, N = 4096, 4096, 4096

A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
B = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")

# Warmup
for _ in range(3):
    C = bf16_gemm(A, B)

torch.cuda.synchronize()

# Profiled run
C = bf16_gemm(A, B)
torch.cuda.synchronize()
