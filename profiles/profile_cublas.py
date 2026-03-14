# Copyright (c) 2026 Darrell Thomas. MIT License. See LICENSE file.
# Profiling script for cuBLAS GEMM (torch.matmul) — runs BF16 4096x4096 for ncu analysis.

import torch

M, K, N = 4096, 4096, 4096

A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
B = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")

# Warmup
for _ in range(5):
    C = torch.matmul(A, B)

torch.cuda.synchronize()

# Profiled run
C = torch.matmul(A, B)
torch.cuda.synchronize()

print(f"A shape: {A.shape}, B shape: {B.shape}, C shape: {C.shape}")
print(f"C dtype: {C.dtype}")
