# Copyright (c) 2026 Darrell Thomas. MIT License. See LICENSE file.

"""Benchmark BF16 GEMM kernel vs cuBLAS (torch.mm)."""

import torch
import time


def benchmark_fn(fn, warmup=10, iters=100):
    """Time a function with CUDA synchronization."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return elapsed / iters * 1000  # ms


def main():
    torch.manual_seed(42)
    device = "cuda:0"

    configs = [
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
        (2048, 512, 2048),
        (4096, 1024, 4096),
        (8192, 4096, 8192),
    ]

    # Primary config for eval.sh metric extraction
    PRIMARY_M, PRIMARY_K, PRIMARY_N = 4096, 4096, 4096

    print(f"{'M':>6} {'K':>6} {'N':>6} | {'cuBLAS (ms)':>12} {'custom (ms)':>12} {'speedup':>8}")
    print("-" * 60)

    for M, K, N in configs:
        A = torch.randn(M, K, device=device, dtype=torch.bfloat16)
        B = torch.randn(K, N, device=device, dtype=torch.bfloat16)

        # cuBLAS baseline (torch.mm dispatches to cuBLAS for BF16)
        cublas_time = benchmark_fn(lambda: torch.mm(A, B))

        # Custom kernel
        try:
            from blackwell_kernels import bf16_gemm
            custom_time = benchmark_fn(lambda: bf16_gemm(A, B))
            speedup = cublas_time / custom_time
            print(f"{M:6d} {K:6d} {N:6d} | {cublas_time:12.3f} {custom_time:12.3f} {speedup:7.2f}x")

            # Emit primary metric for eval.sh
            if M == PRIMARY_M and K == PRIMARY_K and N == PRIMARY_N:
                print(f"primary_custom_ms: {custom_time:.3f}")
                print(f"primary_vs_ref: {speedup:.2f}x")
        except Exception as e:
            print(f"{M:6d} {K:6d} {N:6d} | {cublas_time:12.3f} {'N/A':>12} {'N/A':>8}  ({e})")


if __name__ == "__main__":
    main()
