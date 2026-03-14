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

    print(f"{'M':>6} {'K':>6} {'N':>6} | {'cuBLAS (ms)':>12} {'BF16 (ms)':>12} {'FP8 (ms)':>12} {'BF16/cuBLAS':>12} {'FP8/cuBLAS':>12}")
    print("-" * 82)

    for M, K, N in configs:
        A = torch.randn(M, K, device=device, dtype=torch.bfloat16)
        B = torch.randn(K, N, device=device, dtype=torch.bfloat16)

        # cuBLAS baseline (torch.mm dispatches to cuBLAS for BF16)
        cublas_time = benchmark_fn(lambda: torch.mm(A, B))

        # BF16 custom kernel
        bf16_str = "N/A"
        bf16_ratio_str = "N/A"
        try:
            from blackwell_kernels import bf16_gemm
            bf16_time = benchmark_fn(lambda: bf16_gemm(A, B))
            bf16_ratio = cublas_time / bf16_time
            bf16_str = f"{bf16_time:.3f}"
            bf16_ratio_str = f"{bf16_ratio:.2f}x"
        except Exception:
            bf16_time = None

        # FP8 custom kernel
        fp8_str = "N/A"
        fp8_ratio_str = "N/A"
        try:
            from blackwell_kernels import fp8_gemm
            fp8_time = benchmark_fn(lambda: fp8_gemm(A, B))
            fp8_ratio = cublas_time / fp8_time
            fp8_str = f"{fp8_time:.3f}"
            fp8_ratio_str = f"{fp8_ratio:.2f}x"
        except Exception:
            fp8_time = None

        print(f"{M:6d} {K:6d} {N:6d} | {cublas_time:12.3f} {bf16_str:>12} {fp8_str:>12} {bf16_ratio_str:>12} {fp8_ratio_str:>12}")

        # Emit primary metric for eval.sh
        if M == PRIMARY_M and K == PRIMARY_K and N == PRIMARY_N:
            if bf16_time is not None:
                print(f"primary_custom_ms: {bf16_time:.3f}")
                print(f"primary_vs_ref: {cublas_time / bf16_time:.2f}x")
            if fp8_time is not None:
                print(f"primary_fp8_ms: {fp8_time:.3f}")
                print(f"primary_fp8_vs_ref: {cublas_time / fp8_time:.2f}x")


if __name__ == "__main__":
    main()
