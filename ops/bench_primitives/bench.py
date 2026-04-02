#!/usr/bin/env python3
# Copyright (c) 2026 Darrell Thomas. MIT License. See LICENSE file.
"""Benchmark custom MMA primitives vs cuBLAS references."""

import torch
import time
import sys

torch.backends.cuda.matmul.allow_tf32 = False  # fair FP32 baseline

def bench(fn, warmup=10, iters=50):
    """Returns median time in microseconds."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e6)
    times.sort()
    return times[len(times) // 2]

def check_error(name, custom, ref):
    """Report relative error."""
    err = (custom - ref).abs().max().item()
    scale = ref.abs().max().item()
    rel = err / scale if scale > 0 else err
    status = "OK" if rel < 0.01 else "HIGH"
    print(f"  {name}: max_rel_err={rel:.2e} [{status}]")
    return rel

def main():
    import bench_primitives as bp

    device = 'cuda:0'  # GPU 1 via CUDA_VISIBLE_DEVICES=1
    sizes = [1024, 2048, 4096]

    print("=" * 72)
    print("SYRK FP32: custom MMA vs cuBLAS SSYRK vs torch.mm(A, A.t())")
    print("=" * 72)

    for N in sizes:
        K = N
        A = torch.randn(N, K, device=device, dtype=torch.float32)

        # Reference: torch.mm (full GEMM)
        ref_mm = torch.mm(A, A.t())

        # cuBLAS SSYRK
        ref_cublas = bp.cublas_syrk(A)

        # Custom MMA kernel
        custom = bp.syrk_f32(A)

        print(f"\n  N={N}, K={K}:")
        check_error("custom vs torch.mm", custom, ref_mm)

        t_mm = bench(lambda: torch.mm(A, A.t()))
        t_cublas = bench(lambda: bp.cublas_syrk(A))
        t_custom = bench(lambda: bp.syrk_f32(A))

        print(f"  torch.mm(A, A.t()):  {t_mm:8.1f} us")
        print(f"  cuBLAS SSYRK:        {t_cublas:8.1f} us")
        print(f"  custom MMA SYRK:     {t_custom:8.1f} us")
        print(f"  custom vs torch.mm:  {t_mm/t_custom:.2f}x")
        print(f"  custom vs cuBLAS:    {t_cublas/t_custom:.2f}x")

    print("\n" + "=" * 72)
    print("TRMM FP32: custom MMA vs cuBLAS STRMM vs torch.mm(L.tril(), B)")
    print("=" * 72)

    for N in sizes:
        L = torch.randn(N, N, device=device, dtype=torch.float32).tril()
        B = torch.randn(N, N, device=device, dtype=torch.float32)

        # Reference: torch.mm (full GEMM on already-triangular L)
        ref_mm = torch.mm(L, B)

        # cuBLAS STRMM
        ref_cublas = bp.cublas_trmm(L, B, False)

        # Custom MMA kernel
        custom = bp.trmm_f32(L, B, False)

        print(f"\n  M=N={N} (lower triangular):")
        check_error("custom vs torch.mm", custom, ref_mm)
        check_error("cuBLAS vs torch.mm", ref_cublas, ref_mm)

        t_mm = bench(lambda: torch.mm(L, B))
        t_cublas = bench(lambda: bp.cublas_trmm(L, B, False))
        t_custom = bench(lambda: bp.trmm_f32(L, B, False))

        print(f"  torch.mm(L, B):      {t_mm:8.1f} us")
        print(f"  cuBLAS STRMM:        {t_cublas:8.1f} us")
        print(f"  custom MMA TRMM:     {t_custom:8.1f} us")
        print(f"  custom vs torch.mm:  {t_mm/t_custom:.2f}x")
        print(f"  custom vs cuBLAS:    {t_cublas/t_custom:.2f}x")

    # Also test upper triangular (used by QR for R2 @ R1)
    print("\n" + "-" * 72)
    print("TRMM FP32 upper triangular (QR use case):")
    print("-" * 72)

    for N in [4096]:
        U = torch.randn(N, N, device=device, dtype=torch.float32).triu()
        B = torch.randn(N, N, device=device, dtype=torch.float32)

        ref_mm = torch.mm(U, B)
        custom = bp.trmm_f32(U, B, True)

        print(f"\n  M=N={N} (upper triangular):")
        check_error("custom vs torch.mm", custom, ref_mm)

        t_mm = bench(lambda: torch.mm(U, B))
        t_custom = bench(lambda: bp.trmm_f32(U, B, True))

        print(f"  torch.mm(U, B):      {t_mm:8.1f} us")
        print(f"  custom MMA TRMM:     {t_custom:8.1f} us")
        print(f"  custom vs torch.mm:  {t_mm/t_custom:.2f}x")

if __name__ == '__main__':
    main()
