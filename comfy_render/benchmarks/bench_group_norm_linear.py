# Copyright (c) 2026 Darrell Thomas. MIT License. See LICENSE file.
#
# Benchmark GroupNorm + Linear: custom kernel vs PyTorch reference.
#
# Usage: CUDA_VISIBLE_DEVICES=1 PYTHONPATH=python python3 benchmarks/bench_group_norm_linear.py

import sys
import time
import torch
import torch.nn.functional as F

sys.path.insert(0, "python")
from blackwell_kernels._C import group_norm_forward, fused_group_norm_linear_forward

device = "cuda"
dtype = torch.bfloat16
WARMUP = 20
TIMED = 200


def bench(fn, warmup=WARMUP, timed=TIMED):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(timed):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / timed * 1000


def bench_gn(M, C, groups=32):
    """Benchmark standalone GroupNorm: custom vs PyTorch."""
    x = torch.randn(M, C, device=device, dtype=dtype)
    gamma = torch.randn(C, device=device, dtype=dtype)
    beta = torch.randn(C, device=device, dtype=dtype)

    custom_ms = bench(lambda: group_norm_forward(x, gamma, beta, groups))
    ref_ms = bench(lambda: F.group_norm(x, groups, gamma, beta))
    return custom_ms, ref_ms


def bench_gnl(M, C_in, C_out, groups=32):
    """Benchmark fused GroupNorm + Linear: custom vs PyTorch."""
    x = torch.randn(M, C_in, device=device, dtype=dtype)
    w = torch.randn(C_out, C_in, device=device, dtype=dtype)
    gamma = torch.randn(C_in, device=device, dtype=dtype)
    beta = torch.randn(C_in, device=device, dtype=dtype)
    bias = torch.randn(C_out, device=device, dtype=dtype)

    custom_ms = bench(lambda: fused_group_norm_linear_forward(
        x, w, gamma, beta, bias, groups))

    # Reference: PyTorch GroupNorm + Linear
    def ref_fn():
        gn = F.group_norm(x, groups, gamma, beta)
        return F.linear(gn, w, bias)
    ref_ms = bench(ref_fn)

    return custom_ms, ref_ms


def main():
    print(f"Benchmarking GroupNorm+Linear on {torch.cuda.get_device_name()}\n")

    # Primary benchmark: SD1.5 M=4096 C=320
    c_ms, r_ms = bench_gnl(4096, 320, 320)
    print(f"primary_custom_ms: {c_ms:.3f}")
    print(f"primary_vs_ref: {r_ms/c_ms:.2f}x")

    # GroupNorm-only benchmarks
    print(f"\n{'Config':<35} {'Custom (ms)':<12} {'Ref (ms)':<12} {'Speedup'}")
    print(f"{'='*70}")
    print("--- GroupNorm only ---")
    for M, C in [(4096, 320), (1024, 640), (1024, 1280), (256, 2560), (1024, 3072)]:
        c, r = bench_gn(M, C)
        print(f"  GN M={M:<5} C={C:<5}            {c:<12.3f} {r:<12.3f} {r/c:.2f}x")

    # Full pipeline benchmarks
    print("\n--- GroupNorm + Linear ---")
    configs = [
        ("SD1.5 M=4096 320->320",    4096, 320,  320),
        ("SD1.5 M=1024 320->320",    1024, 320,  320),
        ("SD1.5 M=1024 640->640",    1024, 640,  640),
        ("SDXL  M=1024 1280->1280",  1024, 1280, 1280),
        ("SDXL  M=256  2560->2560",   256, 2560, 2560),
        ("Flux  M=1024 3072->3072",  1024, 3072, 3072),
        ("Flux  M=256  3072->3072",   256, 3072, 3072),
    ]
    for name, M, C_in, C_out in configs:
        c, r = bench_gnl(M, C_in, C_out)
        print(f"  {name:<33} {c:<12.3f} {r:<12.3f} {r/c:.2f}x")


if __name__ == "__main__":
    main()
