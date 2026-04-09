# Copyright (c) 2026 Darrell Thomas. MIT License. See LICENSE file.
#
# Benchmark flash attention vs cuDNN SDPA (via PyTorch).
#
# Usage: CUDA_VISIBLE_DEVICES=1 PYTHONPATH=python python3 benchmarks/bench_comfy_render.py

import sys
import time
import torch
import torch.nn.functional as F

sys.path.insert(0, "python")
from blackwell_kernels import flash_attention

device = "cuda"
dtype = torch.bfloat16

WARMUP = 10
TIMED = 100


def bench(fn, warmup=WARMUP, timed=TIMED):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(timed):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / timed * 1000


def main():
    # Primary config: B=4, H=32, N=1024, D=64 (GPT-2 style)
    B, H, N, D = 4, 32, 1024, 64

    q = torch.randn(B, H, N, D, device=device, dtype=dtype)
    k = torch.randn(B, H, N, D, device=device, dtype=dtype)
    v = torch.randn(B, H, N, D, device=device, dtype=dtype)

    custom_ms = bench(lambda: flash_attention(q, k, v))
    ref_ms = bench(lambda: F.scaled_dot_product_attention(q, k, v))
    vs_ref = ref_ms / custom_ms

    # Required output lines for eval.sh
    print(f"primary_custom_ms: {custom_ms:.3f}")
    print(f"primary_vs_ref: {vs_ref:.2f}x")

    # Additional configs
    configs = [
        ("B4 H32 N1024 D64", 4, 32, 1024, 64),
        ("B1 H8 N2048 D64", 1, 8, 2048, 64),
        ("B2 H16 N512 D128", 2, 16, 512, 128),
        ("B1 H12 N1024 D40", 1, 12, 1024, 40),
        ("B1 H8 N4096 D64", 1, 8, 4096, 64),
        ("B4 H32 N1024 D64 causal", 4, 32, 1024, 64),
    ]

    print(f"\n{'Config':<35} {'Custom (ms)':<15} {'Ref (ms)':<15} {'Speedup'}")
    print(f"{'─'*80}")

    for name, b, h, n, d in configs:
        causal = "causal" in name
        q = torch.randn(b, h, n, d, device=device, dtype=dtype)
        k = torch.randn(b, h, n, d, device=device, dtype=dtype)
        v = torch.randn(b, h, n, d, device=device, dtype=dtype)

        c_ms = bench(lambda: flash_attention(q, k, v, causal=causal))
        r_ms = bench(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=causal))
        ratio = r_ms / c_ms
        print(f"{name:<35} {c_ms:<15.3f} {r_ms:<15.3f} {ratio:.2f}x")


if __name__ == "__main__":
    print(f"Benchmarking on {torch.cuda.get_device_name()}")
    print()
    main()
