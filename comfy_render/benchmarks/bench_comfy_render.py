# Copyright (c) 2026 Darrell Thomas. MIT License. See LICENSE file.
#
# TEMPLATE: Benchmark for [KERNEL_NAME] vs reference implementation.
# Rename this file to bench_<kernel>.py and fill in the benchmarks.
#
# IMPORTANT: This script MUST emit these two lines for eval.sh to parse:
#   primary_custom_ms: <time in milliseconds>
#   primary_vs_ref: <speedup ratio>x
#
# Usage: CUDA_VISIBLE_DEVICES=1 PYTHONPATH=python python3 benchmarks/bench_<kernel>.py

import sys
import time
import torch

sys.path.insert(0, "python")
# from blackwell_kernels import <kernel_fn>

device = "cuda"
dtype = torch.bfloat16

WARMUP = 10
TIMED = 100


def bench(fn, warmup=WARMUP, timed=TIMED):
    """Benchmark a function, return mean time in ms."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(timed):
        fn()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return elapsed / timed * 1000  # ms


def main():
    # ─── Primary config (must match profile script) ──────────────────────────
    # TODO: Set your primary benchmark dimensions
    # M, N, K = 2048, 768, 3072

    # TODO: Create inputs
    # A = torch.randn(M, K, device=device, dtype=dtype)
    # B = torch.randn(K, N, device=device, dtype=dtype)

    # TODO: Benchmark your kernel
    # custom_ms = bench(lambda: my_kernel(A, B))

    # TODO: Benchmark reference (e.g., cuBLAS via torch.mm)
    # ref_ms = bench(lambda: torch.mm(A, B))

    # TODO: Compute speedup
    # vs_ref = ref_ms / custom_ms

    # ─── Required output lines (DO NOT CHANGE FORMAT) ────────────────────────
    # These lines are parsed by eval.sh. Format must be exact.
    # print(f"primary_custom_ms: {custom_ms:.3f}")
    # print(f"primary_vs_ref: {vs_ref:.2f}x")

    # ─── Additional configs (optional) ───────────────────────────────────────
    # print(f"\n{'Config':<30} {'Custom (ms)':<15} {'Ref (ms)':<15} {'Speedup'}")
    # print(f"{'─'*75}")
    # for config in configs:
    #     ...

    print("TODO: Fill in benchmark")


if __name__ == "__main__":
    main()
