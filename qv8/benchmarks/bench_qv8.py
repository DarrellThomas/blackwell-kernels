# Copyright (c) 2026 Darrell Thomas. MIT License. See LICENSE file.
#
# Benchmark QV-8 fused CUDA kernel vs PyTorch reference.
#
# IMPORTANT: emits primary_custom_ms and primary_vs_ref for eval.sh.
# Usage: CUDA_VISIBLE_DEVICES=1 PYTHONPATH=python python3 benchmarks/bench_qv8.py

import sys
import time
import torch

sys.path.insert(0, "python")
from blackwell_kernels.qv8 import (
    qv8_simulate, qv8_simulate_ref, generate_qv8_circuits, generate_qv8_circuits_gpu,
)

device = "cuda"

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
    # ─── Primary config: 1000 circuits (standard QV batch) ─────────────
    C = 1000
    gm, gq, ng = generate_qv8_circuits(C, seed=42)
    gm_d = gm.to(device)
    gq_d = gq.to(device)

    print(f"QV-8 benchmark: {C} circuits, {ng} gates/circuit, 256 amplitudes")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print()

    # Benchmark custom CUDA kernel
    custom_ms = bench(lambda: qv8_simulate(gm_d, gq_d, C))

    # Benchmark PyTorch reference
    ref_ms = bench(lambda: qv8_simulate_ref(gm_d, gq_d, C))

    vs_ref = ref_ms / custom_ms

    # ─── Required output lines (DO NOT CHANGE FORMAT) ──────────────────
    print(f"primary_custom_ms: {custom_ms:.3f}")
    print(f"primary_vs_ref: {vs_ref:.2f}x")

    # ─── Additional configs ────────────────────────────────────────────
    print(f"\n{'Config':<30} {'Custom (ms)':<15} {'Ref (ms)':<15} {'Speedup'}")
    print(f"{'─'*75}")
    print(f"{'1000 circuits (primary)':<30} {custom_ms:<15.3f} {ref_ms:<15.3f} {vs_ref:.2f}x")

    for c_test in [100, 10000]:
        gm_t, gq_t, _ = generate_qv8_circuits(c_test, seed=42)
        gm_td = gm_t.to(device)
        gq_td = gq_t.to(device)
        c_ms = bench(lambda: qv8_simulate(gm_td, gq_td, c_test), warmup=5, timed=50)
        r_ms = bench(lambda: qv8_simulate_ref(gm_td, gq_td, c_test), warmup=5, timed=50)
        ratio = r_ms / c_ms
        print(f"{f'{c_test} circuits':<30} {c_ms:<15.3f} {r_ms:<15.3f} {ratio:.2f}x")

    # ─── End-to-end pipeline (GPU-native generation + simulation) ────
    print(f"\n{'─'*75}")
    print(f"End-to-end pipeline (generate_qv8_circuits_gpu + qv8_simulate):")
    for c_test in [100, 1000, 10000]:
        def gpu_pipeline():
            gm_g, gq_g, ng_g = generate_qv8_circuits_gpu(c_test, seed=42, device=device)
            return qv8_simulate(gm_g, gq_g, c_test)
        e2e_ms = bench(gpu_pipeline, warmup=5, timed=50)
        print(f"  {c_test:>5} circuits end-to-end: {e2e_ms:.3f} ms ({c_test/e2e_ms:.0f} circuits/ms)")


if __name__ == "__main__":
    main()
