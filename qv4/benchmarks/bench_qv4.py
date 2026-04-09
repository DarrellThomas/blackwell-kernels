# Copyright (c) 2026 Darrell Thomas. MIT License.
#
# Benchmark QV-4 CUDA kernel vs sequential PyTorch/NumPy reference.
# The "Aer GPU" baseline is simulated: sequential single-circuit GPU launches
# to model Aer's per-circuit overhead on tiny state vectors.
#
# Usage: CUDA_VISIBLE_DEVICES=1 PYTHONPATH=python python3 benchmarks/bench_qv4.py

import sys
import time
import numpy as np
import torch

sys.path.insert(0, "python")
from blackwell_kernels.qv4 import generate_qv4_circuits, qv4_simulate_cuda

WARMUP = 5
TIMED = 20


def bench_cuda(gate_data_t, pair_ids_t, n_circuits, warmup=WARMUP, timed=TIMED):
    """Benchmark our batched CUDA kernel."""
    for _ in range(warmup):
        qv4_simulate_cuda(gate_data_t, pair_ids_t)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(timed):
        qv4_simulate_cuda(gate_data_t, pair_ids_t)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / timed * 1000


def bench_sequential_gpu(gate_data_np, pair_ids_np, warmup=3, timed=5):
    """Simulate Aer-style sequential per-circuit GPU execution.

    Each circuit launched as a separate tiny kernel — models the overhead
    Aer GPU would face on 4-qubit circuits (kernel launch dominates).
    """
    n_circuits = gate_data_np.shape[0]

    # Warm up
    for _ in range(warmup):
        for c in range(min(n_circuits, 50)):
            gd = torch.from_numpy(gate_data_np[c:c+1]).cuda()
            pi = torch.from_numpy(pair_ids_np[c:c+1]).cuda()
            qv4_simulate_cuda(gd, pi)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(timed):
        for c in range(n_circuits):
            gd = torch.from_numpy(gate_data_np[c:c+1]).cuda()
            pi = torch.from_numpy(pair_ids_np[c:c+1]).cuda()
            qv4_simulate_cuda(gd, pi)
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / timed * 1000


def main():
    # Primary config: 10000 QV-4 circuits (standard QV protocol batch)
    PRIMARY_N = 10000
    rng = np.random.default_rng(42)

    print(f"Generating {PRIMARY_N} QV-4 circuits...")
    gate_data, pair_ids = generate_qv4_circuits(PRIMARY_N, rng=rng)
    gate_data_t = torch.from_numpy(gate_data).cuda()
    pair_ids_t = torch.from_numpy(pair_ids).cuda()

    print(f"Benchmarking batched CUDA kernel ({PRIMARY_N} circuits)...")
    custom_ms = bench_cuda(gate_data_t, pair_ids_t, PRIMARY_N)

    # Reference: sequential per-circuit (Aer-style) — use subset for speed
    REF_N = 1000
    print(f"Benchmarking sequential GPU reference ({REF_N} circuits)...")
    ref_ms = bench_sequential_gpu(gate_data[:REF_N], pair_ids[:REF_N], warmup=2, timed=3)
    # Scale reference to PRIMARY_N
    ref_ms_scaled = ref_ms * (PRIMARY_N / REF_N)

    vs_ref = ref_ms_scaled / custom_ms

    print(f"\n{'Config':<30} {'Time (ms)':<15} {'vs Ref'}")
    print(f"{'─' * 60}")
    print(f"{'batched CUDA (' + str(PRIMARY_N) + ')':<30} {custom_ms:<15.3f} {'—'}")
    print(f"{'sequential GPU (' + str(PRIMARY_N) + ')':<30} {ref_ms_scaled:<15.3f} {f'1.0x (baseline)'}")
    print(f"{'speedup':<30} {'—':<15} {f'{vs_ref:.1f}x'}")

    # Additional batch sizes
    print(f"\n{'Batch Size':<15} {'CUDA (ms)':<15} {'Circuits/ms'}")
    print(f"{'─' * 45}")
    for n in [100, 1000, 10000, 100000]:
        gd, pi = generate_qv4_circuits(n, rng=np.random.default_rng(0))
        gd_t = torch.from_numpy(gd).cuda()
        pi_t = torch.from_numpy(pi).cuda()
        ms = bench_cuda(gd_t, pi_t, n)
        print(f"{n:<15} {ms:<15.3f} {n / ms:<15.0f}")

    # Required output lines for eval.sh
    print(f"\nprimary_custom_ms: {custom_ms:.3f}")
    print(f"primary_vs_ref: {vs_ref:.2f}x")


if __name__ == "__main__":
    main()
