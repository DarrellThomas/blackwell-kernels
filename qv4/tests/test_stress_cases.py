# Copyright (c) 2026 Darrell Thomas. MIT License.
#
# Stress tests for QV-4 kernel: large batch sizes and repeated calls.
# Usage: CUDA_VISIBLE_DEVICES=1 PYTHONPATH=python python3 tests/test_stress_cases.py

import sys
import numpy as np

sys.path.insert(0, "python")
from blackwell_kernels.qv4 import (
    generate_qv4_circuits,
    qv4_simulate_cuda,
    qv4_simulate_numpy,
)

ATOL = 1e-4


def test_large_batch():
    """50K circuits in a single batch — verify correctness on a random subset."""
    rng = np.random.default_rng(7001)
    n = 50000
    gate_data, pair_ids = generate_qv4_circuits(n, rng=rng)
    probs = qv4_simulate_cuda(gate_data, pair_ids).cpu().numpy()

    # Probabilities must sum to 1
    sums = probs.sum(axis=1)
    max_dev = np.abs(sums - 1.0).max()
    assert max_dev < 1e-3, f"prob sum deviation {max_dev}"

    # Spot-check 20 random circuits against NumPy
    check_idx = rng.choice(n, size=20, replace=False)
    ref = qv4_simulate_numpy(gate_data[check_idx], pair_ids[check_idx])
    cuda_subset = probs[check_idx]
    max_err = np.abs(cuda_subset - ref).max()
    print(f"  large_batch_50k: PASS (max_err={max_err:.6f}, sum_dev={max_dev:.6f})")
    assert max_err < ATOL, f"max_err={max_err}"


def test_repeated_calls():
    """Run the same batch 100 times — check determinism."""
    rng = np.random.default_rng(7002)
    gate_data, pair_ids = generate_qv4_circuits(500, rng=rng)

    first = qv4_simulate_cuda(gate_data, pair_ids).cpu().numpy()
    for i in range(99):
        result = qv4_simulate_cuda(gate_data, pair_ids).cpu().numpy()
        max_diff = np.abs(result - first).max()
        assert max_diff == 0.0, f"non-determinism at iter {i+1}: max_diff={max_diff}"
    print(f"  repeated_100x: PASS (deterministic)")


def test_batch_size_1():
    """Single circuit — degenerate batch."""
    rng = np.random.default_rng(7003)
    gate_data, pair_ids = generate_qv4_circuits(1, rng=rng)
    cuda = qv4_simulate_cuda(gate_data, pair_ids).cpu().numpy()
    ref = qv4_simulate_numpy(gate_data, pair_ids)
    max_err = np.abs(cuda - ref).max()
    print(f"  batch_1: PASS (max_err={max_err:.6f})")
    assert max_err < ATOL, f"max_err={max_err}"


if __name__ == "__main__":
    import torch
    print(f"Stress tests for QV-4 on {torch.cuda.get_device_name()}")
    test_large_batch()
    test_repeated_calls()
    test_batch_size_1()
    print("All stress tests passed!")
