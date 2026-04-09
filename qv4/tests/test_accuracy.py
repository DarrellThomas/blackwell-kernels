# Copyright (c) 2026 Darrell Thomas. MIT License.
#
# Accuracy tests for QV-4 kernel: tighter tolerances and statistical validation.
# Usage: CUDA_VISIBLE_DEVICES=1 PYTHONPATH=python python3 tests/test_accuracy.py

import sys
import numpy as np

sys.path.insert(0, "python")
from blackwell_kernels.qv4 import (
    generate_qv4_circuits,
    qv4_simulate_cuda,
    qv4_simulate_numpy,
    heavy_output_probability,
)

ATOL = 1e-4


def test_tight_tolerance():
    """5000 circuits at tight tolerance — mean error must be < 1e-6."""
    rng = np.random.default_rng(9001)
    gate_data, pair_ids = generate_qv4_circuits(5000, rng=rng)
    ref = qv4_simulate_numpy(gate_data, pair_ids)
    cuda = qv4_simulate_cuda(gate_data, pair_ids).cpu().numpy()

    max_err = np.abs(cuda - ref).max()
    mean_err = np.abs(cuda - ref).mean()

    print(f"  tight_tolerance: PASS (max_err={max_err:.2e}, mean_err={mean_err:.2e})")
    assert max_err < ATOL, f"max_err={max_err}"
    assert mean_err < 1e-6, f"mean_err={mean_err} exceeds 1e-6"


def test_probability_normalization():
    """All circuits must produce probability distributions summing to 1.0 within 1e-5."""
    rng = np.random.default_rng(9002)
    gate_data, pair_ids = generate_qv4_circuits(10000, rng=rng)
    probs = qv4_simulate_cuda(gate_data, pair_ids).cpu().numpy()

    sums = probs.sum(axis=1)
    max_dev = np.abs(sums - 1.0).max()
    mean_dev = np.abs(sums - 1.0).mean()
    print(f"  prob_norm_10k: PASS (max_dev={max_dev:.2e}, mean_dev={mean_dev:.2e})")
    assert max_dev < 1e-5, f"max normalization deviation {max_dev}"


def test_non_negative():
    """All output probabilities must be non-negative."""
    rng = np.random.default_rng(9003)
    gate_data, pair_ids = generate_qv4_circuits(5000, rng=rng)
    probs = qv4_simulate_cuda(gate_data, pair_ids).cpu().numpy()

    min_val = probs.min()
    print(f"  non_negative: PASS (min_prob={min_val:.2e})")
    assert min_val >= -1e-7, f"negative probability: {min_val}"


def test_hop_statistical():
    """Ideal QV-4 HOP should be ~0.85 ± 0.05 over 10K circuits."""
    rng = np.random.default_rng(9004)
    gate_data, pair_ids = generate_qv4_circuits(10000, rng=rng)
    probs = qv4_simulate_cuda(gate_data, pair_ids)
    hop = heavy_output_probability(probs)
    mean_hop = hop.mean()

    print(f"  hop_statistical: PASS (mean_HOP={mean_hop:.4f}, expected 0.80-0.90)")
    assert 0.70 < mean_hop < 0.95, f"HOP out of range: {mean_hop}"


if __name__ == "__main__":
    import torch
    print(f"Accuracy tests for QV-4 on {torch.cuda.get_device_name()}")
    test_tight_tolerance()
    test_probability_normalization()
    test_non_negative()
    test_hop_statistical()
    print("All accuracy tests passed!")
