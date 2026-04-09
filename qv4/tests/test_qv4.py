# Copyright (c) 2026 Darrell Thomas. MIT License.
#
# Correctness tests for QV-4 kernel vs NumPy reference.
# Usage: CUDA_VISIBLE_DEVICES=1 PYTHONPATH=python python3 tests/test_qv4.py

import sys
import numpy as np

sys.path.insert(0, "python")
from blackwell_kernels.qv4 import (
    generate_qv4_circuits,
    qv4_simulate_cuda,
    qv4_simulate_numpy,
    heavy_output_probability,
)

ATOL = 1e-4  # absolute tolerance for float32 quantum sim


def test_single_circuit():
    """Single circuit: CUDA vs NumPy reference."""
    rng = np.random.default_rng(42)
    gate_data, pair_ids = generate_qv4_circuits(1, rng=rng)

    ref = qv4_simulate_numpy(gate_data, pair_ids)
    cuda = qv4_simulate_cuda(gate_data, pair_ids).cpu().numpy()

    max_err = np.abs(cuda - ref).max()
    prob_sum_cuda = cuda.sum(axis=1)
    prob_sum_ref = ref.sum(axis=1)

    status = "PASS" if max_err < ATOL else "FAIL"
    print(f"  single_circuit: {status} (max_err={max_err:.6f}, "
          f"sum_cuda={prob_sum_cuda[0]:.6f}, sum_ref={prob_sum_ref[0]:.6f})")
    assert max_err < ATOL, f"max_err={max_err}"


def test_batch():
    """Batch of 1000 circuits: CUDA vs NumPy reference."""
    rng = np.random.default_rng(123)
    gate_data, pair_ids = generate_qv4_circuits(1000, rng=rng)

    ref = qv4_simulate_numpy(gate_data, pair_ids)
    cuda = qv4_simulate_cuda(gate_data, pair_ids).cpu().numpy()

    max_err = np.abs(cuda - ref).max()
    mean_err = np.abs(cuda - ref).mean()

    status = "PASS" if max_err < ATOL else "FAIL"
    print(f"  batch_1000: {status} (max_err={max_err:.6f}, mean_err={mean_err:.8f})")
    assert max_err < ATOL, f"max_err={max_err}"


def test_probabilities_sum_to_one():
    """All output probability distributions must sum to ~1.0."""
    rng = np.random.default_rng(456)
    gate_data, pair_ids = generate_qv4_circuits(500, rng=rng)
    probs = qv4_simulate_cuda(gate_data, pair_ids).cpu().numpy()
    sums = probs.sum(axis=1)

    max_dev = np.abs(sums - 1.0).max()
    status = "PASS" if max_dev < 1e-3 else "FAIL"
    print(f"  prob_sums: {status} (max_dev_from_1={max_dev:.6f})")
    assert max_dev < 1e-3, f"max_dev={max_dev}"


def test_heavy_output():
    """Heavy output probability should be > 2/3 for QV-4 (theoretical threshold)."""
    rng = np.random.default_rng(789)
    gate_data, pair_ids = generate_qv4_circuits(1000, rng=rng)
    probs = qv4_simulate_cuda(gate_data, pair_ids)
    hop = heavy_output_probability(probs)
    mean_hop = hop.mean()

    # For ideal simulation, mean HOP should be ~0.85 for 4 qubits
    status = "PASS" if mean_hop > 0.6 else "FAIL"
    print(f"  heavy_output: {status} (mean_HOP={mean_hop:.4f}, expected >0.6)")
    assert mean_hop > 0.6, f"mean_HOP={mean_hop}"


def test_all_pair_types():
    """Ensure all 6 qubit pair types are exercised and correct."""
    rng = np.random.default_rng(101)
    gate_data, pair_ids = generate_qv4_circuits(200, rng=rng)

    unique_pairs = set(pair_ids.ravel().tolist())
    print(f"  pair_coverage: {len(unique_pairs)}/6 pair types seen")

    ref = qv4_simulate_numpy(gate_data, pair_ids)
    cuda = qv4_simulate_cuda(gate_data, pair_ids).cpu().numpy()
    max_err = np.abs(cuda - ref).max()

    status = "PASS" if max_err < ATOL and len(unique_pairs) == 6 else "FAIL"
    print(f"  all_pair_types: {status} (max_err={max_err:.6f})")
    assert max_err < ATOL, f"max_err={max_err}"
    assert len(unique_pairs) == 6, f"only {len(unique_pairs)} pair types seen"


if __name__ == "__main__":
    import torch
    print(f"Testing QV-4 on {torch.cuda.get_device_name()}")
    test_single_circuit()
    test_batch()
    test_probabilities_sum_to_one()
    test_heavy_output()
    test_all_pair_types()
    print("All tests passed!")
