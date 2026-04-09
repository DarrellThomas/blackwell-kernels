# Copyright (c) 2026 Darrell Thomas. MIT License. See LICENSE file.
#
# Correctness tests for QV-8 fused simulator.
# Usage: CUDA_VISIBLE_DEVICES=1 PYTHONPATH=python python3 tests/test_qv8.py

import sys
import torch

sys.path.insert(0, "python")
from blackwell_kernels.qv8 import qv8_simulate, qv8_simulate_ref, generate_qv8_circuits

device = "cuda"
torch.manual_seed(42)


def test_single_circuit():
    """Single circuit: CUDA vs PyTorch reference."""
    gm, gq, ng = generate_qv8_circuits(1, seed=123)
    gm_d = gm.to(device)
    gq_d = gq.to(device)

    probs_cuda = qv8_simulate(gm_d, gq_d, 1)
    probs_ref = qv8_simulate_ref(gm_d, gq_d, 1)

    max_err = (probs_cuda - probs_ref).abs().max().item()
    prob_sum_cuda = probs_cuda.sum(dim=-1)
    prob_sum_ref = probs_ref.sum(dim=-1)

    print(f"  single_circuit: max_err={max_err:.6e}, "
          f"sum_cuda={prob_sum_cuda.item():.6f}, sum_ref={prob_sum_ref.item():.6f}")

    assert max_err < 1e-4, f"Max error too high: {max_err}"
    assert abs(prob_sum_cuda.item() - 1.0) < 1e-3, f"CUDA probs don't sum to 1: {prob_sum_cuda.item()}"
    print("  single_circuit: PASS")


def test_batch():
    """Batch of 64 circuits: CUDA vs PyTorch reference."""
    C = 64
    gm, gq, ng = generate_qv8_circuits(C, seed=456)
    gm_d = gm.to(device)
    gq_d = gq.to(device)

    probs_cuda = qv8_simulate(gm_d, gq_d, C)
    probs_ref = qv8_simulate_ref(gm_d, gq_d, C)

    max_err = (probs_cuda - probs_ref).abs().max().item()
    sum_errs = (probs_cuda.sum(dim=-1) - 1.0).abs()
    worst_sum = sum_errs.max().item()

    print(f"  batch({C}): max_err={max_err:.6e}, worst_sum_dev={worst_sum:.6e}")
    assert max_err < 1e-4, f"Max error too high: {max_err}"
    assert worst_sum < 1e-3, f"Worst prob sum deviation: {worst_sum}"
    print(f"  batch({C}): PASS")


def test_probabilities_valid():
    """All probabilities non-negative and sum to ~1."""
    C = 32
    gm, gq, ng = generate_qv8_circuits(C, seed=789)
    gm_d = gm.to(device)
    gq_d = gq.to(device)

    probs = qv8_simulate(gm_d, gq_d, C)

    assert (probs >= -1e-6).all(), "Negative probabilities found"
    sums = probs.sum(dim=-1)
    assert ((sums - 1.0).abs() < 1e-3).all(), f"Probability sums deviate from 1: {sums}"
    print("  probabilities_valid: PASS")


if __name__ == "__main__":
    print(f"Testing QV-8 on {torch.cuda.get_device_name()}")
    test_single_circuit()
    test_batch()
    test_probabilities_valid()
    print("All tests passed!")
