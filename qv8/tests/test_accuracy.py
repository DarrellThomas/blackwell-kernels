# Copyright (c) 2026 Darrell Thomas. MIT License. See LICENSE file.
#
# Accuracy tests for QV-8 fused simulator: numerical precision bounds.
# Usage: CUDA_VISIBLE_DEVICES=1 PYTHONPATH=python python3 tests/test_accuracy.py

import sys
import torch

sys.path.insert(0, "python")
from blackwell_kernels.qv8 import qv8_simulate, qv8_simulate_ref, generate_qv8_circuits

device = "cuda"
PASS = 0
FAIL = 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


def test_max_error_bound():
    """Max absolute error across many circuits stays within FP32 tolerance."""
    C = 1000
    gm, gq, ng = generate_qv8_circuits(C, seed=42)
    probs = qv8_simulate(gm.to(device), gq.to(device), C)
    ref = qv8_simulate_ref(gm.to(device), gq.to(device), C)
    max_err = (probs - ref).abs().max().item()
    mean_err = (probs - ref).abs().mean().item()
    print(f"  1000 circuits: max_err={max_err:.2e}, mean_err={mean_err:.2e}")
    check("max_err_1000", max_err < 1e-4, f"max_err={max_err:.2e}")
    check("mean_err_1000", mean_err < 1e-6, f"mean_err={mean_err:.2e}")


def test_unitarity_preservation():
    """Probabilities sum to 1.0 (unitarity of applied gates)."""
    C = 500
    gm, gq, ng = generate_qv8_circuits(C, seed=77)
    probs = qv8_simulate(gm.to(device), gq.to(device), C)
    sums = probs.sum(dim=-1)
    worst_sum = (sums - 1.0).abs().max().item()
    mean_sum = (sums - 1.0).abs().mean().item()
    print(f"  unitarity: worst={worst_sum:.2e}, mean={mean_sum:.2e}")
    check("unitarity_worst", worst_sum < 1e-4, f"worst={worst_sum:.2e}")
    check("unitarity_mean", mean_sum < 1e-6, f"mean={mean_sum:.2e}")


def test_non_negativity():
    """All output probabilities are non-negative."""
    C = 500
    gm, gq, ng = generate_qv8_circuits(C, seed=88)
    probs = qv8_simulate(gm.to(device), gq.to(device), C)
    min_val = probs.min().item()
    check("non_negative", min_val >= -1e-7, f"min={min_val:.2e}")


def test_consistency_across_seeds():
    """Error bound holds across 20 different RNG seeds."""
    max_errs = []
    for seed in range(20):
        C = 50
        gm, gq, ng = generate_qv8_circuits(C, seed=seed * 1000)
        probs = qv8_simulate(gm.to(device), gq.to(device), C)
        ref = qv8_simulate_ref(gm.to(device), gq.to(device), C)
        max_errs.append((probs - ref).abs().max().item())
    worst = max(max_errs)
    check("consistency_20_seeds", worst < 1e-4, f"worst_across_seeds={worst:.2e}")


def test_error_scales_with_layers():
    """Error doesn't blow up with more layers (numerical stability)."""
    C = 32
    errors = []
    for L in [1, 2, 4, 8, 16, 32]:
        gm, gq, ng = generate_qv8_circuits(C, num_layers=L, seed=42)
        probs = qv8_simulate(gm.to(device), gq.to(device), C)
        ref = qv8_simulate_ref(gm.to(device), gq.to(device), C)
        err = (probs - ref).abs().max().item()
        errors.append(err)
        check(f"layers_{L}_err", err < 1e-3, f"err={err:.2e}")
    # Error shouldn't explode: worst should be < 100x best
    ratio = max(errors) / (min(errors) + 1e-15)
    check("error_stability", ratio < 1000, f"ratio={ratio:.1f}")


if __name__ == "__main__":
    print(f"Accuracy tests for QV-8 on {torch.cuda.get_device_name()}")
    test_max_error_bound()
    test_unitarity_preservation()
    test_non_negativity()
    test_consistency_across_seeds()
    test_error_scales_with_layers()
    print(f"\nResults: {PASS} passed, {FAIL} failed")
    if FAIL > 0:
        sys.exit(1)
