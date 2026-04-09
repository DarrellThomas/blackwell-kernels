# Copyright (c) 2026 Darrell Thomas. MIT License. See LICENSE file.
#
# Stress tests for QV-8 fused simulator: large batches and repeated execution.
# Usage: CUDA_VISIBLE_DEVICES=1 PYTHONPATH=python python3 tests/test_stress.py

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


def test_large_batch():
    """Large batch: 10000 circuits."""
    C = 10000
    gm, gq, ng = generate_qv8_circuits(C, seed=42)
    probs = qv8_simulate(gm.to(device), gq.to(device), C)
    sums = probs.sum(dim=-1)
    worst_sum = (sums - 1.0).abs().max().item()
    check(f"large_batch_{C}", worst_sum < 1e-3, f"worst_sum={worst_sum:.2e}")
    check(f"large_batch_{C}_nonneg", (probs >= -1e-6).all().item())
    check(f"large_batch_{C}_shape", probs.shape == (C, 256), f"got {probs.shape}")


def test_repeated_same_input():
    """Same input 50 times: results must be deterministic (no race conditions)."""
    C = 32
    gm, gq, ng = generate_qv8_circuits(C, seed=99)
    gm_d, gq_d = gm.to(device), gq.to(device)

    ref = qv8_simulate(gm_d, gq_d, C)
    all_match = True
    for i in range(50):
        out = qv8_simulate(gm_d, gq_d, C)
        if not torch.equal(ref, out):
            all_match = False
            break
    check("deterministic_50x", all_match)


def test_many_layers():
    """Many layers: 32 layers (128 gates per circuit)."""
    C = 64
    gm, gq, ng = generate_qv8_circuits(C, num_layers=32, seed=42)
    probs = qv8_simulate(gm.to(device), gq.to(device), C)
    ref = qv8_simulate_ref(gm.to(device), gq.to(device), C)
    err = (probs - ref).abs().max().item()
    check(f"many_layers_32", err < 1e-3, f"err={err:.2e}")
    sums = probs.sum(dim=-1)
    worst_sum = (sums - 1.0).abs().max().item()
    check(f"many_layers_32_sum", worst_sum < 1e-3, f"worst_sum={worst_sum:.2e}")


def test_sequential_batches():
    """Multiple independent batches sequentially: no state leakage."""
    for seed in range(10):
        C = 100
        gm, gq, ng = generate_qv8_circuits(C, seed=seed)
        probs = qv8_simulate(gm.to(device), gq.to(device), C)
        sums = probs.sum(dim=-1)
        worst_sum = (sums - 1.0).abs().max().item()
        check(f"seq_batch_seed{seed}", worst_sum < 1e-3, f"worst_sum={worst_sum:.2e}")


if __name__ == "__main__":
    print(f"Stress tests for QV-8 on {torch.cuda.get_device_name()}")
    test_large_batch()
    test_repeated_same_input()
    test_many_layers()
    test_sequential_batches()
    print(f"\nResults: {PASS} passed, {FAIL} failed")
    if FAIL > 0:
        sys.exit(1)
