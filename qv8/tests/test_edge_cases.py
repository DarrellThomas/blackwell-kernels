# Copyright (c) 2026 Darrell Thomas. MIT License. See LICENSE file.
#
# Edge-case tests for QV-8 fused simulator.
# Usage: CUDA_VISIBLE_DEVICES=1 PYTHONPATH=python python3 tests/test_edge_cases.py

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


def test_single_circuit():
    """Minimum batch size: 1 circuit."""
    gm, gq, ng = generate_qv8_circuits(1, seed=1)
    probs = qv8_simulate(gm.to(device), gq.to(device), 1)
    ref = qv8_simulate_ref(gm.to(device), gq.to(device), 1)
    err = (probs - ref).abs().max().item()
    check("single_circuit", err < 1e-4, f"err={err:.2e}")
    check("single_circuit_sum", abs(probs.sum().item() - 1.0) < 1e-3)


def test_two_circuits():
    """Batch of 2: minimal multi-circuit."""
    gm, gq, ng = generate_qv8_circuits(2, seed=2)
    probs = qv8_simulate(gm.to(device), gq.to(device), 2)
    ref = qv8_simulate_ref(gm.to(device), gq.to(device), 2)
    err = (probs - ref).abs().max().item()
    check("two_circuits", err < 1e-4, f"err={err:.2e}")


def test_non_power_of_two_batch():
    """Non-power-of-2 batch sizes."""
    for C in [3, 5, 7, 13, 17, 31, 33, 63, 65, 127, 129]:
        gm, gq, ng = generate_qv8_circuits(C, seed=C)
        probs = qv8_simulate(gm.to(device), gq.to(device), C)
        ref = qv8_simulate_ref(gm.to(device), gq.to(device), C)
        err = (probs - ref).abs().max().item()
        sums = probs.sum(dim=-1)
        worst_sum = (sums - 1.0).abs().max().item()
        check(f"batch_{C}", err < 1e-4, f"err={err:.2e}")
        check(f"batch_{C}_sum", worst_sum < 1e-3, f"worst_sum={worst_sum:.2e}")


def test_different_seeds():
    """Different RNG seeds produce different circuits and all pass."""
    for seed in [0, 1, 999, 12345, 2**31 - 1]:
        gm, gq, ng = generate_qv8_circuits(4, seed=seed)
        probs = qv8_simulate(gm.to(device), gq.to(device), 4)
        ref = qv8_simulate_ref(gm.to(device), gq.to(device), 4)
        err = (probs - ref).abs().max().item()
        check(f"seed_{seed}", err < 1e-4, f"err={err:.2e}")


def test_output_shape():
    """Output tensor shape is [C, 256]."""
    for C in [1, 8, 64]:
        gm, gq, ng = generate_qv8_circuits(C, seed=42)
        probs = qv8_simulate(gm.to(device), gq.to(device), C)
        check(f"shape_C{C}", probs.shape == (C, 256), f"got {probs.shape}")
        check(f"dtype_C{C}", probs.dtype == torch.float32, f"got {probs.dtype}")
        check(f"device_C{C}", probs.is_cuda, f"got {probs.device}")


def test_different_layer_counts():
    """Non-standard layer counts (generate_qv8_circuits supports num_layers)."""
    for L in [1, 2, 4, 16]:
        gm, gq, ng = generate_qv8_circuits(4, num_layers=L, seed=42)
        probs = qv8_simulate(gm.to(device), gq.to(device), 4)
        ref = qv8_simulate_ref(gm.to(device), gq.to(device), 4)
        err = (probs - ref).abs().max().item()
        check(f"layers_{L}", err < 1e-4, f"err={err:.2e}")


if __name__ == "__main__":
    print(f"Edge-case tests for QV-8 on {torch.cuda.get_device_name()}")
    test_single_circuit()
    test_two_circuits()
    test_non_power_of_two_batch()
    test_different_seeds()
    test_output_shape()
    test_different_layer_counts()
    print(f"\nResults: {PASS} passed, {FAIL} failed")
    if FAIL > 0:
        sys.exit(1)
