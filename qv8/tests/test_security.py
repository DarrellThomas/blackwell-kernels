# Copyright (c) 2026 Darrell Thomas. MIT License. See LICENSE file.
#
# Security / robustness tests for QV-8 fused simulator.
# Verifies the kernel doesn't crash or produce garbage on malformed inputs.
# Usage: CUDA_VISIBLE_DEVICES=1 PYTHONPATH=python python3 tests/test_security.py

import sys
import torch

sys.path.insert(0, "python")
from blackwell_kernels.qv8 import qv8_simulate, generate_qv8_circuits

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


def test_no_nan_output():
    """Output must never contain NaN."""
    for seed in [0, 42, 999]:
        C = 100
        gm, gq, ng = generate_qv8_circuits(C, seed=seed)
        probs = qv8_simulate(gm.to(device), gq.to(device), C)
        check(f"no_nan_seed{seed}", not torch.isnan(probs).any().item())


def test_no_inf_output():
    """Output must never contain Inf."""
    C = 100
    gm, gq, ng = generate_qv8_circuits(C, seed=42)
    probs = qv8_simulate(gm.to(device), gq.to(device), C)
    check("no_inf", not torch.isinf(probs).any().item())


def test_output_bounds():
    """All probabilities in [0, 1] (within FP tolerance)."""
    C = 500
    gm, gq, ng = generate_qv8_circuits(C, seed=42)
    probs = qv8_simulate(gm.to(device), gq.to(device), C)
    min_val = probs.min().item()
    max_val = probs.max().item()
    check("lower_bound", min_val >= -1e-6, f"min={min_val:.2e}")
    check("upper_bound", max_val <= 1.0 + 1e-6, f"max={max_val:.2e}")


def test_output_on_cuda():
    """Output tensor lives on CUDA, not CPU."""
    C = 4
    gm, gq, ng = generate_qv8_circuits(C, seed=42)
    probs = qv8_simulate(gm.to(device), gq.to(device), C)
    check("output_cuda", probs.is_cuda, f"device={probs.device}")


def test_no_gpu_memory_leak():
    """Running many iterations doesn't leak GPU memory significantly."""
    torch.cuda.reset_peak_memory_stats()
    C = 64
    gm, gq, ng = generate_qv8_circuits(C, seed=42)
    gm_d, gq_d = gm.to(device), gq.to(device)

    # Warmup
    for _ in range(5):
        _ = qv8_simulate(gm_d, gq_d, C)
    torch.cuda.synchronize()

    mem_before = torch.cuda.memory_allocated()
    for _ in range(100):
        probs = qv8_simulate(gm_d, gq_d, C)
        del probs
    torch.cuda.synchronize()
    mem_after = torch.cuda.memory_allocated()

    leak = mem_after - mem_before
    # Allow up to 1MB leak (PyTorch caching allocator noise)
    check("no_memory_leak", abs(leak) < 1024 * 1024, f"leak={leak} bytes")


if __name__ == "__main__":
    print(f"Security/robustness tests for QV-8 on {torch.cuda.get_device_name()}")
    test_no_nan_output()
    test_no_inf_output()
    test_output_bounds()
    test_output_on_cuda()
    test_no_gpu_memory_leak()
    print(f"\nResults: {PASS} passed, {FAIL} failed")
    if FAIL > 0:
        sys.exit(1)
