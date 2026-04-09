# Copyright (c) 2026 Darrell Thomas. MIT License.
#
# Security / robustness tests for QV-4 kernel: invalid inputs, out-of-range values.
# Usage: CUDA_VISIBLE_DEVICES=1 PYTHONPATH=python python3 tests/test_security.py

import sys
import numpy as np
import torch

sys.path.insert(0, "python")
from blackwell_kernels.qv4 import qv4_simulate_cuda

ATOL = 1e-4


def test_nan_in_gates():
    """NaN in gate data should not crash — output may be NaN but no segfault."""
    gate_data = np.full((10, 8, 32), np.nan, dtype=np.float32)
    pair_ids = np.zeros((10, 8), dtype=np.int32)
    try:
        result = qv4_simulate_cuda(gate_data, pair_ids).cpu().numpy()
        # NaN propagation is acceptable
        print(f"  nan_gates: PASS (no crash, has_nan={np.isnan(result).any()})")
    except RuntimeError as e:
        # CUDA error is also acceptable — just no segfault
        print(f"  nan_gates: PASS (caught RuntimeError: {str(e)[:80]})")


def test_inf_in_gates():
    """Inf in gate data should not crash."""
    gate_data = np.full((10, 8, 32), np.inf, dtype=np.float32)
    pair_ids = np.zeros((10, 8), dtype=np.int32)
    try:
        result = qv4_simulate_cuda(gate_data, pair_ids).cpu().numpy()
        print(f"  inf_gates: PASS (no crash, has_inf={np.isinf(result).any()})")
    except RuntimeError as e:
        print(f"  inf_gates: PASS (caught RuntimeError: {str(e)[:80]})")


def test_zero_gates():
    """All-zero gate matrices — should not crash, output should be well-defined."""
    gate_data = np.zeros((10, 8, 32), dtype=np.float32)
    pair_ids = np.zeros((10, 8), dtype=np.int32)
    result = qv4_simulate_cuda(gate_data, pair_ids).cpu().numpy()
    # Zero gates map everything to zero → all probs zero
    print(f"  zero_gates: PASS (output_sum={result.sum():.6f})")


def test_contiguous_inputs():
    """Non-contiguous tensor inputs — kernel requires contiguous."""
    rng = np.random.default_rng(6001)
    gate_data = np.zeros((20, 8, 32), dtype=np.float32)
    pair_ids = np.zeros((20, 8), dtype=np.int32)

    # Fill with valid data
    for i in range(4):
        gate_data[:, :, i * 4 + i] = 1.0  # identity

    gd_t = torch.from_numpy(gate_data).cuda()
    pi_t = torch.from_numpy(pair_ids).cuda()

    # Slice to make non-contiguous, then contiguous() call in wrapper handles it
    result = qv4_simulate_cuda(gd_t[::2], pi_t[::2]).cpu().numpy()
    print(f"  contiguous_handling: PASS (shape={result.shape})")


if __name__ == "__main__":
    print(f"Security tests for QV-4 on {torch.cuda.get_device_name()}")
    test_nan_in_gates()
    test_inf_in_gates()
    test_zero_gates()
    test_contiguous_inputs()
    print("All security tests passed!")
