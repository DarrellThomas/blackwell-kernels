# Copyright (c) 2026 Darrell Thomas. MIT License. See LICENSE file.
#
# TEMPLATE: Correctness tests for [KERNEL_NAME].
# Rename this file to test_<kernel>.py and fill in the tests.
#
# Usage: CUDA_VISIBLE_DEVICES=1 PYTHONPATH=python python3 tests/test_<kernel>.py

import sys
import torch

sys.path.insert(0, "python")
# from blackwell_kernels import <kernel_fn>

device = "cuda"
dtype = torch.bfloat16
torch.manual_seed(42)

def test_basic():
    """Basic correctness test vs PyTorch reference."""
    # TODO: Set up inputs
    # TODO: Run your kernel
    # TODO: Run PyTorch reference
    # TODO: Compare results

    # Example:
    # output = my_kernel(A, B)
    # reference = torch.mm(A, B)
    # max_err = (output - reference).abs().max().item()
    # rel_err = max_err / reference.abs().mean().item()
    # status = "PASS" if rel_err < 0.05 else "FAIL"
    # print(f"  basic: {status} (rel_err={rel_err:.4f})")
    # assert rel_err < 0.05, f"Relative error too high: {rel_err}"
    pass

def test_primary_config():
    """Test at primary benchmark dimensions."""
    # TODO: Test at the same dimensions used in benchmarks
    pass

if __name__ == "__main__":
    print(f"Testing [KERNEL_NAME] on {torch.cuda.get_device_name()}")
    test_basic()
    test_primary_config()
    print("All tests passed!")
