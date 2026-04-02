# Copyright (c) 2026 Darrell Thomas. MIT License. See LICENSE file.
#
# TEMPLATE: Minimal ncu profile launch for [KERNEL_NAME].
# Rename this file to profile_<kernel>.py.
#
# Two modes:
#   Single-function: sudo ncu --kernel-name <kernel> python3 profiles/profile_<kernel>.py
#   Multi-function:  sudo ncu --kernel-name <kernel> python3 profiles/profile_<kernel>.py --op <op>
#
# For multi-function projects, add argparse with --op and a dispatch table.
# See linalg/profiles/profile_linalg.py for a working example.

import argparse
import sys

import torch

sys.path.insert(0, "python")
# from blackwell_kernels import <kernel_fn>

device = "cuda"
dtype = torch.bfloat16


def profile_single():
    """Single-function: set up data and run one kernel invocation."""
    # TODO: Set dimensions matching your primary benchmark config
    # M, N, K = 2048, 768, 3072
    # A = torch.randn(M, K, device=device, dtype=dtype)
    # B = torch.randn(K, N, device=device, dtype=dtype)

    # Warmup (3 iterations)
    for _ in range(3):
        pass  # TODO: call your kernel
    torch.cuda.synchronize()

    # Profiled run (1 iteration — ncu captures this)
    pass  # TODO: call your kernel
    torch.cuda.synchronize()


def profile_op(op):
    """Multi-function: dispatch to the right op's profile setup."""
    # TODO: Add cases for each operation. Example:
    #
    # if op == "gemv":
    #     A = torch.randn(4096, 4096, device=device, dtype=dtype)
    #     x = torch.randn(4096, device=device, dtype=dtype)
    #     for _ in range(3):
    #         gemv(A, x)
    #     torch.cuda.synchronize()
    #     gemv(A, x)
    #
    # elif op == "dot":
    #     ...
    #
    # else:
    print(f"Unknown op: {op}", file=sys.stderr)
    sys.exit(1)

    torch.cuda.synchronize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", default=None, help="Operation to profile (multi-function)")
    args = parser.parse_args()

    if args.op:
        profile_op(args.op)
    else:
        profile_single()
