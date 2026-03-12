# Copyright (c) 2026 Darrell Thomas. MIT License. See LICENSE file.

"""General-purpose kernel operations for sm_120."""

import torch
from blackwell_kernels._C import bf16_gemm as _bf16_gemm


def bf16_gemm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """BF16 matrix multiply optimized for RTX 5090 (sm_120).

    Args:
        A: [M, K] BF16 tensor
        B: [K, N] BF16 tensor

    Returns:
        C: [M, N] BF16 tensor (A @ B)
    """
    return _bf16_gemm(A, B)
