"""General-purpose kernel operations for sm_120."""

import torch


def bf16_gemm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """BF16 matrix multiply optimized for RTX 5090 (sm_120).

    Args:
        A: [M, K] BF16 tensor
        B: [K, N] BF16 tensor

    Returns:
        C: [M, N] FP32 tensor (A @ B)
    """
    # TODO: Phase 2+ - use custom kernel
    # For now, use PyTorch as reference
    return torch.mm(A.float(), B.float())
