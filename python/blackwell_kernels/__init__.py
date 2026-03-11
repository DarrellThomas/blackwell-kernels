# Copyright (c) 2026 Darrell Thomas. MIT License. See LICENSE file.

"""Custom CUDA kernels optimized for RTX 5090 (sm_120)."""

__version__ = "0.1.0"

from blackwell_kernels.attention import flash_attn_sm120, flash_attn_v2_sm120
from blackwell_kernels.ops import bf16_gemm
