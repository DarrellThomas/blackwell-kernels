# Copyright (c) 2026 Darrell Thomas. MIT License. See LICENSE file.

"""Custom CUDA kernels optimized for RTX 5090 (sm_120)."""

__version__ = "0.1.0"

from blackwell_kernels.comfy_render import flash_attention, fused_group_norm_linear
