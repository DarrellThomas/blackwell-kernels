# Copyright (c) 2026 Darrell Thomas. MIT License. See LICENSE file.

"""Custom CUDA kernels optimized for RTX 5090 (sm_120)."""

__version__ = "0.1.0"

from blackwell_kernels.qv8 import qv8_simulate, qv8_simulate_ref, generate_qv8_circuits
