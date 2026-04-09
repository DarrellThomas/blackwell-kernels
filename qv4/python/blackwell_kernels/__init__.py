# Copyright (c) 2026 Darrell Thomas. MIT License.

"""QV-4: Quantum Volume 4-qubit simulation kernels for RTX 5090 (sm_120a)."""

__version__ = "0.1.0"

from blackwell_kernels.qv4 import (
    generate_qv4_circuits,
    qv4_simulate_cuda,
    qv4_simulate_numpy,
    heavy_output_probability,
)
