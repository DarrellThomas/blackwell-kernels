# Copyright (c) 2026 Darrell Thomas. MIT License.
# Minimal ncu profile launch for QV-4.
# Usage: ncu --kernel-name qv4_simulate_kernel python3 profiles/profile_qv4.py

import sys
import numpy as np
import torch

sys.path.insert(0, "python")
from blackwell_kernels.qv4 import generate_qv4_circuits, qv4_simulate_cuda

rng = np.random.default_rng(42)
gate_data, pair_ids = generate_qv4_circuits(10000, rng=rng)
gate_data_t = torch.from_numpy(gate_data).cuda()
pair_ids_t = torch.from_numpy(pair_ids).cuda()

# Warmup
qv4_simulate_cuda(gate_data_t, pair_ids_t)
torch.cuda.synchronize()

# Profiled run
qv4_simulate_cuda(gate_data_t, pair_ids_t)
torch.cuda.synchronize()
