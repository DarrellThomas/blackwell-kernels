# Copyright (c) 2026 Darrell Thomas. MIT License. See LICENSE file.
#
# Profile launch script for ncu. Runs a single QV-8 batch.
# Usage: ncu --kernel-name qv8_batch_simulate_kernel python3 profiles/profile_qv8.py

import sys
import torch

sys.path.insert(0, "python")
from blackwell_kernels.qv8 import qv8_simulate, generate_qv8_circuits

device = "cuda"

C = 1000
gm, gq, ng = generate_qv8_circuits(C, seed=42)
gm_d = gm.to(device)
gq_d = gq.to(device)

# Warmup
for _ in range(3):
    qv8_simulate(gm_d, gq_d, C)
torch.cuda.synchronize()

# Profiled run
qv8_simulate(gm_d, gq_d, C)
torch.cuda.synchronize()
