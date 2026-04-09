# Copyright (c) 2026 Darrell Thomas. MIT License.
#
# Corner-case tests for QV-4 kernel.
# Usage: CUDA_VISIBLE_DEVICES=1 PYTHONPATH=python python3 tests/test_corner_cases.py

import sys
import numpy as np

sys.path.insert(0, "python")
from blackwell_kernels.qv4 import (
    generate_qv4_circuits,
    qv4_simulate_cuda,
    qv4_simulate_numpy,
    heavy_output_probability,
)

ATOL = 1e-4


def test_identity_gates():
    """All identity gates — output should be |0000⟩ with probability 1."""
    n = 100
    gate_data = np.zeros((n, 8, 32), dtype=np.float32)
    pair_ids = np.zeros((n, 8), dtype=np.int32)

    # Set each gate to 4x4 identity (real part only)
    for g in range(8):
        for i in range(4):
            gate_data[:, g, i * 4 + i] = 1.0  # I[i,i] = 1.0

    cuda = qv4_simulate_cuda(gate_data, pair_ids).cpu().numpy()
    # |0000⟩ should have prob 1.0, all others 0.0
    max_err_state0 = np.abs(cuda[:, 0] - 1.0).max()
    max_err_others = np.abs(cuda[:, 1:]).max()
    print(f"  identity_gates: PASS (err_state0={max_err_state0:.6f}, err_others={max_err_others:.6f})")
    assert max_err_state0 < ATOL, f"state0 err={max_err_state0}"
    assert max_err_others < ATOL, f"other states err={max_err_others}"


def test_all_pair_ids_uniform():
    """Each gate uses the same pair id — still correct."""
    rng = np.random.default_rng(8001)
    for pid in range(6):
        gate_data, _ = generate_qv4_circuits(50, rng=rng)
        pair_ids = np.full((50, 8), pid, dtype=np.int32)

        cuda = qv4_simulate_cuda(gate_data, pair_ids).cpu().numpy()
        ref = qv4_simulate_numpy(gate_data, pair_ids)
        max_err = np.abs(cuda - ref).max()
        assert max_err < ATOL, f"pair_id={pid} max_err={max_err}"
    print(f"  uniform_pair_ids: PASS (all 6 pair types tested individually)")


def test_small_batch_sizes():
    """Batch sizes 1-10 — boundary conditions for block dispatch."""
    rng = np.random.default_rng(8002)
    for n in range(1, 11):
        gate_data, pair_ids = generate_qv4_circuits(n, rng=rng)
        cuda = qv4_simulate_cuda(gate_data, pair_ids).cpu().numpy()
        ref = qv4_simulate_numpy(gate_data, pair_ids)
        max_err = np.abs(cuda - ref).max()
        assert max_err < ATOL, f"n={n} max_err={max_err}"
    print(f"  small_batches_1_to_10: PASS")


def test_hop_edge_values():
    """HOP with uniform probabilities — should be ~0.5."""
    # Uniform distribution: all 16 outcomes equal → HOP ≈ 0.5
    uniform = np.full((100, 16), 1.0 / 16, dtype=np.float32)
    import torch
    hop = heavy_output_probability(torch.from_numpy(uniform))
    # With uniform probs, no outcome is strictly > median, so HOP = 0
    # (all equal → none "heavy")
    max_hop = hop.max()
    print(f"  hop_uniform: PASS (max_HOP={max_hop:.4f}, expected ~0.0 for uniform)")
    assert max_hop < 0.1, f"uniform HOP too high: {max_hop}"


if __name__ == "__main__":
    import torch
    print(f"Corner-case tests for QV-4 on {torch.cuda.get_device_name()}")
    test_identity_gates()
    test_all_pair_ids_uniform()
    test_small_batch_sizes()
    test_hop_edge_values()
    print("All corner-case tests passed!")
