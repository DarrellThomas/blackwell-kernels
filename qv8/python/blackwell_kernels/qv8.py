# Copyright (c) 2026 Darrell Thomas. MIT License. See LICENSE file.
#
# QV-8: Quantum Volume 8-qubit simulation.
#
# Custom CUDA kernel wrapper + PyTorch reference + circuit generator.

import math
import torch
import numpy as np

N_QUBITS = 8
STATE_SIZE = 1 << N_QUBITS  # 256


def _random_su4(rng):
    """Generate a random SU(4) matrix via QR decomposition of a random complex matrix."""
    z = (rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))) / math.sqrt(2)
    q, r = np.linalg.qr(z)
    d = np.diag(r)
    ph = d / np.abs(d)
    q = q @ np.diag(ph)
    # Make it SU(4): det(q) should be 1
    det = np.linalg.det(q)
    q = q / (det ** 0.25)
    return q.astype(np.complex64)


def generate_qv8_circuits(num_circuits, num_layers=N_QUBITS, seed=42):
    """Generate random QV-8 circuits.

    Each layer applies N_QUBITS/2 = 4 random SU(4) gates to random
    qubit pairs (a permutation of all 8 qubits, split into pairs).

    Returns:
        gate_matrices: [C, G, 4, 4, 2] float32 tensor (real/imag last dim)
        gate_qubits:   [C, G, 2] int32 tensor
        num_gates_per_circuit: int (= num_layers * 4)
    """
    rng = np.random.default_rng(seed)
    num_gates = num_layers * (N_QUBITS // 2)

    all_matrices = np.zeros((num_circuits, num_gates, 4, 4, 2), dtype=np.float32)
    all_qubits = np.zeros((num_circuits, num_gates, 2), dtype=np.int32)

    # Generate qubit permutations per layer (shared across all circuits).
    # Only the SU(4) matrices differ per circuit. This is standard for
    # batched QV simulation and enables vectorized reference execution.
    layer_perms = [rng.permutation(N_QUBITS) for _ in range(num_layers)]

    for c in range(num_circuits):
        g_idx = 0
        for layer in range(num_layers):
            perm = layer_perms[layer]
            for pair in range(N_QUBITS // 2):
                q0 = int(perm[2 * pair])
                q1 = int(perm[2 * pair + 1])
                u = _random_su4(rng)
                all_matrices[c, g_idx, :, :, 0] = u.real
                all_matrices[c, g_idx, :, :, 1] = u.imag
                all_qubits[c, g_idx, 0] = q0
                all_qubits[c, g_idx, 1] = q1
                g_idx += 1

    gate_matrices = torch.from_numpy(all_matrices)
    gate_qubits = torch.from_numpy(all_qubits)
    return gate_matrices, gate_qubits, num_gates


def qv8_simulate(gate_matrices, gate_qubits, num_circuits):
    """Run fused CUDA QV-8 simulation.

    Args:
        gate_matrices: [C, G, 4, 4, 2] float32 CUDA tensor
        gate_qubits:   [C, G, 2] int32 CUDA tensor
        num_circuits:  number of circuits

    Returns:
        probs: [C, 256] float32 tensor of output probabilities
    """
    from blackwell_kernels._C import qv8_simulate as _qv8_simulate_cuda
    return _qv8_simulate_cuda(gate_matrices, gate_qubits, num_circuits)


def qv8_simulate_ref(gate_matrices, gate_qubits, num_circuits):
    """PyTorch reference QV-8 simulation (batched, vectorized).

    Same interface as qv8_simulate but uses PyTorch operations.
    """
    device = gate_matrices.device
    C = num_circuits
    G = gate_matrices.size(1)
    N = STATE_SIZE

    # Build complex gate matrices: [C, G, 4, 4]
    gates_complex = torch.complex(gate_matrices[..., 0], gate_matrices[..., 1])

    # Initialize |0⟩^⊗8
    state = torch.zeros(C, N, dtype=torch.complex64, device=device)
    state[:, 0] = 1.0 + 0.0j

    for g in range(G):
        q0 = gate_qubits[0, g, 0].item()  # same qubit layout across batch
        q1 = gate_qubits[0, g, 1].item()

        mask_q0 = 1 << q0
        mask_q1 = 1 << q1

        # Build index arrays for the 4 sub-populations
        all_idx = torch.arange(N, device=device)
        bases = all_idx[((all_idx & mask_q0) == 0) & ((all_idx & mask_q1) == 0)]

        i00 = bases
        i01 = bases | mask_q0
        i10 = bases | mask_q1
        i11 = bases | mask_q0 | mask_q1

        # Gather: [C, N/4, 4]
        amps = torch.stack([state[:, i00], state[:, i01],
                            state[:, i10], state[:, i11]], dim=-1)

        # Per-circuit gate: [C, 4, 4]
        gate = gates_complex[:, g]

        # Apply: [C, N/4, 4] @ [C, 4, 4]^T → [C, N/4, 4]
        new_amps = torch.bmm(amps, gate.transpose(-1, -2))

        # Scatter back
        state[:, i00] = new_amps[..., 0]
        state[:, i01] = new_amps[..., 1]
        state[:, i10] = new_amps[..., 2]
        state[:, i11] = new_amps[..., 3]

    # Return probabilities
    return (state.real ** 2 + state.imag ** 2)
