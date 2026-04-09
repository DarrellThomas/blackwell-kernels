# Copyright (c) 2026 Darrell Thomas. MIT License. See LICENSE file.
#
# Corner-case tests for QV-8 fused simulator.
# Usage: CUDA_VISIBLE_DEVICES=1 PYTHONPATH=python python3 tests/test_corner_cases.py

import sys
import torch
import numpy as np

sys.path.insert(0, "python")
from blackwell_kernels.qv8 import qv8_simulate, qv8_simulate_ref, N_QUBITS, STATE_SIZE

device = "cuda"
PASS = 0
FAIL = 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


def _identity_circuit(C, num_gates=4):
    """Build a circuit of identity SU(4) gates."""
    gm = torch.zeros(C, num_gates, 4, 4, 2, dtype=torch.float32)
    # Identity: real part is eye(4), imag part is zero
    for g in range(num_gates):
        gm[:, g, 0, 0, 0] = 1.0
        gm[:, g, 1, 1, 0] = 1.0
        gm[:, g, 2, 2, 0] = 1.0
        gm[:, g, 3, 3, 0] = 1.0
    gq = torch.zeros(C, num_gates, 2, dtype=torch.int32)
    # Each gate targets a different pair
    for g in range(num_gates):
        gq[:, g, 0] = (2 * g) % N_QUBITS
        gq[:, g, 1] = (2 * g + 1) % N_QUBITS
    return gm, gq, num_gates


def test_identity_gates():
    """Identity gates: output should be |0⟩ state (prob[0]=1, rest=0)."""
    C = 8
    gm, gq, ng = _identity_circuit(C)
    probs = qv8_simulate(gm.to(device), gq.to(device), C)
    # |0⟩^8 through identity gates = |0⟩^8 → prob[0]=1
    check("identity_prob0", (probs[:, 0] - 1.0).abs().max().item() < 1e-5)
    check("identity_rest", probs[:, 1:].abs().max().item() < 1e-5)


def test_identity_vs_ref():
    """Identity circuit: CUDA matches reference exactly."""
    C = 4
    gm, gq, ng = _identity_circuit(C, num_gates=8)
    probs = qv8_simulate(gm.to(device), gq.to(device), C)
    ref = qv8_simulate_ref(gm.to(device), gq.to(device), C)
    err = (probs - ref).abs().max().item()
    check("identity_vs_ref", err < 1e-6, f"err={err:.2e}")


def test_single_layer():
    """Single layer (4 gates) is the minimal useful circuit."""
    from blackwell_kernels.qv8 import generate_qv8_circuits
    C = 16
    gm, gq, ng = generate_qv8_circuits(C, num_layers=1, seed=42)
    probs = qv8_simulate(gm.to(device), gq.to(device), C)
    ref = qv8_simulate_ref(gm.to(device), gq.to(device), C)
    err = (probs - ref).abs().max().item()
    check("single_layer", err < 1e-5, f"err={err:.2e}")


def test_same_qubit_pair_repeated():
    """All gates target qubit pair (0,1): should still be correct."""
    C = 4
    num_gates = 8
    rng = np.random.default_rng(42)

    gm = torch.zeros(C, num_gates, 4, 4, 2, dtype=torch.float32)
    gq = torch.zeros(C, num_gates, 2, dtype=torch.int32)
    for c in range(C):
        for g in range(num_gates):
            # Random SU(4)
            z = (rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))) / np.sqrt(2)
            q, r = np.linalg.qr(z)
            d = np.diag(r)
            ph = d / np.abs(d)
            q = q @ np.diag(ph)
            det = np.linalg.det(q)
            q = q / (det ** 0.25)
            q = q.astype(np.complex64)
            gm[c, g, :, :, 0] = torch.from_numpy(q.real)
            gm[c, g, :, :, 1] = torch.from_numpy(q.imag)
            gq[c, g, 0] = 0
            gq[c, g, 1] = 1

    probs = qv8_simulate(gm.to(device), gq.to(device), C)
    ref = qv8_simulate_ref(gm.to(device), gq.to(device), C)
    err = (probs - ref).abs().max().item()
    check("same_pair_repeated", err < 1e-4, f"err={err:.2e}")


def test_adjacent_qubit_pairs():
    """Gates targeting adjacent qubit pairs (0,1), (2,3), (4,5), (6,7)."""
    from blackwell_kernels.qv8 import generate_qv8_circuits
    C = 8
    gm, gq, ng = generate_qv8_circuits(C, seed=42)
    # Override qubit pairs to be strictly adjacent
    for g in range(ng):
        layer = g // 4
        pair = g % 4
        gq[:, g, 0] = 2 * pair
        gq[:, g, 1] = 2 * pair + 1
    probs = qv8_simulate(gm.to(device), gq.to(device), C)
    ref = qv8_simulate_ref(gm.to(device), gq.to(device), C)
    err = (probs - ref).abs().max().item()
    check("adjacent_pairs", err < 1e-4, f"err={err:.2e}")


if __name__ == "__main__":
    print(f"Corner-case tests for QV-8 on {torch.cuda.get_device_name()}")
    test_identity_gates()
    test_identity_vs_ref()
    test_single_layer()
    test_same_qubit_pair_repeated()
    test_adjacent_qubit_pairs()
    print(f"\nResults: {PASS} passed, {FAIL} failed")
    if FAIL > 0:
        sys.exit(1)
