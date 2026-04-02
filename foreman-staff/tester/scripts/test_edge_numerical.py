#!/usr/bin/env python3
# Copyright (c) 2026 Darrell Thomas. MIT License.
# Edge case tests for Cholesky factorization.

import torch
import sys
sys.path.insert(0, "python")
from blackwell_kernels import _C as bk

DEVICE = "cuda"
passed = 0
failed = 0
errors = []


def run_test(name, fn):
    global passed, failed, errors
    try:
        fn()
        passed += 1
        print(f"  PASS  {name}")
    except Exception as e:
        failed += 1
        errors.append((name, str(e)))
        print(f"  FAIL  {name}: {e}")


def make_spd(N, cond=1.0):
    """Make a symmetric positive definite matrix."""
    A = torch.randn(N, N, dtype=torch.float32, device=DEVICE)
    M = A @ A.T + torch.eye(N, dtype=torch.float32, device=DEVICE) * N * cond
    return M


# ============================================================
# Correctness
# ============================================================
print("\n=== CHOLESKY CORRECTNESS ===")


def test_chol_identity():
    """chol(I) should be I."""
    N = 256
    I = torch.eye(N, dtype=torch.float32, device=DEVICE)
    L = bk.cholesky(I, 64)
    err = (L - torch.eye(N, device=DEVICE)).abs().max().item()
    assert err < 1e-4, f"chol(I) error: {err}"


def test_chol_diagonal():
    """chol(diag(d)) should be diag(sqrt(d))."""
    N = 256
    d = torch.rand(N, dtype=torch.float32, device=DEVICE) + 0.1
    D = torch.diag(d)
    L = bk.cholesky(D, 64)
    L_diag = L.diagonal()
    ref_diag = d.sqrt()
    err = (L_diag - ref_diag).abs().max().item()
    assert err < 1e-3, f"Diagonal Cholesky error: {err}"


def test_chol_reconstruction():
    """L @ L^T should reconstruct the original matrix."""
    for N in [64, 128, 256, 512, 1024]:
        M = make_spd(N)
        L = bk.cholesky(M, 64)
        recon = L @ L.T
        # Only compare lower triangle (upper was zeroed by tril)
        err = (torch.tril(recon) - torch.tril(M)).abs().max().item() / M.abs().max().item()
        assert err < 1e-3, f"Reconstruction error at N={N}: {err}"


def test_chol_lower_triangular():
    """Output must be lower triangular."""
    N = 512
    M = make_spd(N)
    L = bk.cholesky(M, 64)
    upper = torch.triu(L, diagonal=1)
    assert upper.abs().max().item() < 1e-6, "Output has non-zero upper triangle"


def test_chol_positive_diagonal():
    """Diagonal of L must be positive."""
    N = 512
    M = make_spd(N)
    L = bk.cholesky(M, 64)
    min_diag = L.diagonal().min().item()
    assert min_diag > 0, f"Non-positive diagonal: {min_diag}"


# ============================================================
# Edge Cases
# ============================================================
print("\n=== CHOLESKY EDGE CASES ===")


def test_chol_1x1():
    M = torch.tensor([[4.0]], device=DEVICE)
    L = bk.cholesky(M, 64)
    assert abs(L[0, 0].item() - 2.0) < 1e-4, f"chol([[4]]) = {L[0,0].item()}, expected 2.0"


def test_chol_2x2():
    M = torch.tensor([[4.0, 2.0], [2.0, 5.0]], device=DEVICE)
    L = bk.cholesky(M, 64)
    ref = torch.linalg.cholesky(M)
    err = (L - ref).abs().max().item()
    assert err < 1e-4, f"2x2 error: {err}"


def test_chol_non_tile_multiple():
    """N not divisible by NB=64."""
    for N in [100, 150, 200, 300]:
        M = make_spd(N)
        L = bk.cholesky(M, 64)
        recon_err = (L @ L.T - M).abs().max().item() / M.abs().max().item()
        assert recon_err < 1e-2, f"Non-aligned N={N} error: {recon_err}"


def test_chol_small_nb():
    """Different block sizes."""
    N = 256
    M = make_spd(N)
    for nb in [32, 64, 128]:
        L = bk.cholesky(M, nb)
        recon_err = (L @ L.T - M).abs().max().item() / M.abs().max().item()
        assert recon_err < 1e-2, f"NB={nb} error: {recon_err}"


def test_chol_well_conditioned():
    """Well-conditioned matrix (cond ~ 1)."""
    N = 512
    M = make_spd(N, cond=10.0)
    L = bk.cholesky(M, 64)
    recon_err = (L @ L.T - M).abs().max().item() / M.abs().max().item()
    assert recon_err < 1e-4, f"Well-conditioned error: {recon_err}"


def test_chol_ill_conditioned():
    """Ill-conditioned matrix — precision stress test."""
    N = 256
    U, _, _ = torch.linalg.svd(torch.randn(N, N, dtype=torch.float32, device=DEVICE))
    s = torch.logspace(0, -6, N, dtype=torch.float32, device=DEVICE)
    M = U @ torch.diag(s) @ U.T
    L = bk.cholesky(M, 64)
    recon_err = (L @ L.T - M).abs().max().item() / M.abs().max().item()
    assert recon_err < 0.1, f"Ill-conditioned error: {recon_err}"


def test_chol_large_values():
    """Matrix with large values."""
    N = 256
    A = torch.randn(N, N, dtype=torch.float32, device=DEVICE) * 1000
    M = A @ A.T + torch.eye(N, device=DEVICE) * N * 1000
    L = bk.cholesky(M, 64)
    recon_err = (L @ L.T - M).abs().max().item() / M.abs().max().item()
    assert recon_err < 1e-2, f"Large value error: {recon_err}"


def test_chol_vs_torch():
    """Compare against torch.linalg.cholesky."""
    N = 512
    M = make_spd(N)
    L_ours = bk.cholesky(M, 64)
    L_ref = torch.linalg.cholesky(M)
    err = (L_ours - L_ref).abs().max().item() / L_ref.abs().max().item()
    assert err < 1e-2, f"vs torch error: {err}"


for t in [test_chol_identity, test_chol_diagonal, test_chol_reconstruction,
          test_chol_lower_triangular, test_chol_positive_diagonal,
          test_chol_1x1, test_chol_2x2, test_chol_non_tile_multiple,
          test_chol_small_nb, test_chol_well_conditioned,
          test_chol_ill_conditioned, test_chol_large_values, test_chol_vs_torch]:
    run_test(t.__name__, t)


# ============================================================
# Summary
# ============================================================
print(f"\n{'='*50}")
print(f"RESULTS: {passed} passed, {failed} failed out of {passed + failed} tests")
if errors:
    print(f"\nFAILURES:")
    for name, err in errors:
        print(f"  {name}: {err}")
print(f"{'='*50}")
sys.exit(0 if failed == 0 else 1)
