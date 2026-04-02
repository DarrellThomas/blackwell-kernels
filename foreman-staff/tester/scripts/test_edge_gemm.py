#!/usr/bin/env python3
# Copyright (c) 2026 Darrell Thomas. MIT License.
# Edge case tests for GEMM kernels (BF16, FP8, FP64).

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


def rel_error(a, b):
    denom = torch.max(a.abs().max(), b.abs().max())
    if denom == 0:
        return 0.0
    return (a - b).abs().max().item() / denom.item()


# ============================================================
# FP64 DGEMM Edge Cases
# ============================================================
print("\n=== FP64 DGEMM ===")


def test_dgemm_1x1():
    A = torch.tensor([[3.0]], dtype=torch.float64, device=DEVICE)
    B = torch.tensor([[7.0]], dtype=torch.float64, device=DEVICE)
    C = bk.dgemm(A, B)
    assert abs(C[0, 0].item() - 21.0) < 1e-10


def test_dgemm_identity():
    N = 512
    I = torch.eye(N, dtype=torch.float64, device=DEVICE)
    B = torch.randn(N, N, dtype=torch.float64, device=DEVICE)
    C = bk.dgemm(I, B)
    assert rel_error(C, B) < 1e-12, f"I*B != B, error: {rel_error(C, B)}"


def test_dgemm_zeros():
    N = 512
    Z = torch.zeros(N, N, dtype=torch.float64, device=DEVICE)
    B = torch.randn(N, N, dtype=torch.float64, device=DEVICE)
    C = bk.dgemm(Z, B)
    assert C.abs().max().item() < 1e-12


def test_dgemm_rectangular():
    for M, K, N in [(128, 256, 64), (64, 128, 512), (256, 64, 128), (1, 512, 1)]:
        A = torch.randn(M, K, dtype=torch.float64, device=DEVICE)
        B = torch.randn(K, N, dtype=torch.float64, device=DEVICE)
        C = bk.dgemm(A, B)
        ref = A @ B
        err = rel_error(C, ref)
        assert err < 1e-10, f"Rect {M}x{K}x{N} error: {err}"


def test_dgemm_non_tile_aligned():
    """Sizes that don't divide by tile size (64)."""
    for M, K, N in [(100, 50, 70), (33, 17, 5), (127, 63, 129)]:
        A = torch.randn(M, K, dtype=torch.float64, device=DEVICE)
        B = torch.randn(K, N, dtype=torch.float64, device=DEVICE)
        C = bk.dgemm(A, B)
        ref = A @ B
        err = rel_error(C, ref)
        assert err < 1e-10, f"Non-aligned {M}x{K}x{N} error: {err}"


def test_dgemm_large_values():
    N = 256
    A = torch.randn(N, N, dtype=torch.float64, device=DEVICE) * 1e10
    B = torch.randn(N, N, dtype=torch.float64, device=DEVICE) * 1e10
    C = bk.dgemm(A, B)
    ref = A @ B
    assert rel_error(C, ref) < 1e-8, f"Large value error: {rel_error(C, ref)}"


def test_dgemm_tiny_values():
    N = 256
    A = torch.randn(N, N, dtype=torch.float64, device=DEVICE) * 1e-15
    B = torch.randn(N, N, dtype=torch.float64, device=DEVICE) * 1e-15
    C = bk.dgemm(A, B)
    ref = A @ B
    abs_err = (C - ref).abs().max().item()
    assert abs_err < 1e-25, f"Tiny value error: {abs_err}"


def test_dgemm_ill_conditioned():
    N = 256
    U, _, Vt = torch.linalg.svd(torch.randn(N, N, dtype=torch.float64, device=DEVICE))
    s = torch.logspace(0, -14, N, dtype=torch.float64, device=DEVICE)
    A = U @ torch.diag(s) @ Vt
    B = torch.randn(N, N, dtype=torch.float64, device=DEVICE)
    C = bk.dgemm(A, B)
    ref = A @ B
    assert rel_error(C, ref) < 1e-6, f"Ill-conditioned error: {rel_error(C, ref)}"


def test_dgemm_symmetric_result():
    """A @ A^T should be symmetric."""
    N = 256
    A = torch.randn(N, 128, dtype=torch.float64, device=DEVICE)
    C = bk.dgemm(A, A.T.contiguous())
    asym = (C - C.T).abs().max().item()
    assert asym < 1e-10, f"Asymmetry in A@A^T: {asym}"


def test_dgemm_associativity():
    """(A @ B) @ C should equal A @ (B @ C) within FP64 precision."""
    N = 128
    A = torch.randn(N, N, dtype=torch.float64, device=DEVICE)
    B = torch.randn(N, N, dtype=torch.float64, device=DEVICE)
    C = torch.randn(N, N, dtype=torch.float64, device=DEVICE)
    AB_C = bk.dgemm(bk.dgemm(A, B), C)
    A_BC = bk.dgemm(A, bk.dgemm(B, C))
    err = rel_error(AB_C, A_BC)
    assert err < 1e-8, f"Associativity error: {err}"


for t in [test_dgemm_1x1, test_dgemm_identity, test_dgemm_zeros,
          test_dgemm_rectangular, test_dgemm_non_tile_aligned,
          test_dgemm_large_values, test_dgemm_tiny_values,
          test_dgemm_ill_conditioned, test_dgemm_symmetric_result,
          test_dgemm_associativity]:
    run_test(t.__name__, t)


# ============================================================
# BF16 GEMM Edge Cases
# ============================================================
print("\n=== BF16 GEMM ===")


def test_bf16_gemm_zeros():
    N = 256
    A = torch.zeros(N, N, dtype=torch.bfloat16, device=DEVICE)
    B = torch.randn(N, N, dtype=torch.bfloat16, device=DEVICE)
    C = bk.gemm(A, B)
    assert C.float().abs().max().item() < 1e-3


def test_bf16_gemm_identity():
    N = 256
    I = torch.eye(N, dtype=torch.bfloat16, device=DEVICE)
    B = torch.randn(N, N, dtype=torch.bfloat16, device=DEVICE)
    C = bk.gemm(I, B)
    err = rel_error(C.float(), B.float())
    assert err < 0.02, f"BF16 I*B error: {err}"


def test_bf16_gemm_large():
    """BF16 has limited range — test near max."""
    N = 256
    A = torch.randn(N, N, dtype=torch.bfloat16, device=DEVICE) * 100
    B = torch.randn(N, N, dtype=torch.bfloat16, device=DEVICE) * 100
    C = bk.gemm(A, B)
    ref = (A.float() @ B.float()).bfloat16()
    err = rel_error(C.float(), ref.float())
    assert err < 0.1, f"BF16 large value error: {err}"


for t in [test_bf16_gemm_zeros, test_bf16_gemm_identity, test_bf16_gemm_large]:
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
