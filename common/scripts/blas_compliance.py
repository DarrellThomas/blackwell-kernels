#!/usr/bin/env python3
"""
BLAS Compliance Test — run this against any kernel before shipping.

Usage:
    python3 blas_compliance.py <module_path> <function_name> [--op gemm|syrk|trmm|gemv|dot|nrm2|chol|lu|qr]

Example:
    CUDA_VISIBLE_DEVICES=1 python3 blas_compliance.py python/blackwell_kernels/linalg.py dgemm --op gemm

Exit code 0 = all pass. Non-zero = failures found.
This script is called automatically by the watchdog gate at testing_pass.
"""

import argparse
import sys
import importlib.util
import os

os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "1")

import torch

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

def rel_err(A, B):
    """Relative Frobenius error."""
    nB = torch.norm(B).item()
    if nB == 0:
        return torch.norm(A - B).item()
    return (torch.norm(A - B) / nB).item()

def test_gemm_sizes(gemm_fn):
    """Test GEMM at all required sizes including non-tile-aligned and degenerate."""
    print("\n=== GEMM: Matrix Sizes ===")
    sizes = [1, 2, 3, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65,
             127, 128, 129, 255, 256, 257, 511, 512, 513,
             1023, 1024, 1025]
    for N in sizes:
        A = torch.randn(N, N, dtype=torch.float64, device='cuda')
        B = torch.randn(N, N, dtype=torch.float64, device='cuda')
        try:
            C = gemm_fn(A, B)
            ref = torch.mm(A, B)
            err = rel_err(C, ref)
            check(f"N={N}", err < N * 2.3e-16, f"rel_err={err:.2e}")
        except Exception as e:
            check(f"N={N}", False, f"CRASHED: {e}")

def test_gemm_rectangular(gemm_fn):
    """Test non-square matrices."""
    print("\n=== GEMM: Rectangular ===")
    shapes = [(128, 64, 256), (64, 256, 128), (1, 1024, 1), (1024, 1, 1024),
              (73, 129, 47), (1000, 1, 1000), (7, 4096, 13)]
    for M, K, N in shapes:
        A = torch.randn(M, K, dtype=torch.float64, device='cuda')
        B = torch.randn(K, N, dtype=torch.float64, device='cuda')
        try:
            C = gemm_fn(A, B)
            ref = torch.mm(A, B)
            err = rel_err(C, ref)
            check(f"{M}x{K} * {K}x{N}", err < max(M,K,N) * 2.3e-16, f"rel_err={err:.2e}")
        except Exception as e:
            check(f"{M}x{K} * {K}x{N}", False, f"CRASHED: {e}")

def test_gemm_transpose(gemm_fn):
    """Test all 4 transpose cases."""
    print("\n=== GEMM: Transpose Cases ===")
    for N in [64, 65, 256, 1024]:
        A = torch.randn(N, N, dtype=torch.float64, device='cuda')
        B = torch.randn(N, N, dtype=torch.float64, device='cuda')
        cases = [
            ("NN", A, B, torch.mm(A, B)),
            ("TN", A.T, B, torch.mm(A.T.contiguous(), B)),
            ("NT", A, B.T, torch.mm(A, B.T.contiguous())),
            ("TT", A.T, B.T, torch.mm(A.T.contiguous(), B.T.contiguous())),
        ]
        for tag, a, b, ref in cases:
            try:
                C = gemm_fn(a, b, transA=(tag[0]=='T'), transB=(tag[1]=='T'))
                err = rel_err(C, ref)
                check(f"{tag} N={N}", err < N * 2.3e-16, f"rel_err={err:.2e}")
            except TypeError:
                # Function may not accept transA/transB — try with contiguous inputs
                try:
                    C = gemm_fn(a.contiguous(), b.contiguous())
                    err = rel_err(C, ref)
                    check(f"{tag} N={N}", err < N * 2.3e-16, f"rel_err={err:.2e} (no trans flag)")
                except Exception as e:
                    check(f"{tag} N={N}", False, f"CRASHED: {e}")
            except Exception as e:
                check(f"{tag} N={N}", False, f"CRASHED: {e}")

def test_gemm_alpha_beta(gemm_fn):
    """Test alpha/beta scaling."""
    print("\n=== GEMM: Alpha/Beta ===")
    N = 256
    A = torch.randn(N, N, dtype=torch.float64, device='cuda')
    B = torch.randn(N, N, dtype=torch.float64, device='cuda')
    C_init = torch.randn(N, N, dtype=torch.float64, device='cuda')
    cases = [
        (1.0, 0.0, "standard"),
        (2.5, 0.0, "alpha only"),
        (1.0, 1.0, "accumulate"),
        (-1.0, 1.0, "subtract"),
        (0.0, 1.0, "alpha=0"),
        (2.7, -0.3, "arbitrary"),
    ]
    for alpha, beta, tag in cases:
        try:
            ref = alpha * torch.mm(A, B) + beta * C_init
            C = gemm_fn(A, B, alpha=alpha, beta=beta, C=C_init.clone())
            err = rel_err(C, ref)
            check(f"a={alpha} b={beta} ({tag})", err < N * 2.3e-16, f"rel_err={err:.2e}")
        except TypeError:
            check(f"a={alpha} b={beta} ({tag})", False, "function doesn't accept alpha/beta")
        except Exception as e:
            check(f"a={alpha} b={beta} ({tag})", False, f"CRASHED: {e}")

def test_gemm_accuracy(gemm_fn):
    """Test accuracy with challenging inputs."""
    print("\n=== GEMM: Numerical Accuracy ===")
    N = 512
    # Large values
    A = torch.randn(N, N, dtype=torch.float64, device='cuda') * 1e15
    B = torch.randn(N, N, dtype=torch.float64, device='cuda') * 1e15
    C = gemm_fn(A, B)
    ref = torch.mm(A, B)
    err = rel_err(C, ref)
    check("large values (1e15)", err < N * 2.3e-16, f"rel_err={err:.2e}")

    # Small values
    A = torch.randn(N, N, dtype=torch.float64, device='cuda') * 1e-15
    B = torch.randn(N, N, dtype=torch.float64, device='cuda') * 1e-15
    C = gemm_fn(A, B)
    ref = torch.mm(A, B)
    err = rel_err(C, ref)
    check("small values (1e-15)", err < N * 2.3e-16, f"rel_err={err:.2e}")

    # Mixed scale
    A = torch.randn(N, N, dtype=torch.float64, device='cuda')
    A[:N//2] *= 1e10
    A[N//2:] *= 1e-10
    B = torch.randn(N, N, dtype=torch.float64, device='cuda')
    C = gemm_fn(A, B)
    ref = torch.mm(A, B)
    err = rel_err(C, ref)
    check("mixed scale (1e10/1e-10)", err < N * 2.3e-14, f"rel_err={err:.2e}")

    # Identity
    A = torch.eye(N, dtype=torch.float64, device='cuda')
    B = torch.randn(N, N, dtype=torch.float64, device='cuda')
    C = gemm_fn(A, B)
    err = rel_err(C, B)
    check("identity * B = B", err == 0.0 or err < 1e-16, f"rel_err={err:.2e}")

    # Zero
    A = torch.zeros(N, N, dtype=torch.float64, device='cuda')
    B = torch.randn(N, N, dtype=torch.float64, device='cuda')
    C = gemm_fn(A, B)
    check("zero * B = 0", torch.all(C == 0).item(), f"max={C.abs().max().item():.2e}")

def test_chol_sizes(chol_fn):
    """Test Cholesky at all required sizes."""
    print("\n=== Cholesky: Sizes ===")
    sizes = [1, 2, 3, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65,
             127, 128, 129, 255, 256, 257, 511, 512, 513, 1024]
    for N in sizes:
        A = torch.randn(N, N, dtype=torch.float64, device='cuda')
        S = A @ A.T + N * torch.eye(N, dtype=torch.float64, device='cuda')
        try:
            R = chol_fn(S)
            residual = rel_err(R.T @ R, S) if R.shape[0] == N else rel_err(R @ R.T, S)
            check(f"N={N}", residual < N * 2.3e-15, f"residual={residual:.2e}")
        except Exception as e:
            check(f"N={N}", False, f"CRASHED: {e}")

def test_chol_non_spd(chol_fn):
    """Test that non-SPD input gives error, not crash."""
    print("\n=== Cholesky: Error Handling ===")
    N = 64
    A = torch.randn(N, N, dtype=torch.float64, device='cuda')  # not SPD
    try:
        R = chol_fn(A)
        check("non-SPD input", False, "should have raised error")
    except Exception:
        check("non-SPD input", True)

def test_lu_sizes(lu_fn):
    """Test LU at all required sizes."""
    print("\n=== LU: Sizes ===")
    sizes = [1, 2, 3, 7, 16, 17, 63, 64, 65, 128, 129, 256, 512, 1024]
    for N in sizes:
        A = torch.randn(N, N, dtype=torch.float64, device='cuda')
        try:
            L, U, P = lu_fn(A)
            residual = rel_err(P @ L @ U, A)
            check(f"N={N}", residual < N * 2.3e-14, f"residual={residual:.2e}")
        except Exception as e:
            check(f"N={N}", False, f"CRASHED: {e}")

def test_qr_sizes(qr_fn):
    """Test QR at required sizes including rectangular."""
    print("\n=== QR: Sizes ===")
    shapes = [(16, 16), (64, 64), (65, 65), (128, 64), (256, 128),
              (512, 256), (1024, 512), (1024, 1024)]
    for M, N in shapes:
        A = torch.randn(M, N, dtype=torch.float64, device='cuda')
        try:
            Q, R = qr_fn(A)
            residual = rel_err(Q @ R, A)
            orth = torch.norm(Q.T @ Q - torch.eye(N, dtype=torch.float64, device='cuda')).item()
            check(f"{M}x{N} residual", residual < max(M,N) * 2.3e-14, f"residual={residual:.2e}")
            check(f"{M}x{N} orthogonality", orth < max(M,N) * 2.3e-14, f"orth={orth:.2e}")
        except Exception as e:
            check(f"{M}x{N}", False, f"CRASHED: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BLAS Compliance Test")
    parser.add_argument("module", nargs="?", help="Python module path")
    parser.add_argument("function", nargs="?", help="Function name")
    parser.add_argument("--op", default="gemm", help="Operation: gemm|chol|lu|qr|all")
    parser.add_argument("--self-test", action="store_true", help="Test against torch builtins")
    args = parser.parse_args()

    if args.self_test or not args.module:
        # Self-test mode: use torch builtins
        print("=== BLAS Compliance Self-Test (torch reference) ===\n")
        gemm_fn = lambda A, B, **kw: torch.mm(A, B)
        chol_fn = lambda A: torch.linalg.cholesky(A)
        lu_fn = lambda A: torch.linalg.lu(A)
        qr_fn = lambda A: torch.linalg.qr(A)
    else:
        spec = importlib.util.spec_from_file_location("mod", args.module)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        fn = getattr(mod, args.function)
        gemm_fn = fn
        chol_fn = fn
        lu_fn = fn
        qr_fn = fn

    op = args.op.lower()
    if op in ("gemm", "all"):
        test_gemm_sizes(gemm_fn)
        test_gemm_rectangular(gemm_fn)
        test_gemm_transpose(gemm_fn)
        test_gemm_alpha_beta(gemm_fn)
        test_gemm_accuracy(gemm_fn)
    if op in ("chol", "cholesky", "all"):
        test_chol_sizes(chol_fn)
        test_chol_non_spd(chol_fn)
    if op in ("lu", "all"):
        test_lu_sizes(lu_fn)
    if op in ("qr", "all"):
        test_qr_sizes(qr_fn)

    print(f"\n{'='*60}")
    print(f"RESULTS: {PASS} passed, {FAIL} failed out of {PASS+FAIL} tests")
    print(f"{'='*60}")
    sys.exit(1 if FAIL > 0 else 0)
