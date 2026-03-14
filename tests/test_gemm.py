# Copyright (c) 2026 Darrell Thomas. MIT License. See LICENSE file.

"""Test BF16 GEMM kernel against PyTorch reference."""

import torch
import sys


def test_gemm(M, K, N, label=""):
    """Compare custom GEMM output against torch.mm reference."""
    torch.manual_seed(42)
    device = "cuda:0"

    A = torch.randn(M, K, device=device, dtype=torch.bfloat16)
    B = torch.randn(K, N, device=device, dtype=torch.bfloat16)

    # Reference: BF16 matmul (same precision path as our kernel)
    ref = torch.mm(A, B)

    from blackwell_kernels import bf16_gemm
    out = bf16_gemm(A, B)

    # BF16 MMA accumulates in FP32 then converts back to BF16.
    # Tolerance must account for BF16 rounding at each step.
    max_err = (out.float() - ref.float()).abs().max().item()
    mean_err = (out.float() - ref.float()).abs().mean().item()

    # Relative tolerance: scale with K (more accumulations = more error)
    # BF16 has ~7.8e-3 relative precision, sqrt(K) accumulation error
    rel_tol = 0.05  # 5% relative — generous for BF16
    ref_norm = ref.float().abs().mean().item()
    rel_err = mean_err / (ref_norm + 1e-8)

    passed = rel_err < rel_tol
    status = "PASS" if passed else "FAIL"
    desc = f" ({label})" if label else ""
    print(f"{status}: bf16_gemm M={M} K={K} N={N}{desc}  "
          f"max_err={max_err:.4f} mean_err={mean_err:.6f} rel_err={rel_err:.4f}")

    if not passed:
        print(f"  FAILED: rel_err {rel_err:.4f} > {rel_tol}")
        return False
    return True


def test_fp8_gemm(M, K, N, label=""):
    """Compare FP8 GEMM output against torch.mm reference."""
    torch.manual_seed(42)
    device = "cuda:0"

    A = torch.randn(M, K, device=device, dtype=torch.bfloat16)
    B = torch.randn(K, N, device=device, dtype=torch.bfloat16)

    ref = torch.mm(A, B)

    from blackwell_kernels import fp8_gemm
    out = fp8_gemm(A, B)

    max_err = (out.float() - ref.float()).abs().max().item()
    mean_err = (out.float() - ref.float()).abs().mean().item()

    # FP8 e4m3 has ~0.125 relative precision — much lower than BF16.
    # Use 10% relative tolerance as specified.
    rel_tol = 0.10
    ref_norm = ref.float().abs().mean().item()
    rel_err = mean_err / (ref_norm + 1e-8)

    passed = rel_err < rel_tol
    status = "PASS" if passed else "FAIL"
    desc = f" ({label})" if label else ""
    print(f"{status}: fp8_gemm  M={M} K={K} N={N}{desc}  "
          f"max_err={max_err:.4f} mean_err={mean_err:.6f} rel_err={rel_err:.4f}")

    if not passed:
        print(f"  FAILED: rel_err {rel_err:.4f} > {rel_tol}")
        return False
    return True


def main():
    all_pass = True

    # Basic square
    all_pass &= test_gemm(256, 256, 256, "square small")

    # Rectangular
    all_pass &= test_gemm(512, 256, 128, "rectangular")

    # Large square
    all_pass &= test_gemm(1024, 1024, 1024, "square large")

    # Non-tile-aligned (boundary handling)
    all_pass &= test_gemm(200, 300, 250, "non-aligned")

    # Tall and skinny
    all_pass &= test_gemm(2048, 64, 256, "tall-skinny")

    # Short and wide
    all_pass &= test_gemm(64, 256, 2048, "short-wide")

    # FP8 GEMM tests
    print("\n--- FP8 GEMM ---")
    all_pass &= test_fp8_gemm(256, 256, 256, "square small")
    all_pass &= test_fp8_gemm(512, 256, 128, "rectangular")
    all_pass &= test_fp8_gemm(1024, 1024, 1024, "square large")
    all_pass &= test_fp8_gemm(200, 300, 250, "non-aligned")
    all_pass &= test_fp8_gemm(2048, 64, 256, "tall-skinny")
    all_pass &= test_fp8_gemm(64, 256, 2048, "short-wide")

    if all_pass:
        print("\nAll GEMM tests passed!")
    else:
        print("\nSome GEMM tests FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    main()
