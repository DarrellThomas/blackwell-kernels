# Copyright (c) 2026 Darrell Thomas. MIT License. See LICENSE file.
#
# Correctness tests for GroupNorm + Linear kernel vs PyTorch reference.
#
# Usage: CUDA_VISIBLE_DEVICES=1 PYTHONPATH=python python3 tests/test_group_norm_linear.py

import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, "python")
from blackwell_kernels._C import group_norm_forward, fused_group_norm_linear_forward

device = "cuda"
dtype = torch.bfloat16
torch.manual_seed(42)


def check_gn_only(name, M, C, groups=32):
    """Test standalone GroupNorm vs PyTorch reference."""
    x = torch.randn(M, C, device=device, dtype=dtype)
    gamma = torch.randn(C, device=device, dtype=torch.float32)
    beta = torch.randn(C, device=device, dtype=torch.float32)

    out = group_norm_forward(x, gamma, beta, groups)
    ref = F.group_norm(x.float(), groups, gamma, beta, eps=1e-5).to(dtype)

    abs_err = (out.float() - ref.float()).abs()
    max_err = abs_err.max().item()
    close = torch.allclose(out.float(), ref.float(), atol=0.02, rtol=0.05)
    status = "PASS" if close else "FAIL"
    print(f"  {name}: {status}  max_err={max_err:.5f}")
    if not close:
        pct = (abs_err > 0.02 + 0.05 * ref.float().abs()).float().mean().item() * 100
        print(f"    shape: M={M} C={C} groups={groups}")
        print(f"    elements outside tolerance: {pct:.2f}%")
    return close


def check_gnl(name, M, C_in, C_out, groups=32):
    """Test fused GroupNorm + Linear.

    Two checks:
    1. Exact match: fused output == custom_GroupNorm + F.linear (same GEMM path)
    2. Loose match: fused output vs full PyTorch reference (different GroupNorm -> GEMM amplifies)
    """
    x = torch.randn(M, C_in, device=device, dtype=dtype)
    w = torch.randn(C_out, C_in, device=device, dtype=dtype)
    gamma = torch.randn(C_in, device=device, dtype=torch.float32)
    beta = torch.randn(C_in, device=device, dtype=torch.float32)
    bias = torch.randn(C_out, device=device, dtype=torch.float32)

    out = fused_group_norm_linear_forward(x, w, gamma, beta, bias, groups)

    # Check 1: Exact match with custom GN + same GEMM path (F.linear)
    gn_custom = group_norm_forward(x, gamma, beta, groups)
    ref_exact = F.linear(gn_custom, w, bias.to(dtype))
    exact_err = (out.float() - ref_exact.float()).abs().max().item()
    exact_ok = exact_err < 1e-6

    # Check 2: Loose match vs full PyTorch reference
    gn_ref = F.group_norm(x.float(), groups, gamma, beta, eps=1e-5).to(dtype)
    ref_full = F.linear(gn_ref, w, bias.to(dtype))
    full_err = (out.float() - ref_full.float()).abs()
    max_full_err = full_err.max().item()
    # BF16 GEMM noise scales with sqrt(K) * value_magnitude
    # Use generous tolerance for the full reference comparison
    # BF16 GEMM with K inner products: max error ~ sqrt(K) * eps_bf16 * max_val
    # For K=3072, max_val~60: sqrt(3072)*0.004*60 ≈ 13. Use 10% of this.
    import math
    atol_gemm = max(1.0, 0.02 * math.sqrt(C_in) * 4.0)
    full_ok = torch.allclose(out.float(), ref_full.float(), atol=atol_gemm, rtol=0.1)

    ok = exact_ok and full_ok
    status = "PASS" if ok else "FAIL"
    print(f"  {name}: {status}  exact_err={exact_err:.6f}  ref_err={max_full_err:.3f}")
    if not ok:
        if not exact_ok:
            print(f"    EXACT MISMATCH: custom GN + cuBLAS disagrees with fused")
        if not full_ok:
            pct = (full_err > 0.5 + 0.1 * ref_full.float().abs()).float().mean().item() * 100
            print(f"    REF MISMATCH: {pct:.2f}% outside loose tolerance")
    return ok


def test_group_norm_channels():
    """Test GroupNorm across target channel counts."""
    passed = True
    for C in [320, 640, 1280, 2560, 3072]:
        passed &= check_gn_only(f"GN C={C} M=1024", 1024, C)
    return passed


def test_group_norm_batch_sizes():
    """Test GroupNorm across batch sizes."""
    passed = True
    for M in [1, 7, 64, 256, 1024, 4096]:
        passed &= check_gn_only(f"GN M={M} C=640", M, 640)
    return passed


def test_gnl_sd15():
    """SD1.5 configs."""
    passed = True
    passed &= check_gnl("SD1.5 M=4096 320->320", 4096, 320, 320)
    passed &= check_gnl("SD1.5 M=4096 320->640", 4096, 320, 640)
    passed &= check_gnl("SD1.5 M=1024 640->640", 1024, 640, 640)
    passed &= check_gnl("SD1.5 M=1024 640->1280", 1024, 640, 1280)
    return passed


def test_gnl_sdxl():
    """SDXL configs."""
    passed = True
    passed &= check_gnl("SDXL M=1024 1280->1280", 1024, 1280, 1280)
    passed &= check_gnl("SDXL M=256 2560->2560", 256, 2560, 2560)
    return passed


def test_gnl_flux():
    """Flux configs."""
    passed = True
    passed &= check_gnl("Flux M=1024 3072->3072", 1024, 3072, 3072)
    passed &= check_gnl("Flux M=256 3072->3072", 256, 3072, 3072)
    return passed


def test_gnl_edge_cases():
    """Edge cases: M=1, small M, non-square."""
    passed = True
    passed &= check_gnl("M=1 320->320", 1, 320, 320)
    passed &= check_gnl("M=1 1280->1280", 1, 1280, 1280)
    passed &= check_gnl("M=7 640->320", 7, 640, 320)
    passed &= check_gnl("M=64 320->1280", 64, 320, 1280)
    return passed


if __name__ == "__main__":
    print(f"Testing GroupNorm+Linear on {torch.cuda.get_device_name()}")
    all_pass = True
    for label, fn in [
        ("GroupNorm channels", test_group_norm_channels),
        ("GroupNorm batch sizes", test_group_norm_batch_sizes),
        ("SD1.5 configs", test_gnl_sd15),
        ("SDXL configs", test_gnl_sdxl),
        ("Flux configs", test_gnl_flux),
        ("Edge cases", test_gnl_edge_cases),
    ]:
        print(f"\n=== {label} ===")
        all_pass &= fn()

    print()
    if all_pass:
        print("All tests passed!")
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)
