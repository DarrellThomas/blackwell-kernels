# Copyright (c) 2026 Darrell Thomas. MIT License. See LICENSE file.
#
# Correctness tests for fused GroupNorm + Linear kernel vs PyTorch reference.
#
# Usage: CUDA_VISIBLE_DEVICES=1 PYTHONPATH=python python3 tests/test_group_norm_linear.py

import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, "python")
from blackwell_kernels import fused_group_norm_linear

device = "cuda"
dtype = torch.bfloat16
torch.manual_seed(42)


def reference(x, weight, norm_weight, norm_bias, linear_bias, groups):
    """PyTorch two-kernel reference: GroupNorm then Linear."""
    M, C_in = x.shape
    # GroupNorm expects [N, C, *] with C at dim 1
    x_3d = x.unsqueeze(2)  # [M, C, 1]
    gn_out = F.group_norm(x_3d, groups, norm_weight.to(x.dtype), norm_bias.to(x.dtype))
    gn_out = gn_out.squeeze(2)  # [M, C]
    y = F.linear(gn_out, weight, linear_bias.to(x.dtype))
    return y


def check(name, x, weight, norm_weight, norm_bias, linear_bias, groups=32):
    out = fused_group_norm_linear(x, weight, norm_weight, norm_bias, linear_bias, groups)
    ref = reference(x, weight, norm_weight, norm_bias, linear_bias, groups)

    abs_err = (out.float() - ref.float()).abs()
    max_err = abs_err.max().item()
    mean_ref = ref.float().abs().mean().item()
    rel_err = max_err / max(mean_ref, 1e-6)

    # BF16 GEMM tolerance: generous for matmul accumulation errors
    close = torch.allclose(out.float(), ref.float(), atol=0.1, rtol=0.1)
    status = "PASS" if close else "FAIL"
    print(f"  {name}: {status}  max_err={max_err:.4f}  rel_err={rel_err:.4f}")
    if status == "FAIL":
        print(f"    shape: X={list(x.shape)} W={list(weight.shape)} groups={groups}")
        pct_bad = (abs_err > 0.1 + 0.1 * ref.float().abs()).float().mean().item() * 100
        print(f"    elements outside tolerance: {pct_bad:.2f}%")
    return status == "PASS"


def make_inputs(M, C_in, C_out, groups=32):
    x = torch.randn(M, C_in, device=device, dtype=dtype)
    weight = torch.randn(C_out, C_in, device=device, dtype=dtype) * 0.02
    norm_weight = torch.ones(C_in, device=device, dtype=torch.float32) + \
                  torch.randn(C_in, device=device, dtype=torch.float32) * 0.1
    norm_bias = torch.randn(C_in, device=device, dtype=torch.float32) * 0.1
    linear_bias = torch.randn(C_out, device=device, dtype=torch.float32) * 0.1
    return x, weight, norm_weight, norm_bias, linear_bias


def test_diffusion_shapes():
    """Test the primary diffusion model configurations."""
    passed = True
    configs = [
        ("SD1.5",  1024, 320, 320, 32),
        ("SDXL-640",  512, 640, 640, 32),
        ("SDXL-1280", 256, 1280, 1280, 32),
        ("Flux-3072", 256, 3072, 3072, 32),
    ]
    for name, M, C_in, C_out, groups in configs:
        x, w, gw, gb, lb = make_inputs(M, C_in, C_out, groups)
        passed &= check(f"{name} M={M} C={C_in}", x, w, gw, gb, lb, groups)
    return passed


def test_seq_lengths():
    """Test various M (batch*seq) sizes including non-aligned."""
    passed = True
    C = 320
    for M in [1, 7, 15, 32, 63, 64, 65, 127, 128, 129, 256, 512, 1024]:
        x, w, gw, gb, lb = make_inputs(M, C, C)
        passed &= check(f"M={M} C={C}", x, w, gw, gb, lb)
    return passed


def test_non_square():
    """Test C_out != C_in."""
    passed = True
    configs = [
        (128, 320, 640, 32),   # expansion
        (128, 640, 320, 32),   # contraction
        (64, 320, 960, 32),    # 3x for QKV
        (64, 1280, 3840, 32),  # 3x for QKV large
    ]
    for M, C_in, C_out, groups in configs:
        x, w, gw, gb, lb = make_inputs(M, C_in, C_out, groups)
        passed &= check(f"M={M} C_in={C_in} C_out={C_out}", x, w, gw, gb, lb, groups)
    return passed


def test_large():
    """Test large realistic sizes."""
    passed = True
    # SD1.5 full: batch=1, seq=4096, C=320
    x, w, gw, gb, lb = make_inputs(4096, 320, 320)
    passed &= check("SD1.5-full M=4096 C=320", x, w, gw, gb, lb)
    # SDXL: batch=2, seq=1024, C=1280
    x, w, gw, gb, lb = make_inputs(2048, 1280, 1280)
    passed &= check("SDXL-full M=2048 C=1280", x, w, gw, gb, lb)
    return passed


def test_3d_input():
    """Test that 3D input [B, seq, C] works via reshape."""
    B, seq, C = 2, 128, 320
    x = torch.randn(B, seq, C, device=device, dtype=dtype)
    w = torch.randn(C, C, device=device, dtype=dtype) * 0.02
    gw = torch.ones(C, device=device, dtype=torch.float32)
    gb = torch.zeros(C, device=device, dtype=torch.float32)
    lb = torch.zeros(C, device=device, dtype=torch.float32)

    out = fused_group_norm_linear(x, w, gw, gb, lb, 32)
    assert out.shape == (B, seq, C), f"Expected {(B, seq, C)}, got {out.shape}"

    # Also check correctness
    ref = reference(x.reshape(-1, C), w, gw, gb, lb, 32).reshape(B, seq, C)
    close = torch.allclose(out.float(), ref.float(), atol=0.1, rtol=0.1)
    status = "PASS" if close else "FAIL"
    print(f"  3D input [2,128,320]: {status}")
    return status == "PASS"


if __name__ == "__main__":
    print(f"Testing fused_group_norm_linear on {torch.cuda.get_device_name()}")
    all_pass = True
    for label, fn in [
        ("Diffusion shapes", test_diffusion_shapes),
        ("Sequence lengths", test_seq_lengths),
        ("Non-square C_out!=C_in", test_non_square),
        ("Large inputs", test_large),
        ("3D input reshape", test_3d_input),
    ]:
        print(f"\n=== {label} ===")
        all_pass &= fn()

    print()
    if all_pass:
        print("All tests passed!")
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)
