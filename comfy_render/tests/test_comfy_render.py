# Copyright (c) 2026 Darrell Thomas. MIT License. See LICENSE file.
#
# Correctness tests for flash attention kernel vs PyTorch SDPA.
#
# Usage: CUDA_VISIBLE_DEVICES=1 PYTHONPATH=python python3 tests/test_comfy_render.py

import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, "python")
from blackwell_kernels import flash_attention

device = "cuda"
dtype = torch.bfloat16
torch.manual_seed(42)

def check(name, q, k, v, causal=False):
    out = flash_attention(q, k, v, causal=causal)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    # Per-element relative error with safe denominator
    abs_err = (out.float() - ref.float()).abs()
    max_err = abs_err.max().item()
    # Use allclose: atol=0.02, rtol=0.05 (BF16 + online softmax)
    close = torch.allclose(out.float(), ref.float(), atol=0.02, rtol=0.05)
    mean_ref = ref.float().abs().mean().item()
    rel_err = max_err / max(mean_ref, 1e-6)
    status = "PASS" if close else "FAIL"
    print(f"  {name}: {status}  max_err={max_err:.4f}  rel_err={rel_err:.4f}")
    if status == "FAIL":
        print(f"    shape: Q={list(q.shape)} causal={causal}")
        pct_bad = (abs_err > 0.02 + 0.05 * ref.float().abs()).float().mean().item() * 100
        print(f"    elements outside tolerance: {pct_bad:.2f}%")
    return status == "PASS"


def make_qkv(B, H, N, D):
    q = torch.randn(B, H, N, D, device=device, dtype=dtype)
    k = torch.randn(B, H, N, D, device=device, dtype=dtype)
    v = torch.randn(B, H, N, D, device=device, dtype=dtype)
    return q, k, v


def test_head_dims():
    passed = True
    for D in [40, 64, 128]:
        q, k, v = make_qkv(2, 4, 128, D)
        passed &= check(f"D={D} non-causal", q, k, v)
        passed &= check(f"D={D} causal", q, k, v, causal=True)
    return passed


def test_seq_lengths():
    passed = True
    for N in [1, 7, 15, 32, 63, 64, 65, 127, 128, 129, 256, 512, 1024]:
        q, k, v = make_qkv(1, 2, N, 64)
        passed &= check(f"N={N}", q, k, v)
    return passed


def test_batch_heads():
    passed = True
    for B, H in [(1, 1), (1, 8), (2, 4), (4, 16)]:
        q, k, v = make_qkv(B, H, 128, 64)
        passed &= check(f"B={B} H={H}", q, k, v)
    return passed


def test_causal():
    passed = True
    for N in [1, 33, 64, 65, 128]:
        q, k, v = make_qkv(1, 2, N, 64)
        passed &= check(f"causal N={N}", q, k, v, causal=True)
    return passed


def test_large():
    q, k, v = make_qkv(1, 8, 2048, 64)
    passed = check("N=2048 D=64", q, k, v)
    q, k, v = make_qkv(1, 4, 1024, 128)
    passed &= check("N=1024 D=128", q, k, v)
    return passed


if __name__ == "__main__":
    print(f"Testing flash_attention on {torch.cuda.get_device_name()}")
    all_pass = True
    for label, fn in [
        ("Head dimensions", test_head_dims),
        ("Sequence lengths", test_seq_lengths),
        ("Batch/head combos", test_batch_heads),
        ("Causal mask", test_causal),
        ("Large inputs", test_large),
    ]:
        print(f"\n=== {label} ===")
        all_pass &= fn()

    print()
    if all_pass:
        print("All tests passed!")
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)
