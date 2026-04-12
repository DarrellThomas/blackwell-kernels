# Copyright (c) 2026 Darrell Thomas. MIT License. See LICENSE file.
#
# Benchmark fused GroupNorm+Linear vs PyTorch two-kernel reference.
#
# Usage: CUDA_VISIBLE_DEVICES=1 PYTHONPATH=python python3 benchmarks/bench_group_norm_linear.py

import sys
import time
import torch
import torch.nn.functional as F

sys.path.insert(0, "python")
from blackwell_kernels import fused_group_norm_linear

device = "cuda"
dtype = torch.bfloat16

WARMUP = 20
TIMED = 100


def bench(fn, warmup=WARMUP, timed=TIMED):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    # Use CUDA events for precise GPU timing (eliminates Python loop overhead)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(timed):
        fn()
    end_event.record()
    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event) / timed


def reference_fn(x_2d, weight, norm_weight, norm_bias, linear_bias, groups):
    """PyTorch two-kernel: GroupNorm then Linear."""
    x_3d = x_2d.unsqueeze(2)
    gn_out = F.group_norm(x_3d, groups, norm_weight, norm_bias).squeeze(2)
    return F.linear(gn_out, weight, linear_bias)


def main():
    groups = 32

    # Primary benchmark config: SD1.5 style
    M, C_in, C_out = 4096, 320, 320
    x = torch.randn(M, C_in, device=device, dtype=dtype)
    w = torch.randn(C_out, C_in, device=device, dtype=dtype) * 0.02
    gw = torch.ones(C_in, device=device, dtype=dtype)
    gb = torch.zeros(C_in, device=device, dtype=dtype)
    lb = torch.zeros(C_out, device=device, dtype=dtype)

    # Both get BF16 params (fair comparison, matches real-world usage)
    custom_ms = bench(lambda: fused_group_norm_linear(x, w, gw, gb, lb, groups))
    ref_ms = bench(lambda: reference_fn(x, w, gw, gb, lb, groups))
    vs_ref = ref_ms / custom_ms

    # Required output for eval.sh
    print(f"primary_custom_ms: {custom_ms:.3f}")
    print(f"primary_vs_ref: {vs_ref:.2f}x")

    # All diffusion configs
    configs = [
        # name, M, C_in, C_out
        ("SD1.5  M=4096 C=320",    4096, 320, 320),
        ("SD1.5  M=1024 C=320",    1024, 320, 320),
        ("SDXL   M=1024 C=640",    1024, 640, 640),
        ("SDXL   M=1024 C=1280",   1024, 1280, 1280),
        ("SDXL   M=256  C=1280",    256, 1280, 1280),
        ("Flux   M=1024 C=3072",   1024, 3072, 3072),
        ("Flux   M=256  C=3072",    256, 3072, 3072),
        ("QKV    M=1024 C=320→960", 1024, 320, 960),
    ]

    print(f"\n{'Config':<30} {'Custom (ms)':<14} {'Ref (ms)':<14} {'Speedup'}")
    print(f"{'─'*70}")

    for name, m, cin, cout in configs:
        x = torch.randn(m, cin, device=device, dtype=dtype)
        w = torch.randn(cout, cin, device=device, dtype=dtype) * 0.02
        gw = torch.ones(cin, device=device, dtype=dtype)
        gb = torch.zeros(cin, device=device, dtype=dtype)
        lb = torch.zeros(cout, device=device, dtype=dtype)

        c_ms = bench(lambda: fused_group_norm_linear(x, w, gw, gb, lb, groups))
        r_ms = bench(lambda: reference_fn(x, w, gw, gb, lb, groups))
        ratio = r_ms / c_ms
        print(f"{name:<30} {c_ms:<14.3f} {r_ms:<14.3f} {ratio:.2f}x")


if __name__ == "__main__":
    print(f"Benchmarking fused GroupNorm+Linear on {torch.cuda.get_device_name()}")
    print()
    main()
