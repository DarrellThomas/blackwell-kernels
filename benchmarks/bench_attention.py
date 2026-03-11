"""Benchmark flash attention kernel vs PyTorch SDPA."""

import torch
import torch.nn.functional as F
import time


def benchmark_fn(fn, warmup=10, iters=100):
    """Time a function with CUDA synchronization."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return elapsed / iters * 1000  # ms


def main():
    torch.manual_seed(42)
    device = "cuda:0"

    configs = [
        (2, 8, 512, 64),
        (2, 8, 1024, 64),
        (2, 8, 2048, 64),
        (2, 8, 4096, 64),
        (4, 16, 2048, 64),
        (4, 16, 2048, 128),
    ]

    print(f"{'B':>4} {'H':>4} {'N':>6} {'D':>4} | {'SDPA (ms)':>10} {'Custom (ms)':>12} {'Speedup':>8}")
    print("-" * 60)

    for B, H, N, D in configs:
        Q = torch.randn(B, H, N, D, device=device, dtype=torch.bfloat16)
        K = torch.randn(B, H, N, D, device=device, dtype=torch.bfloat16)
        V = torch.randn(B, H, N, D, device=device, dtype=torch.bfloat16)

        # PyTorch SDPA baseline
        sdpa_time = benchmark_fn(
            lambda: F.scaled_dot_product_attention(Q, K, V, is_causal=True)
        )

        # Custom kernel
        try:
            from blackwell_kernels import flash_attn_sm120

            custom_time = benchmark_fn(
                lambda: flash_attn_sm120(Q, K, V, causal=True)
            )
            speedup = sdpa_time / custom_time
            print(f"{B:4d} {H:4d} {N:6d} {D:4d} | {sdpa_time:10.3f} {custom_time:12.3f} {speedup:7.2f}x")
        except Exception as e:
            print(f"{B:4d} {H:4d} {N:6d} {D:4d} | {sdpa_time:10.3f} {'N/A':>12} {'N/A':>8}  ({e})")


if __name__ == "__main__":
    main()
