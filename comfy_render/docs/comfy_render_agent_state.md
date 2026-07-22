# Flash Attention — Optimization State (Job #65)

**Last updated:** 2026-04-22
**Status:** complete (halted — target met)
**Goal:** Close D=64 non-causal gap vs cuDNN SDPA

-----

## Final Performance (GPU 1, clean, 5-trial median)

| Config | Custom (ms) | cuDNN (ms) | Ratio |
|--------|-------------|------------|-------|
| B4H32 N1024 D64 non-causal (primary) | 0.390 | 0.385 | 0.988x |
| B1H8 N2048 D64 non-causal | 0.138 | 0.122 | 0.884x |
| B1H8 N4096 D64 non-causal | 0.513 | 0.443 | 0.865x |
| B4H32 N1024 D64 causal | 0.234 | 0.250 | 1.077x |
| B2H16 N512 D128 non-causal | 0.076 | 0.066 | 0.870x |
| B1H12 N1024 D40 non-causal | 0.068 | 0.078 | 1.147x |

-----

## Changes Made (exp11)

1. **Dispatch d64<false> for non-causal** (was bc128). Bc=64 with K double-buffer
   replaces Bc=128 single-buffer. 166→127 regs/thread, 25%→33% occupancy.
2. **exp2f with LOG2E folded into Q scale.** Direct MUFU.EX2 instead of __expf.
   Only applied to D=64 kernel (D=128/D=40 unchanged).
3. **Pipeline restructure.** V+K[next] prefetch before QK^T, softmax before V wait.
   Gives loads more time to complete under compute.

-----

## Remaining Gaps & Root Causes

- **Small-batch D=64 NC (0.87-0.88x):** Low SM utilization (256 blocks / 680 max = 38%).
  Would need split-K or persistent CTAs. Major rewrite.
- **D=128 NC (0.87x):** Same register/occupancy bottleneck. Could apply exp2f.
- **Primary D=64 NC (0.988x):** Math pipe throttle at 33% occupancy.
  Register pressure (127 regs) prevents >4 blocks/SM.

-----

## Dead Ends (This Session)

1. **__launch_bounds__(128, 5):** Forced 5 blocks/SM → register spills → 0.91-0.94x. Worse.
2. **Grid dimension swap (N/Br, B*H):** Destroyed causal performance (load imbalance). Reverted.
3. **bc128 kernel (Bc=128):** 166 regs, 25% occupancy, single K buffer. Worse than d64<false>.

-----

# Fused GroupNorm+Linear — Optimization State (Job #66)

**Last updated:** 2026-04-22
**Status:** hw_optimizing
**Goal:** >1.2x vs PyTorch GroupNorm+Linear across SD1.5/SDXL/Flux shapes

-----

## Current Performance

| Config | Custom (ms) | Ref (ms) | Speedup |
|--------|-------------|----------|---------|
| SD1.5 M=4096 C=320->320 (primary) | 0.048 | 0.154 | 3.19x |
| SD1.5 M=1024 C=320->320 | 0.023 | 0.048 | 2.06x |
| SD1.5 M=1024 C=640->640 | 0.030 | 0.066 | 2.21x |
| SDXL M=1024 C=1280->1280 | 0.052 | 0.087 | 1.68x |
| SDXL M=256 C=2560->2560 | 0.052 | 0.069 | 1.31x |
| Flux M=1024 C=3072->3072 | 0.236 | 0.280 | 1.18x |
| Flux M=256 C=3072->3072 | 0.092 | 0.105 | 1.14x |

GroupNorm-only: 2.2-5.5x vs PyTorch's F.group_norm

-----

## Architecture

**Strategy: Custom GroupNorm kernel + cuBLAS GEMM (torch::linear)**

The prior approach (hand-rolled MMA GEMM with on-the-fly normalization) was
2-5x slower than cuBLAS for C>=640, making fusion net-negative. Dropped the
custom GEMM entirely.

**GroupNorm kernel:**
- 1 block per row, 256 threads (8 warps)
- Warp-shuffle per-group reductions (no cross-warp sync for stats)
- 128-bit vectorized loads/stores (int4)
- Adaptive dispatch:
  - M >= 512: no-cache variant (stats from global, normalize from L2). One __syncthreads.
  - M < 512: cached variant (row in smem). Two __syncthreads.
- Smem: 256 bytes (no-cache) or C*2 + 256 bytes (cached)

**Linear:** torch::linear → cuBLAS, unbeatable for compute-bound GEMMs.

-----

## Experiments

| # | Description | Primary Speedup | Notes |
|---|-------------|-----------------|-------|
| Prior worker | Hand-rolled MMA GEMM | 0.87-1.00x | GEMM 2-5x slower than cuBLAS for C>=640 |
| v1 | Custom GN + cuBLAS (torch::addmm) | 2.70x | All positive except M=256 regressions |
| v1.1 | Switch to torch::linear | 2.33x | Fixed M=256 regressions |
| v2 | Adaptive dispatch (nocache/cached) | 3.19x | Best of both worlds for all M |

-----

## Key Architectural Decisions

1. **Dropped hand-rolled GEMM.** cuBLAS is unbeatable for large C (640+). The ~5us fusion
   saving from eliminating the intermediate write-read doesn't justify a 2-5x GEMM regression.
2. **torch::linear over torch::addmm.** F.linear uses a more optimized cuBLAS code path.
3. **Adaptive dispatch at M=512.** No-cache variant wins for large M (warm L2, one sync).
   Cached variant wins for small M (cold L2, avoid re-reading from global).
4. **Per-group warp reductions.** Each warp handles ceil(groups/8) groups with full warp
   shuffle — no shared memory cross-warp reduce needed.

-----

## Remaining Gaps

- Flux M=256 C=3072: 1.14x (below 1.2x target, GEMM-dominated, at theoretical limit)
- Flux M=1024 C=3072: 1.18x (borderline 1.2x, confirmed ~1.20x on some runs)
- Would need cuBLAS prologue fusion or cuBLASDx to improve further — both experimental on sm_120

-----

## Dead Ends

1. **Hand-rolled MMA GEMM with on-the-fly normalization** — 2-5x slower than cuBLAS for C>=640
2. **No-cache kernel for all M** — worse for M<512 with large C (cold L2 re-reads)
3. **torch::addmm** — less optimized cuBLAS path vs torch::linear
