# comfy_render — Fused GroupNorm+Linear State (Job #66)

**Last updated:** 2026-04-12
**Status:** stuck_needs_research
**Goal:** >1.2x vs PyTorch GroupNorm+Linear across SD1.5/SDXL/Flux shapes

## Current Performance

| Config | Custom (ms) | Ref (ms) | Speedup |
|--------|-------------|----------|---------|
| SD1.5 M=4096 C=320 | 0.084 | 0.084 | 1.00x |
| SD1.5 M=1024 C=320 | 0.031 | 0.030 | 0.98x |
| SDXL M=1024 C=1280 | 0.054 | 0.048 | 0.89x |
| Flux M=1024 C=3072 | 0.175 | 0.170 | 0.97x |

## What's Built
- Correct fused GroupNorm+Linear with MMA m16n8k16
- TILE_M=128, TILE_N=64, K_STEP=32, double-buffered B, non-volatile MMA
- Hybrid dispatch: fused GEMM for bandwidth-bound, cuBLAS for compute-bound
- All shapes correct (max_err < 0.008 for C=320)

## Blocker
Custom GEMM is 2-5x slower than cuBLAS for C≥640. Fusion saves ~5μs of intermediate
bandwidth, but GEMM gap overwhelms it. Need CUTLASS-based fused GEMM to match cuBLAS
throughput while keeping normalization fusion.

## Experiments
1. v1: 64x64, K=16, no pipeline → 0.11-0.84x
2. v2: K=32, vectorized A, double-buffer B → 0.53-2.00x (varied)
3. v3: 128x64 tiles → 1.00x for primary
4. Hybrid cuBLAS dispatch → 0.87-0.98x consistent
