# NVIDIA Blackwell GPU Tuning Guide

Source: https://docs.nvidia.com/cuda/blackwell-tuning-guide/
Fetched: 2026-03-27

## Architecture Overview

Blackwell maintains the CUDA programming model from Ampere/Hopper while
introducing performance enhancements.

## SM Specifications — Compute Capability 12.0 (sm_120)

| Specification | CC 12.0 (sm_120) | CC 10.0 (comparison) |
|---|---|---|
| Max concurrent warps per SM | **48** | 64 |
| Register file per SM | 64K 32-bit registers | 64K |
| Max registers per thread | 255 | 255 |
| Max thread blocks per SM | 32 | 32 |
| Shared Memory per SM | **128 KB** | 228 KB |
| Max shared memory per block | **99 KB** | 227 KB |

**Key difference from CC 10.0:** sm_120 has fewer warps (48 vs 64) and less
shared memory (128 KB vs 228 KB). This is the consumer Blackwell constraint.

## Memory System

### L1/L2 Cache
- Unified L1/Texture cache (like Ampere)
- L2 cache persistence control via CUDA API
- Acts as coalescing buffer for warp memory requests
- Runtime carveout: `cudaFuncSetAttribute()` with `cudaFuncAttributePreferredSharedMemoryCarveout`
- Supported carveout sizes: 0, 8, 16, 32, 64, 100, 132, 164, 196, 228 KB

### Empirical Findings (from factory experiments)
- **99 KB is the real max shared memory per block** (not 128 KB)
- **48 warps/SM** (not 64 — that's CC 10.0 datacenter Blackwell)
- L1 cache is critically important: 3-stage pipeline (72KB smem) kills L1 and regresses
- 32 KB shared memory sweet spot preserves L1 for data reuse

## Thread Block Clusters

- Maximum portable cluster size: 8
- Distributed Shared Memory for inter-block communication
- Access patterns: follow 32-byte alignment for coalescing

## Optimization Priorities

### High Priority
1. Parallelize sequential code
2. Minimize host-device transfers
3. Maximize device utilization via launch configuration
4. Ensure coalesced global memory access patterns
5. Minimize redundant global memory accesses
6. Avoid warp divergence

### Occupancy on sm_120

For **GEMM (pure compute):** Occupancy-first works.
- Smaller tiles + more blocks/SM beats larger tiles + fewer blocks
- 64x64 tiles, 6 blocks/SM, 80 registers = optimal for BF16 GEMM

For **Attention (softmax overhead):** Occupancy-first FAILS.
- Halving BKV doubles softmax passes
- Occupancy gain cannot compensate for sequential overhead
- Confirmed across experiments 50-53 in attention project

### Register Pressure
- 64K registers / SM, 48 warps max
- At 4 blocks of 128 threads (16 warps): 64K/512 = 128 regs/thread max
- At 6 blocks of 128 threads (24 warps): 64K/768 = 83 regs/thread max
- Use `launch_bounds` to control register allocation
- Register spills to local memory are catastrophic for performance

### Shared Memory Budget
- 99 KB max per block (empirically confirmed)
- Exceeding ~48-64 KB shared memory starts evicting L1 cache
- Double-buffer pipelining needs 2x tile size in shared memory
- Triple-buffer/3-stage pipeline consistently failed across projects (L1 loss)
