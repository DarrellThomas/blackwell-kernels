# GEMM Kernel — Optimization State

**Last updated:** 2026-03-12 (experiment 51)
**Goal:** Match or beat cuBLAS on RTX 5090 (sm_120)

-----

## Hardware

- GPU: RTX 5090, sm_120 (consumer Blackwell, `mma.sync` ISA)
- Host: Threadripper PRO 7995WX, 512GB DDR5, Ubuntu 24.04
- CUDA 13 / PyTorch 2.10

-----

## Current Kernel Architecture

**Instruction:** `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` (PTX inline asm)

**Tile config:** BLOCK_M=128, BLOCK_N=128, BLOCK_K=32, 8 warps (256 threads)

**Data path:**
```
Global --[cp.async 16B]--> Shared (XOR swizzle, double-buffered)
  --[ldmatrix_x4 / x2_trans]--> Registers --[mma.sync]--> FP32 accumulators --> BF16 store
```

**Key structural decisions (load-bearing, do not remove):**
- cp.async with double-buffer pipelining
- XOR swizzle for bank conflict elimination
- ldmatrix_x4 for A, ldmatrix_x2_trans for B
- a1/a2 register swap (r0, r2, r1, r3) — empirically required on sm_120
- 2-tile K-loop unroll (halves barrier count, requires exactly 8 warps)
- `launch_bounds(256, 1)` — relaxed register budget for compiler optimization
- ~32 KB smem total, leaving ~96 KB for L1 cache

-----

## Current Best

| Commit  | Duration (us) | vs cuBLAS | SM%  | Top Stall      |
|---------|---------------|-----------|------|----------------|
| 4ff2425 | **1008.9**    | **0.89x** | 80.5 | math throttle  |
| 73a9a6c | 1064.2        | 0.84x     | 76.9 | math throttle  |

Benchmark: M=N=K=4096, BF16, 10 warmup + 100 timed iterations.

Interesting: on 4096×1024×4096, our kernel achieves **1.09x cuBLAS** (209 us vs 226 us).

-----

## Diagnosis

The kernel is **compute-bound** (math_pipe_throttle is the dominant stall at 44-59%). The SASS shows the compiler already does excellent scheduling — HMMA and LDSM are perfectly interleaved in the inner loop.

**0.89x cuBLAS is the ceiling for compiler-generated mma.sync code.** 51 experiments have exhausted all conventional approaches:

### Exhaustively Explored (all regressed or neutral)
- **Tile sizes**: 128×128 (optimal), 128×64, 256×64, 256×128, 128×256 — all worse
- **BLOCK_K**: 32 (optimal), 64 (too much data/load)
- **Warps**: 4 (too few), 8 (optimal), 16 (race condition)
- **Pipeline stages**: 2 (optimal), 3 (occupancy/L1 loss), 4 (too much smem)
- **K-loop unroll**: 1 (too many syncs), 2 (optimal), 3 (conditional overhead)
- **Warp tiling**: 1D (optimal), 2D (bank conflicts from more A loads)
- **B fragment pipelining**: compiler already interleaves optimally
- **A/B preloading**: compiler already does it; full preload (164 regs) disrupts scheduling
- **Loop reordering**: kc↔nt, compiler produces same code
- **Padding/stride**: alignment issues or perf loss
- **Register swap elimination**: different sub-group mapping → worse bank conflicts
- **Partial unroll**: any non-full unroll → catastrophic (0.12x)
- **Compiler flags**: -O2 vs -O3 identical, __builtin_assume no effect
- **wgmma**: not available on sm_120 ("not supported on .target 'sm_120'")

### cuBLAS Comparison
| Metric | Our Best (128×128) | cuBLAS (256×128) |
|--------|-------------------|------------------|
| Duration | 808 us | ~719 us |
| Registers | 123 | 218 |
| math_throttle | 44-59% | 72% |
| wait | 14-19% | 21% |
| barrier | 4-5% | 0% |
| scoreboard | 8-15% | 0% |
| bank_conflicts | 8.6M | 524K |

cuBLAS achieves 0% barrier/scoreboard through CUTLASS 2.x's hand-tuned pipeline management and 218 registers for aggressive preloading. Our kernel has 8.6M bank conflicts on A loads (inherent with BLOCK_K=32, SWIZZLE_BITS=2, only 4 unique patterns).

-----

## Remaining Path to 1.0x

The only unexplored path to close the 11% gap:

**Full inner-loop PTX** — write the entire compute_tile in a single large `asm` block with manual register allocation and instruction scheduling. This is what CUTLASS/cuBLAS effectively does. Benefits:
1. Eliminate the a1/a2 swap MOVs (saves ~16 instructions per K-iteration)
2. Custom interleaving of LDSM and HMMA with exact stall control
3. Explicit B register rotation without compiler interference
4. Potentially better scoreboard management

This is a major undertaking (~500+ lines of inline PTX). The salykova approach for SGEMM provides a template.

-----

## References

- [spatters.ca MMA matmul](../docs/reference_spatters_mma_matmul.md) — 93% peak on Ada via tiling + cp.async, most applicable reference
- [math throttle guide](../docs/math_throttle_optimization.md) — diagnosis and strategies for the current stall pattern
- [hard-won lessons](../.claude/04_HARD_WON_LESSONS.md) — empirical constraints, do not contradict
- CUTLASS 3.x: https://github.com/NVIDIA/cutlass (wgmma reference — NOT available on sm_120)
