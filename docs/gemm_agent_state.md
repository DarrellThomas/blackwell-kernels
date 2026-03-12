# GEMM Kernel — Optimization State

**Last updated:** 2026-03-12 (experiment 38)
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

-----

## Diagnosis

The kernel is **compute-bound** (math_pipe_throttle is the dominant stall). This means the tensor core input FIFO fills in bursts, then starves during load/sync phases. Memory is not the bottleneck.

**0.89x cuBLAS appears to be near the ceiling for the current mma.sync approach.** 38 experiments have explored scheduling, tiling, buffering, and barrier optimization within the current architecture. The remaining gap likely requires either:

1. Better MMA-to-load interleaving (spreading MMAs across time instead of bursting)
2. Larger output tiles per warp (more independent MMAs between sync points — see spatters.ca 3.0→3.1 jump)
3. A fundamentally different instruction path (wgmma if available on sm_120)

-----

## Next Directions to Explore

**Check `04_HARD_WON_LESSONS.md` before attempting anything** — 38 experiments worth of dead ends are documented there with root causes.

1. **Larger per-warp output tiles** — current: 1x16 MMA tiles (16x128). Increasing M tiling to 2x16 (32x128) gives 2x more independent MMAs per K iteration. Spatters.ca saw their biggest late-stage jump (89%→93% peak) from 2x2→4x4 tiling.

2. **Full inner-loop PTX** — the compiler is good but not perfect at interleaving MMA with loads. A single large `asm` block with manual register allocation could spread MMAs across load phases. This is the salykova/spatters approach for the last ~5%.

3. **Investigate wgmma availability on sm_120** — `wgmma.mma_async` (warp-group level, smem→tensor cores directly) may be available on consumer Blackwell. If so, it bypasses the register bottleneck entirely. Needs PTX ISA investigation and `ptxas --gpu-name sm_120` testing.

-----

## References

- [spatters.ca MMA matmul](../docs/reference_spatters_mma_matmul.md) — 93% peak on Ada via tiling + cp.async, most applicable reference
- [math throttle guide](../docs/math_throttle_optimization.md) — diagnosis and strategies for the current stall pattern
- [hard-won lessons](../.claude/04_HARD_WON_LESSONS.md) — empirical constraints, do not contradict
- CUTLASS 3.x: https://github.com/NVIDIA/cutlass (wgmma reference)
