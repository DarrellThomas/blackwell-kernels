# Attention Kernel — Optimization State

**Last updated:** 2026-03-12 (experiment 37)
**Goal:** Maximize speedup vs cuDNN SDPA on RTX 5090 (sm_120)

-----

## Current Kernel Architecture

**Instruction:** `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` (PTX inline asm)

**Tile config:** BLOCK_Q=64 (dynamic: 128 when grid>=340 blocks), BLOCK_KV=64, 4 warps (128 threads)

**Data path:**
```
Q: loaded once to registers via ldmatrix_x4, reused across all KV blocks
K: loaded via ldmatrix_x4 (was scalar, upgraded iteration 20)
V: loaded via ldmatrix_x2_trans (computes A*B directly, no transpose)
P: FP32 accumulators → register-only BF16 conversion via warp shuffles (no smem round-trip)
```

**Key structural decisions (load-bearing, do not remove):**
- cp.async with double-buffer pipelining for K/V
- XOR swizzle for bank conflict elimination
- Register-only P→A conversion (eliminated dominant bank conflicts)
- a1/a2 register swap (r0, r2, r1, r3) — empirically required on sm_120
- Online softmax with 4-thread group shuffles (XOR masks 1, 2)
- exp2f softmax with LOG2E folded into Q scale
- Skip mask optimization for fully unmasked KV blocks
- Dynamic BLOCK_Q dispatch (128 for large grids, 64 otherwise)
- Prefetch next K/V after QK^T (barrier reduced 6%→3%)

-----

## Current Best

| Commit  | Duration (us) | vs SDPA | SM%  | Top Stall      |
|---------|---------------|---------|------|----------------|
| 2cb22e7 | 93.8          | **1.76x** | 59.8 | math throttle |
| dynamic | 93.5          | **1.76x** | 59.7 | math throttle |

Primary config: B=2 H=8 N=2048 D=64, causal. 10 warmup + 100 timed iterations.

Dynamic BQ dispatch also gives: N=4096 → 2x faster (1.15x SDPA), D=128 → 7x faster (1.00x SDPA).

-----

## Diagnosis

The kernel is **compute-bound** (math_pipe_throttle ~48% is dominant stall). The tensor cores saturate in bursts during QK^T and PV MMA phases, then starve during softmax.

**1.76x SDPA is near the ceiling for the current architecture.** 37 experiments explored scheduling, tiling, buffering, softmax, and dispatch strategies. The remaining gap to theoretical limits is dominated by irreducible softmax overhead between the two MMA phases.

**Ceilings:**
- Compiler ceiling: ~64 us (best achievable with `#pragma unroll` + good scheduling)
- Current: 93.5 us = 68% of compiler ceiling
- Full PTX ceiling: ~55 us (hand-written assembly with perfect MMA/load interleaving)
- Hard floor: ~38 us (tensor math only, unreachable — softmax is irreducible)

-----

## What Worked (cumulative)

1. Prefetch after QK^T — barrier 6%→3%
2. Unconditional exp in softmax — removed 64 compare+select
3. exp2f softmax — fold LOG2E into Q scale, save 34 MULs/iter
4. Skip mask for unmasked KV blocks — 60+ fewer conditionals
5. ldmatrix_x4 for K loads — fewer instructions per MMA pair
6. Dynamic BLOCK_Q dispatch — BQ=128 when grid large enough

-----

## Next Directions to Explore

**Check `04_HARD_WON_LESSONS.md` before attempting anything** — 37 experiments worth of dead ends are documented there with root causes.

1. **Full inner-loop PTX** — hand-scheduled assembly to overlap softmax scalar ops with MMA/load from adjacent phases. Target: ~55 us.
2. **FP8 attention** — `mma.sync.aligned.m16n8k32` gives 2x tensor throughput, making softmax overhead proportionally smaller.
3. **Algorithmic changes** — sigmoid attention or other softmax alternatives that eliminate the sequential dependency between QK^T and PV.

-----

## References

- [math throttle guide](../docs/math_throttle_optimization.md) — diagnosis and strategies
- [hard-won lessons](../.claude/04_HARD_WON_LESSONS.md) — empirical constraints
- [spatters.ca MMA matmul](../docs/reference_spatters_mma_matmul.md) — tiling/scheduling techniques
- [gau-nernst flash attention](../docs/reference_gau_nernst_flash_attention.md) — 94.4% peak on sm_120
