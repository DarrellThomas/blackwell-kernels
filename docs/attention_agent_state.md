# Attention Kernel — Optimization State

**Last updated:** 2026-03-12 (experiment 49)
**Goal:** Maximize speedup vs cuDNN SDPA on RTX 5090 (sm_120)

-----

## Current Kernel Architecture

**Instruction:** `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` (PTX inline asm)

**Tile config:** BLOCK_Q=64 (dynamic: 128 when grid>=340 blocks), BLOCK_KV=64, 4 warps (128 threads)

**Data path:**
```
Q: loaded once to registers via ldmatrix_x4, reused across all KV blocks
K: loaded via ldmatrix_x4 (was scalar, upgraded iteration 20)
V: pre-loaded via ldmatrix_x4_trans before PV MMA (compiler hoists into softmax gap)
P: FP32 accumulators → register-only BF16 conversion via pack_bf16x2 (no smem round-trip)
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
- Separate V preload from PV MMA (compiler hoists V loads into softmax gap)

-----

## Current Best

| Commit  | Duration (bench) | Duration (ncu) | vs SDPA | SM%  | Top Stall      |
|---------|------------------|----------------|---------|------|----------------|
| 8976daf | **68 μs**        | 93.7 μs        | **1.78x** | 59.1 | math throttle |

Primary config: B=2 H=8 N=2048 D=64, causal. 10 warmup + 100 timed iterations.
Register usage: 145 regs, 0 spills, 3 blocks/SM, 12 warps.

Dynamic BQ dispatch also gives: N=4096 → 2x faster (1.15x SDPA), D=128 → 7x faster (1.00x SDPA).

-----

## Diagnosis

The kernel is **compute-bound** (math_pipe_throttle ~48% is dominant stall). The tensor cores saturate in bursts during QK^T and PV MMA phases, then starve during softmax.

**C++ optimization space is exhausted.** 49 experiments (8 kept, 41 discarded) explored every axis: scheduling, tiling (BQ=32/64/128, BKV=32/64/96/128), buffering (double/triple/asymmetric), softmax variants, prefetch timing, V preloading, P pre-packing, loop reorders, launch_bounds, compiler hints, causal templating, and output coalescing. The compiler produces near-identical SASS for most C++ restructurings.

SASS analysis confirms the compiler already:
- Interleaves QK^T HMMA with K LDSM loads
- Hoists 3 of 8 V LDSM loads into the softmax gap (after V preload refactor)
- Interleaves the last ~8 exp2f (MUFU.EX2) with the first ~4 PV HMMA

**68 μs bench = 94% of compiler ceiling (64 μs).** The remaining 6% gap is from suboptimal compiler instruction ordering that cannot be controlled from C++.

**Stall breakdown:** math_throttle 48%, wait 17%, scoreboard 12%, barrier 5%, not_selected 2%

**Ceilings:**
- Compiler ceiling: ~64 μs (best achievable with `#pragma unroll` + good scheduling)
- Current: 68 μs bench = 94% of compiler ceiling
- Full PTX ceiling: ~55 μs (hand-written assembly with perfect MMA/load interleaving)
- Hard floor: ~38 μs (tensor math only, unreachable — softmax is irreducible)
- Achievable ceiling: ~53 μs (~70% tensor utilization, accounting for fundamental overheads)

-----

## What Worked (cumulative)

1. Prefetch after QK^T — barrier 6%→3%
2. Unconditional exp in softmax — removed 64 compare+select
3. exp2f softmax — fold LOG2E into Q scale, save 34 MULs/iter
4. Skip mask for unmasked KV blocks — 60+ fewer conditionals
5. ldmatrix_x4 for K loads — fewer instructions per MMA pair
6. Dynamic BLOCK_Q dispatch — BQ=128 when grid large enough
7. Separate V preload from PV MMA — compiler hoists V loads into softmax gap

-----

## What Didn't Work (selected, full list in results/attention.tsv)

- Any tile size other than BQ=64 BKV=64 for primary config (register pressure or grid too small)
- Loop reorders (nc-outer/dc-inner, fused exp2f into PV) — compiler produces identical SASS
- 3-stage pipeline — smem dropped occupancy 3→2 blocks/SM
- Deferred sum shuffles — bench regression, shuffles may aid scheduling
- Pre-pack all P before PV — 16 extra regs increased barrier stalls
- Pre-load V before softmax — 16 extra live regs hurt register pressure
- Asymmetric K/V buffering — extra __syncthreads overhead
- Block index remapping — destroyed L2 locality
- `-O2` flag — identical code
- Template on CAUSAL — compiler handles runtime branch well

-----

## Next Directions (requires architectural changes)

**Check `04_HARD_WON_LESSONS.md` before attempting anything** — 49 experiments worth of dead ends are documented there with root causes.

1. **Full inner-loop PTX** — hand-scheduled assembly to overlap softmax scalar ops with MMA/load from adjacent phases. Target: ~55 μs. This is the primary remaining opportunity: the ~328 non-MMA instructions between QK^T and PV could be overlapped with MMA from adjacent phases using manual scheduling.

2. **FP8 attention** — `mma.sync.aligned.m16n8k32` gives 2x tensor throughput, making softmax overhead proportionally smaller. The softmax gap stays fixed while MMA throughput doubles.

3. **Algorithmic changes** — sigmoid attention or other softmax alternatives that eliminate the sequential dependency between QK^T and PV.

-----

## References

- [math throttle guide](../docs/math_throttle_optimization.md) — diagnosis and strategies
- [hard-won lessons](../.claude/04_HARD_WON_LESSONS.md) — empirical constraints
- [spatters.ca MMA matmul](../docs/reference_spatters_mma_matmul.md) — tiling/scheduling techniques
- [gau-nernst flash attention](../docs/reference_gau_nernst_flash_attention.md) — 94.4% peak on sm_120
