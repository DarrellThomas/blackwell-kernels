# Hard-Won Lessons — sm_120 Kernel Optimization

Things we learned empirically that the profiler won't tell you.
Do not unlearn these. Do not "fix" them. They are correct.

---

## ISA & Register Layout

**sm_120 is NOT datacenter Blackwell.** It uses `mma.sync` (extended Ampere), not `tcgen05`. There is no TMEM hardware. Data path is registers → tensor cores. Do not attempt datacenter Blackwell approaches (FA3, FA4, CUTLASS tcgen05 examples). They will not compile.

**The a1/a2 register swap is required, not a bug.** `ldmatrix_x4` outputs registers in a different order than `mma.sync` expects. You must swap r1 and r2: pass `(r0, r2, r1, r3)` to MMA. This was empirically verified through `test_mma4.cu`. Every correct kernel in this project does this swap. Removing it will produce silently wrong results that pass casual inspection but fail on non-trivial inputs. **Preferred approach:** Use `ldmatrix_x4_mma()` which bakes the swap into the instruction operand order `{%0, %2, %1, %3}` — outputs land directly in MMA order, eliminating MOV instructions. Both the GEMM and attention kernels now use this.

**ldmatrix_x2_trans computes A*B, not A*B^T.** This is correct for P*V multiplication. Do not add a transpose for V — it's already handled by the instruction semantics.

---

## Load-Bearing Optimizations

These optimizations are structural — they're woven into the kernel's architecture. Removing any one of them will cause major regression, not just a small slowdown.

**cp.async (v3):** All global→shared loads use 16-byte `cp.async.cg` with zero-fill on OOB. This was a 2.54x speedup (313 us → 123 us). Reverting to scalar register-mediated copies would be catastrophic.

**Double-buffer pipelining (v4):** K/V tiles alternate between two shared memory buffer slots. Next block prefetches while current block computes. Removing this re-couples load and compute phases.

**XOR swizzle (v5):** Shared memory addressing uses XOR-based swizzle to spread accesses across banks. This eliminated tens of thousands of bank conflicts. Do not remove or "simplify" the swizzle indexing.

**Register-only P→A conversion (v7):** The softmax output P is converted from FP32 accumulators to BF16 MMA A-fragments entirely in registers via warp shuffles. This eliminated the shared memory round-trip that was the dominant source of bank conflicts. Do not reintroduce a shared memory path for P.

---

## Shared Memory Budget

**99 KB max per block** on sm_120 (not 128 KB — CUDA reserves 1 KB per block). The tuning guide is authoritative. Double-buffering roughly doubles shared memory usage. Always calculate before changing tile sizes:

- Attention (BQ=64, BKV=64): 4 × 64 × 64 × 2 = 32 KB/block → 3 blocks = 96 KB
- Attention (BQ=64, BKV=32): 4 × 32 × 64 × 2 = 16 KB/block → but more softmax overhead
- GEMM (64×64, K=32): 2 × 64 × 32 × 2 + 2 × 64 × 32 × 2 = 16 KB/block → 6 blocks = 96 KB
- **4 blocks × 32 KB = 128 KB is exactly at the SM limit** — empirically too tight, 72 μs vs 69 μs baseline

**Static shared memory declarations are limited to 48 KB.** Anything above requires dynamic allocation with `cudaFuncSetAttribute`.

---

## Occupancy vs Register Pressure — The Dominant Axis on sm_120

sm_120 has only 48 warps/SM (vs 64 on datacenter). Each block's register usage directly constrains how many blocks fit per SM. **Occupancy is the single most important optimization axis for compute-bound kernels on sm_120** — more so than per-warp instruction scheduling or tile size.

### The GEMM Breakthrough (0.88x → 0.98x cuBLAS)

The BF16 GEMM went from 0.88x to 0.98x cuBLAS with one change: **smaller tiles + more blocks/SM**.

| Config | Tiles | Regs | Blocks/SM | Warps/SM | cuBLAS |
|--------|-------|------|-----------|----------|--------|
| 128×128, 8 warps | 4×4 MMA (32 MMAs) | ~125 | 2 | 16 | 0.88x |
| 64×64, 4 warps | 2×4 MMA (16 MMAs) | 80 | 6 | 24 | **0.98x** |

The key insight: **24 warps with 16 MMAs each >> 16 warps with 32 MMAs each**. The hardware warp scheduler benefits more from having warps to choose from than from having more independent instructions within a single warp. The mechanism: with 6 warps per sub-partition, when one warp stalls on math_pipe_throttle, 5 others are available.

**The recipe:**
1. `__launch_bounds__(threads, min_blocks)` — forces compiler to cap registers
2. Target 16KB smem/block × 6 blocks = 96KB (well within 128KB SM limit)
3. 80 regs/thread with 0 spills was the sweet spot
4. Non-volatile MMA (`asm` not `asm volatile`) lets compiler reorder freely
5. `ldmatrix_x4_mma` bakes in a1/a2 swap, eliminates MOV instructions
6. Stream B fragments (load per-tile, not preload-all) to minimize live registers

### Why This Does NOT Work for Attention

The GEMM occupancy strategy was tried on the attention kernel (experiments 50-53) and **failed**:

| Experiment | Change | Result | Why |
|------------|--------|--------|-----|
| BKV=32, launch_bounds 6 | Halve smem → 6 blocks | 80 regs + 16B spill | 85 reg cap too tight |
| BKV=32, launch_bounds 5 | 16KB/block → 5 blocks | 77 μs (-13%) | 2× softmax overhead |
| BKV=64, launch_bounds 4 | 32KB/block → 4 blocks | 72 μs (-5%) | 128KB = SM limit, too tight |
| NV MMA only, launch_bounds 3 | Keep 3 blocks | 69 μs (neutral) | Same SASS, compiler already good |

**Root cause:** GEMM's inner loop is pure MMA — halving tiles just means 2× identical iterations. Attention's inner loop has **softmax** (sequential scalar exp2f + reductions + rescaling) between QK^T and PV. Halving BKV means 2× softmax passes. The softmax overhead grows faster than occupancy can compensate.

**Rule of thumb:** Occupancy-first works for **compute-only** inner loops (GEMM, convolution). For kernels with **irreducible sequential phases** (softmax in attention), the existing 3 blocks/SM (12 warps) is already near-optimal — more blocks reduce per-iteration MMA work without proportionally reducing the fixed overhead.

---

## Benchmark Noise

CUDA kernel benchmarks have noise. The benchmark harness uses 10 warmup + 100 timed iterations and reports mean. Differences under ~2% are within noise. Do not keep or discard based on sub-2% changes — run again to confirm.

GPU 0 has ComfyUI running. Always use `CUDA_VISIBLE_DEVICES=1`. Forgetting this doesn't just use the wrong GPU — it can cause contention and wildly inconsistent benchmark numbers.

---

## PTX-First / Manual Scheduling

**B fragment double-buffering does NOT help on sm_120.** Attempted loading B[nt+1] while MMA executes with B[nt] using separate register sets and `asm volatile` ordering. Result: 0.68x cuBLAS (regression from 0.89x). Root cause: 139 registers (vs 123), 0 spills — the extra registers reduce occupancy, and the `asm volatile` barriers prevent the compiler from reordering instructions freely. The hardware warp scheduler (with 8 warps / 256 threads) already overlaps ldmatrix and mma.sync across warps.

**The compiler's `#pragma unroll` scheduling is surprisingly good.** On sm_120, nvcc with `-O2` and `#pragma unroll` produces near-optimal interleaving of MMA and load instructions. Simple PTX double-buffering tricks that work on Ampere (salykova) add overhead here because: (1) more registers → lower occupancy, (2) `asm volatile` constraints fight the compiler's scheduler, (3) the warp scheduler already hides latency across 8 warps.

**Non-volatile MMA (`asm` not `asm volatile`) is always correct and sometimes helps.** For GEMM, non-volatile MMA was part of the 0.88x→0.98x cuBLAS improvement (combined with occupancy changes). For attention, it's performance-neutral but produces cleaner code — the compiler already schedules volatile MMAs well when given `#pragma unroll`. **Always use non-volatile MMA** (`mma_m16n8k16_bf16_nv`). ldmatrix and cp_async must remain volatile for correctness across sync boundaries.

**To actually beat the compiler, you need a radical approach.** A single large `asm` block with full manual register allocation and instruction scheduling — not incremental PTX additions to a compiler-managed kernel. The salykova approach (writing the entire inner loop in PTX with explicit register naming) is the right idea, but requires committing fully to assembly for the hot path. For the attention kernel specifically, the target is overlapping the ~328 non-MMA softmax instructions between QK^T and PV with MMA/load operations from adjacent phases — something the compiler cannot do because it respects the sequential phase boundaries.

---

## Current Performance (2026-03-13)

### BF16 GEMM
- **0.98x cuBLAS** at 4096³ (best single config)
- **1.23x cuBLAS** at 4096×1024×4096 (beats cuBLAS on non-square)
- 64×64 tiles, 4 warps, 80 regs, 6 blocks/SM, non-volatile MMA, streaming B

### Flash Attention v2
- **1.76x cuDNN SDPA** at B=2 H=8 N=2048 D=64 causal (69 μs)
- **1.35x SDPA** at N=512, **1.15x** at N=4096, **1.00x** at D=128
- 145 regs, 3 blocks/SM, 12 warps. At 94% of compiler ceiling.
- Code cleanup: non-volatile MMA + ldmatrix_x4_mma (performance-neutral, cleaner)

### Reference Points
**gau-nernst** wrote custom flash attention for RTX 5090 and achieved 94.4% of 209.5 TFLOPS peak using BLOCK_Q=128. This is our feasibility proof — we know sm_120 can get there.

**cuDNN SDPA** achieves 97.2% of peak. We beat it on D=64 configs (1.35-1.76x) but match at D=128 (1.00x).
