# Hard-Won Lessons — sm_120a Kernel Optimization

Things we learned empirically that the profiler won't tell you.
Do not unlearn these. Do not "fix" them. They are correct.

---

## CUDA 13.2 / sm_120a Compilation Target

**sm_120a is the correct compilation target for RTX 5090.** The RTX 5090 IS sm_120a
hardware. Always compile with `-gencode arch=compute_120a,code=sm_120a`. Plain
`compute_120,code=sm_120` compiles and runs but misses accelerated features.

**FP8 MMA accumulator throughput depends on the `a` suffix:**
- `mma.sync m16n8k32.f32.e4m3.e4m3.f32` (FP32 accumulators) — runs at **50% throughput** on sm_120a
- `mma.sync m16n8k32.f16.e4m3.e4m3.f16` (FP16 accumulators) — runs at **full speed**
- This means our current FP8 kernels (which use FP32 accumulators for precision) leave
  half the FP8 tensor core throughput on the table. Switching to FP16 accumulators
  doubles FP8 MMA throughput but requires careful numerical analysis for each kernel.

**New instructions available on sm_120a (not yet explored):**
- `ldmatrix.m16n16.b8` — native 8-bit data loading, could eliminate FP8 conversion overhead
  entirely for kernels that accept FP8 inputs. This is the path to solving the BF16-to-FP8
  conversion bottleneck identified in attention FP8 experiments (448 ALU/KV block).
- Block-scaled MMA (MXFP8) — microscaling FP8 with per-block scale factors. Available
  but not yet explored. Could enable higher accuracy than plain FP8 e4m3.

**TORCH_CUDA_ARCH_LIST must also use the `a` suffix:** Set `"12.0a"` (not `"12.0"`)
in build scripts and environment variables for PyTorch extension builds.

---

## ISA & Register Layout

**sm_120 is NOT datacenter Blackwell.** It uses `mma.sync` (extended Ampere), not `tcgen05`. There is no TMEM hardware. Data path is registers → tensor cores. Do not attempt datacenter Blackwell approaches (FA3, FA4, CUTLASS tcgen05 examples). They will not compile.

**The a1/a2 register swap is required, not a bug.** `ldmatrix_x4` outputs registers in a different order than `mma.sync` expects. You must swap r1 and r2: pass `(r0, r2, r1, r3)` to MMA. This was empirically verified through `test_mma4.cu`. Every correct kernel in this project does this swap. Removing it will produce silently wrong results that pass casual inspection but fail on non-trivial inputs. **Preferred approach:** Use `ldmatrix_x4_mma()` which bakes the swap into the instruction operand order `{%0, %2, %1, %3}` — outputs land directly in MMA order, eliminating MOV instructions. Both the GEMM and attention kernels now use this.

**ldmatrix_x2_trans computes A*B, not A*B^T.** This is correct for P*V multiplication. Do not add a transpose for V — it's already handled by the instruction semantics.

**TF32 MMA m16n8k8 B fragment has diagonal broadcast on sm_120.** The B operand broadcasts each register value to two positions: B[k, n] AND B[k+1, n+1]. This makes it unsuitable for general GEMM without decomposition. The output layout also differs from BF16: d0→(gid, tid*2), d1→(gid+8, tid*2), d2→(gid, tid*2+1), d3→(gid+8, tid*2+1) — adjacent rows, not adjacent columns. Use BF16 MMA (m16n8k16) with FP32→BF16 conversion instead, or decompose into two MMA calls. Verified with 25+ empirical tests on RTX 5090 (2026-03-14).

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

GPU 1 is the air-cooled kernel dev GPU. Always use `CUDA_VISIBLE_DEVICES=1`. GPU 0 (water-cooled) is for heavy training. ComfyUI runs intermittently on either — check `nvidia-smi` if benchmark numbers look inconsistent.

---

## PTX-First / Manual Scheduling

**B fragment double-buffering does NOT help on sm_120.** Attempted loading B[nt+1] while MMA executes with B[nt] using separate register sets and `asm volatile` ordering. Result: 0.68x cuBLAS (regression from 0.89x). Root cause: 139 registers (vs 123), 0 spills — the extra registers reduce occupancy, and the `asm volatile` barriers prevent the compiler from reordering instructions freely. The hardware warp scheduler (with 8 warps / 256 threads) already overlaps ldmatrix and mma.sync across warps.

**The compiler's `#pragma unroll` scheduling is surprisingly good.** On sm_120, nvcc with `-O2` and `#pragma unroll` produces near-optimal interleaving of MMA and load instructions. Simple PTX double-buffering tricks that work on Ampere (salykova) add overhead here because: (1) more registers → lower occupancy, (2) `asm volatile` constraints fight the compiler's scheduler, (3) the warp scheduler already hides latency across 8 warps.

**Non-volatile MMA (`asm` not `asm volatile`) is always correct and sometimes helps.** For GEMM, non-volatile MMA was part of the 0.88x→0.98x cuBLAS improvement (combined with occupancy changes). For attention, it's performance-neutral but produces cleaner code — the compiler already schedules volatile MMAs well when given `#pragma unroll`. **Always use non-volatile MMA** (`mma_m16n8k16_bf16_nv`). ldmatrix and cp_async must remain volatile for correctness across sync boundaries.

**To actually beat the compiler, you need a radical approach.** A single large `asm` block with full manual register allocation and instruction scheduling — not incremental PTX additions to a compiler-managed kernel. The salykova approach (writing the entire inner loop in PTX with explicit register naming) is the right idea, but requires committing fully to assembly for the hot path. For the attention kernel specifically, the target is overlapping the ~328 non-MMA softmax instructions between QK^T and PV with MMA/load operations from adjacent phases — something the compiler cannot do because it respects the sequential phase boundaries.

### v3 PTX Experiments (2026-03-13, experiments 54-60)

Six approaches tested, all correct pieces individually validated, but no speedup captured:

| Approach | Operands | Result | Why |
|----------|----------|--------|-----|
| C++ fused exp2f+PV loop | N/A | 0.070 ms (neutral) | Compiler already crosses phase boundaries with `#pragma unroll` |
| Deferred sum shuffle | N/A | 0.070 ms (neutral) | `__shfl_xor_sync` doesn't block non-volatile MMA pipeline |
| Per-kc PTX (CVT+LDSM+MMA) | 44 | 0.070 ms (correct, neutral) | ptxas reorders back to compiler-preferred schedule |
| 2-kc paired PTX blocks | 60 | 0.070 ms (correct, neutral) | Same — ptxas chooses same schedule as v2 within each block |
| Monolithic PTX block | 84 | **0.047 ms** but **WRONG** | 66 "+f" outputs caused register misassignment; "speedup" was computing garbage |
| PTX exp2f only + C++ rest | 6 | 0.070 ms (correct, neutral) | Validates PTX `sub.f32`+`ex2.approx.f32` matches C++ `exp2f()` |

**Key PTX ISA findings for sm_120:**

1. **`cvt.rn.bf16x2.f32 d, a, b` has REVERSED operand order vs C++.** PTX puts first source (a) in HIGH bits [31:16] and second source (b) in LOW bits [15:0]. C++ `__floats2bfloat162_rn(a, b)` puts a in LOW and b in HIGH. To match C++: `cvt.rn.bf16x2.f32 d, b, a` (swap). Verified empirically — without swap, MMA gets garbage P fragments and produces constant output.

2. **`shfl.sync.bfly.b32` c parameter = 31 (0x1f), NOT 0x1f1f.** For `.bfly` mode on sm_120, c[4:0] = warp_width - 1 = 31 for full warp. The packed format (c = (width-1)<<8 | clamp) does NOT work — produces identity shuffles. Only c[4:0] matters for `.bfly`. Verified: c=31 correct, c=0x1f1f broken, c=0x1f00 broken.

3. **`ex2.approx.f32` in PTX matches `exp2f()` in C++ under `--use_fast_math`.** Both map to MUFU.EX2 in SASS. Safe to use interchangeably.

4. **Monolithic asm blocks with >50 "+f" output operands produce wrong results on nvcc/CUDA 13.** The exact threshold is between 34 (works) and 66 (broken). Likely a ptxas register allocator limitation — too many simultaneously-live in/out operands cause silent misassignment. The kernel compiles, runs, and even appears faster (because it's doing less useful work), but produces garbage.

5. **ptxas respects instruction order within `asm volatile` blocks but optimizes aggressively.** For blocks under ~50 operands, ptxas produces the same schedule regardless of the programmer's instruction ordering. The compiler+ptxas combo is already near-optimal for this kernel. The only way to truly control scheduling is the salykova approach: ALL registers PTX-managed (`.reg`), no C++ operand interface, addresses computed inside PTX from a single smem base pointer.

**Conclusion:** Incremental PTX (replacing individual phases with inline asm while keeping the C++ operand interface) cannot beat the compiler for this attention kernel. The next step requires either (a) full salykova-style PTX with zero C++ interface, or (b) FP8 attention which changes the arithmetic intensity ratio fundamentally.

### v4 Zero-Interface PTX (2026-03-13, experiment 61)

Full salykova-style PTX: entire KV loop + output store in a single `asm volatile` block with **ZERO "+f" outputs**. All state in `.reg`, outputs via `st.global` from within PTX. 42 inputs (all "r"/"l"), 0 outputs.

**Result: 69 μs — performance-neutral vs v2.** ptxas still controls instruction scheduling regardless of whether operands cross the asm boundary. The "+f" output interface was NOT the source of the compiler ceiling. The fundamental bottleneck is math_pipe_throttle from the softmax phase (~48% of stalls).

**This definitively closes the PTX optimization path.** All three approaches — incremental PTX (v3), monolithic with "+f" (v4 first attempt), and zero-interface (v4 final) — produce identical performance. The compiler+ptxas combo is at 94% of its theoretical ceiling for BF16 attention.

---

## FP8 Conversion — The Vectorized CVT Discovery

**`cvt.rn.satfinite.e4m3x2.f32` WORKS on sm_120.** The original `fp8_convert.cuh` comment claimed "sm_120 does NOT support PTX cvt.e4m3x2.bf16x2 or cvt.e4m3x2.f32." **This was HALF WRONG:**

| Instruction | sm_120 Support | Verified |
|-------------|---------------|----------|
| `cvt.rn.satfinite.e4m3x2.f32 d, hi, lo` | **YES** | Compiles, runs, correct output |
| `cvt.rn.satfinite.e4m3x2.bf16x2 d, src` | **NO** | ptxas error: "not supported on sm_120" |

**The f32 variant has REVERSED operand order** (same as `cvt.rn.bf16x2.f32`): first source → HIGH byte, second source → LOW byte. To get `{fp8(a), fp8(b)}`, pass `(b, a)`.

**Impact:** The scalar FP8 conversion path (`__nv_fp8_e4m3` constructor + manual byte packing) generated ~11 instructions per uint32. The vectorized path uses one `cvt.e4m3x2.f32` instruction per pair, reducing to ~7 instructions per uint32.

**This wrong comment cost the entire FP8 attention path.** With scalar conversions, FP8 attention ran at 0.080ms (0.87x of BF16 v2 — **slower**). The conversion overhead exceeded the 2x MMA throughput gain. With vectorized CVT, FP8 runs at **0.056ms (1.24x of BF16 v2)** — a 30% speedup.

**Rule: Always test PTX instructions empirically on sm_120.** The ISA docs say sm_89+ but sm_120 is a different microarchitecture. Some instructions work, some don't. The only way to know is to compile and run.

---

## Current Performance (2026-03-13)

### BF16 GEMM
- **0.98x cuBLAS** at 4096³ (best single config)
- **1.23x cuBLAS** at 4096×1024×4096 (beats cuBLAS on non-square)
- 64×64 tiles, 4 warps, 80 regs, 6 blocks/SM, non-volatile MMA, streaming B

### Flash Attention v2 (BF16)
- **1.76x cuDNN SDPA** at B=2 H=8 N=2048 D=64 causal (69 μs)
- **1.35x SDPA** at N=512, **1.15x** at N=4096, **1.00x** at D=128
- 145 regs, 3 blocks/SM, 12 warps. At 94% of compiler ceiling.
- BF16 optimization exhausted: 53 C++ experiments + 7 PTX experiments, all converged.

### Flash Attention FP8 (NEW BEST)
- **2.33x cuDNN SDPA** at B=2 H=8 N=2048 D=64 causal (52 μs)
- **2.79x SDPA** at N=512, **1.85x** at N=1024
- Uses `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32` (2x MMA throughput)
- Vectorized `cvt.rn.satfinite.e4m3x2.f32` for BF16→FP8 conversion (critical discovery)
- Higher error than BF16 (max_err ~0.16 vs ~0.002) — acceptable for training

| B | H | N | D | SDPA (ms) | v2 BF16 (ms) | FP8 (ms) | FP8/v2 | FP8/SDPA |
|---|---|---|---|-----------|-------------|----------|--------|----------|
| 2 | 8 | 512 | 64 | 0.029 | 0.012 | 0.010 | **1.20x** | **2.79x** |
| 2 | 8 | 1024 | 64 | 0.049 | 0.033 | 0.027 | **1.25x** | **1.85x** |
| 2 | 8 | 2048 | 64 | 0.092 | 0.069 | 0.052 | **1.33x** | **2.33x** |
| 2 | 8 | 4096 | 64 | 0.176 | 0.279 | 0.228 | **1.23x** | **0.78x** |

### Reference Points
**gau-nernst** wrote custom flash attention for RTX 5090 and achieved 94.4% of 209.5 TFLOPS peak using BLOCK_Q=128. This is our feasibility proof — we know sm_120 can get there.

**cuDNN SDPA** achieves 97.2% of peak. We beat it on D=64 configs (1.35-2.79x) with both BF16 and FP8.

---

## Universal Dead Ends (Confirmed Across Multiple Projects)

**Full fusion of multi-GEMM operators is a dead end when intermediates exceed tile size.** Fused-MLP Phase 2 attempted fusing gate+up+down into a single kernel. Result: O(D_out/BLOCK_N) redundant recomputation — at D_out=3584 with BLOCK_N=128, that's 28× redundant work for GEMM2. Slowdowns ranged from 7.8× to 51.5×. **Epilogue fusion** (activation fused into GEMM output) is the correct ceiling — 1.22x cuBLAS for fused-mlp. This applies to any fused kernel where intermediate activation dimensions exceed tile dimensions.

**3-stage pipeline kills L1 cache on sm_120.** sm_120 has a unified 128KB L1/smem — more smem = less L1. Triple-buffering pushes total smem past the point where L1 thrashes. Every project that tried it regressed (GEMM direct regression, Fused-MLP scoreboard explosion, Attention avoided based on GEMM findings). **Double-buffer is the sweet spot.**

**RMSNorm barrier stalls (46%) are irreducible** in standalone RMSNorm without occupancy regression. More blocks = more barriers. Fusion with the consuming kernel (GEMM/attention) is the correct solution, not standalone optimization. Additionally, **cp.async causes L1 eviction at large D (≥4096)** — smem caching only helps for D≤1024 where the working set fits in L1.

**Cholesky: cuSOLVER's monolithic single-block kernel is unreachable.** cuSOLVER's batched Cholesky keeps all state on-chip in a single SM-resident block. This approach is unreachable from user code due to the TF32 MMA diagonal broadcast defect. Future Cholesky work requires either a hardware fix or BF16 MMA decomposition (2× the MMA calls).

---

## Reusable Patterns

**Dotproduct streaming loads + atomicAdd single-kernel pattern.** Achieves 1605 GB/s (89.4% of peak 1792 GB/s bandwidth). Reusable for any bandwidth-bound reduction: stream loads with cp.async, reduce in registers, atomicAdd partial results. This pattern beat cuBLAS (1.09x) and PyTorch (1.52x) for large reductions.

**L2 tiled fusion for multi-pass kernels (cuquantum, 2026-03-27).** When a working set
exceeds the 96 MB L2 cache and the kernel makes N sequential passes over the same data,
tile the data into L2-sized chunks and fuse all N passes within each tile. Each tile
stays L2-warm across all passes, turning N DRAM round-trips into 1.

Empirical result (cuquantum Q=24, 128 MB state vector, 4-gate circuit):
- Sequential: 4 × 175 us = 702 us (4 DRAM passes, each at 85% peak)
- Tiled fusion: 186 us (1 DRAM pass, `__syncthreads()` between gates within each tile)
- **3.78x speedup** — purely from eliminating DRAM traffic

ncu confirmed: tiled fusion long_scoreboard dropped from 0.89 to 0.49 (L2 hits replacing
DRAM stalls), with 0.11 barrier cost from inter-gate `__syncthreads()`.

**Requirements for this pattern to apply:**
1. Working set exceeds L2 (>96 MB). If data fits in L2, sequential passes already run
   at L2 speed — fusion adds overhead for zero benefit (confirmed: Q=20 fusion regressed).
2. Multiple passes over the same data (N operations on the same buffer).
3. Operations are tile-independent — each tile's work is self-contained, so block-local
   `__syncthreads()` suffices (no expensive grid-wide sync needed).
4. Tile size must fit comfortably in L2 (target ≤64 MB tiles for 96 MB L2).

**Where this applies beyond cuquantum:**
- Any multi-kernel pipeline where intermediates exceed L2 and are consumed immediately
  (e.g., element-wise chains, normalization + activation sequences on large tensors)
- Attention at very long sequences where KV cache exceeds L2
- Epilogue fusion (fused-mlp Phase 1) is a degenerate case of this — the intermediate
  never leaves registers/L2 between GEMM output and activation

**Where it does NOT apply:**
- Working set fits in L2 (Q<=22 for cuquantum) — already at L2 speed, fusion adds overhead
- Single-pass kernels (GEMM, standalone reductions) — no second pass to amortize
- Operations with cross-tile dependencies requiring grid-wide sync — sync cost dominates

**Primitives must match the BLAS signature or they're not primitives — they're demos
(Cholesky integration, 2026-03-28).** Our SYRK shipped at 2.19x cuBLAS but was
unusable by the Cholesky worker because it only accepted contiguous inputs with no
alpha/beta scaling and no leading dimension stride. The worker needed 4 extra kernels
(copy, transpose, syrk, subtract) to bridge the gap — the overhead killed the advantage.

A real BLAS primitive supports:
```
C = alpha * op(A) + beta * C   with lda, ldb, ldc strides
```

Without `alpha`, `beta`, `lda`, `ldc`, the kernel cannot operate on sub-matrices
of a larger matrix — which is what EVERY factorization algorithm does. This applies
to SYRK, GEMM, TRSM, TRMM — all of them. And for the Octave `.so`, the Fortran ABI
literally requires these parameters in the function signature.

**Rule: before shipping a primitive to `common/csrc/primitives/`, verify it can be
called on a sub-matrix with stride != N and alpha/beta != 1/0. If it can't, it's
not a building block — it's a standalone benchmark kernel.**

**Shipped primitives go stale silently (reship failure, 2026-03-28).** Linalg
fixed 9 bugs (TRSM NaN, SYRK in-place error, non-contiguous input, alpha/beta,
SPD violation) — all 31 edge tests passed in the worktree. But nobody reshipped
to `common/csrc/primitives/`. Numerical kept using the broken versions. The shelf
had 8 out of 22 files stale. Every consumer (Cholesky, QR, Octave .so) was
silently running buggy code.

**Rule: after any primitive change that passes tests, IMMEDIATELY reship to the
shelf AND all consumers. Verify with `diff` — if the shelf differs from the
worktree, someone forgot to ship. The factory test runner (`test-all.sh`) should
check shelf freshness as part of its run.**

**Primitives need version control.** Worker worktrees have git for experiments,
but the shipped shelf (`common/csrc/primitives/`) is just a directory of copied
files with no versioning. When a kernel ships, it needs a version number and
content hash so consumers know exactly what they're getting. If one bit changes,
the hash changes and the version increments. This is the same problem package
managers solve — and we need it for our .so customers.

---

## Pre-Flight Checklist (Do This BEFORE Every Experiment)

The #1 failure mode across all projects is occupancy/register pressure (~45 of ~170 total discards). Before implementing any optimization:

1. **Register count:** Will the new feature's registers fit within `__launch_bounds__`? Calculate before coding. Each extra register can drop a block/SM.
2. **Smem allocation:** Will the total smem (including double-buffer) leave room for L1? On sm_120, L1+smem share 128KB. Target ≤96KB smem.
3. **Sequential phases:** Does the kernel have irreducible sequential phases (softmax, reductions, barriers)? If yes, occupancy-first won't work — the fixed overhead scales with iteration count.
4. **Noise floor:** If your experiment shows <2% change, it's noise. Run 3 trials before keeping anything in the 2-5% range. ~25 discards across projects were noise-floor experiments.
