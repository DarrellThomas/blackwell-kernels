# Cross-Pollination Synthesis — March 2026

**Date:** 2026-03-15
**Author:** researcher-claude
**Purpose:** Workers cannot see each other's work. This document identifies specific
techniques, findings, and dead ends from each active worker that could benefit others.

---

## 1. ATTENTION --> Other Workers

### 1.1 Register-Only P->A Conversion (Who should know: GEMM, Fused-MLP, Linalg)

Attention eliminated the dominant source of bank conflicts by converting FP32
softmax accumulators to BF16 MMA A-fragments entirely in registers via
`pack_bf16x2` warp shuffles — no shared memory round-trip.

**Cross-pollination opportunity:** Any kernel that produces FP32 intermediate
results and then feeds them into BF16 MMA can use this pattern. Specifically:
- **Fused-MLP:** The activation output (FP32 after relu_sq) feeds into GEMM2.
  If GEMM2 ever moves to a custom kernel (currently cuBLAS), this register-only
  conversion would avoid an smem round-trip for the activated intermediate.
- **Cholesky/LU:** If the monolithic kernel approach proceeds with BF16 MMA for
  SYRK/GEMM updates, the FP32 panel results would need BF16 conversion for MMA.
  Register-only conversion avoids smem pressure in an already smem-constrained
  single-block design.

### 1.2 Dynamic BLOCK_Q Dispatch (Who should know: GEMM, Linalg)

Attention uses runtime tile size selection: BLOCK_Q=128 when the grid has >=340
blocks, BLOCK_Q=64 otherwise. This gives 2x speedup at N=4096 and parity at D=128
without sacrificing the primary config.

**Cross-pollination:** GEMM already has dual-dispatch (64x128 / 128x128). Linalg's
batched GEMM currently uses fixed tiles — dynamic dispatch based on batch*M*N
could help at varying problem sizes.

### 1.3 Softmax as Irreducible Sequential Phase (Who should know: LU, QR)

Attention proved that occupancy-first tiling (the GEMM breakthrough strategy) does
NOT work when the inner loop contains irreducible sequential phases. Halving tile
size doubles sequential overhead faster than occupancy gains compensate.

**Cross-pollination:** LU and QR both have sequential panel phases (pivot selection,
Householder reflections). If these workers attempt occupancy-first tiling, they
should know that sequential phases make this strategy counterproductive. The
attention worker exhausted this with 4 experiments (50-53) — do not repeat.

### 1.4 ldmatrix Cannot Be Replaced with Scalar Loads (Who should know: ALL)

Experiments 65-66 proved that replacing `ldmatrix` (warp-collective 128-bit load)
with scalar uint16 loads causes 19-150% regression regardless of instruction count
reduction. ldmatrix is a hardware-optimized path; scalar loads are 8x narrower
with bank conflicts.

**Cross-pollination:** Any worker considering custom fragment loading (e.g., for
non-standard data layouts, FP8 native inputs, or triangular matrix loads) must
find a way to use ldmatrix or equivalent warp-collective loads. This is a hard
hardware constraint, not an optimization preference.

---

## 2. GEMM --> Other Workers

### 2.1 Dual-Dispatch Strategy (Who should know: Attention, Fused-MLP, Linalg)

GEMM uses two tile configs selected at runtime based on data volume and grid pressure:
- 64x128 when data <= 64MB and grid <= 4096 blocks (L2 friendly)
- 128x128 when either threshold is exceeded (reduces L2 contention)

This gave a stable 1.29x cuBLAS at 4096^3.

**Cross-pollination:**
- **Attention:** Already has dynamic BLOCK_Q, but could add a similar L2-pressure
  dispatch for very large N (N=4096 where it drops to 1.15x SDPA).
- **Fused-MLP:** Currently uses cuBLAS for GEMM2 because fixed 64x64 tiles only
  fill 38% of SMs at N=768. If GEMM2 moves to a custom kernel, dual-dispatch
  would address this directly.
- **Linalg batched GEMM:** Fixed tiles; could benefit from size-aware dispatch for
  varying batch dimensions.

### 2.2 CTA Swizzle for L2 Reuse (Who should know: Fused-MLP, Linalg)

FP8 GEMM groups concurrent blocks via CTA swizzle (swizzle=4) so they share B
columns for L2 cache reuse. This is orthogonal to the tile size and was part of
the 1.13x to 1.29x improvement.

**Cross-pollination:** Any kernel with a 2D grid where adjacent blocks read the
same rows/columns benefits from CTA swizzle. Fused-MLP's GEMM1 has a 2D grid
over M x D_ff — swizzling to share weight columns would improve L2 hit rate.

### 2.3 Compute/Load Ratio as Design Metric (Who should know: ALL)

The key FP8 insight was quantitative: widening from 64x64 to 64x128 increased
compute/load ratio from 4.0 to 5.3 MMA/KB. This single metric predicted the
shift from memory-bound (long_scoreboard 45%) to balanced (math_throttle 27%).

**Cross-pollination:** Workers should compute MMA/KB for their tile configs before
experimenting. A ratio below ~4.0 means the kernel is memory-bound regardless of
other optimizations. Above ~5.0 is the balanced zone. This gives a fast filter
for which tile sizes are worth trying.

### 2.4 86K Bank Conflicts from B Loads (Who should know: Attention, Linalg)

GEMM's 64x128 FP8 path still has 86K bank conflicts from B loads with N=128 stride.
This is an open problem. If any worker finds a swizzle pattern for wide N tiles,
it should be shared back to GEMM.

---

## 3. FUSED-MLP --> Other Workers

### 3.1 Full Fusion Post-Mortem: O(D_out/BLOCK_N) Redundancy (Who should know: ALL)

This is the most broadly applicable lesson. Fusing two GEMMs end-to-end (GEMM1 +
activation + GEMM2) by tiling on the OUTPUT dimensions causes each block to
redundantly recompute the FULL intermediate for its M-rows.

**Redundancy = D_out / BLOCK_N.** At GPT-2 scale: 768/64 = 12x redundant GEMM1
compute. At LLaMA-7B: 4096/64 = 64x redundancy. Result: 7.8x to 51.5x SLOWER.

**Why attention escapes this:** Flash attention's output dimension D=64 fits in a
single N-tile, so grid_n=1 and redundancy=1.

**Cross-pollination:**
- **LU:** If considering fusing panel + trailing GEMM into a monolithic kernel,
  verify that the trailing update doesn't create inter-block redundancy. The
  trailing GEMM updates (N-NB) x (N-NB) — each block needs its own slice, not
  the full intermediate.
- **QR:** CholQR2 avoids this by using separate large parallel ops (SYRK, TRSM).
  If anyone proposes fusing SYRK + Cholesky + TRSM, this lesson shows why per-block
  recomputation of the Gram matrix would be catastrophic.
- **General rule:** Fusion only works when the intermediate fits in a single tile
  or when each output block only needs a unique slice of the intermediate.

### 3.2 cuBLAS for Tail Operations (Who should know: Linalg, LU, QR, Cholesky)

Fused-MLP discovered that cuBLAS for GEMM2 beats their custom 64x64 tiles because
cuBLAS uses adaptive tiling. When the custom kernel's tile only fills 38% of SMs,
cuBLAS chooses a better tile.

**Cross-pollination:** Numerical methods workers (LU, QR, Cholesky) should not
replace cuBLAS for trailing updates unless their custom kernel's grid fills
at least 50% of SMs (85+ blocks on 170 SM RTX 5090). cuBLAS is adaptive and
hard to beat at small grid sizes.

### 3.3 FP8 Activation Saturation (Who should know: Attention)

relu_sq output exceeds FP8 max (448). At GPT-2 scale, ~9.5% of activated values
saturate. GEMM2 must stay BF16.

**Cross-pollination:** Attention's FP8 path converts BF16 -> FP8 for QK^T and PV.
The softmax output P is in [0,1] so saturation is not an issue there. But if
attention ever considers FP8 for the Q/K/V inputs directly, values outside
[-448, 448] would saturate. Pre-scaling or clamping would be needed.

---

## 4. DOTPRODUCT --> Other Workers

### 4.1 Streaming Loads (ld.global.cs) for Large Data (Who should know: RMSNorm, Linalg)

Dotproduct uses `ld.global.cs` (cache streaming) to bypass L2 for data > 96MB.
This saved ~2 us on N=16M by avoiding L2 thrashing.

**Cross-pollination:**
- **RMSNorm:** At large D (>=4096), RMSNorm already found that cp.async causes L1
  eviction. Streaming loads via ld.global.cs could help for very large row counts
  where the total data volume exceeds L2. However, RMSNorm's 4.1 us floor is
  set by pipelining, not memory — so this would only matter for batch sizes
  where total data > 96MB (rows * D * 2 bytes > 96MB, i.e., rows > 12288
  at D=4096).
- **Linalg BLAS1 ops (AXPY, SCAL, DOT, NRM2):** These are bandwidth-bound. For
  large vectors, ld.global.cs could give 2-5% improvement by avoiding L2 pollution.

### 4.2 atomicAdd Single-Kernel Pattern (Who should know: Linalg, RMSNorm)

Dotproduct proved that a single kernel with atomicAdd is faster than a two-phase
kernel for reductions. The second kernel launch overhead (~2-3 us) exceeds
atomicAdd contention even with 680 blocks.

**Cross-pollination:**
- **Linalg NRM2:** Currently on the "next directions" list as a "single-pass
  reduce+sqrt" kernel. The dotproduct pattern (float4 loads, FMA accumulation,
  warp shuffle + smem reduction, atomicAdd) is the right architecture. Do NOT
  use a two-phase approach.
- **RMSNorm:** The per-row reduction is already single-kernel, but if a fused
  multi-row reduction is ever needed, atomicAdd is the way.

### 4.3 launch_bounds with min_blocks Causes Register Spills (Who should know: ALL)

Dotproduct tried `launch_bounds(256, 6)` and got 8 bytes of stack spills. For a
bandwidth-bound kernel, even 8 bytes of local memory per thread added ~5 us.

**Cross-pollination:** This confirms what GEMM and attention found independently:
`launch_bounds` min_blocks is a precision tool. If the compiler cannot fit within
the register cap, it spills, and spills are catastrophic for bandwidth-bound
kernels. Only use min_blocks when you have verified the register count fits
without spills. The attention worker found the same (145 regs at cap 128 caused
spills). The GEMM worker found the sweet spot at 80 regs with 0 spills.

---

## 5. CHOLESKY --> Other Workers

### 5.1 TF32 MMA B Fragment Diagonal Broadcast (Who should know: ALL)

**CRITICAL FINDING.** `mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32` on
sm_120 has a B fragment defect: each register value broadcasts to two positions
(B[k,n] AND B[k+1,n+1]), constraining B[k,n] = B[k+1,n+1] along diagonals.

This makes TF32 MMA unusable for general GEMM without decomposition into two calls
(halving throughput).

**Cross-pollination:**
- **LU:** LU needs FP32-precision trailing updates. TF32 MMA cannot be used
  directly — must use either BF16 MMA with FP32->BF16 conversion (losing
  precision) or decomposed TF32 (halving throughput). LU's agent state already
  references this lesson.
- **QR:** CholQR2's SYRK is currently cuBLAS with TF32. If a custom SYRK kernel
  is attempted, it must account for this defect.
- **Linalg SYRK/TRSM:** Linalg delegates to cuBLAS for SYRK (0.96x) and TRSM
  (0.82x). If custom kernels are attempted, TF32 MMA is not an option for the
  general case.

### 5.2 TF32 Output Mapping Differs from BF16 (Who should know: LU, QR, Linalg)

TF32 m16n8k8 output layout: d0,d1 are SAME column in adjacent rows (8-row stride).
BF16 m16n8k16 output layout: d0,d1 are adjacent columns in same row.

Anyone switching between TF32 and BF16 MMA must adjust their store/accumulate logic.

### 5.3 cuSOLVER Monolithic Kernel Architecture (Who should know: LU, QR)

cuSOLVER uses a SINGLE monolithic kernel for batched Cholesky: 1 block, 256 threads,
202 registers, 52KB smem. It achieves ~15 TFLOPS on 1 SM using on-chip tensor cores.

**Cross-pollination:**
- **LU:** LU's strategy doc already references this. The lesson is clear: many
  small kernel launches cannot beat a monolithic kernel that keeps all state
  on-chip. LU should skip the multi-launch phases (v1/v2) and invest directly
  in the monolithic approach, or accept that the multi-launch ceiling is ~0.5x
  cuSOLVER (as Cholesky demonstrated: 190 graph nodes -> 0.55x).
- **QR:** CholQR2 avoids the monolithic problem by choosing a fundamentally
  different algorithm with large parallel ops. This is the right call — QR at
  1.58x cuSOLVER proves that algorithm selection can outperform kernel-level
  optimization.

### 5.4 Launch Overhead Quantification (Who should know: LU, QR, Linalg)

Cholesky measured: 190 graph nodes x ~2 us = ~0.38ms launch overhead. Plus cuBLAS
call overhead: ~10 us minimum per call x 64 calls = ~0.64ms. Total overhead:
~3.1ms vs cuSOLVER's 1.5ms monolithic kernel.

**Cross-pollination:** LU is planning a blocked approach with cuBLAS calls. At
N=4096 with NB=64, that is 64 steps x (panel + TRSM + GEMM) = ~192 kernel
launches. Even with CUDA Graphs, this is ~0.4ms overhead — vs cuSOLVER's 9.4ms,
that is 4% overhead (manageable). But at N=1024 (16 steps), the overhead fraction
grows. LU should prioritize N=4096 where the GEMM-dominated compute amortizes
launch overhead.

---

## 6. QR --> Other Workers

### 6.1 Algorithm Selection as Primary Optimization (Who should know: LU, Cholesky)

QR's CholQR2 result (1.58x-3.19x cuSOLVER) demonstrates that choosing a
fundamentally different algorithm can outperform kernel-level optimization of the
standard algorithm. Blocked Householder QR was 0.43x cuSOLVER (same as Cholesky's
blocked approach). CholQR2 replaced the sequential panel loop with large parallel ops.

**Cross-pollination:**
- **LU:** Is there an LU-equivalent of CholQR2? Communication-avoiding LU (CALU)
  or randomized LU (rLU) could replace sequential panel factorization with large
  parallel operations. The QR result proves the payoff is 3-8x. Research brief
  on alternative LU algorithms would be high-value.
- **Cholesky:** Cholesky IS the inner operation of CholQR2. The current Cholesky
  at 0.55x cuSOLVER is fine for QR because it operates on the small N x N Gram
  matrix, not the full M x N input. But batched small Cholesky (N=32-64) is a
  potential optimization for CholQR2's inner step.

### 6.2 Blocked Householder = Dead End for GPU QR (Who should know: LU)

Blocked Householder QR at 0.43x cuSOLVER confirms that sequential panel algorithms
with many kernel launches are fundamentally slow on GPU. This is the same pattern
as Cholesky's blocked approach (0.55x).

**Cross-pollination:** LU's blocked factorization is the same pattern — sequential
panels with trailing GEMM updates. LU should expect a similar ceiling (~0.4-0.6x
cuSOLVER) from the blocked approach and plan the monolithic kernel or alternative
algorithm path from the start.

---

## 7. RMSNORM --> Other Workers

### 7.1 GPU Pipelining Floor (Who should know: Dotproduct, Linalg BLAS1)

RMSNorm discovered a 4.1 us throughput floor set by GPU kernel pipelining:
- ncu single-kernel latency: ~6 us
- Pipeline overlap: next kernel starts 4.1 us after previous
- Verified with C++ bench loop (zero Python overhead)

**Cross-pollination:**
- **Dotproduct:** At medium N (1M), dotproduct achieves 5.41 us. This is close to
  the pipelining floor. Further optimization may be hitting the same wall.
- **Linalg BLAS1:** AXPY (2 us), SCAL (2 us), DOT (6 us) — the 2 us ops are
  BELOW the pipelining floor, suggesting they are benefiting from extreme
  pipeline overlap or are partially overlapping with launch overhead. DOT at
  6 us is above the floor and may have room.
- **General:** Any standalone kernel with <10 us execution time is approaching
  the pipelining floor. The real optimization for these kernels is FUSION —
  eliminate the standalone launch entirely by folding into the consuming/producing
  kernel. RMSNorm's stated next direction is "fusion with attention."

### 7.2 Barrier Stalls are Irreducible in Standalone Normalization (Who should know: ALL)

RMSNorm has 46% barrier stalls that are irreducible without occupancy regression.
Reducing barriers (smaller blocks, warp-per-row) trades barrier stalls for memory
stalls from reduced occupancy. With 2048 rows on 170 SMs, block-per-row achieves
48 warps/SM (100% occupancy) while warp-per-row maxes at 12 warps/SM (25%).

**Cross-pollination:** Any kernel with a per-element barrier (reduce, broadcast,
sync) will hit this same tradeoff. The answer is not to optimize the standalone
kernel further — it is to fuse with adjacent operations.

### 7.3 cp.async Causes L1 Eviction at Large D (Who should know: Attention, GEMM)

RMSNorm found that `cp.async.cg` evicts L1 cache. For D=768 this was neutral, but
for D>=4096 it regressed to 0.84x.

**Cross-pollination:**
- **Attention:** Uses cp.async for K/V loads. At D=128 (attention's large-D config),
  the working set is small enough that L1 eviction may not matter. But if attention
  ever scales to D=256+, this could become relevant.
- **GEMM:** Uses cp.async for A/B loads. GEMM's working set is controlled by tile
  size (16-48KB smem), leaving 80-112KB for L1. The 86K bank conflicts suggest
  L1 is already under pressure for the 64x128 FP8 path.

---

## 8. LINALG --> Other Workers

### 8.1 cuBLAS Delegation Pattern (Who should know: LU, QR, Cholesky)

Linalg demonstrates the right hybrid strategy: custom CUDA for ops where cuBLAS
is beatable (batched GEMM 1.34x, GEMV 1.75x, BLAS1 1.4-6x), cuBLAS delegation
for ops where it is not (TRSM 0.82x, SYRK 0.96x).

**Cross-pollination:** Numerical methods workers should use the same principle:
- Custom kernels for the panel/factorization (sequential, cuSOLVER-beatable)
- cuBLAS for trailing GEMM/SYRK/TRSM updates (large parallel, cuBLAS-optimal)
- This is exactly what QR's CholQR2 does — all cuBLAS/cuSOLVER ops.

### 8.2 TRSM Stays in FP32 (Who should know: LU, QR, Cholesky)

Linalg found that TRSM at 0.82x using `torch.linalg.solve_triangular` in F32
with cast-back is the ceiling for non-custom approaches. A native BF16 TRSM
would require multi-SM blocked approach with significant complexity.

**Cross-pollination:** LU needs TRSM for the panel solve. QR's CholQR2 uses TRSM
for Q = A @ R^{-1}. Both should use cuBLAS F32 TRSM rather than attempting custom
implementations, unless TRSM becomes the dominant bottleneck.

### 8.3 swap_rows at 6.06x cuBLAS (Who should know: LU)

Linalg's `swap_rows` kernel achieves 6.06x cuBLAS using coalesced 128-bit copies.
LU factorization requires row swaps for pivoting.

**Cross-pollination:** LU should use linalg's swap_rows primitive directly for
pivoting rather than implementing its own row swap. The primitive is already
shipped in `common/csrc/primitives/linalg/`.

---

## 9. LU --> Other Workers

### 9.1 LU is Early-Stage — Lessons Flow IN, Not Out

LU has only established baselines. Its primary value to other workers right now
is as a consumer of their primitives:
- Uses GEMM for trailing updates
- Uses TRSM for panel solves
- Uses swap_rows for pivoting
- Should use Cholesky's monolithic kernel findings to guide strategy

**Cross-pollination opportunity:** LU should study QR's CholQR2 success and
investigate whether alternative LU algorithms (communication-avoiding LU, tile LU,
or randomized LU) could replace the sequential blocked approach, just as CholQR2
replaced blocked Householder.

---

## 10. UNIVERSAL CROSS-CUTTING THEMES

### 10.1 The "Compiler Ceiling" Pattern

Multiple workers independently discovered that the nvcc compiler + ptxas backend
produces near-optimal scheduling for their kernels:
- **Attention:** 94% of compiler ceiling (68 us vs 64 us theoretical)
- **GEMM:** 0.97x cuBLAS (which itself uses hand-tuned SASS)
- **Fused-MLP:** "Compiler ceiling reached — source-level reordering has no effect"
- **RMSNorm:** 16 experiments all converge to 4.1 us

**Lesson for all workers:** Once you hit the compiler ceiling, stop trying C++
restructuring. The remaining paths are (a) full hand-written PTX (500+ lines,
major effort, attention proved it gives 0% for this ISA), (b) algorithmic changes
(CholQR2 proving this works), or (c) FP8 (changing arithmetic intensity).

### 10.2 The "Occupancy vs Sequential Overhead" Tradeoff

Two workers proved opposite sides of the same coin:
- **GEMM:** Occupancy-first (smaller tiles, more blocks) gave 0.88x -> 0.98x cuBLAS
  because GEMM's inner loop is pure MMA with no sequential phase.
- **Attention:** Occupancy-first gave 0.87x -> 0.77x SDPA because softmax is an
  irreducible sequential phase between QK^T and PV.

**Decision rule:** If your inner loop is >80% MMA/loads, try occupancy-first.
If it has a sequential phase (reduction, factorization, softmax), keep tiles
large enough that the sequential phase is amortized.

### 10.3 The "Algorithm Selection" Lever

QR's CholQR2 (1.58-3.19x cuSOLVER) proves that the biggest performance wins
come from choosing the right algorithm, not optimizing the wrong one. Fused-MLP's
full fusion failure (0.02-0.18x) proves the inverse.

**Workers stuck at a ceiling should ask:** Is there a fundamentally different
algorithm for this problem that maps better to GPU execution? The GPU wants
large parallel operations, not sequential panel loops.

### 10.4 FP8 as Architectural Change, Not Incremental Optimization

FP8 is not "BF16 but faster." It changes the kernel's character:
- **Attention FP8:** Shifted from compute-bound (SM 59%) to latency-bound (SM 43.8%)
  due to BF16->FP8 conversion overhead.
- **GEMM FP8:** Shifted from memory-bound (long_scoreboard 45%) to balanced
  (math_throttle 27%) by changing the compute/load ratio.
- **Fused-MLP FP8:** Only 3-4% gain because FP8 only helps GEMM1 (activation
  saturates FP8 range).

**Lesson:** FP8 changes which bottleneck dominates. Workers considering FP8
should first compute whether the conversion overhead or the activation range
limits the benefit.

---

## 11. PRIMITIVE REUSE OPPORTUNITIES

| Primitive | Source | Consumers Who Could Benefit |
|-----------|--------|----------------------------|
| swap_rows (6.06x cuBLAS) | linalg | LU (pivoting) |
| permute_rows (1.93x cuBLAS) | linalg | LU (pivot application), QR (column pivoting) |
| DOT (1.48x cuBLAS) | linalg/dotproduct | QR (CholQR2 orthogonality check), numerical |
| Batched GEMM BF16 (1.34x cuBLAS) | linalg | QR (batched CholQR2), numerical |
| FP8 Batched GEMM (2.02x cuBLAS) | linalg | Future FP8 numerical methods |
| GEMV (1.75x cuBLAS) | linalg | LU panel factorization, iterative solvers |
| atomicAdd reduction pattern | dotproduct | NRM2 (linalg), any new reduction kernel |
| Register-only FP32->BF16 conversion | attention | Any MMA kernel with FP32 intermediates |
| Streaming loads (ld.global.cs) | dotproduct | Any bandwidth-bound kernel with data > L2 |

---

## 12. SPECIFIC RECOMMENDATIONS

### For Attention Worker:
- GEMM's compute/load ratio metric (MMA/KB) could quantify the FP8 conversion
  overhead more precisely. Calculate MMA/KB for FP8 attention — if below 4.0,
  the conversion overhead is making the kernel memory-bound.

### For GEMM Worker:
- Attention's register-only P->A conversion pattern could inspire a register-only
  approach to reduce the 86K B-load bank conflicts (if B fragments can be
  restructured in registers rather than through smem).

### For Fused-MLP Worker:
- GEMM's CTA swizzle could help GEMM1 L2 reuse (weight column sharing across
  blocks). Currently at 0.90x cuBLAS for GEMM1 — CTA swizzle might close the gap.

### For LU Worker:
- Study QR's CholQR2 algorithm selection success before committing to blocked LU.
  Research communication-avoiding LU (CALU) or tile LU algorithms.
- Use linalg's swap_rows (6.06x cuBLAS) for pivoting.
- Expect ~0.5x cuSOLVER from the blocked + CUDA Graph approach (Cholesky's lesson).

### For QR Worker:
- Profile CholQR2 components (SYRK, Cholesky, TRSM) individually. If SYRK
  dominates, consider linalg's batched GEMM as a faster SYRK replacement.
- Cholesky's batched small Cholesky direction could speed up the inner step.

### For Cholesky Worker:
- QR proved that algorithm selection beats kernel optimization. Consider whether
  a different Cholesky variant (e.g., communication-avoiding, or left-looking
  vs right-looking) could avoid the monolithic kernel requirement.

### For RMSNorm Worker:
- Fusion with attention is the right next step. Attention's register-only
  conversion pattern shows how to fold normalization into the MMA input path
  without smem overhead.

### For Dotproduct Worker:
- Optimization is complete. The pipelining floor (4.1-5.4 us) limits further gains.
  Streaming loads pattern should be documented for other bandwidth-bound workers.

### For Linalg Worker:
- NRM2 should use dotproduct's atomicAdd single-kernel pattern (not two-phase).
- Profile each custom kernel with ncu — the "next directions" list includes this
  and it would identify which ops have the most headroom.
