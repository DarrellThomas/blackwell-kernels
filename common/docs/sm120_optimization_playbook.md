# sm_120 Optimization Playbook — Empirical Decision Tree

**Auto-generated from 709 experiments across 12 kernel projects on RTX 5090 (sm_120)**
**334 kept, 375 discarded. Every entry is measured, not theoretical.**
**Last updated: 2026-03-30 18:53 UTC**

This is the optimization vocabulary for this chip. When you see a stall in ncu,
look it up here and pick a technique. Do NOT invent approaches — use what's proven.

---

## How to Use This Document

1. Run ncu, identify your **top stall**
2. Find the matching section below
3. Pick a technique that applies to your kernel type
4. Try it, measure, keep or discard
5. Check "Universal Dead Ends" BEFORE trying anything

---

## LONG_SCOREBOARD — Warps stalled waiting for data from DRAM/L2. Kernel is bandwidth-bound or latency-bound.

*42 kept, 32 discarded across all projects.*

### Proven Techniques

| Technique | Project | vs_ref |
|-----------|---------|--------|
| diagonal_gate: v1 1-amp/thread Q=20 t=10 Rz, 60x PyTorch, matches general gate 4 | cuquantum | 59.95x |
| tiled_fusion: profile 4-gate Q=24 t=0,3,7,10 — barrier 0.11, scoreboard 0.49 | cuquantum | 42.30x |
| two_qubit_gate: v1 float4 for q0=0, adjacent (0,1) 10.3→6.3 us, primary unchange | cuquantum | 38.03x |
| two_qubit_gate: v0 baseline Q=20 q0=5 q1=10 CNOT, 37.8x PyTorch, 6.2 us | cuquantum | 37.78x |
| two_qubit_gate: v3 warp-cooperative loads + half FLOPs, all pairs uniform 6.1-6. | cuquantum | 37.60x |
| diagonal_gate: v3 general+phase paths, Q=20 4.1us 34x PyTorch, phase Q=22 0.67x  | cuquantum | 34.42x |
| single_gate: re-baseline Q=20 t=10 — no regression, 28.8x PyTorch | cuquantum | 28.84x |
| single_gate_stride1: profile Q=20 t=0 — float4 path, scoreboard 0.94 | cuquantum | 27.64x |
| mcgate: v1 baseline Q=20 Toffoli c=[5,10] t=15, 27.64x PyTorch, target >5x met | cuquantum | 27.64x |
| single_gate: v3 gate via device ptr, no CPU sync — Q=20 4087 GB/s from L2 | cuquantum | 25.38x |
| mdn_loss_bwd: v4 remove redundant sync + simplify binding, 10.50x PyTorch | chess_training | 10.50x |
| mdn_loss_bwd: v2 FP32-native (eliminate bf16 round-trip), 10.43x PyTorch fp32 au | chess_training | 10.43x |
| mdn_loss_bwd: v3 single-pass + device-side inv_mask_sum (no .item() sync), 10.12 | chess_training | 10.12x |
| single_gate: v2 int32 indices + float4 stride1, Q=20 t=10 87.0% BW | cuquantum | 9.83x |
| single_gate: v0 baseline Q=20 t=10, 84.4% peak BW, 9.58x PyTorch | cuquantum | 9.58x |

### What FAILED

- **attention** (0.00x): BLOCK_Q=128 8 warps — register spills killed it
- **attention** (0.00x): partial unrolling (#pragma unroll 1) — 192B stack frame, 285K bank conflicts, catastrophic
- **chess_training** (10.43x): mdn_loss_bwd: v3 fused single-pass (cache diff+inv_sigma_sq in regs) — neutral, 229us same as v2
- **chess_training** (10.29x): mdn_loss_bwd: v4 smem cache inv_sigma_sq + constexpr D — eliminates pass2 expf+div but neutral (L2 a
- **chess_training** (10.54x): mdn_loss_bwd: v8 two-pass + launch_bounds(256,6) — neutral 226us, increased occupancy (22->28% SM) b
- **chess_training** (10.54x): mdn_loss_bwd: v9 streaming stores (st.global.cs) for gradient writes — neutral 226us, L2 not the bot
- **cuquantum** (5.41x): single_gate: v1 DISCARD 4 pairs/thread killed occupancy, 2x regression
- **cuquantum** (9.84x): single_gate: v2b DISCARD 512 threads/block, noise-neutral
- **cuquantum** (26.23x): single_gate: v3b DISCARD L2 persistence window, noise-neutral
- **cuquantum** (19.03x): single_gate: v4 DISCARD __constant__ gate, cudaMemcpyToSymbol sync cost +2us
- **cuquantum** (26.54x): single_gate: v5 DISCARD grid-stride 1020 blocks, noise/regression on Q=24
- **cuquantum** (42.53x): tiled_fusion: P2-12 DISCARD smem staging — noise (184 vs 186), bank conflicts 1369, scoreboard worse
- **cuquantum** (39.27x): tiled_fusion: P2-13 DISCARD pragma unroll 4 — 7.7% regression, register pressure
- **cuquantum** (22.36x): tiled_fusion: P2-14 DISCARD 2-pair batched reads — 89% regression, destroyed spatial locality
- **cuquantum** (37.12x): tiled_fusion: P2-15 DISCARD 128 threads/block — 14% regression, fewer threads = more serial work per

---

## MATH_THROTTLE — Tensor core / FMA input FIFO full. Instructions arriving faster than pipe can consume.

*78 kept, 239 discarded across all projects.*

### Proven Techniques

| Technique | Project | vs_ref |
|-----------|---------|--------|
| mlp_bwd: v5 3-stream parallel cuBLAS ops 2+4+5, 1.89x -> 2.02x PyTorch | chess_training | 2.02x |
| mlp_bwd: v4 fuse activated byproduct into GEMM epilogue, 1.88x PyTorch | chess_training | 1.88x |
| mlp_bwd: v6 serial cuBLAS ops (revert 3-stream), 6-10% faster | chess_training | 1.87x |
| ldmatrix_x4_trans for B — halved B loads, bank conflicts -31%, bench 1.07→1.08x | fused_mlp | 1.08x |
| BLOCK_K=64 — barrier 11→6%, bank conflicts 2.4M→62K, all configs improved | fused_mlp | 1.07x |
| cuBLAS for GEMM2 — 0.96x→1.06x on GPT-2, all configs now beat baseline | fused_mlp | 1.06x |
| cuBLAS GEMM2 for FP8 too — all configs beat baseline, FP8 1.09x GPT-2 | fused_mlp | 1.06x |
| baseline — v1 epilogue-fused MLP (GPT-2 scale: 0.148ms, 0.96x cuBLAS) | fused_mlp | 1.00x |
| occupancy-first 64x64 tiles, 4 warps, 6 blocks/SM — BF16 ceiling | gemm | 0.97x |
| FP8 64x128 output tile (was 64x64) — 33% better compute/load ratio, 1.34x cuBLAS | gemm | 0.97x |
| launch_bounds(256,1) — relax register budget, 5% faster, barrier halved | gemm | 0.89x |
| 2-tile K-loop unroll — halve sync count, 2% faster | gemm | 0.84x |
| 8 warps (256 threads) — halve per-warp MMA, double load bandwidth | gemm | 0.83x |
| baseline — initial GEMM kernel | gemm | 0.78x |
| baseline — round 2 (from round 1 final: 1.61x SDPA) | attention | 0.00x |

### What FAILED

- **attention** (0.00x): launch_bounds(128,4) — spill cost > occupancy gain
- **attention** (0.00x): split K/V prefetch — wait stalls increased, no benefit
- **attention** (0.00x): V preload in P*V — compiler already optimizes identically, no change
- **attention** (0.00x): BLOCK_KV=128 launch_bounds(128,2) — occupancy drop 12→8 warps killed latency hiding
- **attention** (0.00x): split-half softmax — extra scalar overhead cancelled wait reduction
- **attention** (0.00x): BLOCK_KV=32 launch_bounds(128,4) — doubled iterations hurt more than occupancy gain
- **attention** (0.00x): Q-in-smem — extra ldmatrix_x4 per kv_block hurt more than register savings helped
- **attention** (0.00x): nc-outer/dc-inner fused mask+max — math_throttle 41→34% but wait 18→24%, net worse
- **attention** (0.00x): P*V nc-outer/kc-inner + pre-packed P — compiler unrolls identically, no change
- **attention** (0.00x): BLOCK_KV=96 — bank conflicts doubled (20K), occupancy likely dropped to 2 blocks
- **attention** (0.00x): skip first-iter rescale — branch overhead outweighs 1-iteration savings
- **attention** (0.00x): deferred sum reduction after P*V — compiler already reorders, no change
- **attention** (0.00x): prefetch before QK^T — cp.async/ldmatrix contention during QK^T, 3% bench regression
- **attention** (0.00x): split prefetch V-before-QKT K-after — extra commit overhead, 3% regression despite 79% fewer bank co
- **attention** (0.00x): BLOCK_Q=128 only — grid too small for primary config (256 blocks < 340 SMs*2), 35% SM utilization

---

## BARRIER — Warps waiting at __syncthreads() for other warps to arrive.

*2 kept, 4 discarded across all projects.*

### Proven Techniques

| Technique | Project | vs_ref |
|-----------|---------|--------|
| v13: compile-time D=768 specialization 26 regs | rmsnorm | 1.36x |
| v2-profiled: smem-cached BLOCK=256 (ncu baseline) | rmsnorm | 1.30x |

### What FAILED

- **attention** (0.00x): FP8 cooperative smem pre-conversion — 129us bench (2.5x regression), extra syncthreads + scalar load
- **rmsnorm** (1.36x): v16: cp.async global→smem + L1 read for sum_sq (D=768 ok, D=4096 REGRESSED to 0.84x — .cg evicts L1)

---

## WAIT — Warps waiting for MMA result (data dependency) or cp.async completion.

*7 kept, 2 discarded across all projects.*

### Proven Techniques

| Technique | Project | vs_ref |
|-----------|---------|--------|
| v3-compliance: 92/92 BLAS compliance tests pass (all sizes, rect, singular, ill- | lu | 1.15x |
| CTA swizzle=4 for L2 B reuse — short_scoreboard 29→26%, 1.14x at 4096^3 | gemm | 1.14x |
| B XOR swizzle + launch_bounds(128,4) — 269M bank conflicts (50% reduction), 1.13 | gemm | 1.13x |
| v3-resume: re-enable look-ahead, 1.13x N=4096 1.09x N=8192 | lu | 1.13x |
| v3-blas: rectangular M×N support, degenerate sizes, info return, 34/34 tests pas | lu | 1.13x |
| v3-opt-study: async left LASWP disproved (BW contention), GEMM_left custom DGEMM | lu | 1.13x |
| v1 baseline — 64x64x16 tiles, TM=8 TN=4, 128 threads, cp.async double-buffer, be | gemm | 1.06x |

### What FAILED

- **gemm** (1.13x): A_STRIDE=17 + reg loads — bank conflicts unchanged (269M, A loads are broadcasts), wait 31→46% regre
- **gemm** (0.62x): BLOCK_K=64 — 2x K-tile too much data per load, wait stalls dominate

---

## NOT_SELECTED — Warp scheduler has nothing to issue. Too few warps in flight.

*0 kept, 0 discarded across all projects.*

---

## UNIVERSAL DEAD ENDS — Never Try These on sm_120

These failed across multiple projects. Do not retry them.

| Dead End | Projects | Root Cause |
|----------|----------|------------|
| **3-stage pipeline** | gemm, attention, fused-mlp | Kills L1 cache on sm_120 — L1 and smem share 128KB. Triple-buffering pushes smem past the point wher |
| **Manual PTX scheduling** | attention (7 attempts) | ptxas reorders back to compiler-preferred schedule. 7 approaches tried across attention, all perform |
| **launch_bounds forcing register spills** | dotproduct, attention, gemm | Even 8 bytes of spill is catastrophic for bandwidth-bound kernels. Spill goes to local memory (backe |
| **Full operator fusion across GEMM boundaries** | fused-mlp | O(D_out/BLOCK_N) redundant recomputation when intermediates exceed tile size. 7.8-51.5x slowdown. |
| **Two-phase reduction kernels** | dotproduct | Second kernel launch overhead (2-3 us) exceeds any reduction benefit for bandwidth-bound kernels. |
| **Block index remapping** | attention | Destroys L2 locality. Sequential block indices map to sequential L2 cache lines. |
| **Preloading all fragments to registers** | gemm, attention | Compiler already interleaves optimally with #pragma unroll. Extra live regs hurt occupancy. |
| **Scalar smem loads replacing ldmatrix** | attention FP8 | ldmatrix is warp-collective 128-bit. Scalar uint16 = 16-bit with bank conflicts. 19% regression. |
| **cp.async with .cg hint for large D** | rmsnorm | L1 eviction at D>=4096. .cg bypasses L1, which is fine for streaming but bad when data is reused. |

---

## PROVEN PATTERNS — Reusable Building Blocks

### Bandwidth-Bound (dotproduct, rmsnorm, BLAS1)
```
float4 vectorized loads (16B per load)
Grid-stride loop with 4-8x unroll (8 independent loads in flight)
FMA intrinsics (__fmaf_rn — one instruction vs MUL+ADD)
Warp shuffle reduction (__shfl_xor_sync butterfly)
Single atomicAdd per block (not per warp — 2720→170 atomics)
Streaming loads (ld.global.cs) for data > L2 (96MB)
Auto-tune block size per problem size
= 89.4% of peak bandwidth (1602/1792 GB/s)
```

### Compute-Bound GEMM (BF16, FP8)
```
64x64 tiles, 4 warps, 80 regs, 6 blocks/SM
cp.async double-buffer pipelining
XOR swizzle for bank conflicts
Non-volatile MMA (asm not asm volatile)
ldmatrix_x4_mma (baked a1/a2 swap, eliminates MOVs)
Stream B fragments per-tile (fewer live regs)
__launch_bounds__(128, 6)
= 0.98x cuBLAS (BF16), 1.34x cuBLAS (FP8)
```

### Compute-Bound Attention (BF16, FP8)
```
BQ=64 BKV=64, 4 warps, 145 regs, 3 blocks/SM
cp.async double-buffer, XOR swizzle
Register-only P→A conversion (no smem round-trip)
exp2f softmax (LOG2E folded into Q scale, saves 34 MULs/iter)
Skip mask for unmasked KV blocks (-60 conditionals/iter)
Dynamic BQ dispatch (128 for large grids)
Prefetch after QK^T not before (avoids load contention)
Vectorized FP8 conversion (cvt.e4m3x2.f32)
= 1.76x SDPA (BF16), 2.33x SDPA (FP8)
```

### Fused Epilogue (fused-mlp)
```
Epilogue fusion only (activation fused into GEMM output)
Do NOT fuse across GEMM boundaries
BLOCK_K=64 for fewer barriers
ldmatrix_x4_trans for B matrix
= 1.07-1.22x PyTorch
```

---

## REGISTER BUDGET (sm_120, 128 threads/block)

| Regs/thread | Max blocks/SM | Warps/SM |
|-------------|---------------|----------|
| 44 | 11 | 44 |
| 64 | 7 | 28 |
| 80 | 6 | 24 |
| 96 | 5 | 20 |
| 128 | 3 | 12 |
| 145 | 3 | 12 |
| 165 | 2 | 8 |
| 255 | 1 | 4 |

---

## COMPILER BEHAVIOR ON sm_120

Things the compiler already does well (don't fight it):
- Interleaves MMA with loads when using `#pragma unroll`
- Hoists loads into compute gaps (V loads into softmax gap)
- Produces identical SASS for most C++ loop restructurings
- Near-optimal register allocation at <=128 registers with non-volatile asm

Things the compiler CANNOT do:
- Cross sequential phase boundaries (softmax between QK^T and PV)
- Choose between streaming and cached loads based on data size
- Reduce register count below the algorithm's fundamental requirements

---

## NOISE FLOOR

**Differences < 2% are noise.** 10 warmup + 100 timed iterations.
If the delta is 1-2%, run 3 more trials before trusting it.

---

## QUICK REFERENCE: What To Try First

| Your top stall | Kernel type | Try first |
|---------------|-------------|-----------|
| long_scoreboard | Bandwidth-bound | More blocks/SM (increase occupancy) |
| long_scoreboard | Compute-bound | Check if data fits L2 → use .ca not .cs |
| math_throttle | GEMM | Occupancy-first (smaller tiles, more blocks) |
| math_throttle | Attention | Probably near ceiling. Check vs our 1.76x. |
| math_throttle | Fused | BLOCK_K=64, ldmatrix_x4_trans for B |
| barrier | Reduction | Warp-level reduction, single atomicAdd per block |
| barrier | Any | Fewer main-loop iterations (larger BLOCK_K) |
| not_selected | Any | Reduce registers via __launch_bounds__ |
| wait | Any | Double-buffer pipelining (NOT triple) |
