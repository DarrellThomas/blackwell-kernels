# Speed-of-Light Flash Attention for RTX 5090 -- gau-nernst Blog

**Source:** [https://gau-nernst.github.io/fa-5090/](https://gau-nernst.github.io/fa-5090/)
**Relevant to:** attention worker, GEMM worker
**Worker's current problem:** BF16 attention at 1.76x SDPA (69 us), FP8 at 2.33x SDPA (52 us). Compiler ceiling at 94%. Need validation of optimization approach and potential new techniques.
**Date:** 2026-03-15

---

## What This Is

A detailed blog post by Thien Tran (gau-nernst) documenting the implementation of
Flash Attention in CUDA C++ specifically targeting the RTX 5090 (sm_120). Achieves
**197.74 TFLOPS (94.39% of 209.5 TFLOPS SOL)** for BF16 at bs=1, heads=8,
len_query=4096, len_kv=8192, head_dim=128. The implementation uses the same
mma.sync m16n8k16 ISA as our kernels.

This is an independent external validation of our optimization approach and provides
several specific data points our workers can use for comparison.

---

## Why It Matters for Us

1. **Validates our ceiling estimate.** gau-nernst reaches 94.4% SOL, closely matching
   our 94% of compiler ceiling estimate. This confirms the ~5-6% gap is fundamental
   to the mma.sync programming model, not a deficiency in our implementation.

2. **Uses head_dim=128** (vs our primary D=64). Different config, but same architecture.
   Shows the techniques scale to larger head dimensions.

3. **Identifies the same bottleneck progression** we observed: bank conflicts first,
   then global memory latency, then instruction scheduling.

4. **Documents specific optimizations** with quantified per-step improvements.

---

## Performance Progression (v1-v5)

| Version | TFLOPS | % SOL | Key Optimization |
|---------|--------|-------|------------------|
| v1 Basic | 142.87 | 68.2% | Baseline with 16-way bank conflicts |
| v2 Swizzling | 181.11 | 86.5% | XOR swizzle eliminates bank conflicts |
| v3 2-stage pipeline | 189.84 | 90.6% | cp.async double-buffer overlap |
| v4 ldmatrix.x4 | 194.33 | 92.8% | Reduced instruction count |
| v5 Better pipeline | 197.74 | 94.4% | Single-buffer V optimization |

**Comparison baselines (same config):**
- F.sdpa() Flash Attention: 186.73 TFLOPS
- CuDNN: 203.61 TFLOPS
- flash-attn package: 190.58 TFLOPS

---

## Key Techniques (with sm_120-specific details)

### 1. XOR Swizzle Pattern (v1 -> v2: +38.2 TFLOPS)

The biggest single improvement. Same technique we use. The blog gives the exact
address calculation:

```cuda
uint32_t row_idx = (index / STRIDE) % 8;
uint32_t bits_to_xor = row_idx / max(64 / STRIDE, 1);
return index ^ (bits_to_xor << 4);
```

This eliminates 8-way bank conflicts in ldmatrix operations. Profiling confirmation:
L1 wavefronts drop from 16 to 2 per access.

**Our status:** Already implemented. No action needed.

### 2. cp.async 2-Stage Pipeline (v2 -> v3: +8.7 TFLOPS)

Uses `cp.async.commit_group` and `cp.async.wait_group N` to overlap global memory
loads with tensor core computation. Strategic placement:
- K prefetch starts after previous iteration's 2nd MMA
- V prefetch starts after 1st MMA of current iteration

**Our status:** Already implemented (cp.async with double-buffer pipelining).
No action needed.

### 3. ldmatrix.x4 for K/V (v3 -> v4: +4.5 TFLOPS)

Replaces ldmatrix.x2 with ldmatrix.x4, issuing 2x fewer load instructions for
the same data. Reduces instruction scheduling pressure.

**Our status:** Already using ldmatrix_x4 for Q and K. Our V uses
ldmatrix_x4_trans. No action needed.

### 4. Single Buffer for V (v4 -> v5: +3.4 TFLOPS)

Key insight: V only needs a single buffer because V is consumed immediately after
loading (no overlap needed with previous iteration's V). Only K needs double
buffering because K from the next iteration is prefetched during current
computation.

This recovers shared memory and allows BLOCK_KV to increase from 32 to 64.

**Our status:** We use double-buffer for both K and V. This is a POTENTIAL
optimization target. However, our tile config is different (BQ=64, BKV=64, D=64
vs gau-nernst's BQ=128, BKV=64, D=128), so the smem savings may not translate
to the same gain.

**Suggested experiment:** Try single-buffering V while keeping K double-buffered.
This would free BKV*D*2 = 64*64*2 = 8KB of shared memory, which could enable
larger BKV or better L1 cache behavior.

### 5. Threadblock Configuration Details

- BLOCK_Q=128, BLOCK_KV=64, DIM=128
- NUM_WARPS=4 (128 threads per block)
- ~40KB dynamic shared memory

Register allocations per warp:
- Q fragments: [WARP_Q/16][8][4] registers
- K fragments: [BLOCK_KV/8][8][2] registers
- Softmax accumulators: [WARP_Q/16][BLOCK_KV/8][4] floats
- Output O: [WARP_Q/16][DIM/8][4] floats

**Comparison with our kernel:**
| | gau-nernst | Our kernel |
|--|-----------|------------|
| BLOCK_Q | 128 | 64 (dynamic 128) |
| BLOCK_KV | 64 | 64 |
| DIM | 128 | 64 |
| Warps | 4 | 4 |
| Blocks/SM | ~2-3 | 3 |
| Regs | ~160 (est.) | 145 |

### 6. Online Softmax Details

Uses the same online softmax approach as our kernel:
- Maintains attention state [m, O_unnorm, sumexp]
- Row reduction: 2 consecutive elements -> 4-thread butterfly via __shfl_xor_sync()
- Rescaling: exp(m_old - m_new) factor

**Our status:** Identical approach. No action needed.

---

## Profiling Insights

**v1 bottleneck:** Stall Short Scoreboard (shared memory access) -- 16-way bank conflicts
**v2 bottleneck:** Stall Long Scoreboard (global memory latency)
**v3+ bottleneck:** Instruction scheduling becomes limiting factor

This matches our profiling progression exactly: bank conflicts -> memory latency ->
math_pipe_throttle. The blog confirms that once bank conflicts and memory latency
are resolved, instruction scheduling is the final wall on sm_120.

---

## FP8 Status

The author mentions MXFP8/NVFP4 MMA support is unavailable in Triton but feasible
in CUDA C++ for sm_120. **No FP8 implementation is provided** -- the blog is BF16
only. Our FP8 attention at 2.33x SDPA appears to be ahead of this work.

---

## What's New That We Should Try

1. **Single-buffer V optimization** -- The only technique we haven't tried. Free up
   8KB smem by using single buffer for V, keeping double buffer for K only.

2. **Benchmark comparison** -- At D=128, our kernel achieves 1.00x SDPA while
   gau-nernst achieves 197.74/186.73 = 1.06x. Our D=128 config may benefit from
   the BQ=128 tile size that gau-nernst uses (they achieve better SM utilization
   with larger Q tiles at D=128).

---

## Caveats

1. **Different primary config.** gau-nernst uses D=128, we primarily benchmark D=64.
   Techniques that help at D=128 may not help at D=64 (different register pressure,
   different smem/compute ratio).

2. **CuDNN beats both implementations.** CuDNN achieves 203.61 TFLOPS (97.2% SOL)
   vs gau-nernst's 197.74 (94.4%). The remaining 3% gap is likely from proprietary
   CuDNN optimizations (possibly 2.5x buffering strategy or different instruction
   scheduling).

3. **No FP8 comparison.** Our FP8 kernel is in uncharted territory -- there is no
   comparable open-source FP8 flash attention targeting sm_120.
