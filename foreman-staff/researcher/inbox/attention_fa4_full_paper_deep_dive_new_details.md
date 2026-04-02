# FlashAttention-4 Full Paper -- New Details Beyond Existing Briefs

**Source:** [https://arxiv.org/html/2603.05451v1](https://arxiv.org/html/2603.05451v1)
**Relevant to:** attention worker
**Worker's current problem:** BF16 at 94% compiler ceiling (68 us). FP8 at 2.33x SDPA (52 us). Math_pipe_throttle 48% from softmax.
**Date:** 2026-03-15
**Supplements:** attention_fa4_portable_softmax_optimizations.md, attention_flashattention4_portable_algorithmic_innovations.md, attention_fa4_precise_coefficients_lazy_rescaling_update.md

---

## What's New in This Brief

The existing FA4 briefs cover the conditional rescaling and polynomial exp2
techniques. This brief covers additional details from the full paper that are
NOT in the existing briefs.

---

## NEW FINDING 1: LPT Scheduling (Portable, Architecture-Agnostic)

FA4 introduces Longest-Processing-Time-first (LPT) scheduling for causal
attention. This is a grid launch optimization, not a kernel change.

**Problem:** In causal attention, blocks near the diagonal process fewer KV
blocks than blocks far from the diagonal. With default grid ordering:
- First blocks (far from diagonal): process all KV blocks -- long runtime
- Last blocks (near diagonal): process few KV blocks -- short runtime
- Result: SMs finish early blocks last, creating tail stalls

**Solution:** Reorder block assignments so the longest-running blocks launch first.
This is the same concept as ProgramID remapping from the NVIDIA cuTile blog,
but formalized.

```cuda
// Default grid ordering:
int block_m = blockIdx.x;  // 0, 1, 2, ... num_m_blocks

// LPT ordering (reverse for causal):
int block_m = num_m_blocks - 1 - blockIdx.x;
```

**FA4 result:** "Validated as improvement to FlashAttention-3 on Hopper" -- the
paper confirms this is portable and beneficial across GPU generations.

**Our status:** Not implemented. The NVIDIA cuTile blog estimated 1-2.6%
improvement on B200. For our RTX 5090 with 170 SMs and relatively small grids
(B=2, H=8 = 16 blocks at BQ=64), the effect is small but free. At larger batch
sizes or longer sequences, the benefit increases.

**Verdict:** Low effort, small but guaranteed improvement. Implement as a
grid launch change -- no kernel code modifications needed.

---

## NEW FINDING 2: CuTe-DSL Compilation Speed (Context Only)

FA4 is implemented entirely in CuTe-DSL (Python, not C++):
- Forward compile time: 2.5 seconds
- Backward compile time: 1.4 seconds
- vs FA3 C++ templates: 55s forward, 45s backward (20-32x faster)

**Relevance to our work:** None directly -- we write CUDA C++. But this data
point suggests NVIDIA is investing heavily in the CuTe-DSL path for kernel
development. If we ever need to iterate faster on experimental kernels, CuTe-DSL
could accelerate the experiment-measure-iterate cycle significantly.

The CuTe-DSL achieves "within 2% of handwritten C++ on Blackwell" according to
CUTLASS release notes. For rapid prototyping of new tile configurations or
softmax variants, this could be valuable.

---

## NEW FINDING 3: FA4 Performance Breakdown (B200 BF16)

| Metric | Value |
|--------|-------|
| Peak forward TFLOPS | 1613 (71% of theoretical peak) |
| vs cuDNN 9.13 | 1.1-1.3x faster |
| vs Triton | 2.1-2.7x faster |
| Sequence lengths tested | 1K, 2K, 4K, 8K, 16K, 32K |
| Head dimensions tested | 64, 128, 192 |

**Key insight for our work:** Even on B200 with tcgen05/TMEM, FA4 only achieves
71% utilization. This means ~29% of cycles are non-MMA overhead (softmax,
rescaling, loads). On sm_120 with mma.sync, our 94% of compiler ceiling (which
is itself perhaps 70-75% of peak) puts us at roughly the same absolute
utilization level -- validating that the mma.sync overhead is comparable to
the tcgen05 overhead for attention workloads.

---

## NEW FINDING 4: Deterministic Backward Pass

FA4 introduces a deterministic backward pass using:
- Semaphore-based CTA ordering (wait for previous CTA to finish dQ writes)
- CTA swizzling to ensure consistent execution order
- Atomic operations for partial dQ accumulation

**Result:** Deterministic backward achieves ~75% speed of non-deterministic
1-CTA baseline.

**Relevance:** If we ever implement backward pass attention, the deterministic
approach trades 25% performance for bit-exact reproducibility. The semaphore
pattern (atomicExch for ordering, fence.mbarrier for visibility) is portable
to sm_120.

---

## NEW FINDING 5: Blackwell SFU Throughput Context

FA4 paper states: "B300 and GB300 GPUs have doubled exponential throughput to
32 ops/clock/SM." This means Blackwell Ultra has 2x SFU throughput vs standard
Blackwell.

**What this tells us about sm_120:** Standard Blackwell (including sm_120) has
16 ops/clock/SM for exponential operations. This is the same as Hopper. The
doubling is a Blackwell Ultra (B300) feature only.

Our softmax uses 64 exp2f calls per KV block (D=64). At 16 ops/clock, this
takes 4 clocks per warp. With 4 warps, that's 16 clock cycles of SFU
contention per KV block -- a small but real contributor to math_pipe_throttle.

---

## WHAT'S NOT PORTABLE (Confirming Previous Briefs)

The following FA4 techniques are datacenter-only (sm_100 tcgen05):
- Tensor Memory (TMEM) for accumulator storage -- NOT on sm_120
- Fully asynchronous MMA to TMEM -- NOT on sm_120
- 2-CTA MMA mode (CTA pair cooperative MMA) -- NOT on sm_120
- 128x128 MMA tiles (vs our 16x8x16/16x8x32) -- NOT on sm_120
- Distributed shared memory (DSMEM) between CTAs -- NOT on sm_120

**Summary of what IS portable:**
1. Conditional rescaling (tau=8.0) -- algorithm only, no hardware deps
2. Polynomial exp2 approximation -- FMA-based, any GPU
3. LPT scheduling -- grid launch ordering, any GPU
4. Deterministic backward semaphore pattern -- atomics, any GPU

---

## Suggested Priority for Our Worker

1. **Conditional rescaling** (highest expected impact, 2-5% estimated)
2. **LPT scheduling** (trivial to implement, 1-2% estimated)
3. **Polynomial exp2** (only if ncu shows SFU saturation; likely minimal for D=64)
4. **Deterministic backward** (future, only if backward pass is implemented)
