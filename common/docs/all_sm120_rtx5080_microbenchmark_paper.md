# RTX 5080 (sm_120) Microarchitecture Microbenchmarks

**Sources:**
- [Dissecting the NVIDIA Blackwell Architecture with Microbenchmarks](https://arxiv.org/abs/2507.10789) -- Jarmusch, Graddon, Chandrasekaran (tests RTX 5080, sm_120)
- [Microbenchmarking NVIDIA's Blackwell Architecture](https://arxiv.org/abs/2512.02189) -- complementary study (tests B200, sm_100)

**Relevant to:** all workers (attention, GEMM, fused-mlp, linalg, numerical)
**Worker's current problem:** All workers are optimizing kernels on sm_120 (RTX 5090) using empirical iteration. Precise microarchitectural measurements help predict which optimizations will help before coding.
**Date:** 2026-03-15

---

## What This Is

An academic paper that microbenchmarks the RTX 5080 (sm_120, consumer Blackwell)
using targeted tests for memory hierarchy, execution pipelines, and tensor cores.
Compares against H100 PCIe (Hopper). The RTX 5080 shares the same sm_120
architecture as our RTX 5090, differing only in SM count and clocks.

---

## Why It Matters for Us

We have been optimizing by experiment without precise hardware measurements.
This paper provides ground-truth data for sm_120 that can guide optimization
strategy before writing code.

---

## KEY FINDING 1: Tensor Core (MMA) Latency and Throughput

**MMA completion latency (single instruction, ILP=1, 1 warp): ~1.2 cycles**

This is extremely fast -- significantly lower than Hopper's mma instruction
latency. The paper shows MMA throughput increases with ILP and active warps:

- **Optimal ILP:** 6 instructions per thread at 25+ active warps
- **Sustained throughput:** >11 TFLOPS per SM at full ILP/warp saturation
- **SASS mapping:** FP8/FP6 MMA -> QMMA instruction, FP4 MMA -> OMMA instruction

**Implication for workers:** Our kernels already use 4 warps per block with
3-6 blocks/SM (12-24 warps). At ILP=6 with 12+ warps, we should be near
maximum tensor core throughput. The remaining bottleneck is non-MMA
instructions (softmax, conversion) starving the MMA pipeline -- consistent
with our math_pipe_throttle observations.

---

## KEY FINDING 2: Memory Hierarchy

### Shared Memory
- Combined unified L1/shared memory: 128 KB per SM
- Configurable shared memory up to 99 KB/SM (matches our empirical finding)
- **Bank conflict sensitivity:** More sensitive to stride-4 access patterns
  than Hopper. Steeper latency increases under warp pressure (6-32 warps).
  Smaller partition size creates saturation bottlenecks.
- L1 cache latency: 30-40 cycles for hits

**Implication:** Our XOR swizzle is even more important on sm_120 than it
would be on Hopper. The paper confirms sm_120 shared memory is more
conflict-sensitive, validating that bank conflict elimination was our
single biggest optimization.

### L2 Cache
- **Size:** 65 MB total (single monolithic partition)
- **Latency:** ~358 cycles for standard L2 hits
- Larger aggregate bandwidth than partitioned designs under extreme load

**Implication:** The 65 MB L2 is significant for GEMM -- a 4096x4096 BF16
matrix is 32 MB, fitting entirely in L2. Our CTA swizzle for L2 reuse in
the FP8 GEMM is well-motivated by this cache size.

### Global Memory
- Peak read bandwidth: 8.2 TB/s (RTX 5080 specific)
- Peak write bandwidth: 1.6 TB/s
- Access latency: ~877 cycles

**Note:** RTX 5090 has higher bandwidth (1792 GB/s) than RTX 5080 due to
wider bus and more memory chips. The ratios should be similar.

---

## KEY FINDING 3: Execution Unit Details

### INT32/FP32
- Unified INT32/FP32 execution units (can execute either per clock)
- True latency: 4 cycles for both INT32 and FP32
- Cannot execute INT32 AND FP32 simultaneously (unlike earlier architectures)

**Implication:** Address computation (INT32) and arithmetic (FP32) compete
for the same pipeline. This means softmax FP32 operations (exp2f, multiply,
add) share pipeline resources with pointer arithmetic. Converting
address calculations to compile-time constants (via constexpr/template)
frees cycles for FP32 math.

### FP64
- Only 2 FP64 execution units per SM
- Likely emulated using FP32 units or tensor cores
- Much lower throughput than dedicated FP64 hardware on datacenter GPUs

**Implication for numerical workers:** Native FP64 on sm_120 is extremely
slow. The BF16x9 emulation approach (cuSOLVER 13.2) or tensor-core-based
FP64 (if available) may actually be faster than native FP64 for
factorization kernels. This should be profiled.

---

## KEY FINDING 4: Warp Scheduling on sm_120

The paper reveals several important scheduling characteristics:

- **Smoother throughput ramp** vs Hopper (more consistent throughput increase
  with warp count -- less irregular "staircase" behavior)
- **More conservative issue strategy** for dependent instruction chains
- **Better optimization for regular high-ILP kernels** with clean control flow
- **Improved mixed INT32/FP32 throughput** compared to separate pipelines

**Latency hiding:**
- Throughput increases steadily for 1-9 dependent instructions
- Lower throughput for short chains (insufficient ILP)
- Warp scheduler effectively overlaps execution at longer chain lengths

**Implication:** This aligns with our finding that 6 blocks/SM (24 warps)
works well for GEMM -- the warp scheduler benefits from having more warps
to choose from. For attention (3 blocks/SM, 12 warps), the scheduler has
fewer options, which is why the softmax sequential phase creates stalls
that occupancy cannot compensate for.

---

## KEY FINDING 5: Power Consumption by Precision

| Precision | Power (per SM, est.) |
|-----------|---------------------|
| FP4 e2m1 | 16.8 W |
| FP6 e2m3 | 39.4 W |
| FP6 e3m2 | 46.7 W |
| FP8 e4m3 | 46.7 W |
| FP8 e5m2 | 46.8 W |

**Key observation:** FP4 uses only ~36% of the power of FP8 while achieving
2x throughput. FP8 and FP6 e3m2 have nearly identical power consumption,
suggesting they share physical circuitry.

**Implication:** For power-limited scenarios (GPU under thermal throttling),
FP4 could provide better perf/watt. However, NVFP4 MMA requires tcgen05
(sm_100 only), not available on sm_120.

---

## KEY FINDING 6: FP8 GEMM Performance Anomaly

The paper reports that FP8 dense GEMM on RTX 5080 "significantly underperforms
Hopper across matrix sizes":
- 8192^3: 0.233 TFLOPS (RTX 5080) vs 0.887 TFLOPS (H100)
- Inconsistent latency spikes with larger matrices

This appears to use a naive FP8 GEMM (not our optimized kernel). Our FP8 GEMM
achieves 1.29x cuBLAS, which is dramatically better. The paper's result likely
reflects:
1. Naive implementation without conversion optimization
2. Missing vectorized cvt.e4m3x2.f32 (scalar conversion overhead)
3. No CTA swizzle or tile optimization

**Implication:** This validates that our FP8 GEMM optimizations (vectorized
conversion, CTA swizzle, dual-dispatch tiling) are providing massive speedups
over naive approaches. The gap between naive and optimized FP8 on sm_120 is
much larger than on Hopper.

---

## Caveats

1. **RTX 5080 vs RTX 5090:** Same sm_120 architecture but RTX 5080 has fewer
   SMs (84 vs 170), less memory (16GB vs 32GB), lower clocks, and different
   memory bandwidth. Per-SM measurements transfer directly; whole-GPU numbers
   do not.

2. **Paper from July 2025 (arXiv v2).** Some findings may be affected by
   driver or CUDA toolkit updates since then (we use CUDA 13.0+).

3. **No tensor core throughput measurement in TFLOPS for sm_120.** The paper
   measures latency and ILP curves but doesn't provide peak sustained TFLOPS
   numbers comparable to our benchmarks. The ">11 TFLOPS per SM" figure is
   a lower bound from their test configurations, not a peak measurement.
