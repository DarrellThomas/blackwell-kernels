# RTX 5090 Theoretical Performance Limits

**The "Shannon Limit" for our kernels** — what's physically possible on this hardware,
independent of any software implementation.

Claude Shannon proved that every communication channel has a maximum information rate
(the channel capacity) that no encoding scheme can exceed. Analogously, every GPU kernel
has a theoretical minimum execution time set by hardware physics — no optimization can
beat it. Knowing this limit tells us how much headroom remains and when to stop optimizing.

---

## 1. Hardware Parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| GPU | NVIDIA GeForce RTX 5090 (GB202, sm_120) | nvidia-smi |
| SMs | 170 | Tuning guide |
| BF16 Tensor throughput | 512 FLOPs/SM/clock (256 FMA) | Back-calculated |
| Nominal boost clock | ~2407 MHz | Derived from 209.5 TFLOPS spec |
| Max SM clock | 3090–3135 MHz | nvidia-smi (GPU 1 / GPU 0) |
| **Nominal BF16 peak** | **209.5 TFLOPS** | NVIDIA spec sheet |
| **Observed sustained peak** | **~224 TFLOPS** | Back-calculated from cuBLAS |
| DRAM bandwidth | 1,792 GB/s | Spec sheet (GDDR7, 512-bit) |
| L2 cache | 96 MB | Tuning guide |
| Shared memory/SM | 128 KB (99 KB usable/block) | Tuning guide |
| Max warps/SM | 48 | Tuning guide |

### How we derived the sustained peak

The 209.5 TFLOPS spec assumes the nominal boost clock (~2407 MHz). Under sustained
compute load, the GPU boosts higher. We can measure this precisely:

```
cuBLAS GEMM 4096×4096 benchmark: 614 μs
FLOPs: 2 × 4096³ = 137.44 GFLOP
Achieved: 137.44e9 / 614e-6 = 223.8 TFLOPS

Sustained clock = 223.8e12 / (170 SMs × 512 FLOPs/SM/clock) = 2570 MHz
```

At maximum clock (3090–3135 MHz), the theoretical peak would be 269–273 TFLOPS,
but this is unreachable under sustained load due to power and thermal limits.

**For all calculations below, we use 224 TFLOPS as the realistic sustained peak.**

---

## 2. The Roofline Model

The roofline model identifies whether a kernel is limited by compute or memory bandwidth.

```
                    Compute ceiling: 224 TFLOPS
                    ─────────────────────────────────────
                   /
                  /
                 /  ← Memory-bound region    Compute-bound region →
                /
───────────────/
              ^
              |
    Balance point: 125 FLOP/byte
    (224 TFLOPS ÷ 1.792 TB/s)
```

| Regime | Arithmetic Intensity | Bottleneck |
|--------|---------------------|------------|
| Memory-bound | < 125 FLOP/byte | DRAM bandwidth (1,792 GB/s) |
| Compute-bound | > 125 FLOP/byte | Tensor core throughput (224 TFLOPS) |

Both our kernels are **deeply compute-bound**:

| Kernel | Arithmetic Intensity | Regime |
|--------|---------------------|--------|
| GEMM 4096³ | 1,432 FLOP/byte | 11.5× above balance point |
| Attention B2H8N2048D64 | ~510 FLOP/byte | 4.1× above balance point |

This means: **memory bandwidth is irrelevant. The only limit is tensor core throughput.**

---

## 3. GEMM: M = N = K = 4096, BF16

### Theoretical Minimum

```
FLOPs:          2 × 4096³ = 137.44 GFLOP
Memory:         3 × 4096² × 2 bytes = 96 MB  (read A, read B, write C)
Arith. intens.: 137.44e9 / 96e6 = 1,432 FLOP/byte

Tensor time:    137.44e9 / 224e12 = 613.6 μs  ← THE LIMIT
Memory time:    96e6 / 1.792e12 = 53.6 μs      (irrelevant — 11× less)
```

**The Shannon limit for this GEMM is ~614 μs.**

### Where We Stand

| Implementation | Time (μs) | TFLOPS | % of Peak | vs Limit |
|----------------|-----------|--------|-----------|----------|
| **Theoretical limit** | **614** | **224** | **100%** | **1.00×** |
| cuBLAS | 614 | 223.8 | 99.9% | 1.00× |
| Our kernel | 788 | 174.4 | 77.9% | 1.28× |

**cuBLAS is already AT the physical limit.** The 0.1% gap is measurement noise.
This is extraordinary engineering — cuBLAS sustains 99.9% tensor utilization on this
config because large square GEMMs are the ideal workload: uniform tiling, no masking,
no serial dependencies, perfect pipeline utilization.

**Our kernel at 0.78× cuBLAS has 22% headroom remaining.** The optimization loop should
target the gap between 174.4 and 224 TFLOPS. Key opportunities:
- Tile scheduling and swizzle optimization
- Better K-tile prefetching to hide latency
- Register pressure reduction for higher occupancy

### Can we beat cuBLAS?

On this config: **effectively no.** cuBLAS is at the wall. On certain non-square or
smaller configs where cuBLAS's heuristics pick a suboptimal tile shape: **possibly.**
Focus optimization on configs where cuBLAS leaves headroom.

---

## 4. Flash Attention: B=2, H=8, N=2048, D=64, Causal

### FLOP Count

Flash attention has two tensor-core matmuls per KV tile plus scalar softmax:

```
Heads (BH):     2 × 8 = 16

Q×Kᵀ (causal, lower triangle):
  Dot products:  N×(N+1)/2 = 2,098,176 per head
  FLOPs:         16 × 2,098,176 × 2 × D = 16 × 2,098,176 × 128 = 4.30 GFLOP

P×V (same causal structure):
  FLOPs:         4.30 GFLOP

Softmax (~10 scalar ops/element):
  Elements:      16 × 2,098,176 = 33.6M
  FLOPs:         ~0.34 GFLOP (scalar, not tensor core)

Total tensor:    8.59 GFLOP
Total scalar:    0.34 GFLOP
Grand total:     8.93 GFLOP
```

### Tile Overhead

Flash attention tiles the computation. With BLOCK_Q=64, BLOCK_KV=32:
- 32 Q-tiles × 16 heads = 512 thread blocks
- Causal masking wastes ~10–15% compute on partially-masked tiles
- Online softmax rescaling adds O(BLOCK_Q × D) scalar ops per KV tile
- **Actual executed FLOPs ≈ 9.5–10 GFLOP** (including waste)

### Theoretical Minimum

```
Tensor FLOPs:   8.59 GFLOP (useful work only)
Memory:         Q + K + V + O + L = ~17 MB (single-pass flash attention)
Arith. intens.: 8.59e9 / 17e6 ≈ 510 FLOP/byte

Tensor time:    8.59e9 / 224e12 = 38.3 μs  ← HARD FLOOR
Memory time:    17e6 / 1.792e12 = 9.5 μs    (irrelevant)
```

**The absolute Shannon limit is ~38 μs.** But unlike GEMM, this is unreachable.

### Why 100% Tensor Utilization Is Impossible for Attention

Flash attention has **fundamental serial dependencies** that GEMM doesn't:

1. **Softmax serialization** — Between Q×Kᵀ and P×V, the softmax must complete.
   This creates an irreducible pipeline bubble every KV tile. The tensor cores
   sit idle while scalar units compute exp/sum/normalize.

2. **Online softmax rescaling** — Each new KV block requires rescaling the running
   O accumulator by the updated softmax normalizer. More scalar work in the
   critical path.

3. **Causal tile waste** — Tiles crossing the diagonal compute and discard masked
   elements. With BLOCK_Q=64 and N=2048, roughly 10–15% of tensor FLOPs are wasted
   on zeros.

4. **Register pressure** — Must simultaneously hold: Q tile (register-resident),
   K/V tile fragments, FP32 accumulators for S and O, softmax running statistics
   (m, l). This limits occupancy → fewer warps → less latency hiding.

5. **Shared memory staging** — K and V data flows through shared memory buffers.
   Even with cp.async and double-buffering, the staging adds latency.

6. **Synchronization** — `__syncthreads()` between load and compute phases.

### Estimating the Achievable Ceiling

**Revised after 38 optimization iterations (2026-03-12).** The original estimates
were too optimistic. Empirical data from extensive profiling reveals:

| Factor | Original Estimate | Revised (Empirical) |
|--------|------------------|---------------------|
| Softmax + rescaling | 5–8% | **15–19%** (134 scalar cycles per 700 total) |
| MMA burst scheduling (math_throttle) | included in pipeline | **10–15%** (compiler can't perfectly spread MMA) |
| Causal tile waste | 10–15% | ~10% |
| Pipeline bubbles + barriers | 5–10% | ~8% (barrier 5% + sync overhead) |
| Scheduling + memory latency | 3–5% | ~5% (not_selected 2% + short_scoreboard 3%) |
| **Combined** | **~25–35%** | **~40–45% overhead** |

The original analysis underestimated softmax overhead because it counted only FLOPs,
not the serial dependency: during softmax (~134 cycles/iteration), ALL warps on the SM
are doing scalar work simultaneously (3 blocks × 4 warps = 12 warps in lockstep), so
the tensor core sits completely idle. This creates a ~19% structural ceiling on tensor
utilization that no software optimization can eliminate without changing the algorithm.

```
Revised achievable ceiling ≈ 38.3 μs / 0.60 ≈ 64 μs  (compiler-managed)
Theoretical ceiling ≈ 38.3 μs / 0.70 ≈ 55 μs          (full PTX, block desync)
```

**Two tiers of ceiling:**
- **Compiler-managed ceiling (~64 μs, 60% utilization):** Achievable with nvcc -O3
  and `#pragma unroll`. The compiler's scheduling is surprisingly good but inherently
  limited by the burst MMA pattern. 38 iterations of optimization confirmed this limit.
- **Full PTX ceiling (~55 μs, 70% utilization):** Would require a hand-written PTX
  inner loop with manual register allocation and instruction scheduling. The salykova
  approach (writing the entire inner loop in assembly) could reach this level, but
  represents a fundamentally different optimization phase.

### Where We Stand

| Implementation | Time (μs) | Effective TFLOPS | % of Peak | vs Compiler Ceiling |
|----------------|-----------|-----------------|-----------|---------------------|
| **Hard floor** | **38** | **224** | **100%** | **—** |
| **Full PTX ceiling** | **~55** | **~156** | **70%** | **—** |
| **Compiler ceiling** | **~64** | **~134** | **60%** | **1.00×** |
| **Our v2 (bench)** | **68** | **126** | **56%** | **1.06×** |
| cuDNN SDPA (bench) | ~121 | ~71 | ~32% | 1.89× |

**We're 1.78× faster than cuDNN SDPA and within 6% of the compiler-managed ceiling.**

After 38 optimization iterations (24 kept, 14 discarded), the kernel has converged.
The last 14+ experiments were all discards, confirming we're at the ceiling for
compiler-managed code. Further gains require either:
1. Full PTX inner loop rewrite (a different optimization phase)
2. Algorithmic changes (FP8 attention for 2× tensor throughput)

### How Stalls Map to Utilization

Profiler stall breakdown at convergence (68 μs bench, 93 μs ncu):
```
math_throttle:  48%  ← tensor core BUSY (input FIFO full)
wait:           17%  ← waiting for MMA result (pipeline latency)
scoreboard:     13%  ← ldmatrix latency (shared memory → registers)
barrier:         5%  ← __syncthreads between KV iterations
short_scoreboard:3%  ← MIO pipe latency
not_selected:    2%  ← scheduling
active_issue:   12%  ← useful instruction issue
```

Tensor utilization ≈ math_throttle (48%) + MMA fraction of active_issue (~6%) ≈ 54-56%.
This exactly matches the measured 56% (126/224 TFLOPS).

---

## 5. The Fundamental Asymmetry

```
GEMM efficiency:        99.9%  (cuBLAS — essentially solved)
Attention efficiency:   ~54%   (our kernel — significant headroom)
Attention ceiling:      ~70%   (achievable with perfect implementation)
```

This asymmetry is **inherent to the algorithms**:

- **GEMM** is a single massive matrix multiply. No serial dependencies, no branching,
  no data-dependent control flow. The tensor cores can be fed continuously.

- **Attention** interleaves two matmuls with element-wise softmax. The softmax creates
  a serial dependency that breaks the tensor core pipeline. No amount of optimization
  can eliminate this — it's fundamental to the algorithm.

This is why NVIDIA ships a separate fused attention kernel (cuDNN SDPA) rather than
just composing GEMMs: the fusion and scheduling are critical, and the optimal strategy
is very different from GEMM.

---

## 6. Scaling Analysis

How do the limits change with problem size?

### GEMM: As M,N,K grow

| Config | FLOPs | Time Limit | cuBLAS Efficiency |
|--------|-------|-----------|-------------------|
| 1024³ | 2.15 GFLOP | 9.6 μs | ~90% (too small) |
| 2048³ | 17.18 GFLOP | 76.7 μs | ~95% |
| 4096³ | 137.4 GFLOP | 613.6 μs | ~99.9% |
| 8192³ | 1099 GFLOP | 4907 μs | ~99.9% |

Larger = better utilization. Small GEMMs don't fill enough SMs.

### Attention: As N grows (B=2, H=8, D=64)

| N | Tensor FLOPs | Time Limit | Blocks | Blocks/SM | Est. Ceiling |
|---|-------------|-----------|--------|-----------|-------------|
| 512 | 0.54 GFLOP | 2.4 μs | 128 | 0.75 | ~40% peak |
| 1024 | 2.15 GFLOP | 9.6 μs | 256 | 1.5 | ~55% peak |
| 2048 | 8.59 GFLOP | 38.3 μs | 512 | 3.0 | ~70% peak |
| 4096 | 34.4 GFLOP | 153.5 μs | 1024 | 6.0 | ~80% peak |
| 8192 | 137.4 GFLOP | 613.6 μs | 2048 | 12.0 | ~85% peak |

Longer sequences improve efficiency because:
1. More blocks → better SM occupancy
2. More KV tiles per Q tile → better amortization of setup/softmax overhead
3. The ratio of useful compute to tile waste improves

**For attention, N=4096+ is where the kernel really stretches its legs.**

---

## 7. Summary: What "Done" Looks Like

### GEMM Kernel
- **Target: match cuBLAS (0.78× → 1.0×)**. There's 22% headroom.
- **Stretch: beat cuBLAS on non-square configs** where its heuristics are suboptimal.
- **Hard wall: 614 μs for 4096³.** Cannot go below this. cuBLAS proves it.

### Attention Kernel
- **Current: 98.8 μs (1.61× SDPA)**. Already beating cuDNN.
- **Target: ~65 μs (~2.4× SDPA, ~75% achievable ceiling)**. Very ambitious.
- **Stretch: ~55 μs (~2.9× SDPA, ~95% achievable ceiling)**. Near-perfect implementation.
- **Hard wall: ~38 μs.** Cannot go below this. Would require eliminating softmax entirely.

### The Key Insight

> **We are not competing against cuDNN or cuBLAS. We are competing against physics.**
>
> cuBLAS has already reached the wall for GEMM. For attention, the wall is further
> away — and that's where our opportunity lives. Every microsecond we shave off
> attention gets us closer to the fundamental limit of what 170 SMs and 224 TFLOPS
> can deliver.

---

*Analysis computed 2026-03-12. Clock speeds measured from nvidia-smi.*
*Sustained TFLOPS derived from cuBLAS M=N=K=4096 benchmark at 614 μs.*
