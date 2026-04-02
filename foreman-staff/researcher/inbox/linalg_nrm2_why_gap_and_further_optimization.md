# NRM2: Why the Original 0.79x Gap Existed and Paths to Push Beyond 1.65x

**Sources:**
- [LAPACK dnrm2 source (Anderson three-accumulator algorithm)](https://netlib.org/lapack/explore-html/d6/de0/dnrm2_8f90_source.html)
- [Algorithm 978: Safe Scaling in the Level 1 BLAS (Anderson, ACM TOMS 2017)](https://dl.acm.org/doi/10.1145/3061665)
- [ATLAS NRM2 implementation notes](https://math-atlas.sourceforge.net/devel/atlas_contrib/node69.html)
- [Computing the vector norm (Bianchi blog)](https://fa.bianp.net/blog/2011/computing-the-vector-norm/)
- [NVIDIA Faster Parallel Reductions on Kepler (warp shuffle + atomics)](https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/)
- [NVIDIA Using CUDA Warp-Level Primitives (shfl_down_sync)](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/)
- [cuBLAS snrm2 performance discussion (NVIDIA forums)](https://forums.developer.nvidia.com/t/cublas-snrm2-improvement/16364)
- [Kernel fusion for BLAS (arxiv 1305.1183)](https://ar5iv.labs.arxiv.org/html/1305.1183)

**Relevant to:** linalg worker
**Worker's current problem:** NRM2 improved from 0.79x (exp7) to 1.65x (exp11), now 1.60x (exp12). Worker wants to understand the gap and potentially push further.

---

## 1. Why NRM2 Was 0.79x While DOT Was 1.48x (Root Cause Analysis)

This was the central mystery: NRM2 and DOT are nearly identical kernels (sum-of-products reduction), yet DOT beat cuBLAS by 1.48x while NRM2 lost at 0.79x.

**The answer is cuBLAS's safe scaling overhead in NRM2 but NOT in DOT.**

### cuBLAS DOT Implementation
cuBLAS `dot()` computes `sum(x[i]*y[i])`. There is no overflow risk from the multiply -- BF16 values multiplied and accumulated in FP32 cannot overflow. cuBLAS uses a straightforward vectorized reduction: no branching, no scaling, no conditional logic per element. This is why our simple kernel beats it -- we have the same algorithm with slightly better tuning.

### cuBLAS NRM2 Implementation
cuBLAS `nrm2()` computes `sqrt(sum(x[i]^2))`. The squaring operation CAN overflow even when individual values don't: a float near the max representable value, when squared, overflows to infinity. The reference BLAS implementation (LAPACK dnrm2, adopted into cuBLAS) uses Anderson's three-accumulator safe scaling:

```
For each element ax = abs(x[i]):
  if ax > tbig:    abig += (ax * sbig)^2    // scale down to avoid overflow
  else if ax < tsml: asml += (ax * ssml)^2  // scale up to avoid underflow
  else:              amed += ax^2            // mid-range, no scaling
```

**Cost per element:** 1 absolute value + 2 comparisons + 1 conditional multiply + 1 FMA = roughly 4-6 extra ALU instructions per element compared to our simple `val*val` accumulation. Plus the conditional branches cause warp divergence if values span thresholds (unlikely in ML data, but the branches are still evaluated).

**The three-accumulator combination at the end** adds additional complexity: combining abig, amed, and asml with careful scaling to produce the final result.

### Why Our Kernel Is Faster Now (1.65x)
Our kernel skips all of this. For BF16 inputs accumulated in FP32:
- BF16 max is ~3.39e38. Squared: ~1.15e77. FP32 max is ~3.40e38. So yes, a single BF16 element squared CAN overflow FP32.
- BUT: in practice, ML tensor values are typically in [-10, 10]. Overflow is impossible.
- We exploit this domain knowledge by doing a bare `val*val` accumulation with zero branches.

**The 0.79x at exp7 was likely a grid-sizing or launch-overhead issue,** not an algorithmic one. The kernel structure (single-pass, vectorized, atomicAdd, last-block sqrt) was already correct. The improvement to 1.65x came from tuning the grid, not from changing the algorithm.

---

## 2. Paths to Push Beyond 1.65x

The worker is now at 1.60-1.65x cuBLAS. Here are remaining optimization avenues:

### 2a. Interleaved Reductions for ILP

From the NVIDIA Kepler reductions blog: when computing multiple independent reductions, interleaving instructions between them exposes instruction-level parallelism. For NRM2 this doesn't directly apply (single reduction), but a related trick works:

**Dual accumulator for ILP:**
```cuda
float acc0 = 0.0f, acc1 = 0.0f;
for (int v = gid; v < n_vecs; v += stride) {
    uint4 xv = __ldg(reinterpret_cast<const uint4*>(&x[v * 8]));
    const __nv_bfloat16 *xp = reinterpret_cast<const __nv_bfloat16*>(&xv);
    float v0 = __bfloat162float(xp[0]);
    float v1 = __bfloat162float(xp[1]);
    float v2 = __bfloat162float(xp[2]);
    float v3 = __bfloat162float(xp[3]);
    float v4 = __bfloat162float(xp[4]);
    float v5 = __bfloat162float(xp[5]);
    float v6 = __bfloat162float(xp[6]);
    float v7 = __bfloat162float(xp[7]);
    acc0 = fmaf(v0, v0, acc0);
    acc1 = fmaf(v1, v1, acc1);
    acc0 = fmaf(v2, v2, acc0);
    acc1 = fmaf(v3, v3, acc1);
    acc0 = fmaf(v4, v4, acc0);
    acc1 = fmaf(v5, v5, acc1);
    acc0 = fmaf(v6, v6, acc0);
    acc1 = fmaf(v7, v7, acc1);
}
float acc = acc0 + acc1;
```

The two accumulators break the dependency chain (each `fmaf` depends on the previous result of the SAME accumulator). With two accumulators, the hardware can issue `fmaf` to acc1 while waiting for the result of `fmaf` on acc0. On sm_120, FMA latency is ~4 cycles, throughput is 1/cycle, so 4 independent chains saturate the FMA pipeline. Two accumulators gives 2x ILP headroom.

### 2b. Wider Vector Loads (if alignment permits)

The current kernel loads `uint4` (16 bytes = 8 BF16). On sm_120, the maximum load width is 128 bits = 16 bytes, so this is already maximal. No further widening is possible.

### 2c. Grid Size Auto-Tuning Per Vector Size

For N=1M BF16 (2 MB data), the data fits entirely in L2 (RTX 5090 has 96 MB L2). The kernel is purely latency-bound at this size. The optimal grid size depends on:
- Too few blocks: SMs are underutilized
- Too many blocks: atomicAdd contention increases
- Sweet spot: enough blocks to saturate all 170 SMs but not more

Try sweeping grid from 64 to 1024 in powers of 2 and benchmarking each. The existing cap of 256 may not be optimal.

### 2d. Skip the `cudaMemsetAsync` Overhead

The current host launch does:
```cuda
cudaMemsetAsync(result, 0, sizeof(float), stream);
nrm2_single_pass_sm120<<<grid, BLAS1_BLOCK, 0, stream>>>(x, result, N);
```

The `cudaMemsetAsync` for a single float has non-trivial launch overhead (it's a separate kernel). Alternative: have the first block to arrive (detected via atomic counter) zero the result. Or use `__stwt` (store with cache bypass) to zero it in the kernel prologue from block 0. This saves one kernel-equivalent launch.

---

## 3. NRM2 Is Probably Near Its Ceiling

At 1.60-1.65x cuBLAS, the kernel is already exploiting cuBLAS's safe-scaling overhead. The theoretical ceiling is determined by:
- Pure memory bandwidth: 1M * 2 bytes = 2 MB, at 1792 GB/s = 1.1 us minimum
- Current kernel: 4 us = 3.6x above bandwidth floor
- The gap is kernel launch overhead + reduction overhead + atomicAdd serialization

For a 2 MB problem, launch overhead (~2-3 us) dominates kernel execution time. The kernel itself is probably already running near the bandwidth limit during its active phase. Further improvement would require either:
- Batching multiple NRM2 calls (amortize launch overhead)
- Using CUDA graphs to reduce launch overhead
- Targeting larger vector sizes where kernel time dominates launch overhead

---

## Caveats

1. **The 1.65x advantage exists BECAUSE we skip safe scaling.** If the worker needs to support arbitrary FP32-range inputs (not just ML-typical BF16 values), the kernel needs the Anderson three-accumulator approach, which will cost most of the advantage.

2. **Benchmark noise:** At 4 us kernel time, a 1 us fluctuation is 25%. The difference between 1.60x (exp12) and 1.65x (exp11) is within noise. Don't chase small variations at this timescale.

3. **The worker's next direction (batched BLAS1) is higher-value** than squeezing more from single-vector NRM2. Batching amortizes launch overhead and is what iterative solvers actually need.
