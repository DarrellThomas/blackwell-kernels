# NRM2 Optimization: Precision, Grid Tuning, and cuBLAS Internals

**Sources:**
- [LAPACK snrm2 (Anderson Blue algorithm)](https://www.netlib.org/lapack/explore-html/d1/d2a/group__nrm2_gad179c1611098b5881f147d39afb009b8.html)
- [Algorithm 978: Safe Scaling in the Level 1 BLAS (Anderson, ACM TOMS 2017)](https://dl.acm.org/doi/10.1145/3061665)
- [Fast and accurate computation of the Euclidean norm (JJIAM 2023)](https://link.springer.com/article/10.1007/s13160-023-00593-8)
- [Accurate Calculation of Euclidean Norms using Double-word Arithmetic (ACM TOMS 2023)](https://dl.acm.org/doi/10.1145/3568672)
- [Kahan compensated summation on GPU (NVIDIA Forums)](https://forums.developer.nvidia.com/t/how-to-improve-float-array-summation-precision-and-stability/67904)
- [Parallel vectorized Kahan and Gill-Moller algorithms (Dmitruk, CCPaE 2023)](https://onlinelibrary.wiley.com/doi/abs/10.1002/cpe.7763)
- [Compensated summation and dot product for Krylov methods (J. Comp. Appl. Math 2022)](https://www.sciencedirect.com/science/article/abs/pii/S0377042722002047)
- [NVIDIA Faster Parallel Reductions on Kepler (blog)](https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/)
- [NVIDIA CUDA Pro Tip: Grid-Stride Loops (blog)](https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/)
- [cuBLAS snrm2 improvement discussion (NVIDIA Forums)](https://forums.developer.nvidia.com/t/cublas-snrm2-improvement/16364)
- [KBLAS: GPU-Optimized BLAS Routines (GitHub)](https://github.com/ecrc/kblas-gpu)

**Relevant to:** linalg worker
**Worker's current problem:** NRM2 was at 0.79x cuBLAS (exp7), has improved to 1.65x (exp11). The worker wants further optimization and robustness.

---

## Status Update

The worker's NRM2 has already improved dramatically from 0.79x to 1.65x cuBLAS through successive experiments. The current kernel (`nrm2_single_pass_sm120`) uses the correct architecture: single-pass grid-stride vectorized loads (8 BF16/load), warp shuffle + shared memory block reduction, atomicAdd to global accumulator, and last-block sqrt. The ncu profile shows the kernel is memory-bound (long_scoreboard 0.66-0.81, SM utilization 3.1%). This brief provides additional techniques for squeezing out the remaining margin.

---

## 1. Grid Size Tuning (Likely the Biggest Remaining Win)

The current kernel caps grid size at 256 blocks (`if (grid > 256) grid = 256`). For a 1M-element BF16 vector:

- Data volume: 1M * 2 bytes = 2 MB
- With 256 blocks * 256 threads = 65,536 threads, each handling 8 elements per vector load
- Effective elements per grid pass: 65,536 * 8 = 524,288 -- about half the vector

This means the kernel needs ~2 grid-stride iterations. The tradeoff is:
- **Too few blocks:** Underutilizes SMs (170 SMs on RTX 5090). With 256 blocks and 6+ blocks/SM capacity, only ~43 SMs are saturated.
- **Too many blocks:** More atomicAdd contention on the global result.

**Recommendation:** For N=1M BF16 (very small data), the kernel is latency-bound, not bandwidth-bound. The 3.1% SM utilization confirms this. Possible improvements:
- **Reduce block count to match data:** For 1M elements / 8 / 256 = 488 vectors/block at 256 blocks, each block does <2 iterations. Try grid=128 or grid=64 to reduce atomicAdd contention and kernel dispatch overhead.
- **Or increase to saturate SMs:** Try grid=680 (170 SMs * 4 blocks/SM) to maximize occupancy, even though each block processes fewer elements. The overhead per block is just the shared memory reduction + one atomicAdd.
- Profile both extremes with ncu to see which direction helps.

The DOT kernel uses the same grid cap (256) and achieves higher vs-ref ratios. The difference may be in how cuBLAS handles NRM2 vs DOT internally.

---

## 2. How cuBLAS NRM2 Works (and Why It's Beatable)

cuBLAS documentation states that cuBLAS functions "may invoke more than one CUDA kernel" for operations that produce scalar results. For NRM2 specifically:

**Likely two-kernel approach:**
1. Kernel 1: Grid-stride sum-of-squares with block-level reductions, writing partial sums to a temporary buffer
2. Kernel 2: Small reduction kernel over the partial sums, plus sqrt

**Why this is slower than single-kernel:**
- Kernel launch overhead: 2-5 us per launch on modern GPUs
- For small vectors (N=1M BF16 = 2 MB), the kernel runtime is ~4-8 us -- the launch overhead is a significant fraction
- The intermediate buffer write/read adds memory traffic

**cuBLAS also uses safe scaling** (Anderson/Blue algorithm style), which adds conditional branches per element. For BF16 inputs accumulated in FP32, overflow/underflow is extremely unlikely (BF16 max ~3.4e38 squared = ~1.2e77, well within FP32 range). Our kernel skips this overhead entirely.

**Key insight:** cuBLAS is designed for correctness across all possible inputs (including extreme values near FP32 overflow). Our kernel is designed for ML workloads where BF16 values are typically in [-10, 10]. This is a legitimate advantage.

---

## 3. The Anderson/Blue Safe Scaling Algorithm

LAPACK's reference NRM2 (since 2017) uses Anderson's improved Blue algorithm. Understanding it helps explain cuBLAS overhead and when we might need it:

**Three-accumulator approach:**
```
For each element ax = abs(x[i]):
  if ax > tbig:    abig += (ax * sbig)^2    // scale down large values
  else if ax < tsml: asml += (ax * ssml)^2  // scale up small values
  else:              amed += ax^2            // no scaling needed
```

- `tbig` / `tsml` are thresholds derived from the floating-point representation
- `sbig` / `ssml` are corresponding scale factors
- After the loop, the three accumulators are carefully combined

**Cost per element:** One absolute value, two comparisons, one conditional multiply, one FMA. This is ~4-6 extra instructions per element compared to our simple `val*val` accumulation.

**When we need it:** Only if computing norms of FP32 or FP64 vectors with values near the extremes of the representable range. For BF16 inputs accumulated in FP32, the dynamic range is so limited that overflow is impossible and underflow is irrelevant. **Our simple approach is both faster and sufficient.**

---

## 4. Compensated Summation (Kahan) for GPU Norm Computation

For iterative solvers where norm accuracy affects convergence, compensated summation can improve results:

**Kahan summation in a reduction kernel:**
```cuda
float sum = 0.0f;
float comp = 0.0f;  // compensation term
for (...) {
    float val = x[i] * x[i];
    float y = val - comp;
    float t = sum + y;
    comp = (t - sum) - y;  // captures lost low-order bits
    sum = t;
}
```

**GPU considerations:**
- Kahan summation adds 3 extra FLOPs per element (subtraction, subtraction, subtraction)
- For bandwidth-bound kernels like NRM2, the extra compute is free -- the ALU is idle waiting for memory
- Parallel tree reductions (warp shuffle) are inherently more accurate than serial summation because they reduce O(N) additions to O(log N) -- each partial sum has fewer terms
- For BF16 inputs accumulated in FP32, the precision loss is dominated by the BF16-to-FP32 conversion (7 bits of mantissa), not the summation. Kahan doesn't help the conversion loss.

**Recommendation:** Not needed for the current BF16 NRM2 kernel. The warp-shuffle tree reduction already provides O(log N) accuracy. If an FP32 input NRM2 is needed in the future, Kahan is worth considering for vectors longer than ~10M elements.

---

## 5. Use `fmaf` Instead of Separate Multiply and Add

The current kernel does:
```cuda
float val = __bfloat162float(xp[i]);
acc += val * val;
```

Replace with:
```cuda
float val = __bfloat162float(xp[i]);
acc = fmaf(val, val, acc);
```

`fmaf` computes `val*val + acc` in a single instruction with higher precision (the intermediate product is not rounded before addition). On sm_120, FMA is the same throughput as separate multiply. Benefits:
- One instruction instead of two (FMUL + FADD -> FFMA)
- No intermediate rounding -- slightly more accurate
- The compiler may already do this optimization, but `fmaf` makes it explicit

---

## 6. Streaming Loads for Large Vectors

The current kernel uses `__ldg` for vectorized loads, which uses the read-only cache path. For large vectors that exceed L2 (N > 32M BF16 = 64 MB), the data will never be reused and caching it wastes L2 capacity.

Use `ld.global.cs` (cache streaming) via inline PTX for large N:
```cuda
// Instead of:
uint4 xv = __ldg(reinterpret_cast<const uint4*>(&x[idx]));
// Use:
uint4 xv;
asm("ld.global.cs.v4.u32 {%0,%1,%2,%3}, [%4];"
    : "=r"(xv.x), "=r"(xv.y), "=r"(xv.z), "=r"(xv.w)
    : "l"(reinterpret_cast<const uint4*>(&x[idx])));
```

This tells the memory system to deprioritize caching this data. For the benchmark N=1M (2 MB), this doesn't matter -- the data fits in L2 easily. But if the benchmark changes to larger sizes, this becomes important.

---

## 7. Potential Race Condition in Last-Block Pattern

The current last-block sqrt has a subtle correctness concern:

```cuda
atomicAdd(result, val);
__threadfence();
unsigned int old = atomicAdd(&nrm2_block_count, 1);
if (old == gridDim.x - 1) {
    *result = sqrtf(*result);
    nrm2_block_count = 0;
}
```

The `__threadfence()` ensures THIS block's atomicAdd to `result` is visible before the counter increment. But it does NOT ensure that OTHER blocks' atomicAdds are visible. Consider:
- Block A: atomicAdd(result, 5.0), threadfence, atomicAdd(counter, 1) -> old=0
- Block B: atomicAdd(result, 3.0), threadfence, atomicAdd(counter, 1) -> old=1 (last block)
- Block B reads `*result` -- but Block A's atomicAdd to result may not yet be visible to Block B on a different SM

**In practice:** atomicAdd on sm_120 goes through L2, which is coherent. All atomicAdds to the same address serialize at the L2 controller. When Block B's atomicAdd to `counter` completes, Block A's atomicAdd to `result` has already completed (because atomicAdds to the same cache line are ordered). So this is actually safe on current hardware. But it's worth understanding why.

**Alternative (bulletproof):** Use `__threadfence_system()` instead of `__threadfence()` if you want to be certain about cross-SM visibility. The performance cost is minimal for one call per block.

---

## 8. Open-Source BLAS1 Implementations for Reference

**KBLAS (King Abdullah University):** GPU-optimized BLAS routines. Includes batched reductions, though primarily targets Kepler/Maxwell era. Source: https://github.com/ecrc/kblas-gpu

**cuTENSOR:** NVIDIA's tensor contraction library. Its reduction primitives achieve near-peak bandwidth. Not open-source but the API shows the optimal interface patterns.

**Ginkgo:** Modern linear algebra framework with GPU BLAS1 implementations. Uses CUB for device-level reductions. Source: https://github.com/ginkgo-project/ginkgo

---

## Summary of Actionable Items

| Item | Expected Impact | Effort |
|------|----------------|--------|
| Grid size tuning (try 64, 128, 680) | 5-20% | Low |
| Use `fmaf(val, val, acc)` | 0-5% (compiler may already do this) | Trivial |
| Profile cuBLAS NRM2 with nsys to count its kernel launches | Diagnostic (understand the gap) | Low |
| Streaming loads for large N | Relevant only for N > 32M | Low |
| Kahan summation | Not needed for BF16 | N/A |

The worker's current kernel architecture (single-pass, vectorized, last-block sqrt) is already correct. The remaining gap to close is likely grid sizing and occupancy tuning, not algorithmic changes.
