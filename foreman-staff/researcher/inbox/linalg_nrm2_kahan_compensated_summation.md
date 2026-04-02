# NRM2: Kahan Compensated Summation for Accurate Single-Pass Norm on GPU

**Sources:**
- [Kahan summation algorithm (Wikipedia)](https://en.wikipedia.org/wiki/Kahan_summation_algorithm)
- [Compensated summation and dot product on parallel architectures (Ogita et al., 2022)](https://www.sciencedirect.com/science/article/abs/pii/S0377042722002047)
- [Parallel Vectorized Compensated Summation Algorithms (Dmitruk et al., PPAM 2022)](https://dl.acm.org/doi/10.1007/978-3-031-30445-3_6)
- [Improving accuracy of summation using parallel vectorized Kahan and Gill-Moller (Dmitruk, 2023)](https://onlinelibrary.wiley.com/doi/abs/10.1002/cpe.7763)
- [Remark on Algorithm 539: Euclidean Norm Reference (ACM TOMS, 2018)](https://dl.acm.org/doi/10.1145/3134441)
- [NVIDIA Forums: Improving float summation precision](https://forums.developer.nvidia.com/t/how-to-improve-float-array-summation-precision-and-stability/67904)

**Relevant to:** linalg worker
**Worker's current problem:** NRM2 fused single-pass kernel needs accurate sum-of-squares accumulation. BF16 inputs have only 8 mantissa bits -- accumulation error in large reductions can lose several ULP of accuracy.

---

## What This Is

Kahan compensated summation maintains a running compensation variable that tracks rounding errors from each addition, effectively doubling the precision of the accumulator at minimal extra cost (one extra FMA per element). When applied to NRM2's sum-of-squares reduction, this gives near-FP64 accuracy from FP32 arithmetic.

---

## Why It Matters for Us

The worker's NRM2 kernel uses a single-pass fused approach (vectorized loads -> FMA accumulation -> warp reduction -> sqrt). For BF16 inputs upcast to FP32 accumulators, the sum-of-squares accumulation loses precision when:
- Vector length is large (N > 10K elements)
- Values have high dynamic range (some x_i^2 >> others)

The reference implementation (cuBLAS nrm2) uses a "blue" algorithm with scaling to handle overflow/underflow. Our BF16 kernel avoids overflow (BF16 max^2 < FP32 max), but accumulation accuracy still matters for solver convergence checks where NRM2 is used to test residual norms.

Kahan summation adds ~3 extra FLOPs per accumulation step (1 subtraction, 1 addition, 1 assignment for the compensation variable). Since NRM2 is memory-bound (not compute-bound), these extra FLOPs are free -- they execute during memory latency.

---

## Key Technique

### Standard accumulation (what you probably have now):
```cuda
float sum = 0.0f;
for (int i = tid; i < n4; i += stride) {
    float4 v = x4[i];
    sum = fmaf(v.x, v.x, sum);
    sum = fmaf(v.y, v.y, sum);
    // ...
}
```

### Kahan compensated accumulation:
```cuda
float sum = 0.0f;
float comp = 0.0f;  // compensation for lost low-order bits

for (int i = tid; i < n4; i += stride) {
    float4 v = x4[i];

    // For each element: compensated FMA
    float y = fmaf(v.x, v.x, -comp);  // add compensation to next term
    float t = sum + y;
    comp = (t - sum) - y;              // algebraically zero, but captures rounding error
    sum = t;

    y = fmaf(v.y, v.y, -comp);
    t = sum + y;
    comp = (t - sum) - y;
    sum = t;

    y = fmaf(v.z, v.z, -comp);
    t = sum + y;
    comp = (t - sum) - y;
    sum = t;

    y = fmaf(v.w, v.w, -comp);
    t = sum + y;
    comp = (t - sum) - y;
    sum = t;
}
```

### Compensated warp reduction:
```cuda
// After the loop, each thread has (sum, comp).
// For the warp reduction, you can either:
// Option A: Just reduce 'sum' (compensation is per-thread, good enough for 32 terms)
// Option B: Reduce (sum - comp) for maximum accuracy
float final_val = sum - comp;  // fold compensation into sum
for (int offset = 16; offset > 0; offset >>= 1)
    final_val += __shfl_down_sync(0xffffffff, final_val, offset);
```

### Why tree reduction helps too:
Parallel tree reduction (warp shuffle) is inherently more accurate than serial accumulation because it adds same-magnitude terms pairwise. A 32-wide warp reduction has O(log2(32)) = 5 rounding errors, vs O(N) for serial. Combined with Kahan compensation in the serial accumulation phase, this gives excellent accuracy.

---

## Practical Guidance

1. **Memory-bound means free compute:** NRM2 at vector sizes >= 4K is memory-bound. The 3 extra FLOPs per element for Kahan are invisible in the roofline -- they hide behind memory latency. There is zero performance penalty.

2. **When it matters:** If the worker is using NRM2 for convergence checks in iterative solvers (batched DOT/NRM2 direction), accuracy in the norm directly affects when the solver terminates. A norm that's off by 0.1% can cause 1-2 extra iterations.

3. **When it doesn't matter:** For standalone NRM2 benchmarks against cuBLAS, accuracy is tested separately. The performance benchmark won't penalize naive accumulation. But adding Kahan is free, so there's no reason not to.

4. **Alternative -- pairwise summation:** If Kahan's per-element overhead is unwanted (shouldn't be for memory-bound code), an alternative is to accumulate into a small array of 4-8 partial sums and reduce them at the end. This gives O(log N) error growth instead of O(N), though not as good as Kahan's O(1).

---

## Caveats

- Kahan summation requires that the compiler NOT optimize away the compensation computation. Use `volatile` or `#pragma clang fp contract(off)` around the critical section if the compiler folds `(t - sum) - y` to zero.
- On NVIDIA GPUs with `--use_fast_math`, FMA contraction can defeat Kahan. Compile the NRM2 kernel without `--use_fast_math`, or use inline PTX for the critical subtract.
- This is NOT needed if accumulating in FP64 -- but FP64 throughput on RTX 5090 is 1/64th of FP32, so FP32 + Kahan is the right choice.
