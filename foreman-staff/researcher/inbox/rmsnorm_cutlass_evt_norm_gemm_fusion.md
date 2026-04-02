# CUTLASS Epilogue Visitor Trees: Composable RMSNorm + GEMM Fusion

**Sources:**
- [Epilogue Fusion in CUTLASS with EVT (Colfax Research)](https://research.colfax-intl.com/epilogue_visitor_tree/)
- [CUTLASS 3.x GEMM Design Blog (NVIDIA)](https://developer.nvidia.com/blog/cutlass-3-x-orthogonal-reusable-and-composable-abstractions-for-gemm-kernel-design/)
- [Crafting Efficient Kernels with Epilogue Fusion (fal.ai)](https://blog.fal.ai/crafting-efficient-kernels-with-epilogue-fusion/)
- [CUTLASS example 35: gemm_softmax with EVT](https://github.com/NVIDIA/cutlass/blob/main/examples/35_gemm_softmax/gemm_with_epilogue_visitor.h)

**Relevant to:** rmsnorm worker
**Worker's current problem:** Industry consensus says fuse RMSNorm into QKV projection GEMM, not attention. But HOW to fuse a reduction (RMSNorm) into a GEMM epilogue?

## What This Is

CUTLASS's Epilogue Visitor Tree (EVT) system provides a composable framework
for fusing arbitrary operations into a GEMM kernel's epilogue (the phase after
the matrix multiply, before writing results to global memory). EVTs support
not just element-wise ops (bias, activation) but also **reductions** -- making
RMSNorm fusion into GEMM epilogues architecturally feasible.

## Why It Matters for Us

The industry consensus brief (`rmsnorm_industry_fusion_consensus`) established
that production systems fuse RMSNorm with the QKV projection GEMM, not with
attention. But our worker writes raw CUDA, not CUTLASS. The EVT architecture
reveals the general engineering pattern for norm-GEMM fusion that the worker
can implement manually:

**The insight:** CUTLASS EVT supports "reduction" nodes (vector/scalar outputs
from elementwise inputs). This means a GEMM epilogue can compute:
1. `D = alpha * A * B + beta * C` (standard GEMM)
2. In the epilogue: `row_sum = sum(D_row^2)` (accumulate sum-of-squares per row)
3. Then: `D_row = D_row * rsqrt(row_sum / N + eps)` (normalize)
4. Write normalized output to global memory

Steps 2-4 are exactly RMSNorm, applied as a GEMM epilogue.

## Key Technique

### How EVT Handles Reductions in Epilogue

The EVT tree decomposes into node types:
- **Compute nodes:** Element-wise ops (multiply, add, activation)
- **Load nodes:** Read auxiliary tensors (bias, scale)
- **Store nodes:** Write outputs
- **Reduction nodes:** Accumulate across a dimension (row-wise or col-wise)

For RMSNorm in epilogue:
```
EVT Tree:
  Store(output)
    |
  Compute(multiply)         -- x * gamma * rsigma
    |         \
  Load(gamma)  Reduce(rsigma)  -- rsigma = rsqrt(sum(x^2) / N + eps)
                |
              Compute(square)   -- x^2
                |
              [GEMM output D]
```

### The Tiling Challenge

GEMM tiles output in blocks (e.g., 64x64). The epilogue processes one tile
at a time. But RMSNorm reduction is across an entire row (dimension D, which
may span multiple tiles in the N dimension).

CUTLASS handles this with:
1. **Partial reduction in each tile's epilogue** -- accumulate sum-of-squares
   for the elements in this tile
2. **Cross-tile reduction** -- after all tiles for a row are computed, combine
   partial sums
3. **Final normalization pass** -- apply rsqrt to the combined sum, then
   normalize all elements

This adds complexity because step 2 requires either:
- A separate reduction kernel (defeating some of the fusion benefit)
- Atomics for cross-CTA partial sum accumulation
- Cooperative groups grid sync

### For Our Worker

Our GEMM kernel uses 64x64 tiles. For a QKV projection (e.g., 2048x768 * 768x2304),
each output row has D_out=2304 elements, tiled as 2304/64 = 36 tiles across N.
A fused RMSNorm in the epilogue would need to reduce across 36 tiles per row.

**Simpler alternative (FlashNorm):** Instead of fusing the full RMSNorm reduction
into the epilogue, use FlashNorm's deferred approach:
1. GEMM produces `z = x @ W'` (weight-absorbed)
2. Separately/in parallel: compute `rms = sqrt(mean(x^2))` (small reduction kernel)
3. Final output: `z / rms` (element-wise, trivially fusible into GEMM epilogue)

Step 3 is a simple element-wise division -- no cross-tile reduction needed.
This is much simpler to implement than full in-epilogue RMSNorm.

## Caveats

- CUTLASS EVT on sm_120 uses the 2.x API (`Sm80EVT`), not the 3.x Hopper
  path (`Sm90EVT`). The 2.x path lacks some composability features but
  supports the basic node types needed for reductions.
- The worker writes raw CUDA, not CUTLASS templates. The EVT pattern is an
  architectural reference, not code to copy. The manual implementation is
  simpler: accumulate partial sums in registers during the epilogue phase,
  then do a final reduction.
- Full in-epilogue RMSNorm (with cross-tile reduction) is complex. The
  FlashNorm deferred approach (separate RMS scalar + element-wise epilogue)
  is the pragmatic path.
