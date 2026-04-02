# LRB, Work Dispatch, and Flat/Line-Enhance: Modern SpMV Load Balancing

**Sources:**
- [Logarithmic Radix Binning: HPEC 2019 (Green & Fox)](https://ieeexplore.ieee.org/document/8916333/)
- [GPU Work Graphs SpMV: ISCA 2025 (Wildgrube et al.)](https://dl.acm.org/doi/10.1145/3695053.3731060)
- [Flat/Line-Enhance: HPDC 2023](https://dl.acm.org/doi/abs/10.1145/3588195.3593002)
- [PERKS: ICS 2023](https://dl.acm.org/doi/abs/10.1145/3577193.3593705)
- [EHYB: arXiv 2204.06666](https://arxiv.org/abs/2204.06666)
**Relevant to:** spmv worker
**Worker's current problem:** Building CSR SpMV kernel; needs practical load-balancing and work-dispatch strategies

---

## What This Is

Three practical alternatives to CSR-Adaptive's row-binning approach that the worker
should be aware of, each with different tradeoffs on preprocessing cost, per-SpMV
performance, and implementation complexity.

---

## 1. Logarithmic Radix Binning (LRB)

### Core Idea

Instead of the sequential analysis pass in CSR-Adaptive, LRB uses a fully parallel
GPU-based binning:

```
Step 1: Compute row lengths (parallel)
  len[i] = row_ptr[i+1] - row_ptr[i]    // one thread per row

Step 2: Compute bin assignment (parallel)
  if len[i] == 0: bin[i] = 0
  else: bin[i] = floor(log2(len[i])) + 1

Step 3: Radix sort rows by bin (GPU sort)
  Use CUB DeviceRadixSort on bin[] with row indices as payload

Step 4: Launch one kernel per non-empty bin
  Each kernel processes rows with similar lengths (within 2x of each other)
```

### Bins in Practice

| Bin | Row Length Range | Threads/Row | Kernel Strategy |
|-----|-----------------|-------------|-----------------|
| 0   | 0 (empty)       | 0           | Skip |
| 1   | 1               | 1           | Thread-per-row |
| 2   | 2-3             | 2           | 2-thread tile |
| 3   | 4-7             | 4           | 4-thread tile |
| 4   | 8-15            | 8           | 8-thread tile |
| 5   | 16-31           | 16          | 16-thread tile |
| 6   | 32-63           | 32          | Full warp |
| 7   | 64-127          | 32          | Warp, 2-4 passes |
| 8   | 128-255         | 32          | Warp, multi-pass |
| 9   | 256-511         | 64          | 2 warps |
| 10  | 512-1023        | 128         | 4 warps |
| 11  | 1024+           | 256         | Full block |

### Performance vs CSR-Adaptive

From the GPU Work Graphs ISCA 2025 paper:
- **LRB preprocessing**: 20x faster than CSR-Adaptive (GPU-parallel vs host-sequential)
- **Per-SpMV performance**: ~0.9x CSR-Adaptive (multiple launches + reordering overhead)
- **Break-even**: LRB is better for < ~20 SpMVs per matrix. CSR-Adaptive is better
  for > ~20 SpMVs (amortized preprocessing).

### Recommendation for Us

Since we target iterative solvers (100+ SpMVs per matrix), CSR-Adaptive's
preprocessing cost is negligible and its per-SpMV performance is higher. **Use
CSR-Adaptive, not LRB, as the primary approach.** LRB is worth knowing as a
fallback for one-shot SpMV workloads.

---

## 2. Flat and Line-Enhance Algorithms (HPDC 2023)

These are the strongest recent results for CSR SpMV without format conversion.

### Flat Algorithm

A pure nonzero-splitting approach (similar to merge-based) but with optimized
memory access patterns:

```
Total nonzeros = nnz
Threads = gridDim.x * blockDim.x
nnz_per_thread = ceil(nnz / Threads)

Each thread t processes nonzeros [t*nnz_per_thread, (t+1)*nnz_per_thread):
  1. Binary search row_ptr to find starting row
  2. Process assigned nonzeros sequentially
  3. Accumulate partial sums per row
  4. Write final sums to y[] (atomic for rows spanning threads)
```

The key difference from Merrill's merge-based approach is in how the "fix-up"
reduction is handled: Flat uses a segmented reduction pass rather than the
merge-path diagonal traversal.

### Line-Enhance Algorithm

Hybrid of row-splitting and nonzero-splitting:

```
if row_length <= threshold:
  Use row-splitting: one or more rows per thread (CSR-Scalar/Vector)
else:
  Use nonzero-splitting: multiple threads per row (Flat-style)

threshold = adaptive, based on median row length
```

### Why Line-Enhance Matters

The adaptive selection between row-splitting and nonzero-splitting is lightweight
(one comparison per row) and captures the best of both worlds:
- Short rows: row-splitting avoids the overhead of binary search and fix-up
- Long rows: nonzero-splitting ensures perfect load balance

### Performance Results (from the paper)

Average speedups over existing methods:
- vs CSR-Vector: **4.24x**
- vs CSR-Adaptive: **7.41x**
- vs HOLA: **1.49x**
- vs cuSPARSE: **1.46x**
- vs merge-based SpMV: **1.72x**

Tested on both AMD and NVIDIA GPUs.

### Implementation Notes

The 1.46x over cuSPARSE is significant and comes from the flat algorithm's better
memory coalescing for the data loading phase. cuSPARSE's merge-based approach
has overhead from the 2D binary search that the flat approach avoids.

**This is worth implementing.** The flat algorithm is conceptually simple and the
1.46x average over cuSPARSE matches our target.

---

## 3. Persistent Kernel Work Dispatch (CUDA Alternative to Work Graphs)

The GPU Work Graphs paper (ISCA 2025) achieves 3.35x mean speedup over rocSPARSE
by using device-driven scheduling. Since Work Graphs are D3D12-only, here is the
CUDA equivalent:

### Persistent Kernel with Global Work Queue

```cuda
__device__ int global_row_counter = 0;

__global__ void persistent_spmv(
    int M, const int* row_ptr, const int* col_idx,
    const float* val, const float* x, float* y)
{
    // Each warp grabs rows dynamically
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;

    while (true) {
        // Warp leader grabs next row
        int row;
        if (lane == 0)
            row = atomicAdd(&global_row_counter, 1);
        row = __shfl_sync(0xFFFFFFFF, row, 0);

        if (row >= M) return;  // All rows processed

        // Process row with warp
        float sum = 0.0f;
        int start = row_ptr[row];
        int end = row_ptr[row + 1];
        for (int j = start + lane; j < end; j += 32)
            sum += val[j] * x[col_idx[j]];

        // Warp shuffle reduction
        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);

        if (lane == 0) y[row] = sum;
    }
}
```

### Advantages

- **Perfect dynamic load balance**: No row is "assigned" -- warps grab work as
  they finish. Fast warps do more work.
- **No preprocessing**: Works directly on CSR.
- **L2 cache friendly**: Since warps process rows in order (approximately),
  spatial locality in val/col_idx is preserved.

### Disadvantages

- **Atomic counter bottleneck**: For matrices with very short rows, the atomic
  counter becomes a serialization point. Mitigate with work-stealing (grab chunks
  of 16-32 rows at once instead of 1).
- **No sub-warp assignment**: All rows get a full warp, wasting threads on short
  rows. Can be combined with LRB for sub-warp bins.

### Variant: Chunk-Based Work Stealing

```cuda
// Grab 32 rows at a time to reduce atomic contention
if (lane == 0)
    chunk_start = atomicAdd(&global_row_counter, CHUNK_SIZE);
chunk_start = __shfl_sync(0xFFFFFFFF, chunk_start, 0);

for (int row = chunk_start; row < min(chunk_start + CHUNK_SIZE, M); row++) {
    // Process row...
}
```

CHUNK_SIZE = 32-128 is optimal. This reduces atomic operations by 32-128x while
maintaining good load balance.

---

## Recommendations for Our Worker

### Priority Order

1. **Row-binned CSR-Adaptive** with sub-warp tiles (primary approach)
   - Best per-SpMV performance for iterative solvers
   - Preprocessing amortized over 100+ iterations

2. **Flat/Line-Enhance** as the fallback for unknown matrices
   - No preprocessing needed
   - 1.46x over cuSPARSE on average (strong default)
   - Implement the flat algorithm first, add line-enhance threshold tuning later

3. **Persistent kernel dispatch** for iterative solver integration
   - Combine with PERKS-style iteration loop
   - Use chunk-based work stealing (CHUNK_SIZE=64)

### Implementation Sequence

```
v1: CSR-Vector with warp shuffle (baseline, match cuSPARSE)
v2: Add sub-warp tiles for short rows (1.1-1.2x improvement)
v3: Add row binning with per-bin kernels (1.2-1.5x over cuSPARSE)
v4: Add BF16 values + 16-bit indices (1.5-2.0x over cuSPARSE)
v5: Add persistent kernel for CG/GMRES (3-5x for iterative on small matrices)
```
