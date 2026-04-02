# EHYB: Explicit X-Vector Caching and Column Index Compression

**Sources:**
- [EHYB Paper: arXiv 2204.06666](https://arxiv.org/abs/2204.06666)
- [EHYB GitHub: Chong-Chen-UNLV/EHYB_SPMV_GPU](https://github.com/Chong-Chen-UNLV/EHYB_SPMV_GPU)
- [FastLoad: IEEE TPDS 2024](https://ieeexplore.ieee.org/document/10713183/)
**Relevant to:** spmv worker
**Worker's current problem:** Building SpMV kernel for RTX 5090; needs techniques to improve x-vector access and reduce memory traffic

---

## What This Is

EHYB (Explicit caching HYBrid) is a SpMV framework that achieves up to 280 GFLOPS
single precision on Tesla V100 through two orthogonal optimizations:
1. **Explicit x-vector caching**: Prefetch portions of the input vector x into
   shared memory before the SpMV computation
2. **Compact column index format**: Store column indices in fewer bytes to reduce
   memory traffic

## Why It Matters for Us

The x-vector is the main bottleneck in SpMV. Column indices are essentially random,
so x[col_idx[j]] accesses are scattered across DRAM. Even with RTX 5090's 96 MB L2,
x-vector misses dominate for large matrices. EHYB's two techniques directly attack
this bottleneck:
- Explicit caching brings relevant x-segments into shared memory (guaranteed hit)
- Index compression reduces the bytes per nonzero by 25%

The EHYB code is open-source on GitHub and can be studied/adapted directly.

---

## 1. Explicit X-Vector Caching into Shared Memory

### The Problem

In standard SpMV, each thread accesses `x[col_idx[j]]` via `__ldg()` or similar,
relying on L1/L2 cache for reuse. But:
- L1 is per-SM (128 KB on sm_120) -- limited capacity
- L2 is shared across all SMs (96 MB) -- good but still has eviction pressure
- Neither cache is software-controlled -- the hardware decides what stays

### The EHYB Solution

Before computing the SpMV for a group of rows, the thread block collaboratively
loads the relevant segment of x into shared memory:

```cuda
// Step 1: Determine the column range for this block's rows
int col_min = INT_MAX, col_max = 0;
for (int row = block_row_start; row < block_row_end; row++) {
    for (int j = row_ptr[row]; j < row_ptr[row+1]; j++) {
        col_min = min(col_min, col_idx[j]);
        col_max = max(col_max, col_idx[j]);
    }
}
// (In practice, use warp-level min/max reductions)

int segment_size = col_max - col_min + 1;

// Step 2: Collaboratively load x-segment into shared memory
__shared__ float x_cache[MAX_SEGMENT_SIZE];
for (int i = threadIdx.x; i < segment_size; i += blockDim.x)
    x_cache[i] = x[col_min + i];
__syncthreads();

// Step 3: Use cached x for SpMV
for (int j = start; j < end; j++) {
    int col = col_idx[j];
    if (col >= col_min && col <= col_max)
        sum += val[j] * x_cache[col - col_min];  // Shared memory hit
    else
        sum += val[j] * x[col];  // Fallback to global (rare)
}
```

### When This Works Well

- **Banded/structured matrices**: Column indices within a block of rows cluster
  in a narrow range. The x-segment is small and reuse is high.
- **Reordered matrices**: After RCM or similar reordering, column locality improves
  dramatically, making x-caching more effective.

### When This Does NOT Work

- **Random/unstructured sparsity**: Column indices span the full range. The
  x-segment would need to be the entire vector, defeating the purpose.
- **Very wide matrices**: If the column range exceeds shared memory capacity
  (228 KB on sm_120 = 57K floats), caching is impossible.

### Adaptation for RTX 5090

On sm_120 with 228 KB shared memory per SM (configurable carveout):
- Max x-segment in shared memory: ~56K floats (224 KB)
- If column range for a block's rows exceeds this, fall back to `__ldg()`
- Use a heuristic: if `col_max - col_min > 50000`, skip caching for that block

**Key insight**: Even partial caching helps. If 70% of accesses hit shared memory
and 30% go to L2, that's still a significant win over 100% going to L2.

---

## 2. Compact Column Index Format

### The Idea

For the ELL portion of the hybrid format, EHYB stores column indices as 16-bit
offsets relative to a per-slice base column:

```
Standard:  col_idx[j] = 32-bit absolute column index
Compact:   col_idx[j] = base_col + 16-bit delta

Storage per nonzero:
  Standard: 4 bytes (val) + 4 bytes (col) = 8 bytes
  Compact:  4 bytes (val) + 2 bytes (delta) = 6 bytes (+ amortized base)
  Saving:   25% reduction in matrix data traffic
```

### Implementation

```cuda
// Per-slice (group of 32-256 rows), store:
//   base_col: 32-bit base column index
//   delta[]: 16-bit offsets from base_col

// During SpMV:
int base = slice_base_col[slice_id];
uint16_t delta = col_delta[j];
int col = base + delta;
float val_j = val[j];
sum += val_j * x[col];
```

### When This Works

- **Matrices with N < 65536**: All column indices fit in 16 bits directly. No base
  needed. This is common for small-medium FEM matrices.
- **Banded matrices**: Column indices within a slice span < 65536 range. Delta
  encoding works even for large N.
- **Power-law matrices**: Often have concentrated column access patterns where
  deltas fit in 16 bits.

### Performance Impact

- **25% reduction in matrix memory traffic** (col goes from 4 to 2 bytes)
- **Better vectorization**: 128-bit loads fetch 8 x 16-bit deltas vs 4 x 32-bit cols
- **Combined with BF16 values**: 4 bytes total per nnz (2 BF16 val + 2 uint16 col)
  vs 8 bytes standard = **50% reduction**

This 50% reduction in matrix traffic is the single biggest optimization available
for bandwidth-bound SpMV.

---

## 3. Combined Strategy for RTX 5090

### Matrix Analysis (Preprocessing)

```
For each matrix, compute:
  1. avg_nnz_per_row, std_nnz_per_row  (for row binning)
  2. column_range = max(col_idx) - min(col_idx)  (for x-caching feasibility)
  3. per_slice_column_range  (for index compression feasibility)

Decision:
  - If column_range < 50K: Use shared memory x-caching
  - If N < 65536 or per_slice_range < 65536: Use 16-bit column indices
  - If values are acceptable in BF16: Use BF16 values
```

### Expected Combined Performance

| Optimization | Bandwidth Saving | Stacks? |
|-------------|-----------------|---------|
| 16-bit col indices | 25% of matrix traffic | Yes |
| BF16 values | 25% of matrix traffic | Yes, with above |
| x-caching in shared mem | 20-60% of x traffic | Yes, orthogonal |
| Row binning | 10-30% from less divergence | Yes, orthogonal |

**Combined: 1.5-2.5x over cuSPARSE FP32 for suitable matrices.**

---

## Caveats

1. **EHYB tested on V100 (sm_70)**: The code needs adaptation for sm_120, but the
   techniques are architecture-independent.

2. **Shared memory x-caching adds a preprocessing step per block**: Computing
   col_min/col_max requires scanning column indices. For short rows this is cheap;
   for long rows it adds overhead. Consider precomputing per-block column ranges
   during the initial analysis pass.

3. **16-bit index compression requires format conversion**: The matrix must be
   converted to the compact format. This is a one-time cost, acceptable for
   iterative solvers.

4. **x-caching is most effective after matrix reordering**: RCM or similar
   bandwidth-reducing reorderings concentrate column indices, making x-segments
   smaller and reuse higher. However, reordering itself is expensive (use MT-METIS
   for GPU-accelerated reordering -- note the EHYB repo includes libmtmetis.a).
