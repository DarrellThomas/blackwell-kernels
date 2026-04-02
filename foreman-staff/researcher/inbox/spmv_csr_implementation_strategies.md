# SpMV CSR Implementation Strategies for GPU

**Sources:**
- [NVIDIA SpMV whitepaper (Bell & Garland, 2008)](https://www.nvidia.com/docs/io/66889/nvr-2008-004.pdf)
- [SC09 SpMV Throughput (Bell & Garland)](https://www.nvidia.com/docs/io/77944/sc09-spmv-throughput.pdf)
- [LightSpMV: Dynamic Warp Distribution](https://ieeexplore.ieee.org/document/7245713/)
- [Merge-Based SpMV (Merrill & Garland, 2016)](https://dl.acm.org/doi/10.1145/3016078.2851190)
- [Block Strategy + Adaptive Storage (2024)](https://link.springer.com/article/10.1007/s10586-024-04966-7)
**Relevant to:** spmv worker
**Worker's current problem:** SpMV not yet started. Need implementation strategy for beating cuSPARSE on sm_120.

---

## 1. CSR Format Recap

CSR stores: `values[]` (nonzeros), `col_idx[]` (column indices), `row_ptr[]` (row boundaries).
SpMV computes `y = A * x` where `y[i] = sum(values[j] * x[col_idx[j]])` for `j in row_ptr[i]..row_ptr[i+1]`.

The fundamental challenge: **row lengths vary wildly** (0 to thousands of nonzeros).

---

## 2. Three Classic CSR Approaches

### 2.1 CSR-Scalar: One Thread Per Row
```cuda
__global__ void spmv_scalar(int M, const int* row_ptr, const int* col_idx,
                            const float* val, const float* x, float* y) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M) {
        float sum = 0.0f;
        for (int j = row_ptr[row]; j < row_ptr[row + 1]; j++)
            sum += val[j] * x[col_idx[j]];
        y[row] = sum;
    }
}
```

**When it works:** Matrices with short, uniform rows (avg nnz/row < 32).
**Problem:** Long rows serialize on a single thread. Uncoalesced `val[]`/`col_idx[]` loads (threads access different row lengths, so addresses diverge).

### 2.2 CSR-Vector: One Warp Per Row
```cuda
__global__ void spmv_vector(int M, const int* row_ptr, const int* col_idx,
                            const float* val, const float* x, float* y) {
    int row = (blockIdx.x * blockDim.x + threadIdx.x) / 32;  // warp-per-row
    int lane = threadIdx.x % 32;
    if (row < M) {
        float sum = 0.0f;
        for (int j = row_ptr[row] + lane; j < row_ptr[row + 1]; j += 32)
            sum += val[j] * x[col_idx[j]];
        // Warp reduction
        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        if (lane == 0) y[row] = sum;
    }
}
```

**When it works:** Matrices with medium-to-long rows (avg nnz/row >= 32).
**Problem:** Short rows waste 31 of 32 threads. Coalesced `val[]`/`col_idx[]` access (consecutive lanes read consecutive elements).

### 2.3 CSR-Adaptive: Best of Both
```
For each row:
  if (nnz_in_row < THRESHOLD) → assign to "stream" group (multiple rows per warp)
  else → assign to "vector" group (one warp per row, or multiple warps)
```

This is the approach used by MAGMA/clSPARSE and achieves the best general-purpose performance. The THRESHOLD is typically 32 (warp size).

**Implementation:** Pre-scan row_ptr to classify rows, then dispatch two sub-kernels or use dynamic branching within a single kernel.

---

## 3. Merge-Based SpMV (State of the Art)

The merge-based approach by Merrill & Garland is the **most robust general CSR SpMV** — it's impervious to row-length heterogeneity.

### Core Idea
Treat SpMV as a merge of two sorted sequences:
1. The sequence of row boundaries: `row_ptr[0], row_ptr[1], ..., row_ptr[M]`
2. The sequence of nonzero indices: `0, 1, 2, ..., nnz-1`

Each thread is assigned an equal-sized portion of the merged space (row boundaries + nonzeros combined). This guarantees perfect load balance regardless of row structure.

### Why It's Better
- **No pathological cases:** Empty rows and very long rows are handled uniformly.
- **Predictable performance:** Bandwidth utilization is consistent across all matrix structures.
- **Works directly on CSR:** No format conversion needed.

### Performance
Achieves near-peak memory bandwidth for most matrices. On A100: within 90% of peak bandwidth for the SuiteSparse dataset.

---

## 4. ELL Format (Alternative)

ELL stores: `values[M][max_nnz_per_row]` and `col_idx[M][max_nnz_per_row]` in column-major.

**Advantage:** Perfectly coalesced — consecutive threads access consecutive rows, all reading the same column of the padded array.

**Disadvantage:** Wastes memory if max_nnz >> avg_nnz (zero padding). Catastrophic for power-law degree distributions.

**Hybrid ELL/COO:** Store the bulk in ELL (up to a threshold), overflow into COO. Best of both worlds.

---

## 5. Recommendations for sm_120 (RTX 5090)

### Start with merge-based CSR
- Most robust general approach
- Works directly on CSR (no format conversion)
- Bandwidth-bound (which is what sm_120 excels at: 1792 GB/s)

### Key optimizations for sm_120:
1. **Vectorized loads:** Use `float4` or `int4` for loading `val[]` and `col_idx[]` when alignment permits. This is 4x the bandwidth per instruction.

2. **L2 cache exploitation:** The `x[]` vector is randomly accessed (indirect through col_idx). On RTX 5090, L2 is 96 MB — for vectors up to 24M floats (96 MB / 4 bytes), the entire vector fits in L2. Repeated access patterns benefit enormously.

3. **Streaming for val/col_idx:** These are accessed sequentially and should bypass L2 when the matrix doesn't fit (`ld.global.cs`).

4. **Grid sizing:** 170 SMs × 4 blocks/SM × 256 threads/block = 174K threads. Each thread should process roughly `nnz / 174K` nonzeros for good load balance.

5. **atomicAdd for partial row results:** When multiple blocks contribute to the same row (merge-based), use atomicAdd on `y[]`. This is fast on sm_120 for low contention.

### Baseline to beat:
Use cuSPARSE's new `SpMVOp` API (CUDA 13.1) as the reference, NOT the legacy `cusparseSpMV`. SpMVOp has better performance. See `spmv_cusparse_spmvop_api_cuda131.md`.

### Test matrices:
Use SuiteSparse Matrix Collection for benchmarking. Key matrix types:
- **Regular structured** (e.g., 5/7/9-point stencil from PDE discretization)
- **Power-law** (e.g., social networks, web graphs — highly irregular)
- **Band-structured** (e.g., FEM matrices)

---

## 6. Implementation Roadmap

1. **v0: cuSPARSE baseline** — benchmark SpMVOp on representative matrices
2. **v1: CSR-Vector** — simplest custom kernel, one warp per row
3. **v2: CSR-Adaptive** — classify rows, dispatch scalar/vector
4. **v3: Merge-based** — equal-work partitioning across all threads
5. **v4: Vectorized loads + streaming** — float4, ld.global.cs for large matrices
6. **v5: Format-specific** — if a specific format dominates, tune for it (e.g., ELL for structured)

## Caveats

1. **SpMV is fundamentally bandwidth-bound.** On sm_120 with 1792 GB/s, the theoretical minimum for a 10M-nnz matrix is ~80 us (loading val + col_idx + x lookups). Beating cuSPARSE likely means better bandwidth utilization, not compute tricks.

2. **The `x[]` vector access pattern dominates.** Irregular col_idx means random reads of `x[]`. Texture cache / L2 hit rate is the key metric. No amount of kernel optimization helps if `x[]` doesn't fit in cache.

3. **Format conversion is usually NOT worth it** for one-shot SpMV (the conversion itself is an SpMV-like operation). Only worth it for iterative solvers where SpMV runs thousands of times on the same matrix.
