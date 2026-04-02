# SpMV Format Selection and Optimization Strategies

**Sources:**
- [cuSPARSE Generic API (SpMV)](https://docs.nvidia.com/cuda/cusparse/generic-api/generic-api-functions.html)
- [cuSPARSE Storage Formats](https://docs.nvidia.com/cuda/cusparse/storage-formats.html)
- [NVIDIA Forum: cuSPARSE SpMV Implementation](https://forums.developer.nvidia.com/t/cusparse-implementation-of-spmv/217591)
- [NVIDIA Forum: SELL Format Performance](https://forums.developer.nvidia.com/t/performance-usinng-sell-format-with-cusparsespmv/286978)
- [BestSF: A Sparse Meta-Format (ACM TACO 2018)](https://dl.acm.org/doi/10.1145/3226228)
- [AUTO-SPMV (arxiv 2302.05662)](https://arxiv.org/pdf/2302.05662)
- [Merge-Based SpMV (Merrill & Garland, SC '16)](https://dl.acm.org/doi/10.1145/3016078.2851190)
- [DASP: MMA-Accelerated SpMV (SC '23)](https://dl.acm.org/doi/10.1145/3581784.3607051)
- [GPUs All Grown-Up: Work Graphs SpMV (ISCA '25)](https://dl.acm.org/doi/10.1145/3695053.3731060)
- [Can Tensor Cores Benefit Memory-Bound Kernels? (No!)](https://arxiv.org/html/2502.16851v2)
- [Systematic SpMV Survey (2024)](https://arxiv.org/html/2404.06047v1)
- [SELL-C-sigma Implementation (UTK 2014)](https://icl.utk.edu/files/publications/2014/icl-utk-772-2014.pdf)
- [Ginkgo ELL/SELL-P Formats (DeepWiki)](https://deepwiki.com/ginkgo-project/ginkgo/4.4-ell-and-sell-p-formats)
- [CSR5 (Liu & Vinter, ICS '15)](https://dl.acm.org/doi/10.1145/2751205.2751209)
- [AMD Lab Notes: SpMV Part 1](https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-spmv-docs-spmv_part1/)
- [Structural Agnostic SpMV / CSR-Adaptive (HiPC 2015)](https://www.computermachines.org/joe/publications/pdfs/hipc2015_spmv.pdf)
- [RTX 5090 Specs (Wikipedia)](https://en.wikipedia.org/wiki/GeForce_RTX_50_series)
- [Is Sparse Matrix Reordering Effective for SpMV?](https://arxiv.org/html/2506.10356)
- [L2 Cache-Aware SpMV on Fermi GPU](https://ieeexplore.ieee.org/document/6299285)
- [Block Strategy and Adaptive Storage (Cluster Computing 2024)](https://link.springer.com/article/10.1007/s10586-024-04966-7)

**Relevant to:** spmv worker
**Worker's current problem:** Building custom SpMV kernel to match or beat cuSPARSE on RTX 5090

---

## What This Is

A comprehensive research brief covering how cuSPARSE selects algorithms internally, the major SpMV format/algorithm families, and concrete strategies for beating cuSPARSE on the RTX 5090 (sm_120, 170 SMs, 98 MB L2, 1792 GB/s GDDR7). SpMV is fundamentally memory-bandwidth-bound with ~2 FLOPs per 12 bytes loaded. The key insight from all the literature: **format selection and load balancing dominate performance**. A well-chosen format with a simple kernel beats a poorly-chosen format with a heroically optimized kernel.

---

## 1. cuSPARSE Format Selection Internals

### What cuSPARSE Actually Does

cuSPARSE's `cusparseSpMV` supports four sparse formats: **COO, CSR, CSC, and Sliced ELL** (SELL). It does NOT auto-select format -- the user must choose the format and create the appropriate descriptor. What cuSPARSE auto-selects is the **algorithm within a format**.

**Algorithm options per format:**

| Algorithm | Format | Behavior |
|-----------|--------|----------|
| `CUSPARSE_SPMV_ALG_DEFAULT` | Any | Auto-selects best for the format |
| `CUSPARSE_SPMV_CSR_ALG1` | CSR/CSC | Higher performance, **non-deterministic** |
| `CUSPARSE_SPMV_CSR_ALG2` | CSR/CSC | Deterministic (bit-wise reproducible), slower |
| `CUSPARSE_SPMV_COO_ALG1` | COO | Higher performance, non-deterministic |
| `CUSPARSE_SPMV_COO_ALG2` | COO | Deterministic, slower |
| `CUSPARSE_SPMV_SELL_ALG1` | Sliced ELL | Deterministic |

**The internal algorithm for CSR_ALG1 is merge-based (nonzero-splitting):**
An NVIDIA engineer (fbusato) confirmed on the developer forums that `cusparseSpMV` follows a **nonzero-splitting approach** based on Merrill & Garland's merge-based SpMV. This is NOT the old scalar-CSR or vector-CSR from the deprecated `cusparsecsrmv` API. The merge-based approach:
- Frames SpMV as a merge of the CSR row-offset array and the nonzero index sequence
- Splits the merged sequence into equal-sized chunks, one per thread
- Each thread does a binary search along its diagonal to find its (row, nnz) starting point
- Achieves **strict load balance** regardless of row length distribution
- Requires no preprocessing or format conversion

**CSR_ALG2** provides deterministic results but is slower -- likely uses a more conservative reduction strategy.

**`cusparseSpMV_preprocess`** (added CUDA 12.4): Optional preprocessing step that improves repeated SpMV performance on the same matrix. The exact optimization is undocumented, but likely involves precomputing the merge-path diagonal starting positions, which eliminates the per-call binary search overhead.

### What This Means for Us

cuSPARSE CSR_ALG1 is essentially merge-based SpMV -- a strong baseline. To beat it, we need either:
1. A **better format** for the specific matrix structure (SELL-P, CSR5, hybrid)
2. **Better load balancing** at finer granularity (warp-level rather than thread-level)
3. **L2 cache exploitation** that cuSPARSE doesn't do (cache-blocking, reordering)
4. **Kernel fusion** (SpMV+DOT, SpMV+AXPY for iterative solvers)

---

## 2. SELL-P (Sliced ELL with Padding) Format

### How It Works

SELL-P divides the matrix into horizontal **slices** of `slice_size` rows. Within each slice:
1. Find the maximum row length (max nnz per row in the slice)
2. Pad all rows in the slice to that maximum
3. Store values and column indices in **column-major order** within each slice
4. Padding entries use column index -1 (or any sentinel)

**Key parameters:**
- `slice_size`: Number of rows per slice. Should be a multiple of warp size (32).
- `stride_factor`: Alignment multiplier for memory layout within slices.
- `sigma` (in SELL-C-sigma): Number of rows to sort by nnz before slicing, to group similar-length rows together and reduce padding waste.

**Element access:** `values[slice_set * slice_size + local_row + i * slice_size]`
where `slice_set` is the cumulative offset for the slice and `i` iterates over columns.

### Optimal Slice Width

| Slice Size | Pros | Cons |
|-----------|------|------|
| 32 (1 warp) | Minimal padding waste, best for irregular matrices | More slice overhead, less occupancy benefit |
| 64 (2 warps) | Good balance | Moderate padding |
| 128 (4 warps) | Better memory coalescing, less slice overhead | More padding waste for irregular matrices |
| 256+ | Approaches ELL problems | Too much padding for most matrices |

**Recommendation for sm_120:** Start with **slice_size = 32** (warp-aligned). This minimizes padding overhead while maintaining perfect memory coalescing within a warp. The Ginkgo library defaults to 32 or 64.

### When SELL-P Beats CSR

SELL-P wins when:
- Row lengths are **moderately variable** (std/mean < 1.0)
- The matrix has enough rows to fill many slices
- Sigma-sorting is applied to group similar-length rows

SELL-P loses when:
- Row lengths are **highly variable** (e.g., power-law, std/mean > 2.0)
- A few rows have extreme length (padding wastes memory and compute)
- The matrix has < 1000 rows (not enough slices to amortize overhead)

**NVIDIA forum finding:** A user tested SELL vs CSR on a 400K x 400K FEM matrix (mean 77.9 nnz/row, std 65.8, max 11723). **CSR was faster.** NVIDIA recommended CSR for matrices with high row-length variance. SELL excels with **uniform sparsity patterns**.

### Implementation Guidance

The SELL-P SpMV kernel is simple:
```
// One warp per slice (slice_size = 32)
// Thread i in warp handles row (slice_start + i)
for col = 0 to slice_max_len - 1:
    idx = base + threadIdx.x + col * slice_size
    if col_idx[idx] >= 0:  // not padding
        sum += values[idx] * x[col_idx[idx]]
atomicAdd or direct write to y[my_row]
```

Memory access is perfectly coalesced: all 32 threads in a warp read consecutive memory locations within a column of the slice.

---

## 3. Meta-Format Approaches (BestSF, AUTO-SPMV)

### BestSF (ACM TACO 2018)

BestSF uses **machine learning (Weighted SVM)** to predict the best format per matrix. It:
1. Computes cheap sparsity features: nnz, num_rows, num_cols, nnz/row (avg, std, max, min), bandwidth, density, etc.
2. Trains a cost-sensitive classifier on benchmark data across formats: COO, CSR, BCSR, ELL, DIA, HYB
3. At runtime, extracts features and predicts the best format

**Results:** Achieved >97% of optimal performance (perfect selection) on Maxwell and Pascal GPUs.

**Applicability to us:** The concept is sound but the trained models are for old architectures. We would need to retrain on RTX 5090. More practically, we can use the **feature-based intuition**: if avg nnz/row is low and uniform, use ELL/SELL-P. If high and variable, use CSR with merge-based algorithm. If the matrix has dense blocks, use BSR.

### AUTO-SPMV (2023)

AUTO-SPMV extends BestSF with:
- Compile-time optimization of kernel parameters (block size, register usage)
- Runtime format selection using a larger feature set (30 matrices, 15K+ configs)
- Optimizes latency, energy, and power consumption

**Key result:** Up to 51.9% latency improvement over default settings in compile-time mode.

### Practical Takeaway

Rather than implementing full ML-based selection, use **simple heuristics** based on matrix properties:

| Condition | Recommended Format |
|-----------|--------------------|
| nnz/row < 4, uniform | CSR-Scalar (thread per row) |
| nnz/row 4-32, uniform (std/mean < 0.5) | SELL-P (slice=32) or ELL |
| nnz/row 4-32, variable (std/mean > 1.0) | CSR merge-based |
| nnz/row > 32, uniform | CSR-Vector (warp per row) |
| nnz/row > 32, variable | Row-binned CSR (see section 5) |
| Dense blocks present (FEM, stencil) | BSR with block_size matching structure |
| Diagonal structure | DIA |
| Rows > 1M OR max_nnz > 64 | CSR with strict NNZ partitioning (CSR5-style) |

---

## 4. CSR5: Tile-Based Balanced SpMV

CSR5 (Liu & Vinter, ICS 2015) is worth understanding because it achieves **perfectly balanced workload** regardless of matrix structure.

### How It Works

1. Flatten all nonzeros into a 1D array (same as CSR values/col_idx)
2. Divide into **fixed-size 2D tiles** of width `w` (warp size, 32) and height `sigma` (typically 32)
3. Each tile holds exactly `w * sigma` nonzeros (except the last)
4. Within each tile, values are stored in **column-major order** for coalesced access
5. A `tile_ptr` array records the starting row of each tile
6. A `tile_descriptor` bitmap marks where row boundaries fall within each tile

**Key properties:**
- Every thread processes exactly the same number of nonzeros (perfect balance)
- Column-major storage within tiles gives coalesced access
- Works on ANY matrix structure without format-specific tuning
- Low preprocessing cost (one-time segmented scan to build tile_ptr + descriptor)

**Performance:** Competitive with cuSPARSE on regular matrices, significantly better on highly irregular matrices (power-law, social network graphs). The overhead of the tile descriptor and extra bookkeeping is small.

**Relevance for us:** CSR5 is a strong candidate for our "general-purpose" format. It handles irregular matrices gracefully where both CSR-merge and SELL-P struggle.

---

## 5. Row-Length Binning and Load Balancing

### The Problem

In CSR SpMV, rows can have wildly different lengths (1 to 100,000+ nonzeros). Assigning one thread or one warp per row leads to catastrophic load imbalance. The literature identifies three strategies:

### Strategy 1: Fixed Mapping (Simple but Limited)

**CSR-Scalar:** One thread per row. Works for very short rows (nnz/row < 4). Poor coalescing.

**CSR-Vector:** One warp (32 threads) per row. Good for medium rows (nnz/row 32-512). Wastes threads on short rows, can't handle very long rows.

### Strategy 2: CSR-Adaptive (Row Binning)

The CSR-Adaptive algorithm (from AMD's rocSPARSE) classifies rows by length into bins:

| Bin | Row Length | Strategy |
|-----|-----------|----------|
| Tiny | 0-1 nnz | Thread per row, pack multiple rows per warp |
| Short | 2-32 nnz | Sub-warp (4/8/16 threads) per row |
| Medium | 33-256 nnz | Full warp per row |
| Long | 257-4096 nnz | Multiple warps per row |
| Very Long | 4096+ | Full thread block per row |

**CSR-Stream:** Multiple short rows packed into a single thread block. Rows are loaded into shared memory, each thread scans consecutive nonzeros, partial sums are reduced via shared memory.

**CSR-Vector:** Heavy rows get a warp each. Threads within the warp iterate over the row's nonzeros with stride 32 and reduce via warp shuffle (`__shfl_down_sync`).

**CSR-VectorL:** Very long rows get a full thread block (or multiple). The thread block collectively processes the row with shared-memory reduction.

**Preprocessing:** A single pass over the row-pointer array classifies rows into bins. This is O(num_rows) and very cheap. Separate kernel launches (or a single kernel with bin-specific code paths) handle each bin.

### Strategy 3: Merge-Based (Nonzero Splitting)

Merrill & Garland's approach (what cuSPARSE uses). No binning needed -- every thread gets exactly the same number of nonzeros to process. Load balance is automatic. The cost is the per-thread binary search on the row-pointer array to find the (row, offset) starting point.

**Performance on Tesla K40:** 17.19 GFLOPS, 181.6 GB/s (62.96% peak bandwidth). The method shows **predictable performance uncorrelated with nonzero distribution**.

### Strategy 4: DASP (Dense MMA Units for SpMV, SC '23)

DASP bins rows into three categories -- **long, medium, short** -- and organizes them into small dense blocks suitable for tensor-core MMA execution. Results on A100: **1.52x over cuSPARSE CSR, 1.46x over CSR5**.

**Caveat for sm_120:** DASP uses tensor cores (mma.sync). The paper "Can Tensor Cores Benefit Memory-Bound Kernels? (No!)" (2025) found that for SpMV, tensor cores provide at most **1.33x theoretical speedup** over CUDA cores for FP64, and in practice cuSPARSE (CUDA core) outperforms DASP (tensor core) for matrices exceeding L2 cache size. Tensor cores don't help when the bottleneck is memory bandwidth, not compute. **Do not pursue tensor-core SpMV.**

### Recommendation

**Row-binned CSR-Adaptive is the most promising approach for beating cuSPARSE.** The merge-based approach cuSPARSE uses is strong on average but suboptimal for specific patterns:
- For matrices with mostly short rows, CSR-Adaptive's thread-packing avoids warp waste
- For matrices with mostly long rows, CSR-Adaptive's warp-per-row is more efficient than merge-path's thread-level splitting
- The preprocessing cost (one pass over row_ptr) is negligible for repeated SpMV in iterative solvers

---

## 6. L2 Cache Exploitation on RTX 5090

### Cache Capacity Analysis

The RTX 5090 has **98 MB L2 cache** (98,304 KB, confirmed by multiple sources -- not 96 MB as sometimes reported).

**X-vector capacity in L2:**

| Precision | Bytes/Element | Max Elements in L2 | Max Matrix Columns |
|-----------|--------------|--------------------|--------------------|
| FP64 | 8 | 12.3M | 12.3M columns |
| FP32 | 4 | 24.6M | 24.6M columns |
| FP16/BF16 | 2 | 49.2M | 49.2M columns |

For most practical matrices (N < 10M), the **entire x-vector fits in L2 in FP32**. This is a massive advantage over older GPUs where x-vector cache misses dominated performance.

### Cache-Blocking Strategy

When the x-vector exceeds L2 capacity (N > 24.6M for FP32), partition the matrix into vertical strips (column blocks):

```
Matrix A (M x N, N > 24M):
  Block 1: columns 0 to 24M-1
  Block 2: columns 24M to N-1

For each block:
  1. The relevant x-vector subset fits in L2
  2. Process all rows for this column range
  3. Accumulate partial y results
```

This ensures x-vector elements are reused from L2 rather than GDDR7. Research showed **up to 5x speedup** on Fermi GPUs with cache blocking. The benefit scales with how much the x-vector exceeds L2.

### Matrix Reordering for L2 Locality

Even when x fits in L2, **column access patterns** affect L2 hit rate. If a row accesses columns scattered across the full range, L2 lines are evicted before reuse. Techniques:

**RCM (Reverse Cuthill-McKee) reordering:** Reduces matrix bandwidth (column spread), improving spatial locality in x-vector access. A 2025 study found RCM consistently helps in sequential execution, but results are mixed for parallel GPU execution.

**Column-based partitioning:** Group columns accessed by nearby rows to improve x-vector reuse across warps/blocks scheduled on the same SM (sharing L2).

**Practical recommendation for our worker:** For matrices with N < 24M (FP32), the x-vector already fits in L2. Focus optimization effort elsewhere. For larger matrices, implement simple column-block partitioning. RCM reordering is probably not worth the complexity for a first version.

### L2 Cache Line Size and Coalescing

RTX 5090 L2 cache line is 128 bytes. For FP32:
- One cache line holds 32 floats
- A warp reading 32 consecutive x-vector elements fills exactly one cache line
- Column-major SELL-P naturally produces coalesced x-vector reads within a slice

---

## 7. GPU Work Graphs for Iterative Solvers

### The ISCA '25 Paper

"GPUs All Grown-Up: Fully Device-Driven SpMV Using GPU Work Graphs" (Wildgrube et al., ISCA 2025) demonstrates using **D3D12 Work Graphs** to schedule SpMV entirely on the GPU, eliminating CPU-GPU round trips.

**Key results:**
- Up to **7.19x faster** than rocSPARSE LRB for single SpMV (mean 3.35x)
- Beats rocSPARSE CSR-Adaptive for up to **92 consecutive** SpMV iterations
- **75% less code** than rocSPARSE LRB
- **Fixed ~25 MiB memory** footprint vs rocSPARSE's scaling to hundreds of MiB
- Much more stable performance across different sparsity patterns

**How it works:**
- Work Graphs allow GPU workgroups to dynamically spawn new workgroups based on runtime data
- The preprocessing phase (row classification, binning) spawns the compute phase directly on GPU
- Different node types in the work graph handle different row-length bins
- The system self-schedules fine-grained work (workgroup-level, wavefront-level, or thread-level)

### Applicability to sm_120 / CUDA

**Bad news:** Work Graphs are a **D3D12 feature** (Windows/DirectX only). There is no direct CUDA equivalent. The paper was implemented on AMD GPUs using the D3D12 API.

**CUDA alternatives for device-side dispatch:**
- **CUDA Dynamic Parallelism:** Supported on sm_120. Allows device-side kernel launches. However, historically has high overhead per child launch. Not suitable for fine-grained dispatch.
- **CUDA Device Graph Launch** (CUDA 12.4+): Enables launching CUDA graphs from device code. Supports fire-and-forget and tail-launch modes. This is the closest CUDA equivalent to Work Graphs, but it operates at graph granularity, not workgroup granularity.
- **Persistent kernels:** A single kernel that stays resident and processes work from a queue. No launch overhead. This is the practical approach for iterative solvers on CUDA.

**Recommendation:** For iterative solvers (CG/GMRES), implement a **persistent kernel** that fuses the entire iteration: SpMV + DOT + AXPY + convergence check in a single kernel launch. This eliminates all CPU-GPU sync points, achieving the same goal as Work Graphs through a different mechanism. Use cooperative groups for inter-block synchronization (grid-wide barrier).

---

## 8. Recommended Approach for Our Worker

### Phase 1: CSR Baseline (Match cuSPARSE)

1. **Implement merge-based CSR SpMV** (thread-level nonzero splitting, Merrill & Garland style). This is what cuSPARSE does internally, so matching their performance validates the approach.
   - Each thread block gets a contiguous chunk of nonzeros
   - Binary search on row_ptr to find (row, offset) boundaries
   - Warp-level reduction for partial row sums
   - Segmented reduction for rows spanning multiple threads

2. **Benchmark against cuSPARSE CSR_ALG1** on the standard test suite: SuiteSparse matrices, random sparse, banded, power-law.

### Phase 2: Row-Binned CSR-Adaptive (Beat cuSPARSE)

3. **Implement row-length binning:**
   - Single pass over row_ptr to classify rows into bins (tiny/short/medium/long/very-long)
   - Separate code paths within a single kernel using warp-level divergence
   - Tiny rows: pack multiple rows per warp (CSR-Stream style)
   - Short rows: sub-warp assignment (4/8/16 threads per row)
   - Medium rows: full warp per row (CSR-Vector)
   - Long rows: thread-block per row (CSR-VectorL)

4. **This is where we beat cuSPARSE.** The merge-based approach is a compromise -- it's good on average but suboptimal for specific patterns. Row-binned dispatch can be tuned for each bin independently.

### Phase 3: SELL-P for Regular Matrices (Specialize Further)

5. **Implement SELL-P** (slice_size=32) for matrices with low row-length variance (FEM, stencil, structured grids).
6. Simple format-selection heuristic: if std(nnz/row)/mean(nnz/row) < 0.5, use SELL-P; else use CSR-Adaptive.

### Phase 4: Kernel Fusion for Iterative Solvers

7. **Fused SpMV+DOT kernel** for CG: compute y=Ax and dot(r,r) in a single pass. Saves one full memory-traffic pass over y.
8. **Persistent kernel** for full CG iteration: eliminates all kernel launch overhead and CPU-GPU sync.

### Arithmetic Intensity Context

SpMV is brutally memory-bound:
- CSR FP32: 2 FLOPs per (4+4+4) = 12 bytes loaded = **0.167 FLOP/byte**
- RTX 5090 bandwidth: 1792 GB/s
- Peak SpMV throughput: 1792 * 0.167 = **299 GFLOPS** (theoretical max)
- RTX 5090 FP32 compute: ~105 TFLOPS
- SpMV uses **0.28% of compute** -- pure bandwidth bound

This means:
- Every byte of unnecessary data movement is directly a performance hit
- Format choice (which determines bytes loaded) is king
- Compute optimizations (instruction scheduling, register pressure) are irrelevant
- L2 cache hit rate for x-vector is the main variable we can optimize

---

## 9. Caveats for sm_120

1. **Tensor cores: not useful.** SpMV is memory-bound. Tensor-core SpMV (DASP) provides at most 1.33x theoretical speedup for FP64, and in practice cuSPARSE CUDA-core outperforms it for matrices exceeding L2. Don't pursue this.

2. **Work Graphs: not available in CUDA.** The ISCA '25 paper uses D3D12 Work Graphs, which are Windows/DirectX only. Use persistent kernels + cooperative groups instead.

3. **Dynamic parallelism: high overhead.** Device-side kernel launches on sm_120 are supported but have significant per-launch overhead. Not suitable for fine-grained row dispatch.

4. **Matrix reordering: mixed results on GPU.** RCM and METIS reordering help on CPU but research shows mixed results on GPU, with >50% of matrices actually getting slower after reordering. Only consider for very large matrices where x-vector exceeds L2.

5. **SELL-P slice size = 32 is optimal for sm_120.** The warp size is 32. Larger slices increase padding waste without improving coalescing. Some implementations use 64 for occupancy reasons but 32 is the safe starting point.

6. **cuSPARSE preprocess:** `cusparseSpMV_preprocess` is available since CUDA 12.4. It optimizes repeated SpMV on the same matrix. For fair benchmarking, always call preprocess for cuSPARSE and amortize it separately, since our row-binning preprocessing is analogous.

7. **FP32 vs FP64:** Most scientific SpMV is FP64. Our DOT/AXPY primitives are FP32. Decide precision early -- FP32 doubles the effective L2 capacity and halves bandwidth requirements, which could be a significant advantage.

8. **98 MB L2, not 96 MB.** Multiple sources confirm the RTX 5090 has 98,304 KB (96 MiB) of L2 cache. Use 98 MB in calculations.
