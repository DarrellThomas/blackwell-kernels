# SpMV Auto-Tuning and Format Selection: Advanced Techniques

**Sources:**
- [Flat/Line-Enhance SpMV (HPDC '23)](https://dl.acm.org/doi/abs/10.1145/3588195.3593002)
- [CB-SpMV: Cache-Friendly Block-Based SpMV (ICS '25)](https://dl.acm.org/doi/10.1145/3721145.3725746)
- [HBP: Hash-Based Partition SpMV (arxiv 2504.08860)](https://arxiv.org/abs/2504.08860)
- [FastLoad: Coalesced SpMV Data Loading (IEEE TPDS '24)](https://ieeexplore.ieee.org/document/10713183/)
- [EC-SpMV: Block Extraction for Sparse LLMs (arxiv 2507.12205)](https://arxiv.org/abs/2507.12205)
- [TileSpMV: Tiled SpMV with Per-Tile Format (IPDPS '21)](https://ieeexplore.ieee.org/document/9460505/)
- [AlphaSparse: Machine-Designed SpMV Formats (SC '22)](https://sc22.supercomputing.org/proceedings/tech_paper/tech_paper_pages/pap145.html)
- [BestSF: ML-Based Format Selection (ACM TACO '18)](https://dl.acm.org/doi/fullHtml/10.1145/3226228)
- [Spaden: Bitmap SpMV with Tensor Cores (ICPP '24)](https://dl.acm.org/doi/10.1145/3673038.3673055)
- [CSR5 (ICS '15)](https://dl.acm.org/doi/10.1145/2751205.2751209)
- [Systematic SpMV Survey (arxiv 2404.06047)](https://arxiv.org/html/2404.06047v1)
- [GPU Work Graphs SpMV (ISCA '25)](https://dl.acm.org/doi/10.1145/3695053.3731060)
- [cuSPARSE 13.2 Documentation](https://docs.nvidia.com/cuda/cusparse/index.html)
- [Merge-Based SpMV (Merrill & Garland, SC '16)](https://images.nvidia.com/events/sc15/pdfs/sc15-Merge-Based-Parallel-Sparse-Matrix-Vector-Multiplication-merrill.pdf)
- [NVIDIA Vectorized Memory Access](https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access)

**Relevant to:** spmv worker
**Worker's current problem:** Building custom SpMV kernel; needs to understand auto-tuning and format selection strategies beyond the basics already documented.

---

## What This Is

A research brief covering **runtime auto-tuning strategies** for SpMV: how to automatically select the best format and algorithm for a given sparse matrix at runtime, and several recent (2023-2025) techniques that push beyond the merge-based and CSR-Adaptive approaches already documented. This complements the existing `spmv_format_selection_optimization.md` (which covers formats in depth) and `spmv_csr_implementation_strategies.md` (which covers CSR kernels) by focusing on the **selection and adaptation layer**.

---

## Why It Matters for Us

The existing docs correctly identify that "format selection dominates performance" and provide a static heuristic table (nnz/row ranges to format). But the state of the art has moved beyond static heuristics into **per-tile, per-region, and per-matrix adaptive selection**. Several 2024-2025 papers demonstrate 2-6x gains over cuSPARSE by exploiting structure at finer granularity than whole-matrix format choice. This brief covers those techniques.

---

## 1. The Auto-Tuning Landscape: Three Levels

Auto-tuning for SpMV operates at three granularities:

### Level 1: Whole-Matrix Format Selection (Coarse)

Pick one format for the entire matrix. This is what the existing heuristic table does.

**Approach:** Extract cheap features (avg/std/max nnz per row, matrix dimensions, bandwidth) and select format.

**Best known method:** BestSF (2018) uses a Weighted SVM classifier trained on benchmark data. Achieves >97% of oracle-optimal performance. The feature set that matters most:
- `avg_nnz_per_row` -- primary discriminator
- `std_nnz_per_row / avg_nnz_per_row` (coefficient of variation) -- uniformity measure
- `max_nnz_per_row / avg_nnz_per_row` -- outlier ratio
- `nnz / (M * N)` -- global density

**Practical heuristic (no ML needed):**

```
coeff_of_variation = std_nnz / avg_nnz
outlier_ratio = max_nnz / avg_nnz

if avg_nnz <= 2:
    use CSR-Scalar (thread per row)
elif coeff_of_variation < 0.3 and avg_nnz <= 64:
    use SELL-P (slice_size=32, sigma-sorted)
elif coeff_of_variation < 0.5 and avg_nnz <= 32:
    use SELL-P or ELL
elif outlier_ratio > 20:
    use Row-Binned CSR-Adaptive
elif avg_nnz > 64:
    use Merge-Based CSR
else:
    use Merge-Based CSR  # safe default
```

### Level 2: Per-Region Format Selection (Medium)

Divide the matrix into 2D blocks/tiles and select a format per tile. This captures local structure that whole-matrix selection misses.

**Key paper: TileSpMV (IPDPS '21)** -- divides the matrix into 16x16 sparse tiles. Each tile gets one of seven format options (CSR, COO, ELL, HYB, dense, dense-row, dense-column). Very sparse tiles are extracted into a separate matrix. Tested on 2,757 SuiteSparse matrices: faster than Merge-SpMV on 1,813, faster than CSR5 on 2,040 matrices.

**Key paper: CB-SpMV (ICS '25)** -- cache-friendly 2D blocking with per-block format selection. Uses "virtual pointers" to aggregate different data types within sub-blocks for better L1/L2 locality. On RTX 4090: L1 hit rate +82%, L2 hit rate +19% vs TileSpMV. Achieves 2.95x over cuSPARSE-BSR, 3.06x over TileSpMV.

**Relevance for us:** CB-SpMV is the most recent and strongest per-region approach. Its cache-friendly design is particularly relevant for RTX 5090 with its large L2.

### Level 3: Per-Matrix Code Generation (Fine)

Generate a custom kernel for each specific matrix. Maximum performance, maximum preprocessing cost.

**Key paper: AlphaSparse (SC '22)** -- uses an "Operator Graph" to search the full design space of format + kernel + parameters. Generates machine-designed formats that don't correspond to any human-designed format. Results: 3.2x average over five state-of-the-art formats, 1.5x over traditional auto-tuning, tested on 843 SuiteSparse matrices. Open source: [github.com/PAA-NCIC/AlphaSparse](https://github.com/PAA-NCIC/AlphaSparse).

**Relevance for us:** The preprocessing cost is too high for one-shot SpMV but acceptable for iterative solvers. More importantly, AlphaSparse's insights about what makes a good format can inform our manual design.

---

## 2. Recent Algorithms Worth Knowing

### 2.1 Flat and Line-Enhance (HPDC '23)

Two new CSR-based algorithms that work directly on CSR without format conversion:

**Flat algorithm:** Pure non-zero splitting. Every thread processes an equal share of non-zeros, with memory access patterns optimized for data loading, storing, and reduction steps. Similar to merge-based but with different partitioning and reduction strategies.

**Line-enhance algorithm:** Hybrid of row-splitting and non-zero splitting. Short rows use row-splitting (one or more rows per thread), long rows use non-zero splitting (multiple threads per row). The transition point is adaptive based on matrix characteristics.

**Adaptive selection:** A lightweight decision function selects flat vs line-enhance based on row-length distribution. The paper reports the selection overhead is negligible.

**Performance:** Average speedups over CSR-Vector (424%), CSR-Adaptive (741%), HOLA (49%), cuSPARSE (46%), and merge-based SpMV (72%). Tested on both AMD and NVIDIA GPUs.

**Why this matters:** The 72% average improvement over merge-based SpMV is significant. Since cuSPARSE internally uses merge-based, this suggests a path to beating cuSPARSE by ~46% on average (as the paper confirms). The key insight is that the flat algorithm's non-zero splitting strategy handles the data loading pattern differently from merge-path, achieving better memory coalescing for certain matrix shapes.

### 2.2 Hash-Based Partition (HBP) Format (arxiv, April 2025)

A novel preprocessing approach that uses nonlinear hash functions to reorder rows within 2D matrix blocks, grouping rows with similar non-zero counts.

**How it works:**
1. Divide the matrix into 2D blocks (column partition = 4096, row partition = 512)
2. Within each block, apply a nonlinear hash function to map row indices based on their nnz count
3. Rows with similar nnz end up in adjacent positions, reducing warp divergence
4. Hash mapping is per-row parallelizable (no inter-row dependencies)

**Key data structures:**
- `col/data`: Contiguous values and column indices per block
- `add_sign`: Distance to next non-zero in same row (-1 = row end)
- `zero_row`: Marks zero rows, tracks preceding zero rows in warp
- `begin_nnz`: Starting positions per block
- `output_hash`: Maps hashed positions back to original row indices

**Load balancing:**
- **Intra-block:** Hash transformation reduces nnz distribution std by 42-79%
- **Inter-block:** Mixed allocation with fixed + competitive parts. Fixed allocations assign equal matrix blocks per warp. Unassigned blocks claimed via ticket locks by idle warps.

**Performance:** 3.53x speedup over sorting-based preprocessing, 3.67x over dynamic programming in preprocessing. Memory throughput jumps from 2.85 to 145.12 GB/s on RTX 4090 after optimization.

**Relevance for us:** The hash-based row reordering within blocks is a lightweight alternative to full matrix reordering (RCM etc.). Could be combined with our CSR-Adaptive approach -- hash-reorder within each bin for reduced warp divergence.

### 2.3 FastLoad (IEEE TPDS, December 2024)

Optimizes data loading patterns for both the sparse matrix and the input vector x.

**Core ideas:**
1. Sort columns within each row by non-zero element count to improve access patterns
2. Organize non-zeros into blocks to avoid thread divergence
3. Use segment-sum and prefix-sum to reduce atomic operations
4. Loads vector x segments into shared memory for reuse

**Implementation note:** Uses CSC (Compressed Sparse Column) as the base format, not CSR. The column-oriented approach enables better coalesced loads of the matrix data.

**Performance:** Geometric mean speedup of 2.12x over CSC-based, 2.98x over cuSPARSE, 2.88x over CSR5, 1.22x over TileSpMV. Tested on RTX 3090 Ti.

**Relevance for us:** The vector segment caching in shared memory is directly applicable. For matrices where x doesn't fit in L2 (N > 16M for FP32 on RTX 5090), prefetching x segments into shared memory per-block can significantly improve performance.

### 2.4 EC-SpMV (arxiv, July 2025)

Designed specifically for sparse LLM weight matrices but the techniques generalize.

**Key innovations:**
1. **Hierarchical block extraction:** Identifies dense sub-blocks at multiple granularities within the sparse matrix. Captures structure that flat CSR/ELL miss.
2. **EC-CSR format:** Uses delta indexing (storing column differences instead of absolute indices) to reduce storage overhead by up to 55.4% vs CSR. Fewer bytes loaded = higher effective bandwidth.
3. **Compressed index arrays:** Since LLM sparsity patterns often have locality (nearby columns), delta encoding compresses column indices significantly.

**Performance:** Up to 6.44x over state-of-the-art SpMV libraries. The delta indexing alone provides measurable bandwidth savings.

**Relevance for us:** Delta indexing for column indices is a simple optimization we could apply to CSR. If column indices within a row tend to be close together (which they are for banded/structured matrices), delta encoding reduces the bytes loaded per non-zero. For FP32: from 12 bytes/nnz (4 val + 4 col_idx + 4 x[col]) to potentially 10 bytes/nnz (4 val + 2 delta_col + 4 x[col]) if deltas fit in 16 bits.

---

## 3. Practical Auto-Tuning Implementation Strategy

Based on all the literature, here is a concrete strategy for the worker:

### Phase 1: Lightweight Feature Extraction (O(M) time)

On the first call with a new matrix, extract:
```
M, N, nnz
avg_nnz = nnz / M
max_nnz, min_nnz  (single pass over row_ptr)
std_nnz            (single pass over row_ptr)
row_length_histogram[buckets]  (0, 1-2, 3-8, 9-32, 33-128, 129-512, 513+)
has_block_structure (sample-based check: pick 64 random 8x8 tiles, check fill ratio)
```

This preprocessing is one pass over `row_ptr` (M+1 reads) and negligible vs the SpMV itself.

### Phase 2: Decision Tree

```
IF has_block_structure AND block_fill > 0.3:
    → BSR with detected block size (evaluate 2x2, 4x4, 8x8)

ELIF std_nnz / avg_nnz < 0.3:
    → SELL-P (slice_size=32, sigma=256 for row sorting)
    # Uniform rows: SELL-P eliminates warp divergence entirely

ELIF histogram shows >80% of nnz in rows of length 1-32:
    → CSR-Adaptive with packed short-row bins
    # Most rows are short: pack multiple rows per warp

ELIF histogram shows >30% of nnz in rows of length 512+:
    → Row-binned CSR with VectorL for long rows
    # Significant long rows need thread-block-level processing

ELIF max_nnz / avg_nnz > 50:
    → Hybrid: ELL (up to P75 row length) + COO (overflow)
    # Extreme outlier rows: separate them out

ELSE:
    → Merge-based CSR (Merrill & Garland style)
    # General case: robust for all patterns
```

### Phase 3: Runtime Validation (Optional)

For iterative solvers doing thousands of SpMV on the same matrix, run a quick 5-iteration benchmark of the top 2 candidate formats. Pick the winner. The cost of 5 extra SpMV iterations is amortized over thousands.

### Phase 4: Per-Region Refinement (Advanced)

If Phase 2 performance is within 20% of cuSPARSE but not beating it, consider TileSpMV-style per-tile format selection. This adds preprocessing cost but captures local structure that whole-matrix selection misses. CB-SpMV's cache-friendly variant is the best known approach.

---

## 4. Key Implementation Details for Warp-Level Processing

### Sub-Warp Assignment for Short Rows

When rows have 1-16 non-zeros, assigning a full warp (32 threads) wastes most threads. Use sub-warp groups:

| Row Length | Threads/Row | Rows/Warp |
|-----------|------------|-----------|
| 1-2       | 2          | 16        |
| 3-4       | 4          | 8         |
| 5-8       | 8          | 4         |
| 9-16      | 16         | 2         |
| 17-32     | 32         | 1         |
| 33+       | 32+ (multi-warp) | <1  |

The sub-warp reduction uses `__shfl_down_sync` with a mask covering only the active sub-warp:
```cuda
// For sub-warp of size `sub_size`
unsigned mask = ((1u << sub_size) - 1) << (lane_id & ~(sub_size - 1));
for (int offset = sub_size / 2; offset > 0; offset >>= 1)
    sum += __shfl_down_sync(mask, sum, offset);
```

### Vectorized Loads for Matrix Data

For CSR, `val[]` and `col_idx[]` are stored contiguously per row. When processing a row with stride-32 access (CSR-Vector), consecutive lanes access consecutive elements, enabling 128-byte transactions. But for CSR-Scalar or sub-warp, consider:

```cuda
// Load 4 val+col pairs at once using float4/int4
// Only when row_start is 16-byte aligned and remaining nnz >= 4
float4 vals = *reinterpret_cast<const float4*>(val + j);
int4 cols = *reinterpret_cast<const int4*>(col_idx + j);
sum += vals.x * x[cols.x] + vals.y * x[cols.y]
     + vals.z * x[cols.z] + vals.w * x[cols.w];
```

Vectorized loads reduce instruction count and improve load efficiency. NVIDIA's blog on vectorized memory access shows this can increase bandwidth utilization by 2-4x for sequential access patterns.

### Shared Memory Vector Caching

For matrices where the x-vector exceeds L2 capacity (N > 16M for FP32), or for improving L2 hit rates on medium matrices:

```cuda
// Each thread block caches a segment of x into shared memory
// Before processing its assigned rows, load the relevant x-segment
__shared__ float x_cache[SEGMENT_SIZE];

// Collaborative load: all threads in block load x_cache
for (int i = threadIdx.x; i < segment_size; i += blockDim.x)
    x_cache[i] = x[segment_start + i];
__syncthreads();

// During SpMV, check if col_idx falls within cached segment
// Use x_cache[col - segment_start] for hits, x[col] for misses
```

This is the core of FastLoad's and HBP's shared memory optimization. The effectiveness depends on column locality within the assigned rows.

---

## 5. Bandwidth Arithmetic for RTX 5090

### Theoretical Peak SpMV Throughput

For FP32 CSR SpMV, each non-zero requires:
- 4 bytes: value (float)
- 4 bytes: column index (int32)
- 4 bytes: x[col] load (float, may hit L2 cache)
- Amortized: ~0.01 bytes for row_ptr (negligible for avg_nnz > 10)

**Best case (x fully cached in L2):**
- Bytes from DRAM per nnz: 8 (val + col_idx only)
- Peak nnz/s: 1792 GB/s / 8 = 224 G-nnz/s
- Peak GFLOP/s: 448 (2 FLOPs per nnz)

**Typical case (x partially cached):**
- Bytes from DRAM per nnz: ~10-12 (val + col_idx + partial x misses)
- Practical nnz/s: 150-180 G-nnz/s
- Practical GFLOP/s: 300-360

**Worst case (x not cached, random access):**
- Bytes from DRAM per nnz: 12 (val + col_idx + x)
- Peak nnz/s: 149 G-nnz/s
- Peak GFLOP/s: 299

**Target bandwidth utilization:** A well-optimized kernel should achieve 60-80% of peak DRAM bandwidth. For RTX 5090: 1075-1434 GB/s effective. cuSPARSE typically achieves 50-70% of peak.

### Index Compression Opportunity

If column indices fit in 16 bits (N < 65536, common for small-medium matrices):
- 4 bytes value + 2 bytes col16 = 6 bytes per nnz (vs 8)
- 33% more nnz/s at the same bandwidth
- RTX 5090 peak: 298 G-nnz/s with 16-bit indices

EC-SpMV's delta indexing goes further: if consecutive column indices within a row differ by < 256, use 8-bit deltas. For banded matrices this is often the case.

---

## 6. Caveats for sm_120

1. **Tensor cores for SpMV: still not worth it.** Spaden (ICPP '24) shows tensor-core SpMV can work with bitmap compression and 4-bit quantized values, achieving 3x over CSR on specific matrices. But it requires format conversion overhead and only helps when the matrix has exploitable block structure. For general SpMV on sm_120, CUDA cores are the right choice (confirmed by the "Can Tensor Cores Benefit Memory-Bound Kernels?" paper).

2. **GPU Work Graphs remain CUDA-unavailable.** The ISCA '25 Work Graphs SpMV achieves 3.35x mean over rocSPARSE but is D3D12-only. Persistent kernels with cooperative groups remain the CUDA alternative for iterative solver loops.

3. **cuSPARSE 13.2 now supports SELL format** in cusparseSpMV. If the worker implements SELL-P, benchmark against cuSPARSE's SELL implementation, not just CSR.

4. **cusparseSpMVOp performance improved on B200 in CUDA 13.2.** The SpMVOp API now returns workspace buffer size separately (cusparseSpMVOp_bufferSize), removing internal allocations. This is the strongest cuSPARSE baseline. Note: SpMVOp improvements are documented for B200 (datacenter Blackwell, sm_100). The RTX 5090 (sm_120) may or may not see the same improvement -- benchmark both cusparseSpMV and cusparseSpMVOp.

5. **AlphaSparse generates CPU-targeted code.** The operator graph search is architecture-agnostic in principle but the current open-source implementation targets CPU. The concept (search over format+kernel+parameter space) is applicable to GPU but would need reimplementation.

6. **CB-SpMV results are on RTX 4090 (Ada, sm_89).** The mma.sync ISA is the same as sm_120, and the cache hierarchy is similar (though RTX 5090 has larger L2). Results should transfer well, possibly even better due to higher bandwidth.

7. **Row reordering within blocks (HBP's hash approach) is cheap.** The hash function operates per-row independently, so it's fully parallelizable on GPU. Preprocessing cost is 3.5x lower than sorting. Worth trying as an add-on to CSR-Adaptive.
