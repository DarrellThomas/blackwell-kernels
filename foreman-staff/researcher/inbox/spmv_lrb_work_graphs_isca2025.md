# SpMV: Logarithmic Radix Binning + GPU Work Graphs (ISCA 2025)

**Sources:**
- https://dl.acm.org/doi/10.1145/3695053.3731060 (GPUs All Grown-Up, ISCA 2025)
- LRB-SpMV: Logarithmic radix binning approach
- LightSpMV: https://ieeexplore.ieee.org/document/7245713/
- CSR-Adaptive: https://dl.acm.org/doi/10.1145/3079079.3079086
- Merge-based SpMV: https://github.com/dumerrill/merge-spmv

**Relevant to:** spmv worker
**Worker's current problem:** Not started yet. Needs foundational understanding of GPU SpMV approaches to choose the right architecture.

---

## What This Is

Two recent advances in GPU SpMV optimization:

1. **LRB-SpMV (Logarithmic Radix Binning)**: Sorts rows by their NNZ count into power-of-two bins, then launches a separate kernel per bin with the optimal parallelism for that row length range.

2. **GPU Work Graphs for SpMV (ISCA 2025)**: Uses CUDA Work Graphs to dynamically dispatch rows to different processing nodes based on row length, achieving fully device-driven SpMV without host-CPU involvement.

---

## Why It Matters for Us

The SpMV project needs to beat cuSPARSE on sm_120. The key challenge in CSR SpMV is **load imbalance** — real sparse matrices have wildly varying row lengths (from 0 to millions of NNZ). A single kernel strategy always leaves performance on the table.

### The Strategy Landscape

| Approach | Pros | Cons | Best For |
|----------|------|------|----------|
| **Thread-per-row** | Simple, low overhead | Load imbalance for long rows | Short rows (NNZ < 32) |
| **Warp-per-row** | Good for medium rows | Warp divergence for short rows | Medium rows (32-1024 NNZ) |
| **Block-per-row** | Handles long rows | Overhead for short rows | Long rows (1024+ NNZ) |
| **Merge-path** | Perfect load balance | Complex, preprocessing overhead | General purpose |
| **LRB (binning)** | Optimal kernel per bin | Multiple launches, preprocessing | Known sparsity patterns |
| **Work Graphs** | Dynamic dispatch, no CPU | Requires CUDA 12+, sm_100+ | General purpose, future |

### LRB-SpMV Details

1. **Preprocessing (GPU-side)**: Scan row offsets, compute NNZ per row, bin into power-of-two buckets
   - Bin 0: rows with 1-2 NNZ (thread-per-row)
   - Bin 1: rows with 3-4 NNZ (thread-per-row, 2 elements/thread)
   - Bin 5: rows with 33-64 NNZ (warp-per-row)
   - Bin 10+: rows with 1024+ NNZ (block-per-row)

2. **Kernel launch**: One kernel per non-empty bin, each optimized for its row length range

3. **Performance**:
   - Preprocessing is 20x faster than CSR-Adaptive
   - Per-SpMV performance is generally lower than CSR-Adaptive (multiple kernel launches)
   - Best when preprocessing is amortized (iterative solvers calling SpMV repeatedly)

### GPU Work Graphs (ISCA 2025)

1. **Architecture**: A binning node computes row lengths and dispatches row IDs to processing nodes
2. **Processing nodes**: Each optimized for a specific NNZ range (thread/warp/block level)
3. **Advantage**: Fully device-driven — no host CPU synchronization between binning and processing
4. **Caveat**: Requires CUDA Work Graphs API (CUDA 12.4+), may not be available on all architectures

---

## Recommended Starter Architecture for sm_120

Based on the literature and our hardware (170 SMs, 48 warps/SM, 1792 GB/s):

### Phase 1: CSR-Adaptive (Recommended First Approach)
- Classify rows as short (<32 NNZ), medium (32-512), or long (>512)
- Short: multiple rows per warp (vectorized warp reduction)
- Medium: one warp per row (shuffle reduction)
- Long: one block per row (shared memory reduction)
- Single kernel with runtime dispatch based on row length

### Phase 2: LRB-SpMV (If Phase 1 Hits Ceiling)
- GPU-side preprocessing to bin rows
- Per-bin specialized kernels
- Amortize preprocessing for iterative solvers

### Key Optimization Axes for sm_120
1. **L2 cache utilization**: 96 MB L2 on RTX 5090 — x vector often fits entirely in L2
2. **Vectorized loads**: float4 or __ldg for coalesced value/column-index loads
3. **Warp shuffle reduction**: __shfl_down_sync for intra-warp sums
4. **Texture/LDG for x vector**: x[col_idx] is random-access — __ldg uses texture cache

---

## Open Source References

| Repo | Description | Relevance |
|------|-------------|-----------|
| `dumerrill/merge-spmv` | Merge-path CSR SpMV | Load-balanced baseline |
| `Ivanrs297/cuda-spmv-csr` | Simple CSR SpMV in CUDA | Minimal starter |
| LightSpMV | Dynamic warp-level CSR SpMV | Warp-based approach |
| cuSPARSE | NVIDIA's production implementation | Benchmark reference |

---

## Caveats

1. **GPU Work Graphs** may not be available on sm_120. The API requires CUDA 12.4+ but the hardware support for Work Graphs may be sm_100+ only. Needs testing.

2. **cuSPARSE 13.x** has been significantly optimized for recent architectures. Beating it requires exploiting specific sparsity patterns that cuSPARSE handles suboptimally.

3. **Format selection matters**: CSR is the standard but ELL (ELLPACK) can be faster for matrices with uniform row lengths. A hybrid CSR+ELL approach (like HYB format) may be optimal for matrices with a mix of row lengths.

4. **The x vector access pattern** is the primary bottleneck for most SpMV kernels. The column indices are random, making x vector loads scatter-gather operations. L2 cache size (96 MB) and texture cache help, but the fundamental memory-boundedness remains.
