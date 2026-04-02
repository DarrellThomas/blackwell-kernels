# Voltrix: Tensor Core SpMM with Asynchronous Pipeline (USENIX ATC 2025)

**Sources:**
- [Voltrix Paper (USENIX ATC 2025)](https://www.usenix.org/conference/atc25/presentation/xia)
- [Voltrix PDF](https://www.usenix.org/system/files/atc25-xia.pdf)
- [RSH-SpMM (arxiv, March 2026)](https://arxiv.org/html/2603.08734)
- [Libra: GPU Heterogeneity for SpMM (arxiv)](https://arxiv.org/html/2506.22714v2)
- [DTC-SpMM (ASPLOS 2024)](https://dl.acm.org/doi/10.1145/3620666.3651378)
**Relevant to:** spmv worker
**Worker's current problem:** Not started yet. Pre-caching research on tensor core
approaches to sparse operations.

---

## What This Is

Voltrix is a state-of-the-art Sparse Matrix-Matrix Multiplication (SpMM) system that
uses tensor cores with an asynchronous data loading pipeline. Published at USENIX ATC
2025. Open-sourced and integrated into PyTorch 2.5.

**Key distinction:** Voltrix is for **SpMM** (sparse x dense matrix), not SpMV
(sparse x vector). SpMM has higher arithmetic intensity and maps better to tensor
cores. However, the techniques may transfer to block-sparse SpMV.

---

## Why It Matters for Us

### Direct Relevance: Medium

Our SpMV worker targets sparse matrix-**vector** multiply, which has much lower
arithmetic intensity than SpMM. Tensor cores are generally not beneficial for SpMV
because the computation is element-wise multiply-add, not matrix multiply.

However, Voltrix's techniques are relevant if:
1. We implement **BSR SpMV** where dense blocks (8x8, 16x8) can use mma.sync
2. We extend to **SpMM** for right-hand-side batched operations
3. We need techniques for **asynchronous data loading** with irregular access patterns

### Transferable Techniques

**1. Warp-Specialized Workflow Control:**
Voltrix separates computation and data loading into distinct warp specializations:
- Data loader warps handle shared memory staging (async cp.async)
- Compute warps consume staged data via tensor cores
- This overlaps memory latency with computation

This pattern is exactly what our GEMM and attention workers already use, but
applying it to sparse data access patterns is novel.

**2. Persistent Kernel with I/O-Balanced Scheduling:**
Voltrix uses a persistent kernel that distributes irregular workloads evenly across
SMs. The I/O co-balanced scheduling considers both compute cost AND memory access
cost when partitioning work to thread blocks. For SpMV, this means:
- Rows with many non-zeros but good cache locality (sequential column indices) get
  balanced against rows with few non-zeros but scattered access patterns
- This is more sophisticated than simple row-length binning

**3. Asynchronous Pipeline for Irregular Access:**
The multi-stage pipeline (similar to our GEMM double-buffering) adapted for sparse
data where buffer sizes vary per tile. Each stage handles a different-sized chunk
of non-zeros.

---

## Performance Numbers

| Comparison | Voltrix Speedup |
|------------|----------------|
| vs cuSPARSE SpMM | 2.7x average |
| vs DTC-SpMM (tensor core) | 1.8x average |
| vs TC-GNN (tensor core) | 36.5x average |
| vs RoDe (CUDA core) | 1.7-1.9x average |

Tested on V100 and A100. Results should transfer to sm_120 since all use mma.sync.

---

## Related SpMM Work (2024-2026)

### RSH-SpMM (arxiv, March 2026)
- "Row-Structured Hybrid Kernel for Sparse Matrix-Matrix Multiplication on GPUs"
- Uses row-structured approach combining tensor core paths and CUDA core paths
- Assigns different kernel strategies based on row non-zero density
- Very recent (March 2026) -- may contain state-of-the-art techniques

### Libra (arxiv, 2025)
- "Unleashing GPU Heterogeneity for High-Performance SpMM"
- Uses both CUDA cores AND tensor cores simultaneously for SpMM
- Assigns dense blocks to tensor cores, sparse elements to CUDA cores
- Exploits GPU's heterogeneous compute units

### DTC-SpMM (ASPLOS 2024)
- "Bridging the Gap in Accelerating General SpMM with Tensor Cores"
- Maps irregular sparse patterns to tensor core operations
- Handles non-block-aligned sparsity with zero-padding and condensation

---

## Applicability to Our SpMV Project

### What Transfers:
- Warp specialization for async data loading in sparse kernels
- Persistent kernel scheduling for irregular workloads
- I/O-balanced work distribution (not just compute-balanced)

### What Doesn't Transfer:
- Tensor core compute (SpMV is too low-intensity for MMA, except BSR)
- Multi-vector parallelism (SpMM spreads work across N columns of dense matrix;
  SpMV has only 1 column)
- The high arithmetic intensity that makes SpMM compute-bound

### Recommendation for SpMV Worker:
Read the Voltrix persistent kernel and I/O scheduling sections for inspiration on
how to handle irregular memory access patterns efficiently. Skip the tensor core
sections unless implementing BSR SpMV. The warp specialization approach could help
for long-row processing where one warp loads the next row segment while another
computes the current one.
