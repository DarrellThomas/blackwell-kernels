# SpMV Open-Source Repositories: Practical Study Guide

**Sources:**
- [EHYB (GitHub)](https://github.com/Chong-Chen-UNLV/EHYB_SPMV_GPU)
- [PERKS (GitHub, BSD-3)](https://github.com/neozhang307/PERKS)
- [DASP (GitHub)](https://github.com/Smilesky18/DASP)
- [Merge-SpMV (GitHub, BSD-3)](https://github.com/dumerrill/merge-spmv)
- [Owens Group Merge-SpMV Fork (GitHub)](https://github.com/owensgroup/merge-spmv)
- [TileSpMV (GitHub)](https://github.com/SuperScientificSoftwareLaboratory/TileSpMV)
- [CSR5 (GitHub)](https://github.com/weifengliu-ssslab/Benchmark_SpMV_using_CSR5)
- [HolaSpMV (GitHub)](https://github.com/pintaguras/holaspmv)
- [FlashSparse (GitHub)](https://github.com/ParCIS/FlashSparse)
- [Ginkgo (GitHub)](https://github.com/ginkgo-project/ginkgo)
- [Mixed-Precision SpMV (GitHub)](https://github.com/ParCoreLab/mixed-and-multi-spmv)
- [rocSPARSE (GitHub)](https://github.com/ROCm/rocSPARSE)
- [cuSPARSE Samples (GitHub)](https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuSPARSE/spmv_csr/spmv_csr_example.c)
**Relevant to:** spmv worker
**Worker's current problem:** Needs reference implementations to study for building custom SpMV kernel

---

## What This Is

A practical guide to which open-source SpMV repositories to study, in what order,
and what to extract from each. Prioritized by relevance to our RTX 5090 target.

---

## Tier 1: Study These First (Directly Applicable)

### 1. EHYB — Explicit Caching HYB SpMV
- **URL**: https://github.com/Chong-Chen-UNLV/EHYB_SPMV_GPU
- **License**: Not specified
- **GPU Tested**: Tesla V100 (sm_70)
- **Peak Performance**: 280 GFLOPS FP32, 175 GFLOPS FP64
- **Key Files**: `spmv.cu`, `kernel.cu`, `kernel.h`
- **What to Study**:
  - How x-vector segments are loaded into shared memory
  - The compact 16-bit column index format
  - Matrix partitioning strategy for the HYB format
  - How MT-METIS reordering is integrated (includes `libmtmetis.a`)
- **Relevance**: Direct inspiration for our x-caching and index compression strategy

### 2. PERKS — Persistent Kernel for Iterative SpMV
- **URL**: https://github.com/neozhang307/PERKS
- **License**: BSD-3-Clause
- **GPU Tested**: A100, V100
- **Key Files**: `conjugateGradient/` (CG solver), `stencil/` (2D/3D stencil)
- **Build**: `config.sh` then `build.sh` in each subdirectory
- **What to Study**:
  - How cooperative groups grid sync is used for iteration barriers
  - How the CG solver fuses SpMV + DOT in a persistent kernel
  - How L2 cache residency is exploited across iterations
  - Register/shared memory usage patterns for persistent execution
- **Relevance**: Direct template for our iterative solver integration

### 3. DASP — Row-Classified SpMV
- **URL**: https://github.com/Smilesky18/DASP
- **License**: Not specified
- **GPU Tested**: A100, H800
- **What to Study**:
  - Row-length binning thresholds (short < 32, medium < 512, long > 512)
  - The dispatch mechanism for different bin types
  - **Ignore the MMA/tensor-core parts** — not applicable for SpMV on sm_120
- **Relevance**: The row-binning strategy transfers directly. The MMA parts do not.

### 4. Merge-SpMV — The cuSPARSE Algorithm
- **URL**: https://github.com/dumerrill/merge-spmv (original)
- **URL**: https://github.com/owensgroup/merge-spmv (maintained fork)
- **License**: BSD-3
- **GPU Tested**: Tesla K40 (old, but algorithm is timeless)
- **What to Study**:
  - The 2D binary search for merge-path decomposition
  - The fix-up pass for partial row sums
  - How nonzeros are evenly partitioned across threads
  - This is what cuSPARSE does internally — understanding it is essential
- **Relevance**: Must understand to beat cuSPARSE

---

## Tier 2: Study for Specific Techniques

### 5. TileSpMV — Per-Tile Format Selection
- **URL**: https://github.com/SuperScientificSoftwareLaboratory/TileSpMV
- **GPU Tested**: V100, A100
- **What to Study**:
  - How 16x16 sparse tiles are created
  - The 7 per-tile format options and selection criteria
  - Deferred COO extraction for outlier nonzeros
- **Relevance**: Useful if we pursue per-region format selection (Phase 4 in existing roadmap)

### 6. CSR5 — Perfectly Balanced CSR
- **URL**: https://github.com/weifengliu-ssslab/Benchmark_SpMV_using_CSR5
- **GPU Tested**: Multi-platform
- **What to Study**:
  - The 2D tile layout (width=32, height=sigma)
  - How tile_ptr and tile_descriptor provide balance
  - Preprocessing cost vs per-SpMV benefit
- **Relevance**: Alternative to merge-based for general-purpose balanced SpMV

### 7. Ginkgo — Production-Quality SpMV Library
- **URL**: https://github.com/ginkgo-project/ginkgo
- **GPU Tested**: V100, A100, AMD GPUs
- **What to Study**:
  - ELL and SELL-P format implementations
  - The hybrid ELL/COO approach
  - Load-balanced CSR with sub-warp assignment
  - Production-quality CUDA code patterns
- **Relevance**: Good reference for SELL-P implementation and quality standards

### 8. FlashSparse — Tensor-Core SpMM (NOT SpMV)
- **URL**: https://github.com/ParCIS/FlashSparse
- **License**: Available
- **GPU Tested**: H100, RTX 4090
- **What to Study**:
  - The swap-and-transpose MMA strategy for sparse matrices
  - Minimum vector granularity (8x1) for unstructured sparsity
- **Relevance**: Only relevant if we ever pursue BSR SpMV with tensor cores.
  **NOT applicable for general SpMV** — this is SpMM, not SpMV.

---

## Tier 3: Reference Only

### 9. HolaSpMV — Load-Balanced CSR
- **URL**: https://github.com/pintaguras/holaspmv
- **What to Study**: Load-balancing between thread blocks with additional GPU buffer
- **Relevance**: Older, less documented. CSR5 and merge-based are better references.

### 10. Mixed-Precision SpMV — Row-Wise Precision Selection
- **URL**: https://github.com/ParCoreLab/mixed-and-multi-spmv
- **GPU Tested**: V100
- **What to Study**: How per-row FP16/FP32 selection is implemented
- **Relevance**: Useful if we implement mixed-precision SpMV

### 11. rocSPARSE — AMD's Sparse Library (Open Source!)
- **URL**: https://github.com/ROCm/rocSPARSE (deprecated, moved to ROCm/rocm-libraries)
- **License**: MIT
- **What to Study**:
  - CSR-Adaptive implementation in `library/src/level2/`
  - Row classification logic and bin thresholds
  - This is the only open-source production SpMV library with the adaptive algorithm
- **Relevance**: The CSR-Adaptive algorithm is our primary strategy. rocSPARSE is
  the reference implementation (written for AMD GPUs but the algorithm transfers).
  **The code is HIP, not CUDA, but HIP is nearly identical to CUDA.**

---

## Study Order for the Worker

```
Week 1: Merge-SpMV (understand what cuSPARSE does)
         + DASP (understand row binning)
         + rocSPARSE CSR-Adaptive (reference implementation)

Week 2: EHYB (x-caching + index compression)
         + PERKS (persistent kernel for CG)

Week 3: CSR5 or TileSpMV (if Phase 2+ needed)
         + Ginkgo SELL-P (if ELL path needed)
```

---

## Caveats

1. **Most repos target sm_70 (V100) or older**: They will need recompilation for
   sm_120. Algorithm-level code transfers directly; PTX-level code does not.

2. **License diversity**: Some repos have no explicit license. Be aware when
   copying code directly. BSD-3 (merge-spmv, PERKS) and MIT (rocSPARSE) are safe.

3. **rocSPARSE uses HIP, not CUDA**: HIP syntax is nearly identical to CUDA.
   Replace `hipMalloc` with `cudaMalloc`, `__syncthreads()` is the same, etc.
   The algorithm logic transfers verbatim.

4. **FlashSparse is SpMM, not SpMV**: Do not confuse sparse matrix-matrix multiply
   with sparse matrix-vector multiply. The techniques do not transfer.
