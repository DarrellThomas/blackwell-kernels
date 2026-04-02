# cuSPARSE CUDA 13.x Full Changelog (SpMV-Relevant)

**Sources:**
- [CUDA 13.2 Release Notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)
- [cuSPARSE 13.2 Documentation](https://docs.nvidia.com/cuda/cusparse/)
- [CUDA 13.1 Release Notes](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-toolkit-release-notes/index.html)
- [CUDA 13.0 Release Notes](https://docs.nvidia.com/cuda/archive/13.0.2/cuda-toolkit-release-notes/index.html)
- [NVIDIA CUDA 13.1 Blog Post](https://developer.nvidia.com/blog/nvidia-cuda-13-1-powers-next-gen-gpu-programming-with-nvidia-cuda-tile-and-performance-gains/)
**Relevant to:** spmv worker
**Worker's current problem:** Need accurate cuSPARSE baseline; must know what cuSPARSE can and cannot do

---

## What This Is

A complete itemized list of every cuSPARSE change across CUDA 13.0, 13.0u1, 13.1,
13.1u1, and 13.2 that is relevant to SpMV. This ensures the worker benchmarks
against the strongest possible cuSPARSE baseline and knows exactly where custom
kernels can win.

---

## CUDA 13.0 (Initial Release)

**Deprecations:**
- Dropped support for pre-Turing architectures (Maxwell, Volta, Pascal)
- sm_120 (RTX 5090) is supported

**Bug Fixes:**
- Fixed `cusparseSparseToDense_bufferSize` requesting up to 16x more memory than needed
- Removed unwanted 16-byte alignment requirements on external buffers (except SpGEMM)
- Fixed CSR SpMV failures with zero-dimension inputs
- Fixed `cusparseCsr2cscEx2` for zero-dimension matrices

**Known Issues:**
- `CUSPARSE_SPMM_CSR_ALG3` produces non-deterministic results
- cuSPARSE logging APIs crash on Windows

## CUDA 13.0 Update 1

**New Features:**
- **BSR format added to generic SpMV API** -- `cusparseSpMV()` now accepts BSR descriptors
- All generic APIs now support zero-dimension and small-dimension matrices/vectors

**Bug Fixes:**
- Fixed incorrect mixed-precision CSR/COO SpMV computation results (important!)

**Deprecations:**
- Legacy BSR SpMV API deprecated (use generic SpMV API with BSR descriptor)

## CUDA 13.1

**New Features (Major):**
- **SpMVOp API introduced** (experimental) -- persistent-kernel-based SpMV with
  user-defined epilogues. Supports:
  - CSR format only
  - 32-bit indices only
  - Double precision (FP64) only
  - Custom epilogue functions (fuse post-SpMV ops)

**Performance:**
- Improved `cusparseXcsrsort` with reduced memory usage and higher performance

**Bug Fixes:**
- Fixed alignment issues in `cusparseCsr2cscEx2`, `cusparseSparseToDense`, and
  CSR/COO `cusparseSpMM`
- Fixed determinism issue in CSR SpMM ALG3
- Extended 2^31 - 1 nnz support to all routines except SpSV and SpSM
- Fixed race condition in dynamic driver API loading

**Known Issues:**
- 32-bit indexing in `cusparseSpSV` and `cusparseSpSM` may crash when nnz
  approaches 2^31 - 1

## CUDA 13.1 Update 1

**New Features:**
- **`cusparseSpMVOp_bufferSize` API** -- returns workspace buffer size for SpMVOp
  separately. Users provide the buffer when creating `cusparseSpMVOpDescr_t`,
  removing internal memory allocations (performance improvement for repeated calls)

**Performance:**
- **Improved SpMVOp performance on B200** (datacenter Blackwell, sm_100)
  - Note: B200 is sm_100 (datacenter), NOT sm_120 (consumer). Performance
    improvement may or may not apply to RTX 5090.

**Bug Fixes:**
- Fixed accuracy issues in mixed-precision CSR/COO SpMM operations
- Fixed CSR SpMM with high column count dense matrices

## CUDA 13.2 (Latest)

**Performance:**
- Improved runtime of `SpMVOp::buffer_size_estimate` API

---

## What cuSPARSE CAN Do (Competitive Baseline)

| Feature | Status |
|---------|--------|
| CSR SpMV (FP32, FP64) | Mature, merge-based, strong baseline |
| CSR SpMV preprocessing | `cusparseSpMV_preprocess()` available since 12.4 |
| COO SpMV | Supported with ALG1 (fast) and ALG2 (deterministic) |
| BSR SpMV | Added in 13.0u1 via generic API |
| SELL SpMV | Supported via `cusparseCreateSlicedEll()` |
| SpMVOp (persistent) | Experimental, FP64 only, CSR only, 32-bit indices |
| Mixed precision CSR/COO | Supported (bug fixed in 13.0u1) |

## What cuSPARSE CANNOT Do (Our Competitive Advantage)

| Gap | Our Opportunity |
|-----|----------------|
| **BF16 SpMV** | cuSPARSE has no BF16 value support. BF16 halves value traffic. |
| **FP32 SpMVOp** | SpMVOp is FP64-only. FP32 persistent SpMV is wide open. |
| **16-bit column indices** | cuSPARSE requires 32-bit indices. 16-bit saves 25%. |
| **Row-binned dispatch** | cuSPARSE uses one-size-fits-all merge-based. |
| **x-vector caching** | cuSPARSE relies on hardware L2, no explicit caching. |
| **Fused SpMV+DOT** | No kernel fusion API for solver operations. |
| **Custom epilogues in FP32** | SpMVOp epilogues are FP64-only. |
| **Matrix reordering** | cuSPARSE uses matrix as-is, no RCM/bandwidth reduction. |

---

## Benchmarking Recommendations

### Strongest cuSPARSE Baseline

```cuda
// Create matrix descriptor
cusparseCreateCsr(&matA, M, N, nnz, row_ptr, col_idx, vals, ...);
cusparseCreateDnVec(&vecX, N, x, ...);
cusparseCreateDnVec(&vecY, M, y, ...);

// Use ALG1 (fastest, non-deterministic)
cusparseSpMV_preprocess(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
    &alpha, matA, vecX, &beta, vecY,
    CUDA_R_32F, CUSPARSE_SPMV_CSR_ALG1, buffer);

// Benchmark
cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
    &alpha, matA, vecX, &beta, vecY,
    CUDA_R_32F, CUSPARSE_SPMV_CSR_ALG1, buffer);
```

### Also Test Against

1. `CUSPARSE_SPMV_ALG_DEFAULT` -- may choose a different algorithm
2. `CUSPARSE_SPMV_CSR_ALG2` -- deterministic, slower (lower bar)
3. SELL format via `cusparseCreateSlicedEll()` for regular matrices
4. BSR format for block-structured matrices

### Fair Comparison Notes

- Always call `cusparseSpMV_preprocess()` for cuSPARSE and amortize separately
- Our row-binning preprocessing is analogous -- both are one-time costs
- Report speedup both with and without preprocessing in the comparison
- For iterative solver benchmarks, amortize preprocessing over iteration count
