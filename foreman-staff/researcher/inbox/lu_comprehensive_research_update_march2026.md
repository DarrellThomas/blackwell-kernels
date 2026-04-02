# LU Factorization Research Update -- Comprehensive Scan (March 14, 2026)

**Relevant to:** numerical/ worker (LU factorization / getrf)
**Worker's current problem:** Building LU factorization kernel for RTX 5090 (sm_120). Strategy: blocked LU -> CUDA Graph -> monolithic kernel. cuSOLVER baseline is 9.4ms at N=4096. TF32 MMA has B fragment defect on sm_120; must use BF16 MMA with FP32 accumulators.

---

## Summary of Scan

Performed comprehensive web search across 10+ research topics. The existing cache
(`/data/src/bwk/foreman-staff/researcher/cache/lu/`) is **remarkably thorough** and
already covers the major findings. This brief documents new details and confirmations
from today's scan that supplement the existing briefs.

---

## 1. NEW: FP8 Ozaki Scheme on Consumer Blackwell (RTX 5060 Ti)

**Source:** https://arxiv.org/abs/2508.00441 (Mukunoki, 2025)
**Supplements:** `lu_cusolver132_bf16x9_getrf_emulation.md`

The Mukunoki paper tested FP64 emulation via FP8 (E4M3) tensor cores on a
**consumer Blackwell GPU (RTX 5060 Ti)** -- same sm_120 ISA as our RTX 5090.

### Key Numbers

| Metric | Value |
|--------|-------|
| FP8 E4M3 tensor core throughput (RTX 5060 Ti) | 94.74 TFLOPS |
| FP16 tensor core throughput | 47.37 TFLOPS |
| Native FP64 throughput | 0.37 TFLOPS |
| FP8 GEMMs needed for one FP64 GEMM | **121** (11 slices squared) |
| FP8 vs FP16 for emulated DGEMM | **1.26x faster** at m=16384 |
| Alignment requirement | **16-element alignment** for FP8 |

### Why This Matters

For FP64-accuracy LU factorization on RTX 5090:
- FP8 tensor cores are 2x faster than FP16 tensor cores
- But FP8 Ozaki needs ~1.49x more GEMMs than FP16 Ozaki (different slice counts)
- Net result: FP8 wins for large trailing matrices (N >= 8192)
- For our N=4096 target, FP16/BF16 Ozaki may still be competitive due to smaller overheads

### Practical Note

The paper uses `cublasLtMatmul` for FP8 GEMMs. Our worker uses custom mma.sync
kernels. Adapting the Ozaki decomposition to work with our custom BF16 mma.sync
GEMM is straightforward (decompose -> multiply -> accumulate), but the 121 GEMM
count for FP64 makes this impractical for our use case. **BF16x9 for FP32 accuracy
(9 GEMMs) remains the right choice for us.**

---

## 2. NEW: cuSOLVER 13.2 FP64 Emulation API Details (Confirmed)

**Source:** https://docs.nvidia.com/cuda/cusolver/index.html (cuSOLVER 13.2)
**Supplements:** `lu_cusolver132_bf16x9_getrf_emulation.md`

Today's scan confirmed the exact API surface for FP64 emulation in cuSOLVER 13.2:

```cpp
// Math mode enum values (confirmed):
CUSOLVER_FP64_EMULATED_FIXEDPOINT_MATH    // INT8 Ozaki for FP64
CUSOLVER_FP32_EMULATED_BF16X9_MATH        // BF16x9 for FP32
CUSOLVER_FP32_FP64_EMULATED_MATH          // Combined

// Emulation strategy control:
cusolverDnSetEmulationStrategy()          // auto vs manual
cusolverDnSetFixedPointEmulationMantissaControl()  // auto vs fixed
cusolverDnSetFixedPointEmulationMaxMantissaBitCount() // default 53
cusolverDnSetFixedPointEmulationMantissaBitOffset()   // dynamic tuning
cusolverDnSetEmulationSpecialValuesSupport()          // NaN/Inf handling
```

**Critical caveat confirmed:** "On Blackwell GPUs, FP64 fixed-point emulation
kernels may produce incorrect results or experience data corruption when executed
concurrently with third-party kernels that allocate tensor memory."

### Worker Action Item

**Re-benchmark cuSOLVER DGETRF and SGETRF with all emulation math modes enabled
on our RTX 5090.** The existing 9.4ms baseline may have been measured without
emulation. The new baseline with emulation enabled could be significantly faster.

---

## 3. CONFIRMED: cuSOLVERDx 0.3.0 -- No New GETRF Features

**Source:** https://docs.nvidia.com/cuda/cusolverdx/release_notes.html
**Status:** Already covered in `lu_cusolverdx_v030_mathdx_update.md`

cuSOLVERDx 0.3.0 added SVD, eigenvalue solvers, and GTSV. No changes to getrf.
The getrf functionality from v0.2.0 (with sm_120 support) remains the current state.

**Matrix size limitation confirmed:** cuSOLVERDx operates at block level with all
data in shared memory. For sm_120 with 99KB usable shared memory, maximum matrix
size is approximately 157x157 (float). This is only useful for panel sub-blocks
(IB=16 or IB=32 sub-panels), not for the full N=4096 matrix.

---

## 4. CONFIRMED: MAGMA 2.9.0 sm_120 Support

**Source:** https://github.com/icl-utk-edu/magma/blob/master/ReleaseNotes
**Status:** Already covered in `lu_magma29_blackwell_and_hpl_pdfact_update.md`

MAGMA 2.9 (Jan 2025) officially supports sm_120 with:
- `magma_sgetrf_native()` -- GPU-only LU with spin-wait panel sync
- Expert interface with user-configurable blocking sizes (nb, recnb)
- Variable-batch no-pivot LU (`magma_<T>getrf_nopiv_vbatched`)
- Small-size LU tuning

No tensor core acceleration for LU trailing updates. No monolithic kernel.
Architecture unchanged: host-side blocked loop + GPU-native panel + cuBLAS trailing GEMM.

---

## 5. CONFIRMED: Numerical Stability of FP64 Emulation for LU (Luszczek et al.)

**Source:** https://arxiv.org/abs/2509.23565
**Status:** Already covered in `lu_cusolver132_bf16x9_getrf_emulation.md`

Key detail confirmed: The Schur complement update in LU (A22 -= A21 * A11^{-1} * A12)
creates large entry disparities in the U factor. For the ParaWilk test matrix:
- 6 INT8 splits: scaled residual 39.14 (FAILS HPL criterion)
- 7 INT8 splits: scaled residual 0.32 (PASSES)
- Standard random matrices: 6 splits sufficient

**For our N=4096 target with FP32 accuracy (not FP64):** BF16x9 emulation uses
9 BF16 GEMMs (not INT8 splits), which provides exact FP32 output. No stability
concerns for well-conditioned matrices.

---

## 6. CONFIRMED: HPL-MxP Mixed-Precision Algorithm Details

**Source:** https://arxiv.org/abs/2509.19618 (Dongarra & Luszczek, 2026 -- published in IJHPCA)
**Status:** Already covered in `lu_hpl_mxp_mixed_precision_scheme.md`

Additional detail from full paper:
- Panel factorization + TRSM done in **FP32 arithmetic**
- Schur complement GEMM done in **FP16 with FP32 accumulation**
- GMRES refinement typically converges in **5-15 iterations**
- If > 50 iterations needed, the code should trigger failure
- Implementations may perform balancing and scaling of input matrix

**For N=4096:** Total operation count ~2/3 * 4096^3 = ~45.8 GFLOP for factorization.
At BF16x9 throughput (~110 TFLOPS effective): ~0.4ms for trailing GEMMs alone.
With panel + LASWP + overhead, target is 2-4ms total.

---

## 7. CONFIRMED: SFLU Single-Kernel Approach (Sparse, but Pattern Relevant)

**Source:** https://www.ssslab.cn/assets/papers/2021-zhao-sflu.pdf
**Status:** Not previously cached (sparse LU, not directly applicable)

SFLU demonstrates a **synchronization-free single-kernel** LU factorization for sparse
matrices. Key technique: global memory flag arrays for inter-block dependency tracking
(identical to MAGMA's `update_flag` pattern). All thread blocks launched in a single
kernel. No inter-kernel synchronization.

**Relevance:** Confirms that the spin-wait atomic flag pattern works at scale
(287x speedup over SuperLU, 6x over GLU). Our cooperative groups approach
(grid.sync()) is cleaner but the flag pattern is a fallback if cooperative launch
has grid size limitations.

---

## 8. No New GTC 2026 Presentations Found

Searched for NVIDIA GTC 2026 presentations on dense linear algebra / GPU factorization.
No specific presentations found in search results. GTC 2026 is March 16-19 (upcoming),
so material may not be indexed yet.

**Recommendation:** Re-scan after March 19 for GTC 2026 proceedings.

---

## 9. Research Landscape Summary

### What's Settled (No Need to Re-Research)

| Topic | State | Cache Brief |
|-------|-------|-------------|
| cuSOLVERDx device-side getrf | v0.2.0+, sm_120, block-level, shmem-limited | `lu_cusolverdx_device_side_factorization.md` |
| MAGMA native panel kernel | register-resident, spin-wait, rowid trick | `lu_magma_native_kernel_internals.md` |
| BF16x9 FP32 emulation | 9 BF16 GEMMs, exact FP32, 3-4x speedup | `lu_cublas_bf16x9_fp32_emulation_blackwell.md` |
| Cooperative groups panel | grid.sync(), single-block panel recommended | `lu_hpl_gpu_panel_factorization_cooperative_groups.md` |
| RBT pivoting avoidance | Pre-transform + no-pivot LU + refinement | `lu_rbt_pivoting_avoidance_and_remifa.md` |
| Pre-pivoted LU (PRP/MPF) | BF16 pivot pre-computation | `lu_monolithic_gpu_factorization_research.md` |
| Monolithic kernel architecture | Full design with pseudocode | `lu_monolithic_gpu_factorization_research.md` |
| FP64 emulation (Ozaki/ADP) | cuSOLVER 13.2, INT8 tensor cores | `lu_cusolver132_bf16x9_getrf_emulation.md` |
| Mustard device-side task graphs | CPU-free task scheduling | `lu_device_side_task_graphs_mustard_ics2025.md` |

### What's Still Worth Tracking

1. **GTC 2026 presentations** (March 16-19) -- may reveal cuSOLVER internal details
2. **cuSOLVERDx getrf matrix size expansion** -- future versions may support larger panels
3. **MAGMA tensor core LU** -- no tensor core acceleration yet, but ICL is actively working on Blackwell support
4. **FP8 Ozaki for consumer Blackwell** -- Mukunoki's work is early but shows the path for FP64-accuracy workloads

---

## Sources

- [Mukunoki: DGEMM via FP8 Ozaki on Consumer Blackwell (arXiv:2508.00441)](https://arxiv.org/abs/2508.00441)
- [Luszczek et al.: FP64 Emulation Numerics for Factorization (arXiv:2509.23565)](https://arxiv.org/abs/2509.23565)
- [Dongarra & Luszczek: HPL-MxP Benchmark (arXiv:2509.19618)](https://arxiv.org/abs/2509.19618)
- [Schwarz et al.: ADP / Guaranteed DGEMM Accuracy (arXiv:2511.13778)](https://arxiv.org/abs/2511.13778)
- [NVIDIA cuBLAS FP Emulation Blog](https://developer.nvidia.com/blog/unlocking-tensor-core-performance-with-floating-point-emulation-in-cublas/)
- [cuSOLVER 13.2 Documentation](https://docs.nvidia.com/cuda/cusolver/index.html)
- [cuSOLVERDx Release Notes](https://docs.nvidia.com/cuda/cusolverdx/release_notes.html)
- [cuSOLVERDx GETRF Documentation](https://docs.nvidia.com/cuda/cusolverdx/get_started/getrf.html)
- [MAGMA 2.9.0 Release Notes](https://github.com/icl-utk-edu/magma/blob/master/ReleaseNotes)
- [MAGMA getrf API](https://icl.utk.edu/projectsfiles/magma/doxygen/group__magma__getrf.html)
- [HPL-MxP Website](https://hpl-mxp.org/)
- [SFLU: Synchronization-Free Sparse LU (DAC 2021)](https://ieeexplore.ieee.org/document/9586141/)
- [CALU: Communication Optimal LU (SIAM)](https://doi.org/10.1137/100788926)
- [Mixed-Precision Pre-Pivoting LU (J. Supercomputing, 2024)](https://link.springer.com/article/10.1007/s11227-024-06523-w)
- [Mixed Precision LU on Tensor Cores (HAL/ICL, 2023)](https://hal.science/hal-02937325)
- [Progressive Optimization of Batched LU (ICL UTK, 2018)](https://www.netlib.org/utk/people/JackDongarra/PAPERS/icl-utk-1237-2018.pdf)
- [Volkov: LU/QR/Cholesky using Vector GPU Capabilities](https://bebop.cs.berkeley.edu/pubs/volkov2008-gpu-factorizations.pdf)
- [CUDA Cooperative Groups](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html)
- [CUDA 13.2 Release Notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)
