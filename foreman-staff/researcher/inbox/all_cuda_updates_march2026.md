# CUDA Ecosystem Updates -- March 2026 Consolidated Brief

**Sources:**
- [CUDA 13.2 Release Notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)
- [CUDA 13.1 Release Notes](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-toolkit-release-notes/index.html)
- [CUDA 13.2 Blog Post](https://developer.nvidia.com/blog/cuda-13-2-introduces-enhanced-cuda-tile-support-and-new-python-features/)
- [CUDA 13.1 Blog Post](https://developer.nvidia.com/blog/nvidia-cuda-13-1-powers-next-gen-gpu-programming-with-nvidia-cuda-tile-and-performance-gains/)
- [cuBLAS FP Emulation Blog](https://developer.nvidia.com/blog/unlocking-tensor-core-performance-with-floating-point-emulation-in-cublas/)
- [Blackwell Tuning Guide 13.2](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html)
- [CUTLASS 4.4.2 Changelog](https://docs.nvidia.com/cutlass/latest/CHANGELOG.html)
- [NVIDIA 595 Linux Driver](https://www.phoronix.com/review/nvidia-595-linux/4)

**Relevant to:** all active workers
**Date:** 2026-03-14

**NOTE:** This brief covers NEW findings not already documented in existing inbox briefs
(`all_cuda13_sm120_new_features.md`, `all_cuda132_ptx_isa92_updates.md`,
`all_cuda131_library_updates.md`, `all_cccl32_cub_new_primitives.md`,
`all_cusolver132_fp64_emulation_factorization.md`, `all_cutlass_42_sm120_geforce_gemm_examples.md`).
Read those first for the full picture.

---

## 1. CUDA Toolkit Status: 13.2 Is Current (March 2026)

We are running **CUDA 13.0**. Two major releases have shipped since:

| Version | Release Date | PTX ISA | Key Theme |
|---------|-------------|---------|-----------|
| 13.0 | Dec 2025 | 9.0 | sm_120 initial support |
| 13.1 | Jan 2026 | 9.1 | CUDA Tile, cuBLAS Grouped GEMM |
| **13.2** | **Mar 2026** | **9.2** | Library perf, FP64 emulation, tools |

**Recommendation for foreman:** Consider upgrading to CUDA 13.2. The bug fixes
alone (FP8 kernel hangs, cublasLtMatmul concurrent execution issues) could affect
our benchmark stability. The library improvements (cuBLAS L3, cuSOLVER emulation)
directly benefit linalg and numerical workers.

---

## 2. cuBLAS 13.2 Critical Bug Fixes (HIGH IMPACT)

These bugs exist in CUDA 13.0 and are fixed in 13.2. They may be affecting our
workers' benchmark reliability:

### a) cublasLtMatmul Concurrent Kernel Correctness
**Bug:** `cublasLtMatmul` with algorithm `CUBLASLT_ALGO_CONFIG_ID=66` produces
incorrect results when run concurrently with other kernels.

**Impact:** If our cuBLAS reference benchmarks use concurrent execution (e.g.,
multiple streams), results could be wrong. Our GEMM worker compares against
`torch.mm` which uses cublasLtMatmul internally.

### b) FP8 Kernel Hang on sm_90 with beta != 0, scale_C = 0
**Bug:** FP8 matmul hangs on compute capability 9.0 when `beta != 0` and
`scale_C = 0`. While this targets sm_90, similar edge cases may exist on sm_120.

### c) Grouped GEMM k=0 Groups Ignored (since CUDA 13.1)
**Bug:** Grouped GEMM API ignores groups with `k=0`, potentially producing
incorrect results. Fixed in 13.2.

### d) Large Leading Dimension Invalid Memory Access
**Bug:** Large leading dimensions cause invalid memory access on compute
capabilities 9.0, 10.x, 11.0.

**Action:** If workers experience any intermittent incorrect results or crashes
with cuBLAS reference benchmarks, upgrading to CUDA 13.2 is the first step.

---

## 3. cuBLAS 13.2: Level 3 Non-GEMM Kernel Improvements (LINALG WORKER)

**NEW in 13.2 (not in existing L3 brief which covers 13.0):**

Improved performance of **SYRK, HERK, TRMM, SYMM, HEMM** for **FP32 and CF32**
precisions on Blackwell GPUs. This is a SECOND round of improvements beyond 13.0.

Also: up to **20% speedup for FP8, FP16/BF16, TF32, INT8 GEMM** on RTX PRO 6000
(Blackwell professional). These improvements likely transfer partially to RTX 5090
since both are sm_120.

**Direct impact on linalg worker:**
- SYRK (currently 0.96x reference) -- reference bar may move UP if using cuBLAS SYRK
- TRMM (currently 1.02x reference) -- reference bar may move UP
- Worker should **re-baseline** after any CUDA upgrade

**Impact on linalg worker TRSM (0.82x reference):**
- TRSM is not listed in the improvements, so reference should stay stable
- But GEMM improvements (used in blocked TRSM) could indirectly help

---

## 4. cuBLAS FP32 Emulation via BF16x9 -- Technical Details

The cuBLAS blog post reveals the BF16x9 technique:

**How it works:** Decomposes each FP32 value into 9 BF16 components using a static
decomposition scheme. The BF16 tensor cores perform 9 matmuls and sum the partial
products to emulate FP32 precision. This is a "split-K" in the precision dimension.

**Performance:** Up to 3x speedup for FP32 GEMM on Blackwell (vs native FP32 on
same hardware), 2.4x for weather simulation (ecTrans).

**FP64 emulation (Ozaki scheme):** Uses Automatic Dynamic Precision (ADP) -- analyzes
inputs at runtime to determine if emulation is safe. Uses INT8 tensor cores
internally for FP64 matmuls.

**Applicability to sm_120/RTX 5090:** The blog explicitly mentions RTX PRO 6000
Blackwell. It does NOT explicitly mention RTX 5090/sm_120. However, since BF16
tensor cores are identical on sm_120, the FP32 emulation should work. The FP64
emulation (using INT8 tensor cores) should also work but performance numbers may
differ.

**Impact:** For numerical workers (LU, Cholesky, QR), this means cuBLAS GEMM
calls with `CUBLAS_COMPUTE_32F` can now be transparently emulated via tensor
cores, potentially shifting baseline performance. The `CUBLAS_EMULATION_SPECIAL_VALUES_SUPPORT_MASK`
environment variable controls special-case handling (NaN, Inf, denormals).

---

## 5. cuSOLVER 13.2: New Emulation APIs (NUMERICAL WORKER)

Beyond what's in the existing cuSOLVER brief, the specific new APIs are:

```c
// Mantissa precision control
cusolverDnSetFixedPointEmulationMantissaControl()
cusolverDnGetFixedPointEmulationMantissaControl()
cusolverDnSetFixedPointEmulationMaxMantissaBitCount()
cusolverDnGetFixedPointEmulationMaxMantissaBitCount()
cusolverDnSetFixedPointEmulationMantissaBitOffset()
cusolverDnGetFixedPointEmulationMantissaBitOffset()

// Special value handling (NaN, Inf, denormals)
cusolverDnSetEmulationSpecialValuesSupport()
cusolverDnGetEmulationSpecialValuesSupport()
```

These provide fine-grained control over the emulation quality/speed tradeoff.
For LU factorization (which needs pivoting with full precision), the mantissa
bit count control is critical -- more bits = more accuracy but slower emulation.

**Also new:** `cusolverDnXsygvd` API for generalized symmetric eigenvalue problems
with larger problem sizes.

---

## 6. CUTLASS 4.4.0 through 4.4.2 (Feb-Mar 2026) -- Newer Than Existing Brief

The existing CUTLASS brief covers 4.2/4.3. Three new releases since:

### CUTLASS 4.4.0 (2026-02-14)
- **CUDA 13.1 support** -- now fully compatible with CTK 13.1
- **GB300 GPU support** (datacenter Blackwell Ultra -- not us, but shows maturity)
- **Ada FP8xFP8 GEMM with blockwise dequantization** -- this is relevant as
  reference implementation. Ada sm_89 is closest ISA to our sm_120.
- **cute.experimental module** -- fragment-free programming model for CuTe DSL
- **Ahead-of-Time (AoT) compilation** -- could speed up our CUTLASS reference builds
- **E2M1 to FP32 optimized conversion** -- for FP4/MXFP4 paths
- **TF32 support for mixed-precision** -- relevant for Cholesky/LU worker's TF32 investigations
- **SM103 batched FP4 blockscaled GEMM** -- not us (sm_103) but shows direction

### CUTLASS 4.4.1 (2026-02-27)
- Fixed segfault with tvm-ffi on aarch64 (not relevant to us)

### CUTLASS 4.4.2 (2026-03-13) -- JUST RELEASED
- **Blackwell SM120f compilation enabled for examples** -- this is important!
  The `f` suffix enables family-specific features. sm_120f may unlock features
  not available with plain sm_120 or sm_120a.
- **NVFP4/MX Grouped GEMM exposed in CUTLASS Profiler** -- benchmarking tool
- **Fixed Hopper FMHA causal attention performance regression** -- not us but shows
  CUTLASS is actively maintaining attention kernel quality
- **Python 3.14 support** for CuTe DSL

**Key takeaway:** The sm_120f compilation target in CUTLASS 4.4.2 is notable.
Our workers should check whether `-arch=compute_120f -code=sm_120f` unlocks
any additional features vs plain sm_120a. The PTX ISA 8.8 release notes mention
sm_120f as a "family-specific" target.

---

## 7. Shared Memory Carveout Sizes on sm_120 (Blackwell Tuning Guide 13.2)

The updated tuning guide for 13.2 clarifies shared memory carveout options:

**Per-SM capacity:** 128 KB (for compute capability 12.0 / sm_120)

**Selectable carveouts via `cudaFuncSetAttribute`:**
```
0, 8, 16, 32, 64, 100, 132, 164, 196, 228 KB
```

**Wait -- 228 KB?** The tuning guide lists carveout sizes up to 228 KB. However,
the per-SM capacity for sm_120 is stated as 128 KB, and per-block max is 99 KB.
The 132-228 KB values are for sm_100 (datacenter Blackwell) which has 228 KB per SM.

**Confirmed for sm_120:** 128 KB per SM, max 99 KB per block (after 1 KB reservation
per block). This is unchanged from what we already know.

**NOTE:** The old hard-won lesson says "99 KB max shared memory per block." This
remains correct for sm_120. The higher carveout sizes are for datacenter Blackwell.

---

## 8. NVIDIA Driver Status: R595 Series Current

The latest driver branch is R595 (specifically 595.71 as of March 2026).

**Key changes in R595 for CUDA/compute:**
- Vulkan compute performance improvements on RTX 5090
- DRI3 v1.2 support with DMA fences (Linux)
- Fixed GPU hang/Xid error bug
- No known CUDA-specific regressions

**Compatibility:** CUDA 13.2 requires minimum driver R560. R595 is fully compatible.
Our current driver should be checked -- if we're on an older branch, updating to
R595 could improve stability.

---

## 9. CUDA 13.2 Math Library: expm1f and erff Faster

**expm1f():** up to 20% faster with minor accuracy improvements.
**erff():** 5-10% faster with minor accuracy improvements.

These are from algorithmic simplifications in the device math library.

**Impact:** Minimal for our current workers. The attention kernel uses `exp2f` (via
MUFU.EX2), not `expm1f`. RMSNorm uses `rsqrtf`. These improvements would matter
for kernels doing error function or expm1 calculations (e.g., GELU approximation).

---

## 10. CUDA 13.2: Host Task Spin-Wait Dispatch

New spin-wait dispatch mode for `cudaLaunchHostFunc()` and graph host nodes.
Reduces execution latency by busy-waiting instead of blocking on GPU interrupt.

**Impact:** Could reduce kernel dispatch overhead for Python-launched kernels.
However, our workers already measure kernel-to-kernel throughput (not dispatch
latency), so impact is minimal unless dispatch overhead is the bottleneck.

---

## 11. CUDA 13.2: New Memory APIs

- `cudaMemcpyWithAttributesAsync` -- single-transfer attribute control
- `cudaMemPoolGetAttribute` -- query pool properties
- Polymorphic `cudaGraphNodeGetParams`

**Impact:** Minimal for kernel optimization work. These are host-side API improvements.

---

## 12. Architecture Deprecations and Removals

**Removed in CUDA 13.0+:**
- Maxwell (sm_50/52/53) -- offline compilation and libraries removed
- Pascal (sm_60/61/62) -- removed
- Volta (sm_70/72) -- removed

**Deprecated (removal planned for CUDA 14.0):**
- Legacy vector types: `double4`, `long4`, `ulong4`, `longlong4`, `ulonglong4`
  (use `_32a` variants instead)

**Other removals:**
- Multi-device cooperative launch APIs
- Various `cudaDeviceProperties` fields (clockRate, computeMode, etc.)
- Legacy surface/texture headers

**Impact:** We target sm_120 only. No impact from architecture removals. The
vector type deprecations are a heads-up for code hygiene.

---

## Summary: What Matters Most for Each Worker

### GEMM Worker (gemm/)
- **cuBLAS 13.2 bug fixes** -- cublasLtMatmul concurrent correctness fix could affect benchmark stability
- **cuBLAS L3 improvements** -- re-baseline after any upgrade
- **CUTLASS 4.4.2 sm_120f** -- investigate new compilation target

### Attention Worker (attention/)
- **cuBLAS bug fixes** -- if using cublasLtMatmul references
- No new instructions or features that change the attention optimization landscape

### Linalg Worker (linalg/)
- **cuBLAS L3 SYRK/TRMM improvements in 13.2** -- HIGH IMPACT, reference bars may move
- **cuBLAS FP32 emulation** -- could transparently accelerate FP32 GEMM calls
- Must re-baseline SYRK (0.96x) and TRMM (1.02x) after upgrading

### LU Worker (lu/)
- **cuSOLVER 13.2 FP64 emulation APIs** -- reference cuSOLVER may get faster with emulation
- **cuBLAS FP32 emulation** -- cuBLAS TRSM/GEMM calls in blocked LU may be faster
- Should re-baseline cuSOLVER sgetrf after upgrading

### Numerical/Cholesky Worker
- **cuSOLVER emulation** -- same as LU
- **TF32 in CUTLASS 4.4.0** -- new mixed-precision TF32 support could inform TF32 MMA work

### RMSNorm Worker (rmsnorm/)
- **expm1f/erff faster** -- not directly relevant (uses rsqrtf)
- No new features that change the optimization landscape

### SpMV Worker (spmv/)
- **cuSPARSE 13.2 SpMVOp buffer_size fix** -- fixes 16x over-allocation bug
- **BSR format in SpMV** -- new format support in generic API

### Fused-MLP Worker (fused-mlp/)
- **cuBLAS concurrent correctness fix** -- if using cuBLAS for GEMM2
- **cuBLAS FP32 emulation** -- could change reference performance

---

## Recommended Upgrade Path

1. **Check current CUDA version and driver version** on the build machine
2. **If upgrading to CUDA 13.2:** Re-run ALL worker baselines (cuBLAS, cuSOLVER, cuSPARSE references) since library performance may have changed
3. **Investigate `sm_120f` compilation target** -- CUTLASS 4.4.2 enables it, may unlock new features
4. **Update NVIDIA driver to R595** for stability improvements
