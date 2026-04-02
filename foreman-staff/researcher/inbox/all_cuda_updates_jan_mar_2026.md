# NVIDIA CUDA Ecosystem Updates: January - March 2026

**Date:** 2026-03-14
**Relevant to:** All workers (attention, GEMM, fused-mlp, numerical, linalg, rmsnorm, dotproduct)
**Scope:** New releases since CUDA 13.0 / PTX ISA 8.8 / CUTLASS 4.2 / cuSolverDx 0.2

---

## 1. CUDA Toolkit Releases

### CUDA 13.1 (January 11, 2026)

The largest CUDA platform update in two decades. Key items for our work:

**CUDA Tile / cuTile Python:**
- New tile-based programming model with a virtual ISA (CUDA Tile IR) that abstracts tensor cores, memory movement, and thread mapping.
- cuTile Python DSL lets you write ~30-line matrix multiply kernels that auto-leverage tensor cores. Achieves >90% of cuBLAS performance.
- Initial support: Blackwell only (sm_100, sm_120). C++ API planned for future.
- **Relevance:** Not directly useful for our hand-tuned PTX kernels, but shows NVIDIA's direction. cuTile-generated code could become a useful baseline to measure against.
- **Source:** [NVIDIA Blog](https://developer.nvidia.com/blog/nvidia-cuda-13-1-powers-next-gen-gpu-programming-with-nvidia-cuda-tile-and-performance-gains/)

**Green Contexts (Runtime API):**
- Fine-grained GPU spatial partitioning by SM count. `split()` API for deterministic SM allocation.
- Static SM partitioning in MPS with `-S` flag.
- **Relevance:** Could enable isolating benchmark runs from ComfyUI on GPU 0 more cleanly, though we already use CUDA_VISIBLE_DEVICES.

**cuBLAS 13.1:**
- Experimental Grouped GEMM supporting FP8 and BF16/FP16 on Blackwell.
- Up to 4x speedup over multi-stream GEMM in MoE use cases.
- Device-side shapes + CUDA Graphs support.
- **Relevance:** Reference point for GEMM worker. Grouped GEMM is relevant if we add MoE kernels.

**cuSOLVER 13.1:**
- Batched ops ~2x faster on RTX PRO 6000 Blackwell vs L40S for SYEV (eigenvalue).
- GEEV up to 1.7x improvement for large matrices (32K+ rows).
- **Relevance:** Baseline comparison for numerical worker's eigenvalue kernels.

**cuSPARSE 13.1:**
- New SpMVOp API for sparse matrix-vector multiply. User-defined epilogues. CSR format.
- **Relevance:** Direct competition for SpMV kernels in linalg/numerical pipeline.

**Nsight Compute 2025.4:**
- Tile kernel profiling with "Tile Statistics" sections.
- Source-page metric mapping to cuTile kernels.
- **Relevance:** Useful if we ever adopt cuTile for prototyping.

**Compute Sanitizer 2025.4:**
- Compile-time patching via `-fdevice-sanitize=memcheck`. Memory error detection integrated into compilation.
- **Relevance:** Could help debug shared memory issues in our kernels.

**CCCL 3.1:**
- Deterministic floating-point reductions (run-to-run bitwise identical option).
- Single-phase CUB APIs eliminating two-phase temp storage pattern.
- **Relevance:** Useful for reduction-heavy kernels (rmsnorm, dotproduct).

### CUDA 13.2 (March 5, 2026)

**cuTile Python expanded to Ampere + Ada:**
- CUDA Tile now works on sm_80+ (Ampere, Ada, Blackwell). Was Blackwell-only in 13.1.
- New Python features: recursive functions, closures, custom reduction/scan, type annotations.
- pip installable: `pip install cuda-tile[tileiras]`

**cuBLAS 13.2:**
- Experimental Grouped GEMM now supports **MXFP8** on Blackwell (sm_100/sm_110). Prior 13.1 had FP8/BF16 only.
- Up to 20% speedup for FP8, FP16/BF16, TF32, INT8 on RTX PRO 6000.
- Up to 3x perf gain for MXFP8/NVFP4 on DGX Spark.
- FP64 fixed-point emulation (Ozaki scheme) now supports SYRK and HERK.
- **Relevance:** The 20% FP8 speedup on RTX PRO 6000 suggests library-level improvements that may raise the bar our GEMM worker competes against. MXFP8 grouped GEMM is a new reference for MoE.
- **Source:** [cuBLAS 13.2 PDF](https://docs.nvidia.com/cuda/pdf/CUBLAS_Library.pdf)

**cuSOLVER 13.2 -- FP64 Emulation for Factorizations (NEW):**
- cuSOLVERDn now has FP64-emulated APIs for **QR (DGEQRF), LU (DGETRF), Cholesky (DPOTRF)**.
- Uses INT8 tensor cores via Ozaki scheme to emulate FP64. Automatic dynamic precision (ADP) framework selects emulation vs native based on input analysis.
- Performance: **up to 2x speedup for QR** at 80K matrix sizes on B200.
- New APIs for mantissa control and special values handling.
- **Relevance:** HIGHLY relevant to numerical worker. This is cuSOLVER doing exactly what we're trying to do -- device-side factorization with tensor cores. The 2x QR speedup is the new bar. Note: demonstrated on B200, unclear if sm_120 gets the same benefit (INT8:FP64 ratio may differ).
- **Source:** [NVIDIA Blog](https://developer.nvidia.com/blog/cuda-13-2-introduces-enhanced-cuda-tile-support-and-new-python-features/)

**Memory Transfer APIs:**
- `cudaMemcpyWithAttributesAsync` and `cudaMemcpy3DWithAttributesAsync` -- attribute-based transfers without batching overhead.
- **Relevance:** Minor. Could help with memory transfer optimization in host code.

**Math Library (libm):**
- `expm1f()` up to 20% faster.
- `erff()` 5-10% faster.
- **Relevance:** Minor. Useful if any kernel uses these functions.

**CCCL 3.2:**
- Modern C++ runtime: `cuda::stream`, `cuda::event`, `cuda::buffer`, `cuda::launch`.
- `cub::DeviceTopK`: up to 5x speedup over radix sort for small K.
- `cub::DeviceSegmentedReduce` (fixed-size): up to 66x speedup for small segments.
- `cub::DeviceSegmentedScan` and binary search primitives.
- **Relevance:** DeviceTopK could be useful for sparse operations. Segmented reduce relevant to reduction kernels.

**Nsight Compute 2026.1:**
- Report clustering/merging tool.
- **Register dependency visualization** -- shows register-level data dependencies.
- Improved CUDA Graphs viewer.
- **Relevance:** Register dependency visualization is directly useful for our PTX-level optimization work.

**CUDA Python:**
- `cuda.core 0.6` with NVML/nvFatbin bindings.
- CUDA Graphs support with conditional execution and fork-join patterns.
- Stream Protocol for zero-copy interop with PyTorch/JAX.

**Deprecated/Removed:**
- Maxwell, Pascal, Volta removed from offline compilation.
- Legacy vector types (`double4`, `long4`, etc.) deprecated, removal in CUDA 14.0.
- Ubuntu 20.04 dropped.

---

## 2. PTX ISA Updates

### PTX ISA 9.0 (ships with CUDA 13.0)
- Shipped with CUDA 13.0. Introduced family-specific targets (`sm_120f`).
- Extended `min` and `max` to support three input arguments.
- Extended `tcgen05.mma` with new scale_vectorsize qualifiers `.block16` and `.block32` and K dimension 96.
- Extended `.field3` of `tensormap.replace` to support 96B swizzle mode.
- Added `tcgen05.ld.red` instruction.
- Extended `ld`, `ld.global.nc`, and `st` to support **256b (256-bit) load/store operations**.
- Added special registers: `%reserved_smem_offset_begin`, `%reserved_smem_offset_end`, `%reserved_smem_offset_cap`.
- **Relevance:** 256-bit loads are interesting but need to verify sm_120 support (may be sm_100-only). The tcgen05 extensions are datacenter-only (not sm_120).

### PTX ISA 9.1 (ships with CUDA 13.1)
- Extended mma instructions to support `.f16` type accumulator and shape `.m16n8k16` with FP8 types `.e4m3` and `.e5m2`.
- Added support for types `.e3m2/.e2m3/.e2m1` and qualifiers `.kind`, `.block_scale`, `.scale_vec_size` on MMA and `mma.sp::ordered_metadata`.
- **Relevance:** **The `.f16` accumulator for m16n8k16 FP8** is potentially useful -- allows FP8 MMA with FP16 accumulation instead of FP32, which could reduce register pressure in attention/GEMM kernels at the cost of precision. The new microscaling types (e3m2, e2m3, e2m1) open up sub-byte quantization paths.

### PTX ISA 9.2 (ships with CUDA 13.2)
- Support for `.u8x4` and `.s8x4` types on `add`, `sub`, `min`, `max`, `neg` -- packed 8-bit integer SIMD.
- `add.sat.{u16x2/s16x2/u32}` -- saturating addition for packed types.
- `.b128` type for `st.async` -- 128-bit async store.
- `.ignore_oob` qualifier for `cp.async.bulk` -- ignore out-of-bounds in bulk async copies.
- `.bf16x2` destination type for `cvt` from various FP8 types (`.e4m3x2`, `.e5m2x2`, `.e3m2x2`, `.e2m3x2`, `.e2m1x2`).
- **Relevance:** The packed u8x4/s8x4 SIMD ops could be useful for quantization kernels. The `.bf16x2` conversion from FP8 pairs is directly relevant -- could simplify FP8-to-BF16 conversion in attention/GEMM pipelines. The `.ignore_oob` qualifier for `cp.async.bulk` could simplify boundary handling.

---

## 3. CUTLASS Updates

### CUTLASS 4.3.0 (November 21, 2025)
- **SM120 mixed input blockscaled grouped GEMM** kernels.
- SM100 and SM120 blockscaled sparse kernel profiler support.
- New MoE grouped GEMM API.
- SM100 convolution stream-K kernel.
- SM100 sparse GEMM compressor with sub-byte and runtime datatype support.

### CUTLASS 4.3.1 (November 26, 2025)
- Blockscaled variant of ragged contiguous grouped GEMM with simplified MoE API (example 92).
- Compatible with all microscaling types.

### CUTLASS 4.3.5 (January 9, 2026)
- Bug fixes and improvements to CuTe DSL.
- Fix for missing SMEM alignment in Blackwell SM120 scale factors.

**Known Issue:** Python DSL `BlockScaledMmaOp` restricts FP4 ops to `sm_100a` only, blocking sm_120/sm_121. C++ API supports it but Python DSL does not yet.

**Relevance:** The SM120 blockscaled grouped GEMM and sparse kernel support shows NVIDIA is actively targeting consumer Blackwell in CUTLASS. The SMEM alignment fix in 4.3.5 could indicate subtle shared memory layout requirements we should check in our own kernels.

---

## 4. MathDx / Device Extensions

### MathDx 25.12.0 / 25.12.1 (Latest as of March 2026)

**cuBLASDx 0.5.0 (NEW since our last check at 0.3.0):**
- **Experimental Pipelining API** for fusable asynchronous execution from global memory.
- Initial support for **WGMMA, 1SM UTCMMA, and TMA** (datacenter Blackwell features, likely NOT sm_120).
- SM100, SM101, SM120 support (added in 0.4.0).
- SM103 and SM121 experimental support.
- PTX 8.7 superMMA and `fma.f32x2` instruction support.
- New accumulator API for fragment operations.
- Wider suggested layouts with analytical swizzling heuristics.
- **Relevance:** cuBLASDx 0.4+ supports sm_120 device-side GEMM. The pipelining API in 0.5 is interesting for fused kernels. The WGMMA/UTCMMA/TMA features are likely sm_100-only. The analytical swizzling heuristics might provide ideas for our own swizzle patterns.

**cuBLASDx 0.5.1:**
- Refined heuristics for Ozaki (FP64) emulation.
- Performance example operators.

**cuSolverDx 0.3.0 (NEW since our last check at 0.2.0):**
- **SVD for bidiagonal and general matrices** (batched and non-batched).
- **Eigenvalue solver for symmetric/Hermitian tridiagonal and general matrices**.
- General tridiagonal linear system solver (GTSV_NO_PIVOT).
- Matrix generation from QR and LQ factorizations (UNGQR, UNGLQ).
- Performance enhancements on Hopper.
- SM120 supported (since 0.2.0).
- **Known issue:** Potential kernel miscompilation with GESV_NO_PIVOT on SM120 with real types in CUDA 12.8-13.0. Workaround: define macro or use PTX optimization flags.
- **Relevance:** HIGHLY relevant to numerical worker. Device-side SVD and eigenvalue solvers on sm_120 are now available as both reference implementations and potential building blocks. The GTSV_NO_PIVOT is useful for tridiagonal systems.

**cuFFTDx 1.6.0:**
- Experimental cuFFT LTO integration.
- Partial BlockDim operator support.
- SM120 support (since 1.5.0).
- **Relevance:** Relevant if/when FFT project starts. Device-side FFT on sm_120 is available.

---

## 5. FP64 Emulation via Tensor Cores (Ozaki Scheme)

This is a significant development across cuBLAS and cuSOLVER:

**How it works:** Decomposes FP64 operations into fixed-point representations, then uses INT8 tensor cores on Blackwell to perform the computation. An Automatic Dynamic Precision (ADP) framework analyzes inputs to determine if emulation is safe and beneficial.

**Performance:**
- RTX PRO 6000 Blackwell: up to **13x speedup** for FP64 DGEMM.
- GB200 NVL72: up to 2.3x for FP64 DGEMM.
- cuSOLVER 13.2: up to 2x for QR factorization at 80K matrix sizes (B200).
- Real-world: BerkeleyGW ~1.5x, Quantum Espresso ~3x.

**Current coverage:**
- cuBLAS: DGEMM, ZGEMM, SYRK, HERK.
- cuSOLVER: DGEQRF (QR), DGETRF (LU), DPOTRF (Cholesky).
- More BLAS-3 and LAPACK routines planned.

**sm_120 status:** Not explicitly confirmed for consumer Blackwell. The INT8:FP64 throughput ratio on sm_120 should still favor emulation given the RTX 5090's strong INT8 tensor core throughput and limited FP64 hardware.

**Relevance:** Sets a new performance bar for our numerical kernels. If cuSOLVER can do 2x on QR via emulation, we need to match or beat that. Also validates the approach of using tensor cores for FP64 work via decomposition.

**Sources:**
- [NVIDIA Blog: FP Emulation in cuBLAS](https://developer.nvidia.com/blog/unlocking-tensor-core-performance-with-floating-point-emulation-in-cublas/)
- [Ozaki Scheme Paper](https://arxiv.org/abs/2508.00441)

---

## 6. External: Flash Attention for RTX 5090

A notable open-source effort by gau-nernst implementing Flash Attention 2 for RTX 5090 in CUDA C++:

**Architecture:** Uses `mma.sync.m16n8k16` (BF16), `cp.async` double-buffer, `ldmatrix.x2`/`.x4`, XOR swizzle -- essentially the same building blocks we use.

**Performance progression:**
- v1: 142.87 TFLOPS (bank conflicts)
- v2: 181.11 TFLOPS (added swizzle + 2-stage pipeline)
- v3: 189.84 TFLOPS (consolidated ldmatrix)
- v4: 194.33 TFLOPS (reduced instruction count)
- v5: 197.74 TFLOPS (**94.4% of SOL**, theoretical max 209.5 TFLOPS)
- PyTorch SDPA: 186.73 TFLOPS
- CuDNN: 203.61 TFLOPS

**Key insight:** On sm_120, **instruction scheduling becomes the bottleneck**, not arithmetic or memory. Reducing `ldmatrix` count by 2x gave measurable speedup even when memory bandwidth wasn't saturated. The MMA output layout for m16n8k16 exactly matches multiplicand A layout, eliminating cross-thread shuffles.

**Config:** BLOCK_Q=128, BLOCK_KV=64, DIM=128, 4 warps (128 threads), warp_q=32.

**Relevance:** HIGHLY relevant to attention worker. This is an independent validation of our approach reaching 94.4% SOL. Their v5 at 197.7 TFLOPS vs our 1.76x SDPA (which is ~329 TFLOPS at our config) shows we're in a similar regime. Their observation that instruction count dominates on sm_120 matches our finding that the kernel is latency-bound.

**Source:** [gau-nernst blog](https://gau-nernst.github.io/fa-5090/)

---

## 7. GTC 2026 (March 16-19, 2026 -- THIS WEEK)

GTC 2026 starts in 2 days. Expected announcements:
- **Vera Rubin GPU architecture** formal debut (next gen after Blackwell).
- Rubin specs: up to 288 GB HBM4, 22 TB/s bandwidth, 35-50 PFLOPS NVFP4.
- Feynman architecture also on roadmap (after Rubin).
- Updated SDKs and frameworks targeting Rubin-class hardware.
- Physical AI and Digital Twins emphasis.
- CUDA library updates likely.

**Relevance:** No immediate impact on our sm_120 work. But GTC sessions may reveal new CUDA features, compiler improvements, or tuning guidance for Blackwell. Worth monitoring session recordings after the event.

**Source:** [The Register GTC 2026 Preview](https://www.theregister.com/2026/03/13/nvidia_gtc_2026_preview_tobias_mann_register/)

---

## 8. Blackwell Tuning Guide (13.2)

The tuning guide has been updated but contains **no new sm_120-specific guidance** beyond what was in 13.0. Key specs remain:
- 48 warps/SM (vs 64 for sm_100)
- 128 KB shared/SM, 99 KB max per block
- 64K registers/SM, 255 max per thread
- 32 thread blocks max per SM

No differential guidance between sm_100 and sm_120 beyond the warp count and the absence of tcgen05/TMEM/TMA.

---

## Summary: What's Actionable

| Finding | Priority | Relevant Worker(s) |
|---------|----------|-------------------|
| cuSOLVER 13.2 FP64-emulated QR/LU/Cholesky via INT8 tensor cores | HIGH | numerical |
| cuSolverDx 0.3.0: device-side SVD + eigenvalue solvers on sm_120 | HIGH | numerical |
| PTX ISA 9.1: FP16 accumulator for m16n8k16 FP8 MMA | HIGH | attention, GEMM |
| PTX ISA 9.2: bf16x2 conversion from FP8 pairs | MEDIUM | attention, GEMM |
| cuBLAS 13.2: 20% FP8/BF16 speedup on RTX PRO 6000 (new baseline) | MEDIUM | GEMM |
| gau-nernst FA for 5090: 94.4% SOL, instruction scheduling bottleneck | MEDIUM | attention |
| cuBLASDx 0.5: pipelining API, sm_120 support | MEDIUM | fused-mlp, numerical |
| Nsight Compute 2026.1: register dependency visualization | MEDIUM | all |
| CUTLASS 4.3: sm_120 blockscaled grouped GEMM + sparse | LOW | GEMM (future) |
| CCCL 3.2: DeviceTopK, segmented reduce/scan | LOW | linalg, numerical |
| PTX ISA 9.2: packed u8x4/s8x4 SIMD, cp.async.bulk ignore_oob | LOW | all |
| GTC 2026 starts March 16 | WATCH | all |
