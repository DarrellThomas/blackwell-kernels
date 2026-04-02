# March 1-14, 2026: New Developments Scan

**Date:** 2026-03-14
**Relevant to:** All workers (attention, gemm, fused-mlp, numerical, linalg, rmsnorm, spmv)

---

## 1. CUDA 13.2 Released (March 5, 2026)

**Source:** [CUDA 13.2 Release Notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html) | [NVIDIA Blog](https://developer.nvidia.com/blog/cuda-13-2-introduces-enhanced-cuda-tile-support-and-new-python-features/)

**sm_120 relevant: YES**

CUDA 13.2 dropped on March 5. Key changes:

- **PTX ISA 9.2** is the new PTX version (see item #2 below).
- **CUDA Tile expanded to sm_80+**: CUDA Tile / cuTile DSL now supports compute capability 8.x (Ampere, Ada) in addition to 10.x, 11.x, and 12.x (Blackwell). This is the tile-based programming model that abstracts tensor core usage.
- **Host task spin-wait dispatch mode**: New dispatch mode to reduce execution latency for host tasks.
- **Compiler updates**: Support for Visual Studio 2026, ARM C Language Extension support for gcc, unified toolkit for Tegra and Desktop GPUs.
- **C++20 standards conformance improvements** in NVCC.

### cuBLAS 13.2 (March 5, 2026)

**Source:** [cuBLAS 13.2 Documentation](https://docs.nvidia.com/cuda/cublas/)

- **MXFP8 Grouped GEMM**: Extended experimental Grouped GEMM API to support MXFP8 inputs on Compute Capability 10.x and 11.0.
- **RTX PRO 6000 improvements**: Up to 20% speedup for FP8, FP16/BF16, TF32, and INT8 precisions. (RTX PRO 6000 is sm_120, same arch as RTX 5090 -- this likely benefits us too.)
- **Bug fix**: Fixed Grouped GEMM API bug that ignored groups with k=0 (existed since CUDA 13.1).
- **FP16 accumulator caveat**: FP8 and FP16 matmuls run at full speed with FP16 accumulation but only half speed with FP32 accumulation on consumer Blackwell. This is a known hardware limitation.

### cuSOLVER 13.2

**Source:** [cuSOLVER 13.2 Documentation](https://docs.nvidia.com/cuda/cusolver/index.html)

- **FP64-emulated factorization APIs (cuSOLVERD)**: New APIs that use INT8 tensor cores to emulate FP64 calculations. Significant performance gains for QR, LU, and Cholesky factorizations on platforms with high INT8-to-FP64 throughput ratio.
- **Batched eigenvalue solver improvements**: New internal algorithm switch on Blackwell GPUs for matrices n <= 32 in cusolverDnXsyevBatched(). Can revert via cusolverDnSetAdvOptions().
- **cusolverDnXsygvd**: New API for larger problem sizes.

### cuSPARSE 13.2

**Source:** [cuSPARSE 13.2 Documentation](https://docs.nvidia.com/cuda/cusparse/)

- **New cusparseSpMVOp_bufferSize API**: Returns workspace buffer size for SpMVOp, removing internal memory allocations.
- **SpMVOp performance improvements** on B200.
- **64-bit index support** in SpGEMM computation.
- **Bug fix**: cusparseSparseToDense_bufferSize was requesting up to 16x more memory than required. Fixed.
- **Zero/small dimension support**: All generic APIs now support zero-dimension and small-dimension matrices/vectors.
- **Mixed-precision CSR/COO SpMV fix**: Corrected incorrect results in mixed-precision computations.

### cuFFT 13.2

- **Known bug**: Half-precision and BF16 size-1 strided R2C and C2R transforms produce incorrect results. Fix pending.
- Removed `_nocallback` static library (now merged into main static lib).

---

## 2. PTX ISA 9.2 (ships with CUDA 13.2)

**Source:** [PTX ISA 9.2 Documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)

**sm_120 relevant: YES**

Key new instructions and extensions:

- **mma instruction extended**: Now supports `.f16` type accumulator with shape `.m16n8k16` for FP8 types `.e4m3` and `.e5m2`. This is directly relevant to our attention and GEMM workers -- FP8 MMA with FP16 accumulation (which runs at full speed on sm_120, unlike FP32 accumulation).
- **New MX types**: `mma` and `mma.sp::ordered_metadata` extended with types `.e3m2`, `.e2m3`, `.e2m1` and qualifiers `.kind`, `.block_scale`, `.scale_vec_size`. These are microscaling (MX) format types for block-scaled operations.
- **256-bit load/store**: `ld`, `ld.global.nc`, and `st` extended to support 256b (32-byte) load/store operations.
- **3-input min/max**: `min` and `max` instructions extended to support three input arguments (useful for clamping).
- **sm_120f and sm_121f**: New family-specific target architecture support added.
- **tcgen05.mma extensions**: New `.block16` and `.block32` scale_vectorsize qualifiers and K dimension 96. (These are for datacenter Blackwell, NOT sm_120.)

### Critical finding for workers:
The FP8 MMA with FP16 accumulator is the key item. On sm_120, FP32 accumulation runs at half speed. If PTX ISA 9.2 enables `.f16` accumulator for FP8 MMA shapes, this could unlock full-speed FP8 tensor core throughput. Workers need to check if `mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16` is now valid on sm_120.

---

## 3. CUTLASS 4.3.3 Released

**Source:** [CUTLASS 4.3.3 Release](https://github.com/NVIDIA/cutlass/releases/tag/v4.3.3) | [Changelog](https://docs.nvidia.com/cutlass/4.3.2/CHANGELOG.html)

**sm_120 relevant: YES**

- **SM120 blockwise GEMM kernel added** (Example 87). This is a new reference implementation for blockwise GEMM specifically targeting consumer Blackwell.
- **SM120 mixed-input blockscaled grouped GEMM** support added.
- **SM121 (DGX Spark)** kernels share code with SM120.
- **Blockwise and groupwise GEMM enhancements** for both Hopper and Blackwell.
- **K major scale factor support** for SM90 blockwise kernels.
- **Relaxed k-dimension constraint**: k dimension no longer needs to be a multiple of tile k dimension.

### Known bug (CUTLASS Python DSL):
BlockScaledMmaOp restricts FP4 operations to sm_100a only, blocking sm_120/sm_121. The `admissible_archs` needs updating to include `Arch.sm_120a` and `Arch.sm_121a`. Filed as [Issue #2800](https://github.com/NVIDIA/cutlass/issues/2800).

---

## 4. NVIDIA GTC 2026 (March 16-19, San Jose)

**Source:** [GTC 2026](https://www.nvidia.com/gtc/) | [Session Catalog](https://www.nvidia.com/gtc/session-catalog/)

**sm_120 relevant: POTENTIALLY**

GTC 2026 starts March 16 (two days from now). Confirmed sessions include:
- An architecture talk by a CUDA architect covering what's new and coming next for CUDA and GPU computing.
- 700+ technical sessions across accelerated computing topics.
- Jensen Huang keynote focusing on AI inference.

The session catalog is live. Worth monitoring for:
- CUDA Tile / cuTile deep dives (new programming model)
- Blackwell kernel optimization sessions
- cuBLAS / cuSOLVER performance talks
- Any sm_120 specific content

Note: NVIDIA has stated no new gaming GPU releases in 2026; focus is on datacenter. But GTC engineering sessions often contain insights applicable to consumer Blackwell.

---

## 5. PyTorch 2.11 (Upcoming)

**Source:** [PyTorch Dev Discussion](https://dev-discuss.pytorch.org/t/transitioning-pypi-cuda-wheels-to-cuda-13-0-as-the-stable-release-2-11/3325)

**sm_120 relevant: YES**

- **CUDA 13.0 becomes stable variant**: PyPI wheels for PyTorch 2.11 will default to CUDA 13.0 (both x86_64 and aarch64). This is the first PyTorch release where CUDA 13.0 is the "stable" variant on PyPI.
- **sm_120 support**: There are ongoing issues ([pytorch#159207](https://github.com/pytorch/pytorch/issues/159207)) for official sm_120 support. The `sm_120a` suffix is needed for accelerated features (block-scaled MMA, NVFP4), but PyTorch's auto-detection sometimes generates `sm_120` without the `a` suffix, breaking these features ([pytorch#172807](https://github.com/pytorch/pytorch/issues/172807)).
- **FP8 kernel improvements**: CUTLASS backend supporting FP8 mm operations achieved up to 16% improvement over Triton/cuBLAS on production workloads.

---

## 6. NVIDIA Driver 595 Series

**Source:** [Phoronix Review](https://www.phoronix.com/review/nvidia-595-linux)

**sm_120 relevant: YES (but cautionary)**

- **Driver 595.45.04** (Linux beta): First public R595 branch build. Incremental performance improvements over R590 series on RTX 50 Blackwell in both graphics and compute benchmarks.
- **Driver 595.71** (Windows): Released to fix issues from 595.59. However, introduced a voltage restriction bug on RTX 50-series cards -- lower voltages at both stock and overclocked frequencies. NVIDIA has not officially acknowledged this.
- **Recommendation**: On our Linux compute setup, the R595 driver branch shows modest compute gains. Worth considering upgrading from current driver, but test carefully. The voltage issue appears Windows-specific.

---

## 7. Nsight Compute 2026.1

**Source:** [Nsight Compute 2026.1 Features](https://developer.nvidia.com/nsight-compute-2026_1-new-features)

**sm_120 relevant: YES**

New features in Nsight Compute 2026.1 (ships with CUDA 13.2):

- **Register Dependencies analysis**: New Source page feature to identify general purpose register dependencies and occupancy issues from live register pressure. Shows attributed live registers and output registers per source line.
- **CUDA Graph Viewer**: Dynamic visualization of CUDA Graphs during interactive profiling.
- **Report Merge Tool**: Combine multiple profiling reports into one -- useful for comparing different kernel configurations.
- **Clustering Window**: Groups similar profiling reports together to identify performance patterns.
- **Improved timeline overlays**: Related metrics shown in single row with max bar backgrounds.

The register dependency analysis is particularly valuable for our workers who are constantly tuning register usage vs occupancy tradeoffs.

---

## 8. NVFP4 on sm_120

**Source:** [vLLM Issue #31085](https://github.com/vllm-project/vllm/issues/31085) | [NVIDIA Blog](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)

**sm_120 relevant: YES (but limited applicability for our kernels)**

- sm_120 hardware supports NVFP4 natively (E2M1 format, 4 bits: 1 sign, 2 exponent, 1 mantissa).
- Groups of 16 FP4 values share an FP8 (E4M3) scale factor, with a global FP32 scale.
- **3.5x memory reduction** vs FP16, 1.8x vs FP8, with <1% accuracy degradation on some LLM tasks.
- Software ecosystem is catching up: vLLM 0.13.0+ has compiled sm_120 NVFP4 kernels, but backend selection logic still has gaps (only checks sm_90 and sm_100 family, not sm_120 family).
- **PTX ISA 9.2** adds `mma` support for `.e2m1` type with `.block_scale` qualifier -- this is the hardware path for NVFP4 on sm_120.

For our kernel work, NVFP4 is potentially relevant for future GEMM exploration (even lower precision than FP8), but not an immediate priority given current FP8 focus.

---

## 9. Flash Attention 4 Paper (February 2026)

**Source:** [FlashAttention-4 Paper](https://arxiv.org/html/2603.05451) | [Together AI Blog](https://www.together.ai/blog/flashattention-4)

**sm_120 relevant: NO (datacenter Blackwell only, uses tcgen05)**

FA4 is optimized for sm_100 (datacenter Blackwell B200), achieving ~20% speedup over cuDNN attention kernels. Uses `tcgen05.mma.cta_group::1` (5th gen tensor cores). This is NOT applicable to sm_120, which uses `mma.sync` (4th gen-style tensor cores).

However, the **algorithmic innovations** in the paper may be portable:
- Algorithm/kernel pipelining co-design for asymmetric hardware scaling
- Techniques for overlapping softmax with MMA
- The observation that tensor core throughput doubles while exponential units don't scale -- this exactly matches our attention worker's bottleneck

---

## 10. Sigmoid Attention (FlashSigmoid)

**Source:** [OpenReview Paper](https://openreview.net/forum?id=Zhdhg6n2OG)

**sm_120 relevant: YES (algorithmic, hardware-independent)**

FlashSigmoid replaces softmax with sigmoid in attention, achieving 17% inference kernel speed-up over FlashAttention2 on H100. Key advantage: eliminates the exponential and reduction operations in softmax, replacing them with sigmoid (which is cheaper on hardware with limited special function units). This directly addresses our attention worker's softmax bottleneck (math_pipe_throttle).

---

## Summary: What Matters Most for Workers

| Finding | Relevance | Priority |
|---------|-----------|----------|
| PTX 9.2: FP8 MMA with FP16 accumulator | attention, gemm | **HIGH** - may unlock full-speed FP8 |
| cuBLAS 13.2: 20% FP8/BF16 speedup on sm_120 | gemm (new baseline) | **HIGH** - need to re-benchmark |
| CUTLASS 4.3.3: SM120 blockwise GEMM (ex. 87) | gemm | **MEDIUM** - reference implementation |
| Nsight 2026.1: Register dependency analysis | all workers | **MEDIUM** - better profiling tool |
| cuSOLVER 13.2: FP64-emulated factorization | numerical (LU, QR, Cholesky) | **MEDIUM** - new baseline to beat |
| CUDA 13.2: 256-bit load/store in PTX 9.2 | all workers | **MEDIUM** - wider memory ops |
| GTC 2026 (Mar 16-19) | all | **WATCH** - monitor for relevant sessions |
| FlashSigmoid | attention | **LOW** - algorithmic alternative |
| NVFP4 on sm_120 | future GEMM | **LOW** - not immediate priority |
