# NVIDIA Ecosystem Updates: March 14-16, 2026

**Date:** 2026-03-14
**Relevant to:** ALL workers
**Context:** Pre-GTC sweep. GTC 2026 keynote is March 16 at 11am PT.

---

## 1. CUDA Toolkit 13.2 (Released ~March 5, 2026)

**Source:** [CUDA 13.2 Release Notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html) | [CUDA 13.2 Blog Post](https://developer.nvidia.com/blog/cuda-13-2-introduces-enhanced-cuda-tile-support-and-new-python-features/)

This is a significant release. We are currently on CUDA 13.0. Key changes:

### PTX ISA 9.2
- **New instructions: BMSK (bitmask creation), SZEXT (sign extension)** -- integer arithmetic, unlikely to affect kernel hot paths directly but useful for index manipulation.
- **Extended min/max:** Now support three input arguments (could simplify softmax row-max reduction).
- **256-bit load/store:** Extended `ld`, `ld.global.nc`, and `st` instructions to support 256b operations. **THIS IS SIGNIFICANT** -- our current max is 128b vector loads. If sm_120 supports 256b loads, this could improve memory bandwidth utilization for all kernels.
- **Family-specific targets:** sm_120f and sm_121f added, allowing single binary across sm_120 family.

### Compiler
- Single unified CUDA Toolkit for Tegra and desktop GPUs.
- C++20 conformance improvements (constraints, requires-expressions, lambdas, noexcept).
- New host compiler support (Visual Studio 2026, GCC ACLE extensions).

### cuBLAS 13.2 (March 5, 2026)
- **Grouped GEMM with MXFP8:** Experimental API in cuBLASLt now supports MXFP8 inputs on compute capability 10.x and 11.0. With CUDA Graphs, up to 4x speedup over multistream GEMM for MoE use cases.
- **Per-batch device-side alpha/beta:** New `cublasLtMatmulDescAttributes_t` entries.
- **BLAS Level 3 non-GEMM improvements:** SYRK, HERK, TRMM, SYMM, HEMM performance improved for FP32 and CF32 on Blackwell. **DIRECTLY RELEVANT to linalg/ worker** -- our TRMM and SYRK baselines may have shifted.
- **FP64 emulation for SYRK/HERK:** Auto-dispatch for large problems when FP64 emulation math mode enabled. Uses Ozaki-1 scheme with INT8 tensor cores.
- **BF16x9 FP32 emulation for SYRK/HERK:** `cublas[SC]syr[2]k` and `cublasCher[2]k` can auto-dispatch to BF16x9-accelerated algorithms.
- **Bug fix:** FP8 matmuls potentially failing on multi-device Blackwell GeForce systems.
- **RTX PRO 6000 optimizations:** Up to 20% speedup for FP8/FP16/BF16/TF32/INT8.
- **Known issue:** cuBLASLt Grouped GEMM ignores groups with k=0, can give incorrect results.

### cuSOLVER 13.2
- **FP64 fixed-point emulation:** New APIs (`cusolverDnSetFixedPointEmulationMantissaControl`) enable emulated FP64 via INT8 tensor cores. Up to 2x speedup for QR at 80K matrix sizes. Uses Ozaki-1 scheme. **DIRECTLY RELEVANT to numerical/ worker** -- our QR and LU baselines will change if we upgrade.
- **New `cusolverDnXsygvd` API** for larger eigenvalue problems.
- **Breaking change (since 13.1):** `cusolverDn{C,Z}sytrf` and `cusolverDnXsytrs` now assume complex input matrix A is Hermitian.

### CCCL 3.2 (CUB/Thrust)
- **DeviceTopK::MaxKeys:** Up to 5x speedup vs radix sort for small K. Useful for any top-k selection in kernels.
- **Fixed-size segmented reduction:** Up to 66x speedup for small segments. **Could benefit batched reductions.**
- **Segmented scan, binary search, FindIf with early-exit:** Up to 7x faster search.

### CUDA Tile (cuTile)
- Now supports compute capability 8.X (Ampere/Ada), 10.X, 11.X, and **12.X (Blackwell)**.
- Enhanced Python DSL with recursive functions, closures, custom reductions, type-annotated assignments.
- Not directly relevant to our hand-written PTX kernels, but worth knowing about.

### CUDA Python/CuPy
- CUDA Graphs graduated from experimental to main `cuda.core` namespace.
- Stream Protocol for zero-copy interop between CuPy/PyTorch/JAX.

---

## 2. Nsight Compute 2026.1

**Source:** [Nsight Compute 2026.1 New Features](https://developer.nvidia.com/nsight-compute-2026_1-new-features)

**Highly relevant to all workers.** This ships with CUDA 13.2.

- **Report Merge Tool:** Combine multiple .ncu-rep files into one. Useful for multi-run analysis and comparing experiments. File > Merge Reports menu.
- **Register Dependencies Analysis:** Added to Source page. Shows **"Attributed Live Registers"** and **"Output Registers"** columns per instruction. **THIS IS HUGE** -- directly shows which instructions are causing register pressure. No more guessing which MMA/ldmatrix/cp.async is spilling.
- **Metric Pipeline per instruction:** Source metrics page now shows which pipeline each instruction maps to.
- **CUDA Graph Viewer:** Dynamic visualization of CUDA Graphs during interactive profiling.
- **Thread-level instruction statistics charts:** Granular per-thread performance in instruction stats section.
- **Enhanced timelines:** Overlay related metrics in single rows, max bar backgrounds, toggle Y-axis between theoretical peak and collected max.
- **.ncu-repz format:** Compressed report files.
- **Concurrent kernel profiling across processes:** `--communicator shmem` flag.
- **Nsight Copilot (early access):** AI-assisted profiling analysis.

---

## 3. PyTorch 2.11

**Source:** [PyTorch 2.11 Schedule](https://dev-discuss.pytorch.org/t/release-2-11-schedule-update-release-day-moved-to-monday-march-23/3323) | [CUDA 13 transition](https://dev-discuss.pytorch.org/t/transitioning-pypi-cuda-wheels-to-cuda-13-0-as-the-stable-release-2-11/3325)

- **Release date moved to March 23** (was March 18, delayed to avoid GTC conflict and allow more testing).
- **CUDA 13.0 becomes the stable CUDA variant** for PyPI wheels (both x86_64 and aarch64). This means `pip install torch` will default to CUDA 13 binaries.
- **sm_120 support status:** Still via nightly/CUDA 13 builds. Official stable sm_120 support in prebuilt wheels remains an open request (issues #159207, #164342). PyTorch 2.11 with CUDA 13 should work on sm_120 but may not be fully optimized.
- **Impact on us:** When 2.11 drops (March 23), we should test whether our benchmarking harness gets any changes from the new CUDA 13 default. Our custom kernels compile independently, but the PyTorch SDPA baseline we compare against may change.

---

## 4. CUTLASS Updates

**Source:** [CUTLASS Releases](https://github.com/NVIDIA/cutlass/releases) | [CUTLASS Changelog](https://docs.nvidia.com/cutlass/latest/CHANGELOG.html)

### CUTLASS 4.4.0 (February 14, 2026)
- CuTe DSL now supports CUDA Toolkit 13.1.
- Host-side swizzling heuristics moved to device-side, applied per group based on problem shape and max swizzle size.
- GB300 support added.
- Bug fixes: nvfp4 grouped GEMM core dump, profiler issues, L1 functional test refactoring.

### CUTLASS 4.3.5 (January 9, 2026)
- Fixed unexpected CPU overhead introduced by 4.3.4.
- CuTe DSL bug fixes.

### sm_120 Status in CUTLASS
- **Example 79 (Blackwell GeForce GEMM):** Four variants demonstrating sm_120 block-scaled GEMMs:
  - 79a: NVFP4 + BF16 GEMM
  - 79b: NVFP4 + NVFP4 GEMM
  - 79c: Mixed MXFP8/MXFP6 + BF16 GEMM
  - 79d: NVFP4 grouped GEMM
- Uses `mma.sync.aligned.kind::mxf8f6f4.block_scale` instruction family.
- **NVFP4 MMA has 2x throughput vs MXFP8, 4x vs Ada FP8 MMA.**
- **Known bug (issue #2800):** Python DSL `BlockScaledMmaOp` restricts FP4 operations to sm_100a only, blocking sm_120/sm_121. Fixed in C++ API but not yet in Python DSL.
- **Known bug (issue #2820):** SM120 block-scaled MMA runtime assertion failure in some configurations.

### Block-Scaled MMA on sm_120a (from NVIDIA Forums)
- Instruction: `mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0`
- Operands: A = uint32_t[4], B = uint32_t[2], C/D = float[4], scale_A = uint8_t, scale_B = uint8_t
- Scale factors stored in registers (not tensor memory) on sm_120a.
- **Scale factor UE8M0 has 127-bias** -- must be accounted for when initializing.
- **ldmatrix limitation for FP8:** Standard ldmatrix doesn't support 8x32 tile shape needed for B matrix. Workarounds: manual element loading, or load with compatible ldmatrix shapes and reshuffle.

---

## 5. GTC 2026 Preview (March 16-19)

**Source:** [GTC 2026](https://www.nvidia.com/gtc/) | [GTC 2026 Blog](https://blogs.nvidia.com/blog/gtc-2026-news/)

- **Keynote:** Jensen Huang, Monday March 16, 11am PT, from SAP Center.
- **Expected topics:** AI inference with new chips, agentic AI, CPU-only racks, CUDA ecosystem updates, NemoClaw (open source enterprise AI agents platform).
- **Relevant sessions to watch for:**
  - Any "Programming Blackwell Tensor Cores" sessions (follow-up to GTC 2025 S72720)
  - CUDA kernel optimization / PTX sessions
  - CUTLASS / cuBLAS performance sessions
  - Consumer GPU programming sessions
- **700+ sessions** scheduled. Session catalog at nvidia.com/gtc/session-catalog/.
- **No leaked content found.** Keynote hasn't happened yet (March 16). We should re-scan after the keynote for announcements.

---

## 6. Community Flash Attention for RTX 5090

**Source:** [gau-nernst: Writing Speed-of-Light Flash Attention for 5090](https://gau-nernst.github.io/fa-5090/)

**DIRECTLY RELEVANT to main/ (attention) worker.** Thien Tran wrote a detailed Flash Attention implementation for RTX 5090 in CUDA C++, achieving 94.4% of theoretical peak (197.7 TFLOPS out of 209.5 TFLOPS for BF16).

### Architecture & Performance
- Config: batch=1, heads=8, Q_len=4096, KV_len=8192, head_dim=128, 5090 @ 400W
- Uses mma.m16n8k16 (same as our attention worker), 4 warps, BLOCK_Q=128, BLOCK_KV=64
- **Beats flash-attn library (190.6 TFLOPS / 91.0% SOL) and PyTorch SDPA Flash (186.7 / 89.1%)**
- **cuDNN still wins at 203.6 TFLOPS (97.2% SOL)**

### Key Techniques (in order of impact)
1. **XOR swizzle for shared memory:** Eliminated 8-way bank conflicts, L1 wavefronts ratio from 8:1 to 1:1. Uses address XOR with row-index-derived bits. +27% TFLOPS.
2. **Two-stage cp.async pipeline:** Overlaps global-to-shared transfers with MMA compute. cp.async.wait_group 2 synchronization. +4.7%.
3. **ldmatrix.x4 instead of ldmatrix.x2:** Halved instruction count for K/V loads. Speedup suggests instruction dispatch is the bottleneck at this throughput level. +2.4%.
4. **Refined pipeline (V doesn't need double-buffer):** Only K needs second buffer; freed shared memory allowed BLOCK_KV back to 64. +1.8%.

### sm_120-Specific Findings
- **sm_120 supports cp.async.bulk (TMA) but author didn't use it** -- stayed with Ampere-compatible cp.async.
- **Instruction scheduling is the critical bottleneck** at high throughput -- reducing instruction count (ldmatrix.x4 vs x2) helped even without changing arithmetic intensity.
- **Register spilling sometimes faster** than aggressive conservation -- compiler makes good decisions here.
- **No MXFP8/NVFP4 exploration** -- noted as future work.

### Comparison to Our Attention Worker
Our attention worker is at 1.76x SDPA (BF16). This community implementation shows:
- The ceiling is cuDNN at 97.2% SOL (~203.6 TFLOPS)
- Custom kernels can reach 94.4% SOL with the right pipeline
- Same instruction set (mma.m16n8k16) and tile strategy (4 warps, BLOCK_Q=128)
- Key differentiator appears to be pipeline overlap and instruction count reduction

---

## 7. RTX 5090 Decode Optimization (1000+ tok/s)

**Source:** [Alpin's Blog: Hitting 1,000 tokens per second on a single RTX 5090](https://blog.alpindale.net/posts/5090_decode_optimization/)

**Relevant to:** ALL workers (architecture insights)

### Architecture Insights
- RTX 5090: 170 SMs, 96 MB L2 cache (16x vs 3090), GDDR7 @ 1,674 GB/s
- 128-bit vector loads are the max (256-bit unsupported in practice despite PTX 9.2 extension -- needs verification)
- L2 cache large enough to keep KV caches resident at moderate sequence lengths

### Techniques Worth Noting
- **L1 cache bypass via `L1::no_allocate` PTX hint:** Prevents cache pollution for single-read data (weights). Applicable to our attention K/V loads.
- **`ex2.approx.ftz.f32` for fast exponential:** ~10x faster than `expf`. If our softmax isn't using this, it's leaving performance on the table.
- **`prefetch.global.L2` during idle phases:** Warm weights into L2 while other blocks do bandwidth-light work. Applicable to attention's softmax phase.
- **Custom atomic barriers instead of cooperative kernel launch:** Monotonic generation counter avoids ABA races with lower per-barrier latency.
- 128 thread blocks x 512 threads = persistent megakernel approach.
- 170 SMs but 128 blocks performed better than 170 -- extra atomic overhead from more blocks.

---

## 8. cuSOLVERDx / MathDx Status

**Source:** [MathDx Release Notes](https://docs.nvidia.com/cuda/mathdx/release_notes.html) | [cuSolverDx Docs](https://docs.nvidia.com/cuda/cusolverdx/)

- Latest version: **MathDx 25.06.1** (for CUDA 12 and 13). No 26.x release found yet.
- cuSOLVERDx 0.3.0 was the last known version (supports device-side QR, LU, Cholesky).
- **No 0.3.1 or newer release detected.** Next update may coincide with GTC or a future CUDA release.

---

## 9. FP64 Emulation Research Papers (Relevant to numerical/ worker)

**Source:** Multiple arXiv papers

### "Performance and Numerical Aspects of Decompositional Factorizations with FP64 Emulation in INT8" (arXiv:2509.23565, Sep 2025)
- INT8 advantage over FP64 grows from 30-fold (Hopper) to **100-fold+ on Blackwell**.
- Blackwell: 40 TFLOPS FP64 vs ~5 PFLOPS INT8 tensor cores.
- Directly informs cuSOLVER 13.2's FP64 emulation feature.

### "DGEMM without FP64 Arithmetic" (arXiv:2508.00441, Aug 2025)
- FP64 emulation via FP8 tensor cores using Ozaki scheme on Blackwell.

### "Guaranteed DGEMM Accuracy" (arXiv:2511.13778, Nov 2025)
- Modified cusolverDnGeqrf to use emulated DGEMM for trailing matrix updates.
- Shows ADP (Adaptive Dynamic Precision) as transparent drop-in for FP64 in HPC workflows.
- **Directly describes the mechanism behind cuSOLVER 13.2's emulation feature.**

---

## Action Items / Recommendations

### Immediate (before GTC keynote March 16)
1. **No action needed before keynote.** Re-scan after keynote (March 16 evening or March 17) for announcements.

### Short-term (this week)
2. **Consider CUDA 13.2 upgrade:** The Nsight Compute 2026.1 register dependency analysis alone justifies the upgrade. Workers currently guess at register pressure -- this would show it per-instruction.
3. **Verify 256b load/store on sm_120:** PTX ISA 9.2 extends ld/st to 256b. If this works on sm_120, it could improve all memory-bound kernels. Needs testing with CUDA 13.2.
4. **Re-baseline linalg/ TRMM and SYRK:** cuBLAS 13.2 improved BLAS Level 3 non-GEMM kernels on Blackwell. Our baselines may be stale.
5. **Re-baseline numerical/ QR and LU:** cuSOLVER 13.2 FP64 emulation gives up to 2x speedup on QR. If we upgrade, the baseline shifts.
6. **Attention worker:** Review gau-nernst's FA implementation for pipeline overlap techniques. Our BF16 attention is at 1.76x SDPA; this implementation reaches 94.4% SOL with similar instruction set.
7. **All workers:** Check if `ex2.approx.ftz.f32` is used for all exponential operations. If anyone is using `expf`, switch.

### Post-GTC (March 17+)
8. **Re-scan for GTC announcements:** New CUDA releases, CUTLASS updates, kernel optimization talks.
9. **Watch for CUDA 13.3 announcement** at GTC.
10. **Watch for new Blackwell / sm_120 programming sessions** in GTC catalog.
