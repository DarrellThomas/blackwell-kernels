# Monolithic Cholesky Kernel: BF16 MMA for Device-Side SYRK/GEMM

**Source:** Multiple (see references at bottom)
**Relevant to:** numerical worker (Cholesky)
**Worker's current problem:** At 0.55x cuSOLVER with blocked approach (190 CUDA Graph nodes). cuSOLVER uses a single monolithic kernel on 1 SM. TF32 MMA (m16n8k8) has a B fragment broadcasting defect on sm_120, making it unusable. Worker needs device-side GEMM/SYRK using BF16 MMA (m16n8k16) inside a monolithic kernel.

---

## 1. cuBLASDx: Device-Side GEMM on sm_120 (CONFIRMED WORKING)

cuBLASDx (cuBLAS Device Extensions) is NVIDIA's official library for calling GEMM from within a CUDA kernel. This is the most direct path to a device-side GEMM for the monolithic Cholesky.

**Architecture support:** sm_120 is explicitly supported as of cuBLASDx v0.4.0. The release notes state: "SM100, SM101, SM120 support" with CUDA 13 support added in v0.4.1.

**BF16 support:** BF16 is a supported computation type. However, there is a **critical compiler bug** to watch for: CUDA 12.8.0, 12.8.1, and 12.9.0 can miscompile cuBLASDx code when computation types include bf16 (or fp8, fp16, int8) AND any of M, N, K is not a multiple of 16 OR a custom static leading dimension is used. **Workarounds:**
- Use CUDA 12.9.1+ or CUDA 13.x (worker has CUDA 13 -- should be fine)
- Define `CUBLASDX_IGNORE_NVBUG_5218000_ASSERT` if needed
- Add `-Xptxas -O1` compilation flag

**API pattern for device-side GEMM:**

```cuda
// Define the GEMM at compile time
using GEMM = decltype(Size<M, N, K>()
                    + Precision<__nv_bfloat16>()
                    + Type<type::real>()
                    + Function<function::MM>()
                    + Arrangement<row_major, col_major>()
                    + SM<1200>()
                    + Block());

// Inside kernel:
// 1. Allocate shared memory: cublasdx::get_shared_storage_size<GEMM>()
// 2. Create tensors: cublasdx::make_tensor(...)
// 3. Copy global -> shared: cublasdx::copy(gmem_tensor, smem_tensor)
// 4. Execute: GEMM().execute(alpha, a_smem, b_smem, beta, c_smem)
// Or with register accumulator:
//   auto acc = GEMM::get_accumulator();
//   GEMM().execute(a_smem, b_smem, acc);
```

**Key detail:** cuBLASDx operates at the CUDA block level. All threads in the block cooperatively execute the GEMM. The block dimension is obtained via `GEMM::block_dim`.

**A*A^T example exists:** The `simple_gemm_aat` example demonstrates C = A * A^T where both A and A^T share the same shared memory, directly relevant to SYRK. This reduces shared memory by ~50% since A and A^T don't need separate storage.

**Limitations:** cuBLASDx supports ONLY GEMM (no SYRK, no TRSM). SYRK must be implemented as GEMM + lower-triangle masking. TRSM must be hand-written.

---

## 2. cuSolverDx: Device-Side POTRF + TRSM on sm_120 (CONFIRMED WORKING)

cuSolverDx provides device-side Cholesky (potrf) and triangular solve (trsm) that run entirely within a kernel.

**Architecture support:** sm_120 is supported as of cuSolverDx v0.2.0 (experimental Blackwell support). CUDA 13.0 support added in v0.2.1.

**Supported operations directly relevant:**
- `potrf` — Cholesky factorization (panel factorization)
- `trsm` — Triangular solve
- `posv` — Combined Cholesky + solve

**Execution model:** "All the matrices reside in the shared memory." The user handles data movement between global and shared memory. Operations execute at the CUDA block level.

**Supported precisions:** float and double documented. BF16 is NOT documented for cuSolverDx — it's FP32/FP64 only, which is actually fine for the panel factorization (potrf/trsm need full precision anyway).

**The blocked_potrf example** in CUDALibrarySamples demonstrates exactly what the worker needs: a left-looking blocked algorithm with a single thread block per matrix, using cuSolverDx for potrf/trsm and cuBLASDx for the GEMM update. This is the reference implementation for a monolithic Cholesky kernel.

**Blocked algorithm structure (from cuSolverDx docs):**
```
For step i = 0 to N/NB:
  1. potrf: Factor diagonal block A[i,i] (NB×NB) — cuSolverDx
  2. trsm:  Solve A[j,i] for j > i — cuSolverDx
  3. gemm:  Update A[j,k] -= A[j,i] * A[k,i]^T for j,k > i — cuBLASDx
```
All three operations run device-side in shared memory within a single thread block.

---

## 3. NVIDIA Warp: Higher-Level Blocked Cholesky Reference

NVIDIA's Warp library (Python) wraps cuBLASDx and cuSolverDx with a tile-based API. An example exists at `warp/examples/tile/example_tile_cholesky.py` that demonstrates the blocked Cholesky algorithm using tile operations:
- `wp.tile_cholesky()` — potrf
- `wp.tile_lower_solve()` / `wp.tile_upper_solve()` — trsm
- `wp.tile_matmul()` — gemm

While the worker won't use Warp directly (needs raw CUDA), this example is a readable reference for the algorithm flow and operation composition. Requires CUDA 12.6.3+ and the MathDx library.

---

## 4. BF16 MMA for SYRK: Precision Analysis

### The Core Approach: BF16 Input, FP32 Accumulation

BF16 mma.sync.aligned.m16n8k16 performs: D(f32) = A(bf16) * B(bf16) + C(f32)

For SYRK in Cholesky (C -= L * L^T):
- Convert FP32 panel columns to BF16 before MMA
- MMA accumulates in FP32 (no precision loss in accumulation)
- Only the BF16 conversion introduces error: ~1e-3 relative error per element
- BF16 has 7-bit mantissa = ~2 decimal digits of precision in the multiplied terms

### Precision Acceptability

Research from ICL/UTK (Haidar et al., SC18) confirms this approach works: "mixed-precision SGEMM updates replaced by cuBLAS FP16→FP32, all other steps in FP32." The Cholesky factor L is affected by the low precision in the rank-k update, but FP32 accumulation preserves the critical diagonal values. For N=4096 with NB=64, the accumulated error from 64 BF16 rank-k updates is bounded and acceptable for single-precision targets.

**Key insight:** BF16 (8-bit exponent, 7-bit mantissa) has the SAME dynamic range as FP32. This means no overflow/underflow issues — only precision loss. This is strictly better than FP16 (5-bit exponent) for numerical stability in factorizations.

### BF16x9: FP32-Accurate SYRK via Tensor Cores (Advanced Path)

NVIDIA's BF16x9 algorithm in cuBLAS 13.0+ achieves exact FP32 accuracy using BF16 tensor cores. Each FP32 value is decomposed into exactly 3 BF16 values (9 = 3×3 sub-GEMMs). This is a **static decomposition** (unlike the dynamic Ozaki scheme for FP64).

**Key facts:**
- "An FP32 value can be exactly represented as three BF16 values without any loss of accuracy"
- Only provides advantage when BF16 throughput > 9× FP32 throughput
- cuBLAS 13.0 Update 2 added BF16x9 to `cublas[SC]syr[2]k` specifically
- Performance: ~2-3x speedup over native FP32 on Blackwell

**For the worker:** If BF16 precision is insufficient (unlikely for single-precision Cholesky), the BF16x9 splitting could be implemented in the device-side GEMM for exact FP32 accuracy. However, this requires 9 MMA calls per GEMM output element instead of 1, so throughput drops to ~1/9th of peak BF16. Given that the bottleneck is launch overhead (not compute), this may still be viable.

---

## 5. Implementing SYRK as GEMM with Lower-Triangle Masking

Since cuBLASDx has no SYRK, the worker must implement it from GEMM.

**SYRK definition:** C = C - A * A^T (lower triangular output only)

**Approach 1: Full GEMM + mask.** Compute the full GEMM C = A * A^T, then only write the lower triangle back. Wastes ~50% compute but is the simplest path. For NB=64 blocks on a single SM, the wasted compute is small in absolute terms.

**Approach 2: Skip upper-triangle tile blocks.** In a blocked GEMM, output tiles that are entirely above the diagonal can be skipped. For a 64×64 SYRK with 16×8 MMA tiles, roughly half the tiles are skippable. Implementation: the blocked loop simply checks `if (tile_row >= tile_col)` before computing each MMA.

**Approach 3: Use cuBLASDx's `simple_gemm_aat` pattern.** This example already computes A * A^T with shared A/A^T memory. Modify the output write to only store the lower triangle.

**Recommendation:** Start with Approach 1 (full GEMM + mask). The SYRK at N=4096 NB=64 involves tiny matrices (at most 64×64), and the wasted compute is negligible compared to the launch overhead being eliminated. Optimize to Approach 2 only if profiling shows compute is the bottleneck.

---

## 6. Monolithic Kernel Design: Lessons from MAGMA and cuSOLVER

### What cuSOLVER Does (from worker's nsys profile)

cuSOLVER's `getrf_wo_pivot_params_<float, 0, 256, 1, 64, 64, 68>` runs on Grid 1×1×1, Block 256×1×1, uses 202 registers and 52KB shared memory. It achieves ~15 TFLOPS on 1 SM using tensor cores for the GEMM/SYRK updates internally.

**Template parameters decoded:** `<float, 0, 256, 1, 64, 64, 68>` likely means:
- float precision, no pivoting (0), 256 threads, 1 block
- NB=64, NB=64, and 68 likely relates to padded leading dimension (64+4 for bank conflict avoidance)

### MAGMA's Batched Small Cholesky Design

MAGMA's research (Haidar et al., 2017-2018) provides the most detailed single-CTA Cholesky kernel designs:

**For N ≤ 32:** "One thread per element" mapping. Each thread owns one element of the matrix in registers. The factorization loop runs sequentially through columns. At each column k:
1. Thread (k,k) computes sqrt for diagonal
2. Column k threads divide by diagonal (TRSM column)
3. All threads in lower triangle do rank-1 update: A[i,j] -= A[i,k] * A[j,k] (SYRK)

**For N = 33-64:** Register blocking with 2×2 or 4×4 thread tiles. Matrix stored partly in registers, partly in shared memory. The factorization alternates between shared memory panel factorization and register-based trailing matrix updates.

**Key optimization: register blocking for SYRK.** Each thread holds a small tile of the output matrix in registers. The rank-k update accumulates directly in registers without going through shared memory. This eliminates shared memory bank conflicts and reduces synchronization.

**Performance:** Up to 6× speedup over cuBLAS batched routines for N ≤ 32 on Pascal. Up to 11.8× on V100.

### Panel Factorization (potrf/potf2)

The unblocked panel factorization (potf2) for NB×NB diagonal blocks uses:
- Column-by-column Cholesky: for k = 0..NB-1
  - A[k,k] = sqrt(A[k,k])
  - A[k+1:NB, k] /= A[k,k]  (TRSM: division by scalar)
  - A[k+1:NB, k+1:NB] -= A[k+1:NB, k] * A[k+1:NB, k]^T  (rank-1 SYRK)

For NB=64, this is 64 iterations with decreasing rank-1 updates. Each rank-1 update is small enough to be done with CUDA cores (no tensor cores needed for panel). Tensor cores are used for the LARGE trailing matrix SYRK/GEMM update between panels.

---

## 7. Recommended Monolithic Kernel Architecture

Based on all research, here is the recommended approach for the worker:

### Kernel Configuration
```
Grid: 1×1×1 (single SM)
Block: 256×1×1 (matches cuSOLVER)
Shared memory: ~52-64KB (panel + update buffers)
Registers: ~200-256 per thread
```

### Algorithm (Left-Looking Blocked Cholesky)
```
Load full N×N matrix to shared memory (for N ≤ ~360 in FP32 with 99KB shmem)
For N=4096: out-of-core, process NB=64 panels

For each panel j = 0 to N/NB - 1:
  // 1. PANEL FACTORIZATION (FP32 CUDA cores)
  //    Factor diagonal block A[j*NB : (j+1)*NB, j*NB : (j+1)*NB]
  //    Use sub-blocked potf2 with IB=16 (worker's existing approach)
  //    Includes TRSM for sub-panel below diagonal

  // 2. TRSM (FP32 CUDA cores or cuSolverDx)
  //    Solve: A[below, j] = A[below, j] * L[j,j]^{-T}
  //    For each block row below panel:
  //      Load A block from global to shmem
  //      Triangular solve against L[j,j] in shmem

  // 3. SYRK/GEMM UPDATE (BF16 MMA tensor cores)
  //    A[below, below] -= A[below, j] * A[below, j]^T
  //    For each pair of block rows (i, k) where i >= k:
  //      Convert FP32 → BF16 in registers
  //      Execute mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
  //      Accumulate in FP32 registers
  //      Write back lower triangle to global memory

  __syncthreads();  // Barrier between panels
```

### Key Implementation Details

**FP32 → BF16 conversion:** Use `__float2bfloat16_rn()` intrinsic. This is a single instruction on sm_120. Convert just before MMA, keep all storage in FP32.

**SYRK as in-register GEMM:** For the trailing matrix update within a single thread block:
- Each thread computes a small tile of the output (e.g., 4 output elements)
- Load A panel column tiles into shared memory
- Loop over K dimension in chunks of k=16 (BF16 MMA k-dimension)
- Execute MMA with FP32 accumulation
- Only write results for lower-triangle positions

**Memory management for N=4096 out-of-core:**
- Keep current panel (64 columns) in shared memory
- Stream trailing matrix blocks through shared memory for SYRK update
- Double-buffer: while computing SYRK on one block, load next block via cp.async

### Alternative: Use cuBLASDx + cuSolverDx Directly

Instead of hand-writing MMA, use the NVIDIA device libraries:
```cuda
// Panel: cuSolverDx potrf (FP32, NB×NB)
// TRSM: cuSolverDx trsm (FP32)
// SYRK: cuBLASDx GEMM (BF16, A*A^T pattern from simple_gemm_aat)
```

This is less work but gives less control. The cuSolverDx blocked_potrf example does exactly this. The worker could start with this approach and hand-optimize later.

---

## 8. Batched Small Cholesky: Alternative Win Path

The worker's agent state mentions "Batched small Cholesky" as direction 3. This is actually a strong opportunity:

**Why:** cuSOLVER's batched potrf is NOT optimized for tiny matrices (N=32-64). MAGMA showed 6-11× speedups over cuBLAS for these sizes. The worker's existing panel kernel already handles 64×64 efficiently.

**Approach:** Launch one thread block per matrix in the batch. Each block:
1. Loads its N×N matrix (N ≤ 64) entirely into shared memory
2. Runs column-by-column Cholesky (potf2) with register blocking
3. Writes factored L back to global memory

For N=32: 32 threads (one per column), each thread owns one column in registers. For N=64: 256 threads with 2×2 register blocking.

No tensor cores needed at these sizes — the matrices are too small for MMA to help. Pure FP32 CUDA cores with register blocking.

**This could be a quick win** while the monolithic N=4096 kernel is being developed.

---

## References

- [cuBLASDx Documentation](https://docs.nvidia.com/cuda/cublasdx/index.html) — Device-side GEMM API
- [cuBLASDx Release Notes](https://docs.nvidia.com/cuda/cublasdx/release_notes.html) — sm_120 support confirmed, BF16 compiler bug
- [cuBLASDx Examples](https://docs.nvidia.com/cuda/cublasdx/examples.html) — simple_gemm_aat (A*A^T pattern)
- [cuBLASDx GEMM Usage](https://docs.nvidia.com/cuda/cublasdx/using_cublasdx.html) — API patterns and shared memory management
- [cuSolverDx Cholesky](https://docs.nvidia.com/cuda/cusolverdx/get_started/potrf.html) — Device-side potrf
- [cuSolverDx Release Notes](https://docs.nvidia.com/cuda/cusolverdx/release_notes.html) — sm_120 support, operations list
- [cuSolverDx Blocked Potrf Example](https://docs.nvidia.com/cuda/cusolverdx/examples/advanced_example.html) — Reference monolithic kernel
- [cuBLAS FP Emulation Blog](https://developer.nvidia.com/blog/unlocking-tensor-core-performance-with-floating-point-emulation-in-cublas/) — BF16x9 algorithm details
- [Haidar et al. "Harnessing GPU Tensor Cores for Fast FP16 Arithmetic" (SC18)](https://www.netlib.org/utk/people/JackDongarra/PAPERS/haidar_fp16_sc18.pdf) — Mixed-precision Cholesky with tensor cores
- [Haidar et al. "Fast Cholesky on GPUs" (MAGMA)](https://www.sciencedirect.com/science/article/abs/pii/S1877750316305154) — Batched small Cholesky kernel design
- [Haidar et al. "Guide for Small Matrices" (IEEE TPDS)](https://ieeexplore.ieee.org/document/8214236/) — Register blocking for N ≤ 64
- [ICL/UTK "Optimizing GPU Kernels for Irregular Batch"](https://icl.utk.edu/files/publications/2018/icl-utk-1123-2018.pdf) — Tiny matrix Cholesky techniques
- [arXiv:2601.08082 "Hierarchical Precision Recursive Cholesky"](https://arxiv.org/html/2601.08082v1) — Recursive SYRK, mixed-precision hierarchy
- [arXiv:2601.03754 "GPU-Accelerated Block Tridiagonal Cholesky"](https://arxiv.org/html/2601.03754) — Fused kernel using Warp/cuBLASDx/cuSolverDx on RTX 5090
- [arXiv:2203.03341 "Recovering FP32 from Tensor Cores"](https://arxiv.org/abs/2203.03341) — Error-correcting tensor core GEMM
- [NVIDIA Warp Cholesky Example](https://github.com/NVIDIA/warp/blob/main/warp/examples/tile/example_tile_cholesky.py) — Tile-based blocked Cholesky reference
- [NVIDIA CUDALibrarySamples cuSolverDx](https://github.com/NVIDIA/CUDALibrarySamples/tree/main/MathDx/cuSolverDx) — Source code for blocked_potrf
