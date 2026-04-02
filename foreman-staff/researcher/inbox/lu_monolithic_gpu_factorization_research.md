# Monolithic GPU LU Factorization — Research Brief

**Source:** Multiple (see per-section citations)
**Relevant to:** numerical/ worker (LU factorization / getrf)
**Worker's current problem:** cuSOLVER does N=4096 LU in 9.4ms with a single monolithic kernel. Worker needs to understand how to build a competitive monolithic LU kernel on sm_120 using mma.sync ISA.

---

## 1. cuSOLVERDx Device-Side GETRF (NVIDIA Official)

**Source:** https://docs.nvidia.com/cuda/cusolverdx/get_started/getrf.html
**Source:** https://docs.nvidia.com/cuda/cusolverdx/release_notes.html

### What It Is

cuSOLVERDx provides device-callable LU factorization that runs inside your own CUDA kernel. Added in v0.2.0 with sm_120 (Blackwell consumer) support.

### Key Details

- **Two variants:** `cusolverdx::function::getrf_no_pivot` and `cusolverdx::function::getrf_partial_pivot`
- **Execution model:** Block-level — the entire factorization runs within a single thread block
- **Data must reside in shared memory** — user is responsible for loading from global to shared before calling, and saving results after
- **Block config:** Use `Solver::block_dim` for optimal thread count, `Solver::shared_memory_size` for shmem requirement
- **Supported types:** float, double, complex
- **Supported SM:** sm_70 through sm_90 (v0.1.0), sm_100/sm_101/sm_120 (v0.2.0+)
- **Operator API:** Compose with `Size<M,N>() + Precision<float>() + Function<getrf_partial_pivot>() + SM<1200>() + Arrangement<col_major>()`

### Known Issue on sm_120

CUDA 12.8-13.0 may miscompile kernels using `gesv_no_pivot` with high register pressure when sm_120 and real types are combined. Mitigations involve compiler flags. The `getrf_partial_pivot` variant is not mentioned as affected.

### Why It Matters

This is the building block for the monolithic kernel. cuSOLVERDx GETRF handles the panel factorization (GETF2 equivalent) entirely on-device. Combined with cuBLASDx GEMM for trailing updates, you can build a blocked LU kernel that never returns to the host — exactly what cuSOLVER's internal monolithic kernel does.

### Matrix Size Limitation

cuSOLVERDx operates at block level with data in shared memory. For sm_120 with 128KB shared memory, a float matrix fits up to ~180x180 (180*180*4 = 129.6KB). For the panel (NB columns of N rows), you need N*NB*4 bytes. At N=4096, NB=32: 512KB — far too large. **The panel must be sub-blocked or the matrix must be processed in tiles.**

### Blocked Algorithm Pattern (from cuSOLVERDx Cholesky example)

The advanced Cholesky example demonstrates a **left-looking blocked algorithm** for large matrices:
1. Divide NxN matrix into NB x NB blocks, process in N/NB steps
2. Each step: cuSOLVERDx unblocked factorization + cuBLASDx TRSM + cuBLASDx GEMM
3. Single thread block per batch item — out-of-core (data streams through shmem)
4. All operations fused in one kernel — no global memory round-trips between steps

This pattern directly applies to LU. Replace POTF2 with GETF2, SYRK with GEMM, and add LASWP.

---

## 2. MAGMA Native GPU LU Factorization

**Source:** https://icl.utk.edu/projectsfiles/magma/doxygen/group__magma__getrf.html
**Source:** https://icl.utk.edu/projectsfiles/magma/doxygen/group__magma__getf2__batched.html

### What It Is

MAGMA provides `magma_dgetrf_native()` — a GPU-only LU factorization that does NO computation on the CPU. This is the closest open-source equivalent to what cuSOLVER does internally.

### Architecture

- **Algorithm:** Right-looking Level 3 BLAS blocked LU
- **Panel factorization:** `dgetf2_native_kernel` — a SINGLE CUDA kernel that performs the entire panel factorization
  - Pivot selection for column i is done by thread block i, while other thread blocks wait
  - The entire panel is fused: pivot search + row swap + scale + rank-1 update
  - Panel width constrained to max 1024 columns
- **Trailing update:** GEMM on GPU (separate kernel calls)
- **Expert interface:** `magma_dgetrf_expert_gpu_work()` accepts `MagmaNative` mode, custom block sizes (nb, recnb), and user-provided workspaces
- **Streams:** Uses 2 queues to overlap communication and computation (look-ahead)

### Storage Strategies for Panel

MAGMA implements multiple strategies:
1. **Register-based:** Load entire m*n panel into registers, factorize with pivoting, copy back. Best for tiny panels.
2. **Shared memory:** Load C, B into shared memory, solve, copy back. Good for medium panels.
3. **Hybrid:** Slice along dimensions — active portion in registers, inactive in shared memory.

### Batched Variants (Small Matrices)

For small matrices (N <= ~512), MAGMA offers batched getrf with aggressive optimizations:
- **Column blocking:** A column cached in shmem during update + factorize
- **Panel blocking:** Entire panel in register file, m threads each hold a row of length nb
- **Matrix blocking:** Entire matrix cached throughout factorization (optimal traffic)
- **Kernel fusion:** Panel factorize + TRSM + GEMM all in one kernel
- Panel width nb in [1:32] as compile-time template parameter for loop unrolling
- Matrix-blocking kernel is 2x faster than panel-blocking
- Multi-level blocking achieves 3.28x/2.69x speedup over generic design (single/double)
- Up to 8.72x/7.2x faster than cuBLAS for single/double precision

### Why It Matters

MAGMA's native getrf proves that GPU-only LU factorization is achievable and competitive. The key insight: **the panel factorization must be a single fused kernel** to avoid launch overhead. Their register/shmem storage strategies directly inform how to build the panel kernel.

---

## 3. Blocked Right-Looking LU Algorithm (Standard Approach)

**Source:** https://www.netlib.org/benchmark/hpl/algorithm.html (HPL reference)
**Source:** MAGMA/LAPACK standard algorithm

### The Algorithm

```
for k = 0 to ceil(N/NB) - 1:
    // Panel factorization (compute-bound for small NB, memory-bound for large NB)
    GETF2(A[k*NB:(k+1)*NB, k*NB:N])    // unblocked LU on NB columns

    // Apply pivots to entire row
    LASWP(A[:, 0:k*NB], ipiv)            // left-side swaps (already factored)
    LASWP(A[:, (k+1)*NB:N], ipiv)        // right-side swaps (trailing)

    // Solve for U block
    TRSM(L[k], A[k*NB:(k+1)*NB, (k+1)*NB:N])  // L \ A -> U

    // Trailing matrix update (THIS IS WHERE TENSOR CORES GO)
    GEMM(A[(k+1)*NB:N, k*NB:(k+1)*NB],         // -L21 * U12
         A[k*NB:(k+1)*NB, (k+1)*NB:N],
         A[(k+1)*NB:N, (k+1)*NB:N])             // update trailing
```

### Panel Factorization (GETF2) Detail

Within each panel of NB columns:
```
for j = 0 to NB-1:
    // 1. Find pivot: parallel argmax over column j, rows j..M-1
    pivot_row = parallel_reduce_argmax(A[j:M, j])

    // 2. Swap rows: pivot_row <-> row j (all columns)
    swap_rows(A, j, pivot_row)

    // 3. Scale: A[j+1:M, j] /= A[j, j]  (compute multipliers)

    // 4. Rank-1 update: A[j+1:M, j+1:NB] -= A[j+1:M, j] * A[j, j+1:NB]
```

### Look-Ahead Technique

HPL uses look-ahead to overlap panel factorization with trailing GEMM:
- **Depth 0:** No overlap — factorize panel k, then update trailing, then factorize k+1
- **Depth 1:** After broadcasting panel k, update the NEXT panel's columns first (k+1 panel), then bulk-update the rest of the trailing matrix. Panel k+1 factorization can start while the bulk GEMM runs.
- **Depth 2:** Rarely better than depth 1
- Requires extra memory for buffering panels in the look-ahead pipe

For a monolithic kernel, look-ahead translates to: **factorize the next panel's columns immediately after the current panel completes, before doing the full trailing GEMM.** This keeps the panel factorization on the critical path as short as possible.

---

## 4. Mixed-Precision Pre-Pivoting LU (PRP/MPF)

**Source:** https://link.springer.com/article/10.1007/s11227-024-06523-w (Journal of Supercomputing, Oct 2024)
**Source:** https://www.researchgate.net/publication/381913182

### What It Is

Two novel algorithms that use half-precision (FP16) to accelerate double-precision LU factorization:

1. **PRP (Pre-Pivoted LU):** Compute pivot lists in FP16, then do the actual LU in FP64 without pivoting
2. **MPF (Mixed-precision Panel Factorization):** Use PRP internally for panel factorization within a standard blocked LU

### How PRP Works

1. Compute LU factorization of A in half precision (FP16) — this is fast on tensor cores
2. Extract the pivot permutation list from the FP16 factorization
3. Apply the pivot permutation to the original FP64 matrix
4. Perform LU factorization WITHOUT pivoting on the pre-pivoted FP64 matrix

The insight: pivoting is the serial bottleneck in GPU LU (argmax + row swap per column). By pre-computing pivots in low precision, the actual factorization can proceed without any pivoting synchronization.

### Two PRP Variants

- **hPRP:** Pivot list computed entirely in FP16. Cheaper but less stable.
- **xPRP:** Pivot list computed in mixed FP16/FP32. More stable, slightly slower.

### MPF Panel Factorization

Within a standard blocked LU, replace the GETF2 panel step with:
1. Copy panel to FP16 buffer
2. Run hPRP on the FP16 panel to get pivot list
3. Apply pivot list to FP64 panel
4. Run unpivoted GETF2 on the pre-pivoted FP64 panel

This eliminates the serial pivot search from the critical path of the panel factorization.

### Performance

Tested on V100, A100, and **RTX 3090** (Ampere consumer GPU — relevant precedent for sm_120):
- MPF achieves accuracy on par with standard DGETRF
- Speed improvement comes from eliminating pivot synchronization overhead
- The FP16 pre-factorization is fast because it uses tensor cores

### Why It Matters for Us

The worker faces the same pivot bottleneck on sm_120. Using BF16 mma.sync m16n8k16 to pre-compute pivot lists, then doing the actual factorization without pivoting, could significantly reduce the serial dependency in the panel. This is especially valuable in a monolithic kernel where you want to minimize synchronization points.

**Concrete approach for sm_120:**
1. Panel columns arrive in shmem as FP32
2. Convert to BF16, run a fast BF16 panel factorization using mma.sync to get pivot order
3. Apply pivot permutation to FP32 panel
4. Run unpivoted FP32 panel factorization (no argmax, no swap — pure compute)
5. Use BF16 mma.sync for trailing GEMM with FP32 accumulation

---

## 5. Mixed-Precision LU with Tensor Core Trailing Updates

**Source:** https://icl.utk.edu/files/publications/2020/icl-utk-1414-2020.pdf (ICL UTK, 2020)
**Source:** https://hal.science/hal-02937325

### What It Is

A left-looking LU that stores the matrix in FP16 and uses tensor cores for trailing GEMM updates, with FP32 buffers for numerical stability.

### Algorithm (Doubly-Partitioned)

1. **Outer loop:** Process panels left-to-right
2. **For each panel:** Update it using all previously factored panels (left-looking)
3. **Panel factorization:** In FP32 with partial pivoting
4. **Trailing GEMM updates:** Use FP16 tensor cores (wmma or mma.sync)
5. **Key trick:** Doubly-partitioned — the update is split into an FP32 panel update (for accuracy) and an FP16 bulk update (for speed)

### Performance

- Up to 2x faster than state-of-the-art FP32 LU
- Half the memory footprint (FP16 storage)
- Half the data movement
- Accuracy comparable to standard FP32 factorization

### Why It Matters

For our FP32 getrf on sm_120, the trailing GEMM is the compute-heavy part. Using BF16 mma.sync m16n8k16 for the trailing update (with FP32 accumulation) would:
- Get tensor core throughput for the O(N^3) GEMM portion
- Keep FP32 precision in the panel where pivoting needs accuracy
- Match what cuSOLVER likely does internally

---

## 6. MAGMA Batched GETF2 Kernel Internals

**Source:** https://icl.utk.edu/projectsfiles/magma/doxygen/group__magma__getf2__batched.html
**Source:** Search results on dgetf2_native_kernel.cu

### Panel Kernel Architecture

The `dgetf2_native_kernel` is the core of MAGMA's GPU-native LU:

1. **Single kernel** for the entire panel (not separate kernels for argmax/swap/scale/update)
2. **Thread block i handles pivot selection for column i**, other blocks wait
3. **Pivot search:** Parallel reduction over column elements to find argmax
4. **Row swap:** All threads cooperate to swap rows across all columns
5. **Scale:** Threads parallelize division A[i,k] /= A[k,k]
6. **Rank-1 update:** Each thread handles one column of trailing panel

### Fused Variants

- `getf2_fused_batched`: Loads entire m*n panel into **registers** — one row per thread, length nb. Best for tiny panels (nb <= 32).
- `getf2trsm_batched`: Loads L, B into **shared memory** for triangular solve, copies back.
- `getf2_native`: Uses mixed register/shmem strategy based on panel size.

### Thread Organization

- 256 threads per block (typical for panel kernel)
- Shared arrays for pivot reduction: `sh_val[256]` and `sh_idx[256]`
- Tree-based parallel reduction for argmax
- `__syncthreads()` between phases (argmax → swap → scale → update)

### Key Constraint

For a panel of M rows and NB columns:
- Register approach: Need M threads, each holding NB floats. M=4096, NB=32 → 4096 threads × 32 regs = 131K registers. Exceeds 64K register file.
- **Solution:** Sub-blocking. Process the panel in sub-panels of IB columns (IB=8 or 16). Each sub-panel fits in registers. After each sub-panel, do a mini-TRSM and mini-GEMM to update the remaining columns.

---

## 7. Practical Single-Block GPU LU (Julia Discourse)

**Source:** https://discourse.julialang.org/t/improving-performance-of-cuda-gpu-kernel-lu-factorization/132971

### Implementation Details

A practical single-block LU kernel achieving correct results for N up to ~2000:

- **256 threads, single block** (same as MAGMA panel kernel)
- **Shared memory:** Small arrays for pivot reduction (`sh_val`, `sh_idx`)
- **Matrix in global memory** (not shared — too large)
- **Sequential phases per column k:**
  1. Parallel argmax scan (threads cooperate on rows k..N)
  2. Tree reduction to find pivot
  3. Cooperative row swap
  4. Parallel scale (multipliers)
  5. Rank-1 trailing update (thread per column)

### Critical Performance Insight

Naive rank-1 updates make single-block LU 135x slower than cuSOLVER for N=30000. The fix: **blocked algorithm with panel factorization + GEMM trailing update**. The GEMM is where all the FLOPS concentrate and where tensor cores provide benefit.

### Memory Coalescing

Column-major storage: `threadIdx.x` must map to the row index (first dimension) for coalesced access. Having threads scan across columns within a row causes uncoalesced reads.

---

## 8. HPL Look-Ahead for Monolithic Kernel

**Source:** https://www.netlib.org/benchmark/hpl/algorithm.html

### Translating Look-Ahead to Single Kernel

In a multi-kernel HPL, look-ahead overlaps panel factorization with trailing GEMM using streams. In a monolithic kernel, the equivalent is:

```
for k = 0 to ceil(N/NB) - 1:
    GETF2(panel k)                    // panel factorization
    LASWP + TRSM(panel k)             // pivots + triangular solve

    // LOOK-AHEAD: update next panel's columns FIRST
    GEMM(update columns (k+1)*NB : (k+2)*NB only)

    // Then bulk trailing update
    GEMM(update columns (k+2)*NB : N)

    // Now panel k+1 is ready to factorize immediately
```

Within a monolithic kernel using cooperative groups or `__syncthreads()` barriers, this means:
- After the panel k trailing TRSM, immediately GEMM-update only the next NB columns
- The next iteration's GETF2 can start right away on those updated columns
- Meanwhile, the rest of the trailing GEMM finishes

For N=4096 with NB=64, there are 64 iterations. Look-ahead saves one GEMM latency per iteration for the panel portion.

---

## 9. Communication-Avoiding LU (CALU) with Tournament Pivoting

**Source:** https://epubs.siam.org/doi/10.1137/100788926 (SIAM J. Matrix Anal. Appl.)
**Source:** https://icl.utk.edu/files/publications/2022/icl-utk-1533-2022.pdf

### What It Is

CALU replaces partial pivoting with tournament pivoting to reduce communication (synchronization). Instead of finding one pivot per column sequentially, it finds all NB pivots for the panel at once through a tournament.

### Tournament Pivoting Algorithm

1. **Partition** the panel into P block-rows
2. **Local factorization:** Run GEPP (standard partial pivoting) on each block-row independently
3. **Tournament tree:** At each level of a binary tree, combine two sets of pivot candidates by running GEPP on the combined pivot rows
4. **Result:** After log(P) levels, you have NB pivots that are "good enough" for stability

### Why It Matters

On a single GPU with a monolithic kernel, the "communication" being avoided is **synchronization between thread blocks**. Standard partial pivoting requires a global sync (or single-block bottleneck) for each column's argmax. Tournament pivoting:
- Lets multiple warps/blocks find local pivots independently
- Combines them in a tree — less synchronization depth (log2 vs linear)
- Proven to be as stable as partial pivoting in practice

### GPU-Specific Benefit

For a panel of M=4096 rows with NB=64:
- **Partial pivoting:** 64 sequential argmax reductions over 4096 elements, each requiring global sync
- **Tournament pivoting:** Partition into 16 sub-panels of 256 rows. Each does local LU (parallel). Then 4 levels of tournament (log2(16)=4 reductions). Much less serial work.

### Caveat

Tournament pivoting adds complexity. The row permutation is not the same as partial pivoting, so the LASWP pattern changes. For a first implementation, standard partial pivoting is simpler. Tournament pivoting is the optimization for when pivoting becomes the bottleneck.

---

## 10. Synthesis: Recommended Architecture for N=4096 Monolithic LU

Based on all findings, here is the recommended approach:

### Phase 1: Blocked LU with CUDA Graph (Baseline)

Reuse Cholesky infrastructure:
- Panel: cuSOLVERDx `getrf_partial_pivot` for sub-panel + custom LASWP kernel
- Trailing: cuBLAS GEMM with TF32/BF16 tensor cores
- CUDA Graph captures and replays the loop
- **Expected:** ~0.5-0.6x cuSOLVER (similar to Cholesky result)

### Phase 2: Monolithic Kernel (Target)

Single persistent kernel, all operations fused:
```
Kernel config: 1 block, 256 threads (or cooperative launch with multiple blocks)
NB = 64, IB = 16 (sub-blocked panel)

for k = 0 to 63:  // 4096/64 = 64 iterations
    // Panel factorization in shmem (IB=16 sub-panels within NB=64)
    for ib = 0 to 3:  // 64/16 = 4 sub-panels
        load_panel_to_shmem(A, k, ib)    // 4096 x 16 chunk
        getf2_in_shmem(panel, pivots)     // argmax + swap + scale + rank-1
        store_panel_to_gmem(A, k, ib)
        mini_trsm_in_shmem()              // update remaining IB columns
        mini_gemm_in_shmem()              // update remaining IB columns

    // Full trailing update with tensor cores
    laswp(A, pivots)                      // row swaps across trailing
    trsm_with_mma(L_panel, U_row)         // BF16 mma.sync m16n8k16
    gemm_with_mma(L_col, U_row, trailing) // BF16 mma.sync m16n8k16
```

### Key Design Decisions

1. **Panel in shared memory:** Sub-blocked IB=16 within NB=64. Each sub-panel is 4096*16*4 = 256KB — still too large for shmem. Must process in row-tiles: load 128 rows at a time (128*16*4 = 8KB), do argmax/scale/update, write back, load next 128 rows.

2. **Trailing GEMM with BF16 MMA:** Use mma.sync m16n8k16 for the trailing matrix update. Convert FP32 tiles to BF16, accumulate in FP32. This is where 90%+ of FLOPs go.

3. **Pivoting strategy:** Start with standard partial pivoting. If pivot search becomes the bottleneck, switch to pre-pivoting (PRP) or tournament pivoting.

4. **Multi-block coordination:** For N=4096, a single block cannot efficiently GEMM the full trailing matrix. Options:
   - Cooperative groups grid sync between blocks
   - Persistent kernel with device-side work queues
   - Grid-stride loop pattern within a single launch

5. **Look-ahead:** After each panel, update the next panel's columns before the bulk trailing GEMM. This hides panel factorization latency.

### Performance Targets

- cuSOLVER: 9.4ms at N=4096
- Panel factorization: ~64 panels × 4 sub-panels × ~1us each = ~0.25ms
- LASWP: 64 iterations × ~10us = ~0.64ms (bandwidth-bound row swaps)
- TRSM: Small relative to GEMM
- Trailing GEMM: ~2/3 * N^3 FLOPs = ~45.8 GFLOP. At 50% of theoretical BF16 throughput (~330 TFLOPS / 2 = 165 TFLOPS effective FP32-equivalent): ~0.28ms
- **Achievable target: 2-4ms** (if kernel overhead is eliminated)

---

## Sources

- [cuSOLVERDx GETRF Documentation](https://docs.nvidia.com/cuda/cusolverdx/get_started/getrf.html)
- [cuSOLVERDx Release Notes](https://docs.nvidia.com/cuda/cusolverdx/release_notes.html)
- [cuSOLVERDx Blocked Cholesky Example](https://docs.nvidia.com/cuda/cusolverdx/examples/advanced_example.html)
- [cuSOLVERDx Introduction](https://docs.nvidia.com/cuda/cusolverdx/0.1.0/get_started/introduction.html)
- [MAGMA getrf Variants](https://icl.utk.edu/projectsfiles/magma/doxygen/group__magma__getrf.html)
- [MAGMA getf2 Batched Panel](https://icl.utk.edu/projectsfiles/magma/doxygen/group__magma__getf2__batched.html)
- [MAGMA DeepWiki](https://deepwiki.com/icl-utk-edu/magma)
- [Mixed Precision LU on Tensor Cores (ICL UTK 2020)](https://icl.utk.edu/files/publications/2020/icl-utk-1414-2020.pdf)
- [Mixed-Precision Pre-Pivoting LU (J. Supercomputing 2024)](https://link.springer.com/article/10.1007/s11227-024-06523-w)
- [HPL Algorithm](https://www.netlib.org/benchmark/hpl/algorithm.html)
- [Progressive Optimization of Batched LU on GPUs (ICL UTK 2018)](https://www.netlib.org/utk/people/JackDongarra/PAPERS/icl-utk-1237-2018.pdf)
- [CALU: Communication Optimal LU (SIAM)](https://epubs.siam.org/doi/10.1137/100788926)
- [CALU with Tournament Pivoting on GPUs (ICL UTK 2022)](https://icl.utk.edu/files/publications/2022/icl-utk-1533-2022.pdf)
- [Julia Discourse: CUDA LU Kernel Implementation](https://discourse.julialang.org/t/improving-performance-of-cuda-gpu-kernel-lu-factorization/132971)
- [GPU-based Batched LU for Band Matrices (SC'23)](https://dl.acm.org/doi/10.1145/3624062.3624247)
- [Batched LU Small Matrices (ICL UTK 2014)](https://icl.utk.edu/files/publications/2014/icl-utk-792-2014.pdf)
- [MAGMA 2024 Exascale Paper](https://journals.sagepub.com/doi/10.1177/10943420241261960)
