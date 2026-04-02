# Monolithic Factorization Kernel Approaches for sm_120

**Source:** Multiple (see per-section citations)
**Relevant to:** Cholesky, LU, and QR workers (numerical/, lu/, qr/)
**Worker's current problem:** All three factorization workers hit the same wall: launch overhead from many small kernel calls vs cuSOLVER's single monolithic kernel. Cholesky is at 0.55x cuSOLVER with 190 CUDA Graph nodes. LU is just starting (cuSOLVER baseline 9.4ms at N=4096). QR has not started. The path to parity requires eliminating kernel launch overhead through a monolithic or persistent kernel design.

---

## 1. cuSOLVERDx Device-Side API: Full Function Coverage for Factorizations

**Source:** [cuSOLVERDx Release Notes](https://docs.nvidia.com/cuda/cusolverdx/release_notes.html), [cuSOLVERDx GETRF](https://docs.nvidia.com/cuda/cusolverdx/get_started/getrf.html), [cuSOLVERDx GEQRF](https://docs.nvidia.com/cuda/cusolverdx/get_started/functions/geqrf.html), [cuSOLVERDx UNMQR](https://docs.nvidia.com/cuda/cusolverdx/get_started/functions/unmqr.html), [cuSOLVERDx Blocked Potrf Example](https://docs.nvidia.com/cuda/cusolverdx/examples/advanced_example.html)

### What cuSOLVERDx Provides (v0.2.0+, sm_120 supported)

cuSOLVERDx v0.2.0 added sm_100, sm_101, and **sm_120** support, with CUDA 13.0 support in v0.2.1. The following device-callable factorization functions are available:

| Function | Purpose | Version Added | sm_120 |
|----------|---------|---------------|--------|
| **potrf** | Cholesky factorization | v0.1.0 | Yes (v0.2.0+) |
| **getrf_no_pivot** | LU without pivoting | v0.1.0 | Yes (v0.2.0+) |
| **getrf_partial_pivot** | LU with partial pivoting | v0.2.0 | Yes, but see caveat |
| **geqrf** | QR factorization (Householder) | v0.2.0 | Yes |
| **unmqr** | Apply Q from QR factorization | v0.2.0 | Yes |
| **trsm** | Triangular solve | v0.2.0 | Yes |
| **ungqr** | Generate explicit Q from QR | v0.3.0 | Yes |

**Known sm_120 Bug:** CUDA 12.8-13.0 may miscompile kernels using `gesv_no_pivot` with high register pressure on sm_120 with real types. The `getrf_partial_pivot` variant is NOT mentioned as affected, but caution is warranted. Workaround: `-Xptxas -O1` flag.

### Device-Side API Pattern

```cpp
// Define solver type at compile time
using Solver = decltype(
    cusolverdx::Size<M, N>()
    + cusolverdx::Precision<float>()
    + cusolverdx::Function<cusolverdx::function::getrf_partial_pivot>()
    + cusolverdx::SM<1200>()
    + cusolverdx::Arrangement<cusolverdx::arrangement::col_major>()
);

// Inside kernel:
__shared__ __align__(16) char smem[Solver::shared_memory_size];
auto [As, ipivs] = cusolverdx::shared_memory::slice<float, int>(smem);

// Load matrix A from global to shared memory
// ...

Solver().execute(As, ipivs, &info);  // Entire factorization in shared memory

// Store results back to global memory
```

### GEQRF Device-Side API

```cpp
// QR factorization
__device__ void execute(data_type* A, data_type* tau);
__device__ void execute(data_type* A, const unsigned int lda, data_type* tau);
```

After execution, the upper triangular part of A contains R. Elements below the diagonal, together with the tau array (size min(M,N)), represent Q as a product of Householder vectors.

### UNMQR Device-Side API (Apply Q)

```cpp
// Apply Q from QR to matrix C: C = op(Q) * C or C = C * op(Q)
__device__ void execute(const data_type* A, const data_type* tau, data_type* C);
```

Supports left/right side multiplication, transpose/conjugate transpose. This is the key building block for blocked QR trailing updates -- it replaces the LARFT+LARFB sequence with a single device call.

### Shared Memory Constraint

All data must reside in shared memory before calling `execute()`. For sm_120 with 99KB usable shared memory:
- FP32 max square matrix: ~157x157 (157*157*4 = 98.6KB)
- FP64 max square matrix: ~111x111
- This constrains the NB (panel block size) for the blocked algorithm

### Blocked Algorithm Composition (from blocked_potrf example)

The advanced cuSOLVERDx example demonstrates a **left-looking blocked algorithm** that processes an NxN matrix in N/NB steps, each step composing:
1. **cuSOLVERDx** for panel factorization (potrf/getrf/geqrf) in shared memory
2. **cuSOLVERDx trsm** for triangular solve in shared memory
3. **cuBLASDx GEMM** for trailing matrix update in shared memory

All within a **single thread block**, out-of-core (data streams through shared memory from global memory). This is the reference pattern for monolithic factorization kernels.

---

## 2. cuBLASDx Device-Side GEMM: The Trailing Update Engine

**Source:** [cuBLASDx Release Notes](https://docs.nvidia.com/cuda/cublasdx/release_notes.html), [cuBLASDx Examples](https://docs.nvidia.com/cuda/cublasdx/examples.html)

### What cuBLASDx Provides

cuBLASDx v0.4.0+ supports sm_120. It provides **GEMM only** -- no TRSM, no SYRK. Key features:

- **BF16 GEMM** with FP32 accumulation (for tensor core trailing updates)
- **A*A^T pattern** (`simple_gemm_aat` example) -- directly useful for Cholesky's SYRK
- Block-level execution: all threads in a block cooperate on the GEMM
- Suggested layouts for optimal performance per SM architecture

**Critical compiler bug (cuBLASDx v0.4.0):** CUDA 12.8-12.9 can miscompile cuBLASDx code when computation types include bf16/fp8/fp16/int8 AND M, N, or K is not a multiple of 16 OR a custom static leading dimension is used. **Workaround:** CUDA 13.x should be fine; add `-Xptxas -O1` if issues arise.

### TRSM Gap

cuBLASDx does NOT provide device-side TRSM. Options:
1. **cuSOLVERDx trsm** (v0.2.0+, sm_120 supported) -- the official device-side TRSM
2. **Hand-written TRSM** using our proven BF16 MMA infrastructure
3. **Invert-and-multiply** for small triangular blocks (NB <= 64)

---

## 3. Composing cuSOLVERDx + cuBLASDx: The Monolithic Pattern

**Source:** [cuSOLVERDx Blocked Potrf Example](https://docs.nvidia.com/cuda/cusolverdx/examples/advanced_example.html), [MathDx Installation](https://docs.nvidia.com/cuda/mathdx/installation.html)

### The Proven Composition Pattern

The blocked_potrf example demonstrates the exact pattern all three factorization workers need:

```
Single kernel, single thread block per batch item:

for step = 0 to N/NB - 1:
    // Load panel from global to shared memory

    // 1. Panel factorization: cuSOLVERDx (potrf/getrf/geqrf)
    //    Operates on NB x NB block in shared memory
    Solver().execute(panel_smem, ...);
    __syncthreads();

    // 2. Triangular solve: cuSOLVERDx trsm
    //    Solve against factored panel for off-diagonal blocks
    Trsm().execute(L_smem, B_smem);
    __syncthreads();

    // 3. Trailing update: cuBLASDx GEMM
    //    Update trailing matrix with BF16 tensor cores
    GEMM().execute(alpha, A_smem, B_smem, beta, C_smem);
    __syncthreads();

    // Store results back to global memory
```

### Adaptation by Factorization Type

| Step | Cholesky (potrf) | LU (getrf) | QR (geqrf) |
|------|-----------------|------------|------------|
| Panel | cuSOLVERDx potrf | cuSOLVERDx getrf_partial_pivot | cuSOLVERDx geqrf |
| Solve | cuSOLVERDx trsm | cuSOLVERDx trsm + LASWP | cuSOLVERDx unmqr |
| Update | cuBLASDx GEMM (A*A^T) | cuBLASDx GEMM | cuBLASDx GEMM |

### MathDx Installation Requirement

MathDx (containing cuBLASDx + cuSOLVERDx) is NOT installed at `/usr/local/cuda-13` by default. It must be downloaded from https://developer.nvidia.com/mathdx and installed. cuBLASDx is header-only; cuSOLVERDx requires linking to LTO libraries. CUDA 13.0 is supported (MathDx v0.2.1+/v0.3.0+).

---

## 4. MAGMA's Native GPU Factorization Architecture

**Source:** [MAGMA getrf_native](https://icl.utk.edu/projectsfiles/magma/doxygen/group__magma__getrf.html), [MAGMA Batch QR (ICCS 2022)](https://www.iccs-meeting.org/archive/iccs2022/papers/133500064.pdf), [MAGMA Fast Cholesky](https://www.sciencedirect.com/science/article/abs/pii/S1877750316305154), [MAGMA 2024 Exascale](https://journals.sagepub.com/doi/10.1177/10943420241261960)

### MAGMA's Three-Strategy Approach

MAGMA uses different kernel strategies based on matrix size. This applies to all three factorizations:

**Strategy 1: Fully Fused (N <= ~32)**
- Entire matrix in register file across all threads
- One thread per row, each thread holds NB values in registers
- Column-by-column factorization entirely in registers
- **rowid trick** for LU: track permutation as metadata, no physical row swaps
- Single warp (N <= 32) means no `__syncthreads()` needed

**Strategy 2: Fused Panel + Update Kernel (N = 33-512)**
- Panel cached in register file (m threads, each holding nb registers)
- Fused dlarft+dlarfb (QR) or fused scale+rank1 (LU) -- avoids forming T
- Trailing update via batch GEMM
- Tree reduction in shared memory for column norms (QR) and pivot search (LU)

**Strategy 3: LAPACK-Style Blocked (N > 512)**
- Panel factorization as a fused kernel
- Trailing update via large external GEMM (cuBLAS or custom)
- Look-ahead for panel/trailing overlap

### MAGMA LU Native Panel Kernel (dgetf2_native_kernel)

This is the most relevant MAGMA pattern for our LU worker:

- **Grid:** N blocks (one per column), TX threads per block (256 typical)
- **Each thread holds NPAGES elements** of its column in registers
  - TX=256, M=4096: NPAGES = ceil(4096/256) = 16 registers per thread for data
- **Column-by-column loop:**
  1. Block i finds pivot (parallel argmax via shared memory tree reduction)
  2. All blocks swap rows (via shared memory staging)
  3. Block i scales column, writes to global memory, sets `atomicExch(&update_flag[i], 1)`
  4. Blocks j > i **spin-wait** on `update_flag[i]`, then do rank-1 update

**Inter-block synchronization:** Global memory atomic flag array. MAGMA forces 1 block per SM occupancy (via large shared memory allocation) to prevent deadlock from blocks waiting on unscheduled blocks.

### MAGMA QR Fused Panel Kernel

For QR, MAGMA's fused kernel merges dlarft and dlarfb:

- **Thread-per-row assignment:** Each thread holds one row of the panel in registers
- **Two critical reductions in shared memory:**
  1. Column norm (for Householder reflector generation): tree reduction of squared elements
  2. v^T * A product (for reflector application): multi-column tree reduction
- **Avoids forming T factor:** Applies elementary reflectors directly to trailing matrix
- **Compile-time nb:** Panel width must be known at compile time for register allocation

### Key MAGMA Insight for Our Workers

MAGMA's fused kernels read/write each matrix element exactly once from global memory. The entire factorization happens in register + shared memory. This is exactly what our monolithic kernels should aim for.

---

## 5. Inter-SM Synchronization for Blocked Factorization

**Source:** [CUDA Cooperative Groups](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html), [Inter-Block GPU Communication (Xiao et al., IPDPS 2010)](https://synergy.cs.vt.edu/pubs/papers/xiao-ipdps2010-gpusync.pdf), [GPU Sync Methods (arxiv 2004.05371)](https://arxiv.org/pdf/2004.05371)

### Option A: Cooperative Groups Grid Sync (Recommended)

**How it works:**
```cpp
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void monolithic_factorization(float* A, int N, int NB) {
    cg::grid_group grid = cg::this_grid();

    for (int k = 0; k < N/NB; k++) {
        // Panel factorization (blocks assigned to panel columns)
        if (blockIdx.x < NB) {
            panel_factorize(A, k, NB);
        }

        grid.sync();  // ALL blocks barrier -- panel is done

        // Trailing GEMM update (all blocks participate)
        trailing_gemm_update(A, k, NB);

        grid.sync();  // ALL blocks barrier -- trailing update done
    }
}

// Launch:
void* args[] = {&d_A, &N, &NB};
int numBlocks;
cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, kernel, blockSize, smemSize);
cudaLaunchCooperativeKernel((void*)kernel, numBlocks * numSMs, blockSize, args, smemSize);
```

**Requirements:**
- Compute capability 6.0+ (sm_120 qualifies)
- Grid must fit entirely on the GPU (all blocks resident simultaneously)
- Launch via `cudaLaunchCooperativeKernel` (not `<<<>>>`)
- Query max blocks: `cudaOccupancyMaxActiveBlocksPerMultiprocessor`

**For sm_120 (170 SMs, 32 max blocks/SM, 48 warps/SM):**
- With 256 threads (8 warps) and ~50KB shared memory: ~2 blocks/SM = **340 blocks max**
- With 128 threads (4 warps) and ~32KB shared memory: ~3 blocks/SM = **510 blocks max**
- Enough for trailing GEMM parallelism at N=4096

**Advantage:** Clean, correct, deadlock-free. No spin-waiting.
**Disadvantage:** All blocks must be resident = limited grid size. Barrier cost ~microseconds.

### Option B: Atomic Flag Spin-Wait (MAGMA Pattern)

**How it works:**
```cpp
__device__ volatile int* completion_flags;  // global memory, size N

// Panel block sets flag after completing column k:
atomicExch(&completion_flags[k], 1);

// Trailing blocks spin-wait:
while (atomicAdd(&completion_flags[k], 0) == 0) { /* spin */ }
```

**Critical deadlock prevention:** Must guarantee all blocks are simultaneously resident. Achieve this by:
1. Large shared memory allocation to force 1 block/SM occupancy
2. Or use `cudaOccupancyMaxActiveBlocksPerMultiprocessor` to size the grid

**Advantage:** Fine-grained synchronization (per-column, not global barrier). Blocks waiting on column k can proceed as soon as k is done, without waiting for other columns.
**Disadvantage:** Wastes cycles spinning. Can deadlock if blocks aren't all resident.

### Option C: Single-Block (cuSOLVER/cuSOLVERDx Pattern)

**How it works:** One thread block, 256 threads. No inter-block synchronization needed. All work serialized within the block. Panel factorization and trailing GEMM both done by the same 256 threads.

**Advantage:** Simplest. No synchronization issues. Proven to work (cuSOLVER uses this).
**Disadvantage:** Only 256 threads doing the trailing GEMM. At N=4096 with NB=64, the trailing GEMM is up to 4032x64x4032 -- doing this with 256 threads on 1 SM is slow.

**However:** cuSOLVER achieves 9.4ms at N=4096 with a single block. The trailing GEMM uses tensor cores internally (device-side cuBLASDx GEMM or equivalent). 256 threads with tensor cores is not trivial -- each warp has 4 tensor cores, 8 warps = 32 tensor cores on one SM. At ~330 TFLOPS BF16 / 170 SMs = ~1.94 TFLOPS per SM, a 4032x64x4032 GEMM is ~2.1 GFLOP, taking ~1ms on one SM. Across 64 iterations: ~64ms. But the matrices shrink each iteration, so total trailing GEMM is ~N^3/3 = ~22.9 GFLOP, taking ~12ms on one SM. cuSOLVER doing 9.4ms suggests either proprietary tensor core usage or undisclosed multi-SM coordination.

### Recommendation for Each Factorization

| Factorization | Recommended Approach | Rationale |
|---------------|---------------------|-----------|
| **Cholesky** | Single-block (cuSOLVERDx pattern) | cuSOLVER does it in 1.5ms on 1 block. SYRK is symmetric = half the GEMM work. |
| **LU** | Cooperative groups multi-block | cuSOLVER takes 9.4ms. Full GEMM (not symmetric). Multi-block trailing GEMM essential. |
| **QR** | Start single-block, graduate to cooperative | QR trailing update is 2 GEMMs. Test single-block first, measure if trailing GEMM dominates. |

---

## 6. Real-World Validation: Block Tridiagonal Cholesky on RTX 5090

**Source:** [arXiv:2601.03754 (Jan 2026)](https://arxiv.org/abs/2601.03754)

### What This Is

A GPU-accelerated block tridiagonal Cholesky factorization using cuBLASDx + cuSOLVERDx via NVIDIA's Warp library, tested on **RTX 5090 (sm_120)**. This is the only published paper we know of that uses the MathDx device-side composition pattern on our exact hardware.

### Key Implementation Details

- **Fused kernels:** Load entire n x n blocks into shared memory, run potrf + trsm + syrk/gemm, write back
- **Blocked kernels:** For larger blocks, tile operations with sub-matrix size b x b
- **Inter-block coordination:** CUDA streams (3 parallel streams) + atomic operations + CUDA graphs
- **Optimal tile sizes:** b=8 or b=16 for FP64; n divisible by 16 for FP32
- **Memory alignment to 128 bits** is critical for leveraging special GPU load instructions

### Performance Results

- 100x faster than sparse solver QDLDL
- 25x faster than optimized CPU (BLASFEO)
- 2x faster than NVIDIA cuDSS
- RTX 5090 achieves 500x speedup over QDLDL in single precision on long horizons

### Why It Matters

This paper proves that the cuSOLVERDx + cuBLASDx composition pattern **works on sm_120**. The Warp tile API wraps MathDx and generates CUDA/C++ code. While the workers won't use Warp (they write raw CUDA), the underlying MathDx calls are the same. The paper encountered no sm_120-specific issues beyond memory alignment sensitivity.

---

## 7. Minimum Viable Monolithic Kernel Architecture

### For LU (getrf) at N=4096

**Phase 1: cuSOLVERDx + cuBLASDx single-block (quick baseline)**

```
Kernel: 1 block, Solver::block_dim threads
Shared memory: max(Solver::shared_memory_size, GEMM::shared_memory_size)

for k = 0 to N/NB - 1:
    // Load panel (NB columns) from global to shared
    cuSOLVERDx_getrf_partial_pivot(panel_smem, ipiv_smem, &info);
    __syncthreads();

    // Apply pivots to left and right of panel (in global memory)
    laswp_device(A_global, ipiv_smem, k*NB);
    __syncthreads();

    // TRSM: solve for U block
    cuSOLVERDx_trsm(L_panel_smem, U_row_smem);
    __syncthreads();

    // Trailing GEMM: stream tiles through shared memory
    for each tile pair (i, j) in trailing matrix:
        load A[i, k] and A[k, j] tiles to shared memory
        cuBLASDx_GEMM(-1, A_smem, B_smem, 1, C_smem);
        store C_smem back to global memory
        __syncthreads();
```

**Expected:** ~0.5-0.6x cuSOLVER (same as Cholesky baseline). The single-block trailing GEMM is the bottleneck.

**Phase 2: Cooperative groups multi-block**

```
Kernel: numBlocks * numSMs blocks, 256 threads each
Launch: cudaLaunchCooperativeKernel

for k = 0 to N/NB - 1:
    // Panel factorization: block 0 only
    if (blockIdx.x == 0):
        load panel to shared memory
        // Can use cuSOLVERDx getrf OR custom fused panel kernel
        panel_factorize(panel_smem, ipiv_smem);
        store panel to global memory

    grid.sync();  // Panel done

    // LASWP + TRSM: distributed across blocks
    // Each block handles a subset of row swaps and TRSM columns
    distributed_laswp_trsm(A, ipiv, k);

    grid.sync();  // Pivots and TRSM done

    // Trailing GEMM: each block handles a tile
    // Grid-stride loop over trailing matrix tiles
    for tile = blockIdx.x; tile < num_tiles; tile += gridDim.x:
        load A and B tiles
        gemm_mma_bf16(A_tile, B_tile, C_tile);  // Our proven BF16 MMA
        store C tile

    grid.sync();  // Trailing update done
```

### For QR (geqrf) at N=4096

**Phase 1: cuSOLVERDx single-block (baseline)**

Same as LU but with:
- `geqrf` for panel factorization (returns reflectors in A and tau)
- `unmqr` for trailing update (applies Q^T to trailing matrix)
- No LASWP needed (QR has no pivoting)

**Phase 2: Recursive QR (target)**

The tensor core recursive QR approach (from existing research brief) replaces the standard blocked trailing updates with recursive splits that produce near-square GEMMs. Combined with cooperative groups:

```
function recursive_qr(A, m, n):
    if n <= NB:
        // Base: cuSOLVERDx geqrf on panel
        return geqrf(A)

    n1 = n/2
    recursive_qr(A[:, :n1], m, n1)        // Factor left half

    grid.sync();

    // Apply Q1^T to right half -- THIS IS THE WINNING GEMM
    // Dimensions: n1 x m x n1 -- NEAR-SQUARE, tensor cores efficient
    unmqr_or_gemm(Q1^T, A[:, n1:]);

    grid.sync();

    recursive_qr(A[n1:, n1:], m-n1, n-n1) // Factor right half
```

### For Cholesky (potrf) at N=4096

**Single-block with cuSOLVERDx is the primary path.** cuSOLVER achieves 1.5ms on a single block. The SYRK trailing update is half the work of a full GEMM. Using cuSOLVERDx potrf + cuBLASDx GEMM (A*A^T pattern) in a single kernel should approach cuSOLVER's performance.

The Cholesky worker's current 0.55x ratio is entirely due to launch overhead (190 graph nodes). Eliminating launches with a monolithic kernel should close most of the gap.

---

## 8. Critical Path: MathDx Installation

**Source:** [MathDx Installation](https://docs.nvidia.com/cuda/mathdx/installation.html), [cuBLASDx Downloads](https://developer.nvidia.com/cublasdx-downloads)

Before any worker can attempt the cuSOLVERDx + cuBLASDx composition pattern, MathDx must be installed. This is a prerequisite that should be escalated to Darrell if not already available.

**Steps:**
1. Download MathDx from https://developer.nvidia.com/mathdx (requires NVIDIA developer account)
2. Extract to a system location (e.g., `/usr/local/mathdx`)
3. cuBLASDx: header-only, just add include path
4. cuSOLVERDx: requires linking to LTO libraries (`-dlto` flag, link against `cusolverdx_lto_<arch>.a`)
5. Build with: `-I/path/to/mathdx/include -arch=sm_120`

**Alternative if MathDx unavailable:** Build custom monolithic kernels using our existing BF16 MMA infrastructure (proven at 0.97x cuBLAS) for the trailing GEMM, with hand-written panel factorization kernels following the MAGMA patterns described above. This is more work but gives full control and avoids the MathDx dependency.

---

## 9. Summary: Recommended Approach per Worker

### Cholesky Worker (currently 0.55x cuSOLVER)

1. **Install MathDx** (prerequisite)
2. **Implement blocked_potrf pattern:** Single-block, cuSOLVERDx potrf + cuBLASDx GEMM (A*A^T)
3. **NB=64** (matching cuSOLVER's template parameters)
4. **Expected result:** 0.8-1.0x cuSOLVER (eliminating launch overhead closes most of the gap)
5. **Fallback:** Custom BF16 MMA SYRK in the monolithic kernel if cuBLASDx GEMM is suboptimal

### LU Worker (baseline only, 9.4ms at N=4096)

1. **Phase 1:** Single-block cuSOLVERDx getrf + cuBLASDx GEMM (quick baseline, expect 0.5-0.6x)
2. **Phase 2:** Cooperative groups multi-block with BF16 MMA trailing GEMM
3. **Panel kernel options:**
   - cuSOLVERDx getrf_partial_pivot (simplest)
   - Custom MAGMA-style fused panel with rowid trick (best performance)
   - Mixed-precision pre-pivoting PRP (if pivoting becomes bottleneck)
4. **NB=64 initially**, test NB=128, NB=256 (MAGMA uses nb=512 for float at N=4096)

### QR Worker (not started)

1. **Baseline:** cuSOLVER sgeqrf measurement
2. **Phase 1:** Single-block cuSOLVERDx geqrf + unmqr + cuBLASDx GEMM
3. **Phase 2:** Recursive QR with BF16 MMA tensor core GEMMs (the winning architecture)
4. **Key insight:** Recursive QR converts tall-skinny GEMMs (poor tensor core utilization) into near-square GEMMs (excellent tensor core utilization). This is the unique advantage QR has over LU/Cholesky.
5. **Panel:** cuSOLVERDx geqrf handles Householder reflector generation, avoids hand-writing the complex column norm reductions and reflector applications

---

## Caveats for sm_120

1. **TF32 MMA is broken on sm_120.** B fragment diagonal broadcasting makes m16n8k8 TF32 MMA unusable for general GEMM. All tensor core trailing updates must use BF16 MMA (m16n8k16) with FP32 accumulators. This is a known hard-won lesson.

2. **cuSOLVERDx sm_120 compiler bugs.** The `gesv_no_pivot` miscompilation issue on sm_120 with CUDA 12.8-13.0 suggests caution. Test each cuSOLVERDx function empirically before building on it.

3. **sm_120 is consumer Blackwell (mma.sync), NOT datacenter (tcgen05).** cuBLASDx v0.5.0 has experimental pipelining with WGMMA and TMA support, but these are sm_100 (datacenter) features. Our sm_120 code path in cuBLASDx/cuSOLVERDx uses the mma.sync backend, which is correct.

4. **Cooperative groups on sm_120:** Supported (compute capability 12.0 > 6.0 requirement). Max 32 blocks per SM, 48 warps/SM. The grid must fit entirely on the GPU for cooperative launch.

5. **99KB shared memory limit per block** (not 128KB). This constrains the NB size for panel factorization and the tile size for trailing GEMM in the single-block pattern.

---

## Sources

- [cuSOLVERDx Release Notes](https://docs.nvidia.com/cuda/cusolverdx/release_notes.html)
- [cuSOLVERDx GETRF Documentation](https://docs.nvidia.com/cuda/cusolverdx/get_started/getrf.html)
- [cuSOLVERDx GEQRF Documentation](https://docs.nvidia.com/cuda/cusolverdx/get_started/functions/geqrf.html)
- [cuSOLVERDx UNMQR Documentation](https://docs.nvidia.com/cuda/cusolverdx/get_started/functions/unmqr.html)
- [cuSOLVERDx Blocked Potrf Example](https://docs.nvidia.com/cuda/cusolverdx/examples/advanced_example.html)
- [cuBLASDx Release Notes](https://docs.nvidia.com/cuda/cublasdx/release_notes.html)
- [cuBLASDx Examples](https://docs.nvidia.com/cuda/cublasdx/examples.html)
- [MathDx Installation Guide](https://docs.nvidia.com/cuda/mathdx/installation.html)
- [MAGMA getrf Variants](https://icl.utk.edu/projectsfiles/magma/doxygen/group__magma__getrf.html)
- [MAGMA Batch QR on GPUs (ICCS 2022)](https://www.iccs-meeting.org/archive/iccs2022/papers/133500064.pdf)
- [MAGMA Fast Cholesky (ScienceDirect 2017)](https://www.sciencedirect.com/science/article/abs/pii/S1877750316305154)
- [MAGMA 2024 Exascale Paper](https://journals.sagepub.com/doi/10.1177/10943420241261960)
- [GPU Block Tridiagonal Cholesky on RTX 5090 (arXiv:2601.03754)](https://arxiv.org/abs/2601.03754)
- [CUDA Cooperative Groups](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html)
- [Inter-Block GPU Communication (Xiao et al.)](https://synergy.cs.vt.edu/pubs/papers/xiao-ipdps2010-gpusync.pdf)
- [GPU Sync Methods Study](https://arxiv.org/pdf/2004.05371)
- [Persistent Threads GPU Programming](https://escholarship.org/content/qt3j76d3td/qt3j76d3td_noSplash_1b206c94eb21559ac9ee806431718cdb.pdf)
- [NVIDIA Cooperative Groups Blog](https://developer.nvidia.com/blog/cooperative-groups/)
- [CUDA Occupancy API](https://developer.nvidia.com/blog/cuda-pro-tip-occupancy-api-simplifies-launch-configuration/)
