# MAGMA Native GPU LU Kernel Internals — Deep Dive

**Source:** MAGMA source code analysis (icl-utk-edu/magma), cuSOLVERDx samples (NVIDIA/CUDALibrarySamples)
**Relevant to:** numerical/ worker (LU factorization / getrf)
**Worker's current problem:** Building monolithic LU kernel for N=4096 on sm_120. Need detailed understanding of how MAGMA's fused panel kernel works internally, especially device-side pivoting, row swap mechanics, and inter-block synchronization.

---

## 1. MAGMA zgetf2_native_kernel — The Fused Panel Kernel

**Source:** `magmablas/zgetf2_native_kernel.cu` (icl-utk-edu/magma)

### Architecture

The kernel factorizes an M x N panel entirely on GPU in a SINGLE kernel launch. Grid has N blocks (one per column), each block has TX threads (warp-aligned, set by `ZGETF2_FUSED_NTH`).

**Template parameters:**
- `TX` = thread block width (multiple of 32)
- `NPAGES` = ceil(M / TX) = how many elements each thread holds vertically

### The "Pages" Concept (Register-Resident Column)

Each thread holds NPAGES elements of its column in registers:
```
rA[0] = A[bx * ldda + tx]              // rows 0 to TX-1
rA[1] = A[bx * ldda + 1*TX + tx]       // rows TX to 2*TX-1
...
rA[NPAGES-1] = A[bx * ldda + (NPAGES-1)*TX + tx]
```

The entire M x N panel lives in registers across all N blocks. This eliminates global memory traffic during the column-by-column factorization loop.

**Register budget for our case (N=4096, NB=64):**
- If TX=256, NPAGES = ceil(4096/256) = 16
- Each thread holds 16 float values = 16 registers for data
- Plus control variables (~20 regs)
- Total: ~36 regs/thread = well within budget, 6+ blocks/SM possible

### Column-by-Column Loop

```
for (int i = 0; i < n; i++):
    // Phase 1: Block i finds the pivot (blocks j != i skip this)
    if (bx == i):
        // Each thread computes |rA[page]| for its elements
        // Local max tracked across all NPAGES pages
        // Shared memory reduction: all threads write to sx[], sabs[], smax_id[]
        // First warp (tx < 32) does final warp-level reduction
        // Thread 0 broadcasts pivot location max_id and value rx_max

    // Phase 2: Row swap (ALL blocks participate)
    // If max_id != i, swap row i with row max_id:
    //   - Thread owning max_id writes its rA value to shared memory sx[tx]
    //   - Thread owning row i reads from sx[max_id % TX] and swaps
    //   - Synchronize between steps

    // Phase 3: Scale and rank-1 update
    // Block i: rA[page] *= 1/pivot (for rows below diagonal)
    //          then writes column back to global memory
    //          then sets update_flag[i] = 1 via atomic exchange

    // Other blocks (bx > i):
    //   spin-wait: while(update_flag[i] == 0) { flag = update_flag[i]; }
    //   then: rA[page] -= A[i*ldda + page*TX + tx] * scaled_pivot_element
```

### Inter-Block Synchronization via update_flag

This is the critical mechanism. CUDA has no built-in cross-block barrier, so MAGMA uses a global memory flag array:

1. `zgetf2_native_init_kernel` zeroes out `update_flag[0..n-1]`
2. Block i completes scaling column i, writes to global memory
3. Block i does `atomicExch(&update_flag[i], 1)` to signal
4. Blocks j > i spin-wait on `update_flag[i]` until it becomes 1
5. Then blocks j > i read the scaled column from global memory and do rank-1 update

**This is a spin-lock pattern.** It works because MAGMA forces one block per SM via large shared memory allocation (75% of device max), ensuring all blocks make progress.

### Shared Memory Layout

```
__shared__ magmaDoubleComplex sx[TX];    // pivot value exchange
__shared__ double sabs[TX];               // absolute values for argmax
__shared__ int smax_id[TX];               // pivot row indices
__shared__ magmaDoubleComplex sreg;       // broadcast pivot element
```

Total: ~TX * 28 + 16 bytes. For TX=256: ~7.2 KB. The rest of shared memory is deliberately wasted to force one block per SM occupancy.

### Panel Width Constraint

The kernel is templated on N (column count) with a switch/case dispatcher. MAGMA supports up to N=53 for complex double, larger for float. The `ZGETF2_FUSED_MAX_M` is 7168 (max panel height).

---

## 2. MAGMA getf2_native — Blocked and Recursive Variants

**Source:** `src/zgetf2_native.cpp` (icl-utk-edu/magma)

MAGMA's native panel factorization has TWO variants selected at runtime:

### Blocked Variant (for m > FUSED_MAX_M or older GPUs)

Uses inner block size IB = BATF2_NB = 8.

```
for j = 0 to min_mn step nb:
    for step = 0 to ib:
        izamax_native()           // find pivot in column
        zswap_native()            // swap pivot row across ALL columns
        zscal_zgeru_native()      // scale + rank-1 update
    zgetf2trsm_2d_native()        // triangular solve on remaining cols
    zgemm()                       // rank-ib update on trailing matrix
```

Each inner step is a separate kernel launch (izamax, swap, scal+ger). This is the multi-kernel approach — slower but handles arbitrarily large panels.

### Recursive Variant (for m <= FUSED_MAX_M, GPU arch >= 300)

Uses the fused kernel above plus recursive splitting:

```
zgetf2_native_recursive(m, n):
    if n <= nb:
        magma_zgetf2_native_fused(m, n, ...)  // single fused kernel
    else:
        n1 = n/2, n2 = n - n1
        zgetf2_native_recursive(m, n1, ...)    // factor left half
        zlaswp_native(...)                      // apply pivots to right half
        zgemm(...)                              // trailing update
        zgetf2_native_recursive(m-n1, n2, ...)  // factor right half
        adjust_ipiv(dipiv+n1, n2, n1)           // offset pivot indices
```

Uses dual CUDA queues (queue_0 for panels, queue_1 for updates) with event synchronization for overlap.

---

## 3. MAGMA Block Sizes for Native LU

**Source:** `control/get_nb.cpp` (icl-utk-edu/magma)

For `arch >= 800` (which includes sm_120):

| Precision | Matrix Size | Panel Width (nb) |
|-----------|-------------|-------------------|
| float | all | **512** |
| double | N <= 7000 | **256** |
| double | N > 7000 | **512** |

These are MUCH larger than the Cholesky panel widths (NB=64). The reason: LU's trailing update is a GEMM (not SYRK), so larger panels mean larger, more efficient GEMM calls.

For the inner panel sub-blocking: **IB = BATF2_NB = 8** (compile-time constant).

**Implication for our N=4096 FP32 case:** MAGMA would use nb=512. With N=4096, that's only 8 outer iterations. Each panel factorization handles a 4096 x 512 sub-matrix (shrinking as k increases).

---

## 4. MAGMA Small Batched GESV — Register-Only LU

**Source:** `magmablas/zgesv_batched_small.cu` (icl-utk-edu/magma)

For very small matrices (N <= 53 complex or ~60 float), MAGMA has a register-only fused LU + solve kernel:

### The rowid Trick (Virtual Pivoting)

Instead of actually swapping rows in memory, each thread tracks which logical row it "owns" via a `rowid` variable:

```
int rowid = tx;  // initially, thread tx owns row tx

for each pivot column i:
    // Find pivot (max |A[j,i]| for j >= i)
    // If max is at thread with rowid == max_id:
    //   Thread with rowid == max_id sets rowid = i
    //   Thread with rowid == i sets rowid = max_id
    //   (No actual data movement!)

    // Scale: if(rowid > i) rA[i] *= (1/pivot)
    // Rank-1: rA[j] -= rA[i] * sA(i,j)  for j > i
```

At the end, each thread writes its row to `A[rowid, :]` — the permutation is applied only during the final write-back.

**This eliminates ALL row swap overhead.** No shared memory staging, no register-to-register swaps. The pivot permutation is tracked as metadata only.

### Thread/Block Config

- `thread_x = n` (one thread per row)
- Grid: one block per batch item
- For n <= 32: all data in registers (warp-level syncs only)
- For n > 32: shared memory for cross-warp communication

### Applicability to Our Case

The rowid trick works for the panel factorization within a larger blocked LU. If the panel has M rows and we assign one thread per row (up to 1024 threads), each thread holds NB register values. The rowid trick eliminates row swaps during panel factorization. At write-back, threads write their row to the permuted position.

---

## 5. cuSOLVERDx GETRF — Device-Side API

**Source:** NVIDIA/CUDALibrarySamples, MathDx/cuSolverDx/01_Linear_Solve/

### Available Variants

- `getrf_wo_pivot.cu` — LU without pivoting (60x64 matrix, 256 threads)
- `getrf_partial_pivot.cu` — LU with partial pivoting (48x32 matrix, 33 threads)
- `gesv_batched_wo_pivot.cu` — batched linear solve via LU (no pivot)
- `gesv_batched_partial_pivot.cu` — batched linear solve via LU (partial pivot)

### Single-Kernel Pattern

```cpp
kernel<Solver><<<1, Solver::block_dim, Solver::shared_memory_size, stream>>>(...)

// Inside kernel:
__shared__ __align__(16) char smem[Solver::shared_memory_size];
auto [As, ipivs] = cusolverdx::shared_memory::slice<DataType, int>(smem);

// Load matrix A from global to shared memory
// ...

Solver().execute(As, ipivs, info);  // entire factorization in shared memory

// Store results back to global memory
```

### Key Constraints

- Data must be in shared memory before calling `execute()`
- Matrix size limited by shared memory (99KB on sm_120 = ~157x157 float)
- Thread block config determined by Solver type at compile-time
- **No composition with cuBLASDx GEMM in these examples** — the factorization is self-contained

### Template Composition

```cpp
using Solver = decltype(
    Size<48, 32>()
    + Precision<float>()
    + Type<type::complex>()
    + Function<getrf_partial_pivot>()
    + SM<Arch>()
    + BlockDim<33, 1, 1>()
    + Arrangement<arrangement::col_major>()
);
```

### MathDx Not Installed

MathDx (cuBLASDx + cuSOLVERDx) is NOT currently installed at `/usr/local/cuda-13`. It would need to be downloaded from https://developer.nvidia.com/mathdx and integrated into the build system.

---

## 6. Synthesis: What the Worker Should Know

### cuSOLVER's Internal Kernel (from Cholesky nsys profiling)

The Cholesky worker found cuSOLVER launches `getrf_wo_pivot_params_<float, 0, 256, 1, 64, 64, 68>` for potrf. Note the name contains "getrf" — cuSOLVER internally implements Cholesky as a special case of LU without pivoting. For actual LU with pivoting (dgetrf/sgetrf), expect a similar monolithic kernel pattern but with pivot-related template parameters.

Template parameter meanings (likely):
- `float` = precision
- `0` = some algorithm variant flag
- `256` = threads per block
- `1` = blocks in grid
- `64, 64` = tile dimensions (NB x NB)
- `68` = some internal parameter (shared memory alignment?)

### The Multi-Block vs Single-Block Question

**Single block (cuSOLVER/cuSOLVERDx pattern):**
- One block, 256 threads, processes entire matrix
- Panel factorization: sub-blocked in shared memory
- Trailing update: device-side GEMM within the same kernel
- No inter-block synchronization needed
- Limited to 256 threads = limited GEMM parallelism

**Multi-block (MAGMA native pattern):**
- N blocks for N-column panel, each handles one column
- Inter-block sync via spin-wait on global memory flags
- More parallelism for trailing updates
- But spin-wait wastes cycles

**For N=4096 on sm_120 (170 SMs):**
The single-block approach leaves 169 SMs idle during panel factorization. But the panel is O(N*NB^2) while trailing GEMM is O(N^2*NB). For large N, even with one SM doing the panel, the GEMM dominates total time. cuSOLVER's approach (single block doing everything) works because:
1. Panel factorization uses tensor cores for sub-panel TRSM/GEMM
2. Trailing GEMM is done by the same block, streaming tiles through shmem
3. No kernel launch overhead between phases

### Recommended Starting Architecture

```
Option A: cuSOLVERDx-based (if MathDx can be installed)
  - Use cuSOLVERDx getrf_partial_pivot for panel factorization
  - Use cuBLASDx GEMM for trailing updates
  - Compose in a single kernel (blocked_potrf pattern)
  - Advantage: proven device-side factorization code
  - Risk: MathDx not installed, may have bugs on sm_120

Option B: Custom MAGMA-style (recommended)
  - Write custom fused panel kernel (register-resident columns, rowid trick)
  - Use our existing BF16 mma.sync GEMM for trailing updates
  - Single persistent kernel with cooperative groups for multi-block coordination
  - Advantage: full control, can use BF16 tensor cores, no dependency on MathDx
  - Risk: more engineering effort

Option C: Hybrid blocked (quick baseline)
  - cuSOLVER/cuBLAS calls for panel + GEMM
  - CUDA Graph capture for launch overhead reduction
  - Expected: ~0.5x cuSOLVER (same as Cholesky baseline)
  - Purpose: establish baseline and profiling data
```

### Critical Path Items

1. **Profile cuSOLVER sgetrf N=4096** with nsys to see the actual kernel name, thread config, and whether it's truly monolithic or uses multiple kernel launches
2. **Measure panel vs trailing GEMM time** to understand where the bottleneck is
3. **Test cuSOLVERDx getrf_partial_pivot** on sm_120 if MathDx can be obtained
4. **Implement rowid-based panel kernel** as described above, verify correctness
5. **LASWP (row swaps) is bandwidth-bound** — this is where N=4096 pays a cost that smaller matrices avoid. Measure separately.

---

## Sources

- MAGMA source: https://github.com/icl-utk-edu/magma (BSD-3-Clause)
  - `magmablas/zgetf2_native_kernel.cu` — fused panel kernel
  - `magmablas/zgesv_batched_small.cu` — register-only small LU
  - `src/zgetf2_native.cpp` — blocked/recursive dispatch
  - `control/get_nb.cpp` — block size tuning
  - `control/batched_kernel_param.h` — constants (BATF2_NB=8, FUSED_MAX_M=7168)
- cuSOLVERDx examples: https://github.com/NVIDIA/CUDALibrarySamples/tree/main/MathDx/cuSolverDx
- MathDx download: https://developer.nvidia.com/mathdx
