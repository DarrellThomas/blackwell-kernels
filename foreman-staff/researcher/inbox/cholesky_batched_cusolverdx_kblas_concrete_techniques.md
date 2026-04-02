# Batched Small Cholesky: cuSolverDx API + KBLAS Register Kernels — Concrete Implementation Guide

**Source:** NVIDIA CUDALibrarySamples (cuSolverDx examples), KBLAS-GPU source code, cuSolverDx 0.3.0 docs
**Relevant to:** numerical worker (Cholesky)
**Worker's current problem:** 0.55x cuSOLVER at N=4096 (monolithic kernel gap). Worker considering pivot to batched small Cholesky (N=32-64) where cuSOLVER is weak.

---

## 1. cuSolverDx: Device-Side POTRF with Concrete API (THE BIG FINDING)

cuSolverDx v0.3.0 (part of MathDx 25.12.1) provides device-side Cholesky that
runs INSIDE your CUDA kernel. It supports sm_120 explicitly. This is the most
important finding in this brief.

### Confirmed sm_120 Support

From release notes (v0.2.0+): "Blackwell architectures sm_100, sm_101, sm_120"
with experimental sm_103 and sm_121. **Our RTX 5090 (sm_120) is fully supported.**

### The API Pattern (from actual NVIDIA example code)

The API is compile-time configured via C++ operator overloading:

```cpp
#include <cusolverdx.hpp>

// Define the solver at compile time
using Solver = decltype(
    cusolverdx::Size<32, 32>()           // Matrix dimensions (compile-time!)
  + cusolverdx::Precision<double>()      // Precision
  + cusolverdx::Type<cusolverdx::type::real>()  // Real or complex
  + cusolverdx::Function<cusolverdx::potrf>()   // Operation = Cholesky
  + cusolverdx::FillMode<cusolverdx::fill_mode::upper>()
  + cusolverdx::Block()                  // Thread-block level execution
  + cusolverdx::BlockDim<256>()          // Threads per block
  + cusolverdx::SM<120>()               // Target architecture
  + cusolverdx::LeadingDimension<33>()   // Optional padding for bank conflicts
);

// Kernel uses shared memory
__global__ __launch_bounds__(Solver::max_threads_per_block)
void potrf_kernel(float* A, Solver::status_type* info) {
    extern __shared__ unsigned char shared_mem[];
    float* As = reinterpret_cast<float*>(shared_mem);

    // Load global -> shared (strided for coalescing)
    common::io<Solver>::load_a(A, Solver::m_size, As, Solver::lda);

    // Execute factorization IN SHARED MEMORY
    Solver().execute(As, info);

    // Store shared -> global
    common::io<Solver>::store_a(As, Solver::lda, A, Solver::m_size);
}

// Launch: shared memory size from Solver trait
cudaFuncSetAttribute(potrf_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                     Solver::shared_memory_size);
potrf_kernel<<<batch_count, Solver::block_dim, Solver::shared_memory_size>>>(d_A, d_info);
```

**Key traits exposed by the Solver type:**
- `Solver::shared_memory_size` — exact bytes needed
- `Solver::block_dim` — thread block dimensions
- `Solver::max_threads_per_block` — for launch bounds
- `Solver::lda` — padded leading dimension in shared memory
- `Solver::m_size`, `Solver::n_size` — matrix dimensions
- `Solver::batches_per_block` — suggested batches per block (for POSV)
- `Solver::suggested_block_dim` — auto-tuned block dim

### Batched Pattern (from POSV example)

For batched execution, cuSolverDx supports multiple matrices per block:

```cpp
using POSV = decltype(
    cusolverdx::Size<32, 32, 1>()  // m, n, nrhs
  + cusolverdx::Precision<double>()
  + cusolverdx::Function<cusolverdx::function::posv>()
  + cusolverdx::FillMode<cusolverdx::lower>()
  + cusolverdx::SM<120>() + cusolverdx::Block()
);

constexpr unsigned bpb = POSV::batches_per_block;  // auto-determined!

// Kernel processes bpb matrices per block
kernel<<<(batches + bpb - 1) / bpb, POSV::block_dim, POSV::shared_memory_size>>>(
    d_A, lda, d_B, ldb, d_info, batches);
```

The `batches_per_block` trait tells you how many small matrices to pack per
thread block for optimal throughput. This is exactly what we need.

### Blocked POTRF for Larger Matrices (from advanced example)

For matrices larger than what fits in shared memory, NVIDIA provides a blocked
composition using cuSolverDx + cuBLASDx primitives:

```cpp
// Compose POTRF + TRSM + GEMM for blocked Cholesky
using POTRF = decltype(cusolverdx::Function<cusolverdx::function::potrf>()
    + cusolverdx::Size<NB>() + cusolverdx::Precision<T>()
    + cusolverdx::Block() + cusolverdx::BlockDim<NT>() + cusolverdx::SM<Arch>()
    + cusolverdx::FillMode<cusolverdx::fill_mode::upper>());

using TRSM = decltype(cusolverdx::Function<cusolverdx::function::trsm>()
    + cusolverdx::Size<NB, NB, NB>() + cusolverdx::Precision<T>()
    + cusolverdx::Side<cusolverdx::side::left>()
    + cusolverdx::TransposeMode<cusolverdx::transpose::transposed>()
    + cusolverdx::Block() + cusolverdx::BlockDim<NT>() + cusolverdx::SM<Arch>());

using GEMM = decltype(cublasdx::Size<NB, NB, NB>() + cublasdx::Precision<T>()
    + cublasdx::Block() + cublasdx::BlockDim<NT>() + cublasdx::SM<Arch>());

// Single kernel, single CTA, left-looking blocked algorithm:
// Shared memory: 4 x NB x NB tiles (sA, sB, sC, sD)
for (int k = 0; k < n_tiles; ++k) {
    // Load diagonal tile, apply Schur complement from previous steps
    for (int i = 0; i < k; ++i) {
        GEMM().execute(T(-1.0), sC, sC, T(1.0), sA);  // SYRK update
    }
    POTRF().execute(sA, lds, sinfo);  // Factor diagonal block

    // Update panel tiles
    for (int j = k+1; j < n_tiles; ++j) {
        for (int i = 0; i < k; ++i) {
            GEMM().execute(T(-1.0), sC, sD, T(1.0), sB);  // GEMM update
        }
        TRSM().execute(sA, lds, sB, lds);  // Triangular solve
    }
}
```

**The blocked example uses NB=32, NT=128, N=512, batch=400.**
Shared memory = 4 * NB * NB * sizeof(double) + sizeof(int) = 32KB for double.

This is EXACTLY the monolithic single-kernel approach that cuSOLVER uses
internally, but now accessible via public API. For our N=32-64 batched case,
we don't even need the blocking — a single POTRF call per matrix suffices.

### Why cuSolverDx Bypasses Our TF32 MMA Blocker

cuSolverDx handles its own internal compute. It doesn't expose what MMA
instructions it uses. Since NVIDIA wrote it, they presumably either:
- Use a proprietary MMA variant for sm_120
- Use BF16 MMA internally with appropriate precision management
- Use a scalar FP32 path for small matrices

Either way, the TF32 B fragment broadcasting defect that blocks our custom
device-side GEMM is **not our problem** when using cuSolverDx.

---

## 2. KBLAS Register-Based Kernel Architecture (Concrete Source Code)

**Source:** https://github.com/ecrc/kblas-gpu/tree/master/src/batch_triangular

KBLAS provides the most detailed open-source reference for register-based
batched Cholesky. The source code reveals three distinct kernel paths:

### Path 1: Fixed-N Register Kernel (N=8, N=16)

```cpp
// kernel_potrf_U_registers_fixN<T, N, BS>
// One matrix per warp. Each thread holds one row in registers.
// N threads per matrix, multiple matrices per block via blockDim.y.

for (int j = 0; j < N; j++) {
    s = sqrt(shfl(rA[j], j, N));    // Broadcast diagonal via warp shuffle
    rA[j] /= s;                     // Scale column
    for (int i = 0; i < N; i++)
        if (j < i)
            rA[i] = FMA(rA[j], s, rA[i]);  // Rank-1 update
}
```

**No shared memory.** All data in registers. Communication via `__shfl_sync`.
Grid: `batch_count / blockDim.y` blocks. Multiple matrices per block.

### Path 2: Variable-N Register Kernel (N <= 32)

Same as fixed-N but with runtime size parameter. Constraint: `n <= TX` where
TX is the warp size (32). Loop bounds are runtime-checked.

### Path 3: Blocked Register Kernel (32 < N <= 64)

```cpp
// kernel_potrf_U_registers_varN_blocked_2
// Constraint: BS < n <= 2*BS where BS is typically 32
// Decomposes into four stages:
//   1. POTRF on A[0,0] (32x32, register-resident)
//   2. TRSM on A[1,0] (triangular solve)
//   3. SYRK on A[1,1] (symmetric rank-k update)
//   4. POTRF on A[1,1] (32x32, register-resident)
```

### KBLAS Driver: Size-Based Dispatch

```cpp
// For n <= 16: direct register kernel (fixed or variable N)
//   func_idx = 2 * (n > 8) + (nvar == 1)
//   Dimensions: 8x4 through 32x2 thread blocks
//
// For n > 16: recursive divide-and-conquer
//   Split into n1, n2 at register-friendly boundaries
//   Apply POTRF recursively, then TRSM and SYRK
```

**Block dimension tuning by precision:**
- Single precision: different thread configs than double
- Each size gets its own (blockDim.x, blockDim.y) pair

---

## 3. Recommended Strategy for Our Worker

Given these concrete findings, here is the priority-ordered approach:

### Option A: cuSolverDx (Highest Priority — Try First)

**Effort: LOW.** The API is clean and sm_120 is supported.

1. Install MathDx 25.12.1 (includes cuSolverDx 0.3.0)
2. Write a batched POTRF kernel using the pattern above
3. Template on N=32 and N=64
4. Use `Solver::batches_per_block` for automatic multi-matrix packing
5. Benchmark against `cusolverDnSpotrfBatched` for batch_size = 1K-100K

**Key advantage:** cuSolverDx is NVIDIA's own optimized code. It likely
already uses the best available approach for sm_120 internally. We get
NVIDIA-quality factorization with minimal effort.

**Potential issue:** Matrix size limits. cuSolverDx uses shared memory, and
the maximum supported N for a single-block POTRF may be limited. If N=64
doesn't fit, use the blocked composition pattern from the advanced example.

### Option B: Custom Register-Based Kernel (If cuSolverDx is insufficient)

**Effort: MEDIUM.** Follow the KBLAS pattern.

For N=32:
- 32 threads per matrix, all data in registers
- Warp shuffles for communication (no shared memory)
- Template-unrolled factorization loop
- Pack multiple matrices per block via blockDim.y
- Use `rsqrtf()` instead of `sqrtf()` for the diagonal

For N=64:
- Blocked 2x2 approach: two 32x32 register-resident sub-problems
- Or: 64 threads + shared memory for cross-warp communication
- Sub-blocked IB=16 panel within 64x64 (reuse existing panel kernel logic)

### Option C: Hybrid — cuSolverDx Panel + Custom Trailing Update

**Effort: MEDIUM-HIGH.** Use cuSolverDx for the panel POTRF and TRSM,
but write custom SYRK. This avoids the TF32 MMA issue for SYRK while
leveraging cuSolverDx for the sequential factorization steps.

---

## 4. MathDx Installation

```bash
# MathDx 25.12.1 includes cuSolverDx 0.3.0
# Download from: https://developer.nvidia.com/mathdx
# Supports CUDA 12.6.3+ and CUDA 13
# Header-only for device code: #include <cusolverdx.hpp>
# Also includes cuBLASDx for device-side GEMM/TRSM
```

**Build requirement:** The examples use CMake. cuSolverDx is header-only for
the device-side API (compile-time code generation via templates).

---

## 5. Performance Expectations

| Approach | N=32, batch=10K | N=64, batch=10K | Engineering Effort |
|----------|-----------------|-----------------|-------------------|
| cusolverDnSpotrfBatched | baseline | baseline | none |
| cuSolverDx batched | 3-8x (estimated) | 2-4x (estimated) | low |
| Custom register kernel | 4-10x (estimated) | 2-5x (estimated) | medium |
| KBLAS reference | 4-8x (published) | 2-4x (published) | n/a (reference) |

The estimates are based on MAGMA published results (6-11.8x on P100/V100 vs
cuBLAS) scaled conservatively for modern hardware where vendor libraries have
improved.

---

## References

- cuSolverDx 0.3.0 docs: https://docs.nvidia.com/cuda/cusolverdx/
- cuSolverDx POTRF example: https://github.com/NVIDIA/CUDALibrarySamples/blob/master/MathDx/cuSolverDx/01_Cholesky/potrf.cu
- cuSolverDx blocked POTRF: https://github.com/NVIDIA/CUDALibrarySamples/blob/master/MathDx/cuSolverDx/06_Advanced/blocked_potrf.cu
- cuSolverDx batched POSV: https://github.com/NVIDIA/CUDALibrarySamples/blob/master/MathDx/cuSolverDx/00_Introduction/posv_batched.cu
- KBLAS batched potrf kernels: https://github.com/ecrc/kblas-gpu/blob/master/src/batch_triangular/Xpotrf_batch_kernels.cuh
- KBLAS batched potrf drivers: https://github.com/ecrc/kblas-gpu/blob/master/src/batch_triangular/Xpotrf_batch_drivers.cuh
- MathDx download: https://developer.nvidia.com/mathdx
- cuSolverDx release notes (sm_120 confirmed): https://docs.nvidia.com/cuda/cusolverdx/release_notes.html
- MAGMA batched potrf API: https://icl.utk.edu/projectsfiles/magma/doxygen/group__magma__potrf__batched.html
- Dong et al., "A Fast Batched Cholesky Factorization on a GPU" (IEEE ICPP 2014): https://doi.org/10.1109/icpp.2014.52
- Haidar et al., "A Guide for Achieving High Performance with Very Small Matrices on GPU" (IEEE TPDS 2018): https://doi.org/10.1109/tpds.2017.2783929
- Kurzak et al., "Implementation and Tuning of Batched Cholesky Factorization and Solve for NVIDIA GPUs" (IEEE TPDS 2016): https://doi.org/10.1109/tpds.2015.2481890
- Charara et al., "Batched Triangular Dense Linear Algebra Kernels for Very Small Matrix Sizes on GPUs" (ACM TOMS 2019): https://doi.org/10.1145/3267101
- Lemaitre & Lacassagne, "Batched Cholesky factorization for tiny matrices" (DASIP 2016): https://doi.org/10.1109/dasip.2016.7853809
