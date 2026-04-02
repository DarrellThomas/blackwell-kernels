# cuSOLVERDx v0.2.1: sm_120 Support Confirmed + Critical Compiler Bug Warning

**Source:** https://docs.nvidia.com/cuda/cusolverdx/release_notes.html
**Relevant to:** numerical worker (Cholesky monolithic kernel)
**Worker's current problem:** Building monolithic Cholesky using cuSolverDx potrf/trsm on sm_120 with CUDA 13.0.

## What This Is

cuSOLVERDx release notes confirm sm_120 support and document a critical compiler bug that the worker MUST know about.

## Why It Matters for Us

The worker plans to use cuSolverDx for device-side potrf and trsm in the monolithic kernel. This brief confirms support and warns about a specific bug to avoid.

## Key Facts

### sm_120 Support Timeline

| Version | Change |
|---------|--------|
| v0.2.0 | Added Blackwell architectures sm_100, sm_101, **sm_120**; experimental sm_103, sm_121 |
| v0.2.1 | Added support for CUDA 13.0 |
| v0.3.0 | Current release (MathDx 25.12.0) |

### Available Functions (Relevant to Cholesky)

- **potrf** -- Cholesky factorization (SPD matrix)
- **trsm** -- Triangular system solve
- **posv** -- Combined Cholesky factorization + solve (potrf + potrs)
- **potrs** -- Cholesky-based solve (given pre-computed L)
- **getrf** -- LU factorization (with/without pivoting)
- **gesv** -- LU-based solve
- **gesv_no_pivot** -- LU solve without pivoting
- **geqrf** -- QR factorization
- **heev** -- Eigenvalue solver (symmetric matrices)
- **gesvd** -- SVD

### CRITICAL COMPILER BUG (NVBUG 5288270)

**Affects: CUDA 12.8, 12.9, AND 13.0** (all versions the worker might use)

**Condition:** Using `gesv_no_pivot` function when SM is 120 AND type is real (i.e., float, not complex).

**Impact:** "CUDA 12.8, 12.9 and 13.0 could miscompile kernels using gesv_no_pivot function."

**Workarounds:**
1. Define `CUSOLVERDX_IGNORE_NVBUG_5288270_ASSERT` to suppress the assertion
2. Add `-Xptxas -O1` compiler flag to reduce optimization level

**Does this affect potrf/trsm?** The bug specifically targets `gesv_no_pivot`, NOT potrf or trsm. However, the miscompilation mechanism (ptxas optimization level) could theoretically affect other functions. **Recommendation:** Add `-Xptxas -O1` as a precaution when using ANY cuSolverDx function on sm_120 with CUDA 13.0, and test results against cuSOLVER reference.

### Breaking API Change (v0.2.0)

The `FillMode` operator for Cholesky functions (potrf, posv) changed from implicit default to **explicit required specification**:

```cuda
// OLD (v0.1.x): FillMode was optional, defaulted to lower
auto POTRF = cusolverdx::potrf(Size<N>() + SM<1200>() + Type<type::real>());

// NEW (v0.2.0+): FillMode MUST be specified
auto POTRF = cusolverdx::potrf(Size<N>() + SM<1200>() + Type<type::real>()
                              + FillMode<fill_mode::lower>());
```

### Precision Support

cuSolverDx supports `type::real` (float/double) and `type::complex` (complex float/double). BF16/FP16 are NOT supported for factorization functions -- they are FP32/FP64 only. This is fine for our use case: potrf and trsm need full FP32 precision anyway; only the SYRK/GEMM update benefits from BF16 tensor cores.

### Shared Memory and Block Dimensions

- Shared memory size: obtained via `Solver::shared_memory_size`
- Block dimensions: obtained via `Solver::block_dim` (library-chosen optimal)
- Custom block dims: possible via `BlockDim` operator but NOT recommended

### Execution Model

```cuda
// Inside kernel:
__shared__ char smem[Solver::shared_memory_size];
auto* A_smem = reinterpret_cast<float*>(smem);
// ... load data from global to shared memory ...
Solver().execute(A_smem, info);    // in-place factorization
// ... write results back to global memory ...
```

All data must be in shared memory before calling execute(). The user manages global-to-shared transfers.

## Caveats

1. **No tensor core acceleration mentioned.** cuSolverDx documentation doesn't mention tensor cores for potrf/trsm. These likely run on CUDA cores (FP32 FMA), which is fine since panel factorization is not the bottleneck.

2. **Maximum matrix size limited by shared memory.** For FP32 potrf of N=64: 64*64*4 = 16KB. For N=128: 64KB. For N=256: 256KB (exceeds sm_120's 99KB limit). The blocked_potrf example handles larger matrices by streaming blocks through shared memory.

3. **The blocked_potrf example is the closest reference.** It combines cuSolverDx potrf+trsm with cuBLASDx GEMM in a single kernel. Source in CUDALibrarySamples/MathDx/cuSolverDx/.

## Sources

- [cuSOLVERDx Release Notes](https://docs.nvidia.com/cuda/cusolverdx/release_notes.html) -- sm_120 support, compiler bug, API changes
- [cuSOLVERDx POTRF Documentation](https://docs.nvidia.com/cuda/cusolverdx/get_started/functions/potrf.html) -- API reference
- [cuSOLVERDx Introduction](https://docs.nvidia.com/cuda/cusolverdx/get_started/introduction.html) -- Execution model
- [cuSOLVERDx Blocked Potrf Example](https://docs.nvidia.com/cuda/cusolverdx/examples/advanced_example.html) -- Reference implementation
