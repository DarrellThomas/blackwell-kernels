# cuSolverDx: Device-Side LU Factorization for sm_120

**Source:** https://docs.nvidia.com/cuda/cusolverdx/get_started/getrf.html | https://docs.nvidia.com/cuda/cusolverdx/release_notes.html
**Relevant to:** lu worker, numerical worker (cholesky)
**Worker's current problem:** Multi-kernel blocked LU cannot beat cuSOLVER's monolithic single-kernel approach. Kernel launch overhead (~0.38ms from CUDA Graph) is the fundamental barrier. Worker needs a path to device-side factorization.

## What This Is

cuSolverDx (cuSOLVER Device Extensions) v0.3.0 is an NVIDIA library that enables
matrix factorization routines to be **called from within CUDA kernels as device
functions**. It supports sm_120 (RTX 5090) and includes LU factorization with
partial pivoting (`getrf_partial_pivot`) and without pivoting (`getrf_no_pivot`).

## Why It Matters for Us

The LU worker's strategy document identifies the monolithic kernel as "the only path
to beating cuSOLVER." The Cholesky worker proved this empirically — the multi-kernel
approach topped out at 0.55x cuSOLVER. The entire gap comes from kernel launch overhead.

cuSolverDx provides a potential shortcut: instead of writing a monolithic LU kernel
from scratch (which requires device-side GEMM, TRSM, and panel factorization all
in one kernel), the worker could:

1. Write a single kernel that calls cuSolverDx `getrf` for panel factorization
2. Use cuSolverDx `trsm` for triangular solves (also available device-side)
3. Use our own MMA-based GEMM for the trailing matrix update
4. Compose all of these in a single kernel launch → zero kernel launch overhead

This is exactly what cuSOLVER's own monolithic kernel does internally, but now
the building blocks are exposed via a public API.

## Key Technical Details

### Supported Operations (device-side)
- `getrf_partial_pivot` — LU factorization with partial pivoting (our primary need)
- `getrf_no_pivot` — LU without pivoting (less useful, numerical stability issues)
- `trsm` — Triangular solve (needed for LU trailing update)
- `potrf` — Cholesky factorization (relevant for numerical/cholesky worker)
- `geqrf` — QR factorization (relevant for future QR work)
- `gesv` — Combined factorize + solve

### Architecture Support
- sm_120 (RTX 5090): **SUPPORTED** as of v0.3.0
- Also supports sm_70 through sm_100+

### API Pattern
```cuda
// Device function — called from within your kernel
__device__ void execute(data_type* A, const unsigned int lda,
                        int* ipiv, status_type* info);
```

The key insight: this is a `__device__` function, not a kernel launch. You embed it
in YOUR kernel. No launch overhead. No host synchronization.

### Known Bug (CRITICAL)
CUDA 12.8-13.0 may miscompile `gesv_no_pivot` kernels with SM120 and real types
under high register pressure. Workarounds:
- Use `-Xptxas -O0` compilation flag
- Or define `CUSOLVERDX_IGNORE_NVBUG_5288270_ASSERT`
- This bug is for `gesv_no_pivot` specifically — `getrf_partial_pivot` may not be affected

### Composition Strategy for LU
```
Single kernel:
  for each panel column block:
    1. cuSolverDx::getrf on panel (device-side, no launch)
    2. cuSolverDx::trsm on U row (device-side, no launch)
    3. Our MMA-based GEMM on trailing submatrix (registers + smem)
    4. __syncthreads() between steps
```

This eliminates ALL kernel launch overhead while using NVIDIA's optimized panel
factorization. The trailing GEMM can use our proven BF16 MMA primitives.

## Caveats

1. **Matrix size limits unknown.** cuSolverDx may only support panel-sized matrices
   in device mode (e.g., N<=64 or N<=128). The documentation doesn't specify. The
   worker should test with NB=64 panel size first.

2. **Thread block constraints.** cuSolverDx may impose specific thread block dimensions.
   The worker needs to check if 128 or 256 threads are compatible.

3. **Shared memory usage.** cuSolverDx likely uses shared memory internally. The worker
   must account for this when budgeting the 99KB SM limit.

4. **Precision.** The docs mention real and complex types. FP32 should be available
   (matching cuSOLVER). BF16/FP8 support is unclear.

5. **Performance.** cuSolverDx is a library, not hand-optimized for our specific
   problem. It may not match cuSOLVER's internal monolithic kernel quality. But
   eliminating launch overhead alone could close the gap significantly.

## Recommendation

This should be investigated IMMEDIATELY. The worker should:
1. Install cuSolverDx v0.3.0 (check if included in CUDA 13 toolkit)
2. Write a minimal test: single kernel calling `getrf` on a 64x64 panel
3. Verify it compiles and produces correct results on sm_120
4. Benchmark the device-side call vs cuSOLVER host-side call
5. If viable, compose into the blocked LU loop as a single kernel
