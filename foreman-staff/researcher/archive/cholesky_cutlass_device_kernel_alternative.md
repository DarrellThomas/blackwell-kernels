# CUTLASS device_kernel: Alternative to cuBLASDx for Inlined GEMM

**Source:** https://github.com/NVIDIA/cutlass/discussions/985 | https://docs.nvidia.com/cutlass/media/docs/cpp/gemm_api_3x.html
**Relevant to:** cholesky worker (numerical/)
**Worker's current problem:** Needs device-callable GEMM for monolithic Cholesky. cuBLASDx is one path; CUTLASS is the other.

## What This Is

CUTLASS 3.x provides `cutlass::device_kernel<>()` which can be called from within another CUDA kernel to execute GEMM operations. This gives full control over the GEMM implementation (tile sizes, pipeline depth, precision) while running inside the Cholesky kernel context.

## Why It Matters for Us

If cuBLASDx doesn't work well on sm_120 (it's relatively new), CUTLASS is the proven fallback. The worker already knows CUTLASS-style patterns (ldmatrix, cp.async, swizzle, double-buffer) from the GEMM and attention kernels. CUTLASS device functions would let them compose these familiar building blocks into the Cholesky kernel.

## Key Technique

### CUTLASS 3.x approach:
```cpp
// Within your kernel:
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<...>;
typename GemmKernel::Params params = ...;  // manually constructed
cutlass::device_kernel<GemmKernel><<<grid, block, smem>>>(params);
```

### CUTLASS 2.x approach:
```cpp
using GemmKernel = cutlass::gemm::kernel::Gemm<...>;
cutlass::Kernel<GemmKernel><<<grid, block, smem>>>(params);
```

### Key requirements:
1. Construct `Params` manually (bypasses host-side utilities)
2. Launch grid matching CUTLASS's internal tiling
3. Ensure dynamic shared memory matches kernel requirements
4. Allocate and initialize any workspace

### Our advantage:
We already have hand-written mma.sync GEMM kernels. Instead of using CUTLASS, the worker could refactor our existing GEMM into a `__device__` function:

```cpp
// Convert our GEMM kernel to a device function
__device__ void device_gemm_tf32(
    float* C, const float* A, const float* B,
    int M, int N, int K, int lda, int ldb, int ldc,
    float* smem) {
    // Same mma.sync m16n8k8 tiling as our GEMM kernel
    // but callable from within the Cholesky kernel
}
```

This avoids the CUTLASS dependency entirely and reuses our battle-tested GEMM code.

## Caveats

- **CUTLASS device_kernel launches a sub-grid.** It's not truly inlined — it creates a child kernel launch (dynamic parallelism). This adds ~5-10 us per launch, which may negate the benefit for small trailing updates.
- **cuBLASDx is the simpler path.** It's designed specifically for device-side BLAS without dynamic parallelism. CUTLASS device_kernel is more of a workaround.
- **Our own device function is likely the best path.** The worker already has a TF32 GEMM kernel (or can adapt the BF16 kernel to use m16n8k8 MMA). Refactoring it to `__device__` avoids all external dependencies and gives full control.
- **sm_120 CUTLASS support.** CUTLASS 4.x has sm_120 support, but the device_kernel API may not be tested for this path. Verify before committing.
