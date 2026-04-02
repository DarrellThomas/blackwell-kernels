# Dynamic Parallelism (CDP2) for LU Factorization: Not Recommended

**Source:** https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/dynamic-parallelism.html
**Source:** https://ieeexplore.ieee.org/document/8430608 (Dynamic GPU Parallel Sparse LU, 2018)
**Source:** https://www.sciencedirect.com/science/article/pii/S2590123025028634 (GPU Dynamic Parallelism SPH, 2025)
**Relevant to:** LU worker
**Worker's current problem:** Evaluating whether CDP2 (CUDA Dynamic Parallelism v2) could be used to launch the trailing GEMM from within the panel kernel, avoiding host-side kernel launch overhead.

## What This Is

CUDA Dynamic Parallelism (CDP2, default since CUDA 12.0) allows device-side
kernel launches. The idea: the panel factorization kernel could launch the
trailing GEMM as a child kernel from the GPU, avoiding host roundtrip latency.

## Why CDP2 Is Tempting for LU

Traditional blocked LU requires host orchestration:
```
for k = 0 to N/NB:
    panel_factorize(panel_k)       // kernel launch
    apply_pivots(A, ipiv_k)        // kernel launch
    trailing_gemm(L_k, U_k, A)    // kernel launch (the big one)
```

Each iteration requires 3 kernel launches + host synchronization. At N=4096
with NB=64, that's 64 * 3 = 192 kernel launches. Even with CUDA Graphs, the
per-launch overhead (~2-5 us) adds up.

CDP2 would allow:
```
__global__ void lu_monolithic(float* A, int N, int NB) {
    for (int k = 0; k < N/NB; k++) {
        panel_factorize_device(A, k, NB);
        __syncthreads();
        // Launch trailing GEMM as child kernel
        dim3 grid(trailing_tiles_m, trailing_tiles_n);
        trailing_gemm<<<grid, 256>>>(A, k, NB);
        cudaDeviceSynchronize();  // Wait for child
    }
}
```

## Why CDP2 Is NOT Recommended

### 1. CDP2 Launch Overhead Is Still Significant

CDP2 improved over CDP1 but device-side kernel launches still have ~5-20 us
overhead. For 64 iterations with 1-2 child launches each, that's ~300-1200 us
of launch overhead -- comparable to cooperative grid sync overhead but with
more complexity.

### 2. sm_120 Support Confirmed but Limited

CDP2 is supported on CC 7.0+ (sm_120 is CC 12.0, so supported). However:
- "CDP2 is the only version of CUDA dynamic parallelism available on devices
  of CC 9.0 and higher"
- The old CDP1 tail launch optimization is gone
- Device-side memory allocation for child kernels adds pressure

### 3. Cooperative Groups Is Strictly Better for This Use Case

Cooperative launch with `grid.sync()` achieves the same goal (all blocks
participate in trailing GEMM) without the overhead of child kernel launches:

| Factor | CDP2 | Cooperative Groups |
|--------|------|--------------------|
| Launch overhead | 5-20 us per child | 1-5 us per grid.sync() |
| Memory management | Child needs workspace | Shared allocation |
| Synchronization | cudaDeviceSynchronize | grid.sync() |
| Code complexity | Medium (nested kernels) | Low (single kernel) |
| Block scheduling | New grid, may not co-schedule | All blocks already resident |

### 4. cuSOLVERDx/cuBLASDx Eliminate the Need

The MathDx device-side APIs (cuSOLVERDx getrf + cuBLASDx GEMM) provide
device-side factorization primitives callable from within a single kernel.
This is cleaner than CDP2 because:
- No child kernel launch overhead
- All operations share the same thread block and shared memory
- No device-side memory allocation needed

### 5. Sparse LU Research Not Applicable

The main CDP research for LU (IEEE 2018, GPU sparse LU) targets sparse matrix
factorization where the task graph has irregular parallelism. Dense LU has
regular structure that cooperative groups handle better.

## What CDP2 IS Good For (Not Our Case)

- Irregular workloads with unpredictable parallelism (adaptive mesh, tree search)
- Problems where different iterations need different grid configurations
- Recursive algorithms where the subproblem size varies

Dense blocked LU has regular, predictable structure. Cooperative groups or
cuSOLVERDx composition is the right tool.

## Recommendation

**Do not use CDP2 for the monolithic LU kernel.** Use either:
1. **Cooperative groups** (single kernel, grid.sync between phases), or
2. **cuSOLVERDx + cuBLASDx composition** (single kernel, device-side calls)

Both approaches eliminate host-side launch overhead without CDP2's drawbacks.

## Caveats

1. CDP2 might be useful later for a multi-precision adaptive factorization
   where the precision strategy changes based on pivot quality (launching
   different child kernels for different precision paths). But this is
   far-future territory.
