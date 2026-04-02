# Batched BLAS1: Strided Batched Reductions and Single-Kernel Convergence Tests

**Sources:**
- [cuBLAS Strided Batched Matrix Multiply (NVIDIA Blog)](https://developer.nvidia.com/blog/cublas-strided-batched-matrix-multiply/)
- [Batched sparse iterative solvers on GPU (Anzt et al., ICL/UTK 2022)](https://icl.utk.edu/files/publications/2022/icl-utk-1608-2022.pdf)
- [Preconditioners for Batched Iterative Linear Solvers on GPUs (2022)](https://link.springer.com/chapter/10.1007/978-3-031-23606-8_3)
- [NVIDIA Forums: batched dot products on L4](https://forums.developer.nvidia.com/t/how-to-get-the-most-dot-products-of-batched-vectors-out-of-l4-gpu/333499)
- [Dot product outperforming cuBLAS (Avci, Medium)](https://emre-avci.medium.com/dot-product-in-cuda-c-which-might-outperform-cublas-t-dot-732047aa5ec5)
- [cuBLAS 13.2 documentation](https://docs.nvidia.com/cuda/cublas/)

**Relevant to:** linalg worker
**Worker's current problem:** Batched BLAS1 ops for iterative solver convergence. Individual DOT (1.48x) and NRM2 ops are fast, but iterative solvers need multiple reductions per iteration with host synchronization between each.
**Supplements:** `linalg_batched_blas1_iterative_solvers.md` (covers pipelined CG, MAGMA merge, Ginkgo patterns)

---

## What's New Here (vs. existing brief)

The existing brief covers pipelined solver algorithms (Chronopoulos-Gear CG, MAGMA kernel merge, Ginkgo batch::MultiVector). This brief adds:

1. **Strided batched layout** for avoiding pointer-array overhead
2. **Single-kernel multi-reduction** pattern for computing DOT + NRM2 together
3. **Device-side convergence test** to avoid host round-trip for scalar comparison

---

## Key Technique 1: Strided Batched Layout

cuBLAS's `cublasSgemmStridedBatched` avoids the pointer-to-pointer overhead of `cublasSgemmBatched`. The same principle applies to BLAS1:

**Problem with pointer-batched approach:**
```
// Need to allocate + transfer array of pointers to GPU
float** d_x_ptrs;  // GPU array of pointers
float** d_y_ptrs;
// Overhead: cudaMalloc + cudaMemcpy for pointer arrays each iteration
```

**Strided approach (zero overhead):**
```cuda
// All vectors packed contiguously with known stride
// x[batch_i] starts at x_base + batch_i * stride_x
// No pointer array needed -- just arithmetic
__global__ void batched_dot_strided(
    const float* x_base, int stride_x,
    const float* y_base, int stride_y,
    float* results, int n, int batch_count
) {
    int batch_id = blockIdx.y;  // one Y-block per batch
    const float* x = x_base + batch_id * stride_x;
    const float* y = y_base + batch_id * stride_y;
    // ... standard dot product for this batch ...
}
```

Use grid dimensions `(num_blocks_per_vector, batch_count, 1)` to dispatch all batches in a single launch.

---

## Key Technique 2: Single-Kernel Multi-Reduction

In CG iteration, you need `dot(r, z)` and `nrm2(r)` -- both touch vector `r`. Fuse them:

```cuda
__global__ void fused_dot_nrm2(
    const float* r, const float* z,
    float* dot_result, float* nrm2_result,
    int n
) {
    float dot_sum = 0.0f;
    float nrm2_sum = 0.0f;

    // Grid-stride loop with float4
    const float4* r4 = reinterpret_cast<const float4*>(r);
    const float4* z4 = reinterpret_cast<const float4*>(z);
    int n4 = n / 4;
    for (int i = tid; i < n4; i += stride) {
        float4 rv = r4[i];
        float4 zv = z4[i];
        dot_sum  += rv.x * zv.x + rv.y * zv.y + rv.z * zv.z + rv.w * zv.w;
        nrm2_sum += rv.x * rv.x + rv.y * rv.y + rv.z * rv.z + rv.w * rv.w;
    }

    // Warp reduction for BOTH values simultaneously
    for (int offset = 16; offset > 0; offset >>= 1) {
        dot_sum  += __shfl_down_sync(0xffffffff, dot_sum, offset);
        nrm2_sum += __shfl_down_sync(0xffffffff, nrm2_sum, offset);
    }

    // Block reduction + atomicAdd for both
    // ...
    if (threadIdx.x == 0) {
        atomicAdd(dot_result, dot_sum);
        atomicAdd(nrm2_result, nrm2_sum);
    }
}
```

This reads `r` once from DRAM, computing both reductions from the same data. Since both ops are memory-bound, fusing them nearly doubles effective throughput vs. two separate kernels.

---

## Key Technique 3: Device-Side Convergence Test

The iterative solver pattern is:
```
while (nrm2(residual) > tolerance):
    // ... CG/GMRES iteration ...
```

Currently this requires: launch NRM2 kernel -> copy scalar to host -> host comparison -> launch next iteration. The host round-trip is ~5-10 us.

**Device-side convergence flag:**
```cuda
// In the reduction kernel, after computing nrm2:
__global__ void nrm2_with_convergence(
    const float* x, float* nrm2_result,
    int* converged_flag, float tolerance, int n
) {
    // ... standard nrm2 reduction ...

    // Last block to finish writes the final result
    if (is_last_block) {
        float norm = sqrtf(*nrm2_result);
        *nrm2_result = norm;
        // Device-side convergence test -- no host round-trip
        *converged_flag = (norm <= tolerance) ? 1 : 0;
    }
}
```

The host checks `converged_flag` via `cudaMemcpyAsync` overlapped with the next iteration's kernel launch. Or use a persistent kernel that loops internally until convergence:

```cuda
__global__ void persistent_cg_iteration(
    /* solver state in global memory */
    int* iteration_count, int* converged, float tolerance
) {
    while (!*converged) {
        // SpMV phase (cooperative across grid)
        // Fused DOT+NRM2 reduction
        // AXPY updates
        // Convergence check (device-side)
        __grid_sync();  // cooperative groups grid sync
        atomicAdd(iteration_count, 1);
    }
}
```

This eliminates ALL host round-trips during the solve. Launch with `cudaLaunchCooperativeKernel` for grid-wide sync.

---

## Caveats

- **Strided layout** requires that solver vectors are packed contiguously. If the solver allocates vectors independently (e.g., via torch.empty), they won't be strided. Need to allocate a single buffer and slice it.
- **Persistent kernel** with cooperative groups requires occupancy planning -- the kernel must fit in available SMs without over-subscribing. On RTX 5090 with 170 SMs, this is feasible but register pressure must be managed.
- **Device-side convergence** means the iteration count is only known after the kernel completes. If you need per-iteration monitoring (for debugging), use the non-persistent approach.
- **atomicAdd for multi-block reduction** introduces non-determinism. For reproducible convergence, use a two-pass approach (partial sums -> final reduction) instead of atomicAdd.
