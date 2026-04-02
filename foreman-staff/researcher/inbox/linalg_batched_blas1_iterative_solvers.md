# Batched BLAS1 Operations for Iterative Solvers on GPU

**Sources:**
- [Pipelined Iterative Solvers with Kernel Fusion for GPUs (Rupp et al., TU Wien)](https://www.iue.tuwien.ac.at/pdf/ib_2016/JB2016_rupp_1.pdf)
- [Systematic Fusion of CUDA Kernels for Iterative Sparse Linear System Solvers (Anzt, Dongarra)](https://link.springer.com/chapter/10.1007/978-3-662-48096-0_52)
- [PERKS: Persistent Kernels for Iterative Memory-bound GPU Applications](https://arxiv.org/abs/2204.02064)
- [Optimizing CUDA Code By Kernel Fusion - Application on BLAS](https://ar5iv.labs.arxiv.org/html/1305.1183)
- [Ginkgo: A Modern Linear Operator Algebra Framework (GitHub)](https://github.com/ginkgo-project/ginkgo)
- [Multi-GPU Communication Schemes for Iterative Solvers: When CPUs are Not in Charge (ICS 2023)](https://dl.acm.org/doi/10.1145/3577193.3593713)
- [CUB DeviceSegmentedReduce API](https://nvidia.github.io/cccl/cub/api/structcub_1_1DeviceSegmentedReduce.html)
- [CUDA Cooperative Groups (Lei Mao)](https://leimao.github.io/blog/CUDA-Cooperative-Groups/)
- [Communication-reduced CG Variants for GPU-accelerated Clusters (2025)](https://arxiv.org/html/2501.03743)
- [Two-Stage Block Orthogonalization for s-step GMRES (2024)](https://arxiv.org/html/2402.15033v1)
- [NVIDIA AmgX GPU Solver Library](https://github.com/NVIDIA/AMGX)
- [Performance engineering for tall & skinny matrix multiplication kernels on GPUs](https://journals.sagepub.com/doi/full/10.1177/1094342020965661)

**Relevant to:** linalg worker
**Worker's current problem:** Individual DOT/NRM2/AXPY ops are fast (DOT 1.48x, AXPY 1.41x cuBLAS), but iterative solvers need multiple reductions per iteration -- each currently a separate kernel launch with host synchronization.

## What This Is

Iterative Krylov solvers (CG, BiCGSTAB, GMRES) spend a significant fraction of their runtime on BLAS1 operations: dot products, norms, and vector updates. The bottleneck is not the arithmetic -- it is the kernel launch overhead and host-device synchronization for each reduction. A single CG iteration requires 2 dot products, 1 norm, 2 AXPYs, and 1 SpMV -- that is 6 separate kernel launches with 3 requiring host round-trips for scalar results. This document surveys how production libraries solve this and what patterns apply to our sm_120 setup.

## How Production Libraries Handle Batched Reductions

### MAGMA Sparse (ICL/UTK)

MAGMA's pipelined solver variants (CG, BiCGSTAB, GMRES) use **kernel merge** to fuse multiple BLAS1 ops into compound kernels:

- **Standard CG:** 6 kernels/iteration (SpMV, dot, dot, axpy, axpy, norm). Three reductions require host sync.
- **Pipelined CG:** The algorithm is rearranged (Chronopoulos-Gear reformulation) so that both dot products share the same vectors and can be computed in a **single merged dot kernel** (one launch, one reduction, two scalar results). This cuts from 3 global syncs to 1.
- **Kernel count reduction:** Systematic fusion reduces preconditioned CG from 10 to 5 kernels, BiCG from 13 to 5, BiCGSTAB from 14 to 8.
- **Fusion rule:** All BLAS1 ops that touch the same vectors (e.g., `dot(r,z)` and `dot(r,r)` when r is shared) are fused into a single kernel. The SpMV result can also be fused with the subsequent dot product by computing the inner product inline as the SpMV writes its output.
- **Performance:** Up to 3x speedup for systems with 10^4-10^5 unknowns where launch overhead dominates.

### Ginkgo (KIT/UTK)

Ginkgo explicitly implements a **pipelined CG solver** with merged dot products (PR #1908 in their GitHub). Key design decisions:

- `batch::MultiVector` class enables fused operations: dot, norm, scale on batched vectors in a single call.
- For batched iterative solvers (solving many small independent systems), the **entire solver runs in a single kernel** -- one thread block per system. All BLAS1 ops (dot, nrm2, axpy, scal) are inlined as device functions within the solver loop. No kernel launches inside the iteration.
- For large single-system solvers, they use the Chronopoulos-Gear pipelined CG with merged allreduce.
- Prefers vendor (cuBLAS) implementations for `Dense::dot`, `Dense::conj_dot`, `Dense::norm2` when operating on single vectors, but switches to custom fused kernels for the solver inner loop.

### AmgX (NVIDIA)

NVIDIA's AmgX library implements CG, BiCGSTAB, and GMRES with GPU-optimized Krylov methods. It uses fused kernels for the vector operation sequences within each iteration. The library achieves up to 10x acceleration for the linear solver portion by minimizing kernel launch overhead and keeping data on-device.

### cuSOLVER

cuSOLVER's iterative refinement (GMRES-based) keeps the iteration loop on GPU, using cuBLAS calls internally. It does not expose a batched dot product API. For custom solvers, you must build your own fusion.

## Multi-Reduction Kernel Design

The core pattern for computing N dot products in one kernel:

### Architecture 1: One Block Per Dot Product (Small N, Large Vectors)

```
Grid: N blocks (one per dot product)
Block: 256 threads

__global__ void multi_dot(const float* __restrict__ vecs_a,  // N vectors, each of length L
                          const float* __restrict__ vecs_b,
                          float* __restrict__ results,
                          int L, int N) {
    int dot_idx = blockIdx.x;  // which dot product
    const float* a = vecs_a + dot_idx * L;
    const float* b = vecs_b + dot_idx * L;

    // Grid-stride accumulation with 128-bit vectorized loads
    float sum = 0.0f;
    for (int i = threadIdx.x; i < L/4; i += blockDim.x) {
        float4 va = __ldg((float4*)(a + i*4));
        float4 vb = __ldg((float4*)(b + i*4));
        sum += va.x*vb.x + va.y*vb.y + va.z*vb.z + va.w*vb.w;
    }

    // Warp shuffle reduction
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);

    // Block reduction via shared memory
    __shared__ float warp_sums[8];  // 256/32 = 8 warps
    if (threadIdx.x % 32 == 0) warp_sums[threadIdx.x / 32] = sum;
    __syncthreads();

    if (threadIdx.x < 8) {
        sum = warp_sums[threadIdx.x];
        for (int offset = 4; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xff, sum, offset);
        if (threadIdx.x == 0) results[dot_idx] = sum;
    }
}
```

This is the simplest and most practical for our use case. For CG with 2-3 dot products, N=3 blocks is tiny -- the GPU is underutilized. Must combine with other work.

### Architecture 2: All Dots Interleaved in One Block (Small N, Medium Vectors)

Each thread computes partial products for ALL N dot products, using N accumulators in registers. Then N separate reductions happen in shared memory. This keeps all results in a single block's shared memory, avoiding a second reduction pass:

```
// Each thread holds N partial sums
float sums[N];
for (int k = 0; k < N; k++) sums[k] = 0.0f;

for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < L; i += gridDim.x * blockDim.x) {
    for (int k = 0; k < N; k++) {
        sums[k] += a[k*L + i] * b[k*L + i];
    }
}

// Then reduce each of the N sums across the block
for (int k = 0; k < N; k++) {
    // warp shuffle + shared memory reduction for sums[k]
    // atomicAdd(&results[k], block_sum_k);
}
```

This is better when N is small (2-5) and vectors are large enough that multiple blocks are needed. The key insight: **a single thread reads the same memory positions for each dot product only once** if the vectors overlap (e.g., `dot(r,z)` and `dot(r,r)` both read `r`).

### Architecture 3: Fused SpMV + Dot (The Real Win)

The biggest performance gain in iterative solvers comes from fusing the SpMV with the subsequent dot product. After computing `y = A*x`, the result `y` is still in registers/L1. Computing `dot(y, r)` immediately avoids writing `y` to global memory and reading it back:

```
// Inside SpMV kernel, after computing y[i]:
float yi = /* SpMV result for row i */;
y[i] = yi;  // write to global memory
dot_partial += yi * r[i];  // fuse dot product
```

This saves one full vector read (bandwidth-bound operations dominate for sparse solvers). The Rupp et al. pipelined solvers and the Anzt systematic fusion paper both identify this as the single highest-impact optimization.

## Fused Convergence Checking

### The Problem

Standard CG requires checking `||r|| < tol` each iteration. Computing `nrm2(r)` produces a scalar on GPU. Getting it to the host for the `if` check requires `cudaMemcpy` + implicit sync. This serializes every iteration.

### Solution 1: Device-Side Flag (Simplest)

Write the convergence result to a device-side flag, check it with a lightweight memcpy only every K iterations:

```
// In the reduction kernel that computes ||r||^2:
if (threadIdx.x == 0 && blockIdx.x == 0) {
    float norm_sq = results[0];  // after full reduction
    if (norm_sq < tol_sq) {
        *d_converged = 1;  // device flag
    }
    *d_norm_sq = norm_sq;  // store for potential host read
}

// Host side: check every 10-50 iterations
if (iter % 50 == 0) {
    cudaMemcpyAsync(&h_converged, d_converged, sizeof(int), D2H, stream);
    cudaStreamSynchronize(stream);
    if (h_converged) break;
}
```

This amortizes the sync cost. Most solvers converge in 100-1000 iterations, so checking every 50 adds at most 1 extra iteration of wasted work.

### Solution 2: Persistent Kernel with Cooperative Groups (Advanced)

The PERKS approach moves the **entire solver loop** inside a single persistent kernel using cooperative groups for grid-wide sync:

```
__global__ void persistent_cg(/* all solver data */) {
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();

    for (int iter = 0; iter < max_iter; iter++) {
        // SpMV (each block handles its rows)
        // ...
        grid.sync();  // device-wide barrier

        // Fused dot products (grid-wide reduction)
        // ...
        grid.sync();

        // AXPY updates
        // ...
        grid.sync();

        // Convergence check (single thread reads results)
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            if (norm_r < tol) *d_converged = 1;
        }
        grid.sync();  // all threads see convergence flag
        if (*d_converged) return;
    }
}
```

**Performance:** PERKS achieves 4.67x speedup for CG on small SpMV datasets (SuiteSparse), 1.39x on larger ones. The speedup comes from eliminating kernel launch overhead between iterations AND caching vectors in registers/shared memory across iterations.

**Requirements:**
- Must use `cudaLaunchCooperativeKernel` (supported on sm_120)
- Grid size limited to what can simultaneously occupy the GPU (170 SMs on RTX 5090, so max ~170*N_blocks_per_SM blocks)
- `grid.sync()` is a true global barrier -- all blocks must reach it

**Caveat for our worker:** This is the most complex approach. The grid-wide reduction requires careful design -- you cannot use the simple "one block per row" SpMV pattern because the grid must be small enough for cooperative launch. Better suited for medium-sized problems where the grid naturally fits.

### Solution 3: Two-Phase Reduction with Atomic Flag (Practical Middle Ground)

Phase 1: Block-level partial reductions write to a small global array via atomicAdd.
Phase 2: The last block to finish (detected via atomic counter) reads all partial results and writes the final scalar + checks convergence.

```
__shared__ bool is_last_block;
// ... reduction code ...

if (threadIdx.x == 0) {
    // Atomic increment of a block counter
    unsigned int old = atomicInc(&block_count, gridDim.x - 1);
    is_last_block = (old == gridDim.x - 1);
}
__syncthreads();

if (is_last_block) {
    // This block has access to all partial results
    // Reduce them, write final scalar, check convergence
    float final_norm = /* reduce partial_results[0..gridDim.x-1] */;
    if (threadIdx.x == 0) {
        results[0] = final_norm;
        if (final_norm < tol) *d_converged = 1;
    }
}
```

This is a **single-kernel dot product with built-in convergence check**, no host sync needed. The "last block" pattern is well-known and used in production (NVIDIA samples, CUB internals).

## CUB/Thrust Batched Primitives

### CUB DeviceSegmentedReduce

CUB provides `DeviceSegmentedReduce::Reduce()` which computes reductions on multiple contiguous segments of an array in a single call. For batched dot products:

1. Pre-multiply the vectors element-wise: `c[i] = a[i] * b[i]` for all vector pairs (concatenated)
2. Call `DeviceSegmentedReduce::Sum()` with segment offsets marking each vector's boundaries

**Limitations:**
- Requires the element-wise multiplication as a separate step (or a custom iterator/transform)
- The segment boundaries are defined by offset arrays, not stride -- good for variable-length segments
- Performance is good for large segments with many segments, but custom kernels beat it for small N (2-5 dot products) because the overhead of segment management dominates

### CUB BlockReduce

More useful as a building block: `cub::BlockReduce<float, BLOCK_SIZE>` provides optimized within-block reduction that you can use inside your custom multi-dot kernel. This is strictly better than hand-rolled shared memory reduction.

### Thrust reduce_by_key

`thrust::reduce_by_key` can compute segmented reductions but is fastest only for small N (N <= 10 segments) with very large segments. For the 2-5 dot products typical in CG, a custom kernel is faster.

### Recommendation

For our use case (2-5 simultaneous dot products of long vectors), **use CUB BlockReduce inside a custom multi-dot kernel**. Don't use DeviceSegmentedReduce -- it adds unnecessary indirection for such small N.

## Gram Matrix / Orthogonalization Patterns (GMRES)

### The Problem

GMRES iteration k requires orthogonalizing a new vector v against the existing k-vector orthogonal basis Q = [q_1, ..., q_k]. This means computing k dot products: `h_i = dot(q_i, v)` for i=1..k, which is equivalent to `h = Q^T * v` -- a tall-skinny matrix-vector multiply.

### Approach 1: cuBLAS GEMV (Q^T * v)

Store Q as a column-major matrix (n x k). Use `cublasSgemv` with `CUBLAS_OP_T` to compute the k-element result in one call. cuBLAS handles the internal parallelization.

**Performance note:** cuBLAS GEMV is poorly optimized for tall-skinny shapes (n >> k). The worker's custom GEMV already beats cuBLAS by 1.75x. For GMRES orthogonalization, the custom GEMV kernel is the right building block.

### Approach 2: Block Dot Products (Custom Kernel)

Each block computes one or more dot products. With k small (10-50 for typical GMRES restarts), assign one block per dot product. Each block does a grid-stride dot product of its assigned q_i with v.

### Approach 3: s-step / Block GMRES (Communication-Avoiding)

The s-step GMRES variant (Hoemmen, Carson) generates s basis vectors at once via a Matrix Powers Kernel (MPK), then orthogonalizes the entire block using BLAS-3 (GEMM) operations:

- `R = Q^T * V` where V is n x s -- this is a GEMM, not s separate dot products
- BLAS-3 operations have much better arithmetic intensity than BLAS-1
- Two-stage block orthogonalization (2024 paper by Bielich et al.) achieves 2.6x speedup for the orthogonalization phase on V100 GPUs

**For our worker:** If implementing GMRES, use the custom GEMV kernel for classical Gram-Schmidt. For block GMRES or s-step GMRES, use the batched GEMM kernel (already at 1.34x cuBLAS). The block variants turn BLAS-1 problems into BLAS-3 problems -- a fundamental win.

### Approach 4: Classical Gram-Schmidt with Reorthogonalization (CGS-2)

CGS-2 maintains orthogonality near machine precision and requires only one global synchronization per iteration (vs MGS which requires k syncs). This is the preferred algorithm for GPU GMRES because:
- One GEMV call (`h = Q^T * v`) + one GEMV call (`v = v - Q*h`) + repeat once = 4 BLAS calls total
- All are BLAS-2 operations on the same data -- fusible in principle

## Recommended Approach for Our Worker

### Phase 1: Multi-Dot Kernel (Immediate Win)

Build a `batched_dot` kernel that computes N dot products in a single launch:

```
// Compute results[k] = dot(a[k], b[k]) for k = 0..N-1
// where a[k] and b[k] are pointers to vectors of length L
__global__ void batched_dot(const float** a_ptrs, const float** b_ptrs,
                            float* results, int L, int N);
```

Design choices:
- **N <= 8:** Use Architecture 2 (all dots interleaved, N accumulators per thread). Reads each element once even when vectors overlap.
- **N > 8:** Use Architecture 1 (one block per dot product).
- Use 128-bit vectorized loads (`__ldg` with `float4`/`bf16x8`), warp shuffle reduction, `cub::BlockReduce` for the final block-level step.
- For the final cross-block reduction, use the atomic counter "last block" pattern to avoid a second kernel launch.

### Phase 2: Fused CG Iteration Kernel (Medium Effort)

Implement a Chronopoulos-Gear pipelined CG where:
1. The two dot products per iteration are merged into one multi-dot kernel launch
2. The AXPYs are fused with each other (they share vectors)
3. The SpMV result feeds directly into the fused dot kernel (if using custom SpMV)
4. Convergence check is embedded in the reduction kernel (device-side flag, checked every K iterations from host)

This should reduce kernel launches from 6/iteration to 2-3/iteration.

### Phase 3: Persistent Solver (Advanced, High Reward)

For problems that fit in GPU memory and have moderate matrix size:
- Use `cudaLaunchCooperativeKernel` to run the entire CG loop as a persistent kernel
- All BLAS1 ops become device functions called within the loop
- Grid-wide sync via `cooperative_groups::this_grid().sync()` replaces kernel launches
- Vectors cached in registers/shared memory across iterations
- Expected speedup: 2-5x for small-to-medium systems (per PERKS results)

### Vectorization Note for BF16

The worker's existing DOT kernel uses 128-bit vectorized loads. For BF16 vectors, this means loading 8 BF16 values per load (`__ldg` with `uint4`). The multi-dot kernel should maintain this. For the reduction, accumulate in FP32 (convert BF16 to FP32 after load), reduce in FP32, then the scalar result stays FP32.

## Caveats

### sm_120 Specifics

- **Cooperative groups grid sync** is supported on sm_120 (requires compute capability >= 6.0). The RTX 5090 with 170 SMs can launch cooperative kernels with up to ~170 * blocks_per_SM blocks.
- **No TMA/tcgen05** -- all loads must use `cp.async` or `__ldg`. The multi-dot kernel is memory-bound, so this is fine.
- **Shared memory budget:** 99 KB/block. For a multi-dot kernel with N=5 dot products, shared memory is negligible (just warp reduction buffers). Not a constraint.
- **Register pressure:** N accumulators per thread (N floats) for Architecture 2. With N <= 8, this is 8 registers -- negligible impact on occupancy.

### cuBLAS Does NOT Have Batched Dot

There is no `cublasSdotBatched` or `cublasSdotStridedBatched` in cuBLAS. Computing multiple dot products requires either N separate `cublasSdot` calls or a custom kernel. This is an opportunity -- our custom batched_dot kernel has no library competition.

### The Real Bottleneck is SpMV, Not BLAS1

For large sparse systems, SpMV dominates iteration time (80-90%). Fusing BLAS1 ops helps most for small-to-medium systems (10^4-10^5 unknowns) where launch overhead is a significant fraction. For large systems, the BLAS1 fusion still helps but the relative impact is smaller.

### Numerical Stability of Pipelined CG

The Chronopoulos-Gear reformulation introduces auxiliary recurrences that can accumulate rounding error. For BF16 arithmetic, this may require periodic residual replacement (recompute `r = b - A*x` every K iterations). The worker should start with standard CG + multi-dot, then try pipelined CG and monitor convergence behavior.
