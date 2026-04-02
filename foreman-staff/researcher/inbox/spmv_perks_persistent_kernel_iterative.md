# PERKS: Persistent Kernels for Iterative SpMV (L2 Cache Exploitation)

**Sources:**
- [PERKS: ICS 2023 (Zhang et al.)](https://dl.acm.org/doi/abs/10.1145/3577193.3593705)
- [PERKS arXiv preprint](https://arxiv.org/abs/2204.02064)
- [PERKS GitHub](https://github.com/neozhang307/PERKS) (BSD-3 license)
- [Persistent Kernel SpMV concept paper](https://www.researchgate.net/publication/359757295_Persistent_Kernels_for_Iterative_Memory-bound_GPU_Applications)
**Relevant to:** spmv worker
**Worker's current problem:** Building SpMV kernel for RTX 5090; needs L2 cache exploitation strategy for iterative solvers

---

## What This Is

PERKS (PERsistent KernelS) is an execution model from Oak Ridge National Lab
that keeps iterative memory-bound GPU applications (including CG solvers with
SpMV) running inside a single persistent kernel. Instead of launching separate
kernels for each SpMV iteration, the entire iteration loop runs on the GPU with
cooperative group grid-wide barriers for synchronization.

## Why It Matters for Us

The RTX 5090 has a 96 MB L2 cache. For matrices that fit (partially or fully) in
L2, a persistent kernel that keeps the matrix resident across iterations eliminates
DRAM re-fetches. PERKS demonstrates:
- **4.55x-4.87x speedup** on A100 when matrix fits in L2 (single/double precision)
- **1.30x-1.44x speedup** when matrix exceeds L2 capacity
- **4.86x speedup** for CG solver on smaller SpMV datasets

These are massive gains from a technique that does NOT change the SpMV algorithm
itself -- it only changes how iterations are orchestrated.

---

## Key Technique: How PERKS Works

### 1. Traditional Approach (Multiple Kernel Launches)

```
Host code:
  for iter = 1 to max_iter:
    launch spmv_kernel(A, x, y)     // SpMV: y = A*x
    launch dot_kernel(r, r, &rr)     // DOT: rr = r^T * r
    launch axpy_kernel(alpha, p, x)  // AXPY: x = x + alpha*p
    check convergence on host
```

Each kernel launch:
- Has ~5-10us launch overhead
- Flushes L2 cache contents between launches (data may be evicted)
- Requires CPU-GPU synchronization for convergence check
- Matrix A must be re-fetched from DRAM each iteration if evicted from L2

### 2. PERKS Approach (Single Persistent Kernel)

```
Host code:
  launch persistent_cg_kernel(A, x, b, tol, max_iter)  // ONE launch

Device code (persistent_cg_kernel):
  // Grid-wide initialization
  cooperative_groups::grid_group grid = cooperative_groups::this_grid();

  for iter = 1 to max_iter:
    // SpMV: y = A*x (each block processes its assigned rows)
    spmv_my_rows(A, x, y);
    grid.sync();  // Grid-wide barrier

    // DOT: rr = r^T * r (block-level partial sums + atomic)
    partial_dot = dot_my_rows(r, r);
    atomicAdd(&global_rr, partial_dot);
    grid.sync();

    // AXPY + convergence check
    axpy_my_rows(alpha, p, x);
    if (global_rr < tol * tol) break;
    grid.sync();
```

### 3. Why This Exploits L2 Cache

- **Matrix A stays in L2**: Since the kernel never exits, matrix data loaded in
  iteration 1 may still be in L2 for iteration 2. For matrices smaller than L2
  (nnz * 8 bytes < 96 MB, i.e., nnz < 12M), the entire matrix lives in L2 after
  the first iteration.

- **Vector x stays in registers/L1**: Each block's portion of x and y can be cached
  in registers or shared memory across iterations, avoiding global memory round-trips.

- **No launch overhead**: A CG solver doing 1000 iterations saves 3000+ kernel
  launches (SpMV + DOT + AXPY per iteration). At 5-10us per launch, that's 15-30ms
  saved.

### 4. L2 Cache Capacity Analysis for RTX 5090

| Matrix Size (nnz) | Storage (val+col, bytes) | Fits in 96MB L2? | Expected PERKS Benefit |
|-------------------|--------------------------|-------------------|----------------------|
| 1M                | 8 MB                     | Yes (8%)          | ~4-5x per iteration  |
| 5M                | 40 MB                    | Yes (42%)         | ~3-4x per iteration  |
| 10M               | 80 MB                    | Mostly (83%)      | ~2-3x per iteration  |
| 15M               | 120 MB                   | No (125%)         | ~1.3-1.5x per iteration |
| 50M               | 400 MB                   | No (417%)         | ~1.1-1.3x per iteration |

---

## Implementation for sm_120

### Requirements

1. **Cooperative launch**: Use `cudaLaunchCooperativeKernel()` instead of `<<<>>>`
2. **Grid size constraint**: Must launch exactly the number of blocks that can
   simultaneously reside on the GPU. For sm_120: 170 SMs * blocks_per_SM.
   With 256 threads/block and ~30 registers per thread: ~4 blocks/SM = 680 blocks.
3. **Grid sync**: Use `cooperative_groups::this_grid().sync()` for grid-wide barrier.

### Skeleton Code

```cuda
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void persistent_cg(
    int M, int nnz,
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    const float* __restrict__ val,
    float* __restrict__ x,
    const float* __restrict__ b,
    float tol, int max_iter,
    float* __restrict__ workspace)  // for r, p, Ap, etc.
{
    cg::grid_group grid = cg::this_grid();
    int block_id = blockIdx.x;
    int num_blocks = gridDim.x;

    // Each block "owns" a contiguous range of rows
    int rows_per_block = (M + num_blocks - 1) / num_blocks;
    int my_row_start = block_id * rows_per_block;
    int my_row_end = min(my_row_start + rows_per_block, M);

    // ... CG initialization (r = b - A*x, p = r) ...
    grid.sync();

    for (int iter = 0; iter < max_iter; iter++) {
        // SpMV: Ap = A * p
        float local_pAp = 0.0f;
        for (int row = my_row_start; row < my_row_end; row++) {
            float sum = 0.0f;
            for (int j = row_ptr[row]; j < row_ptr[row + 1]; j++)
                sum += val[j] * x[col_idx[j]];
            workspace[row] = sum;  // Ap[row]
            local_pAp += workspace[row] * p[row];  // fused DOT
        }

        // Block-level reduction of local_pAp
        // ... (warp shuffle + shared memory reduction) ...
        atomicAdd(&global_pAp, block_pAp);
        grid.sync();

        // ... alpha, x update, r update, convergence check ...
        grid.sync();
    }
}

// Launch:
void* args[] = { &M, &nnz, &row_ptr, &col_idx, &val, &x, &b, &tol, &max_iter, &workspace };
int num_blocks;
cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks, persistent_cg, 256, 0);
num_blocks *= 170;  // 170 SMs on RTX 5090
cudaLaunchCooperativeKernel((void*)persistent_cg, num_blocks, 256, args);
```

### Key Optimization: Fused SpMV + DOT

In the persistent kernel, after computing `Ap[row]`, the result is still in
registers. Immediately computing the local contribution to `p^T * Ap` avoids
writing Ap to global memory and reading it back. This saves one full vector
read+write per iteration -- a 2x reduction in vector traffic.

---

## Caveats for sm_120

1. **Cooperative launch grid size is limited**: You can only launch as many blocks
   as can simultaneously reside on the GPU. With 170 SMs and ~4 blocks/SM, that's
   ~680 blocks = ~174K threads. For large matrices (M > 1M), each block handles
   ~1500 rows. This is fine -- each block has plenty of work.

2. **Grid sync is a hard barrier**: All blocks must reach the sync point before any
   can proceed. If work is imbalanced (some blocks finish SpMV early), they wait.
   This is mitigated by even row-range assignment.

3. **Register pressure**: The persistent kernel holds CG state (r, p, Ap, scalars)
   across iterations. With ~30 registers per thread, this fits in sm_120's register
   file (64K registers / 48 warps = 1365 regs/warp = 42 regs/thread).

4. **Debugging complexity**: Persistent kernels are harder to debug than separate
   launches. Use the PERKS repo as a reference implementation.

5. **Not applicable for single SpMV**: PERKS only helps for iterative workloads
   (CG, GMRES, BiCGSTAB). For single SpMV calls, use the standard approach.

---

## Performance Expectations for RTX 5090

| Scenario | Expected Speedup | Basis |
|----------|-----------------|-------|
| CG with matrix fitting in L2 (nnz < 12M) | 3-5x per iteration | PERKS A100 results |
| CG with matrix exceeding L2 (nnz > 15M) | 1.2-1.5x per iteration | PERKS large-matrix results |
| SpMV+DOT fusion alone (no persistence) | 1.3-1.5x per iteration | Saves one vector pass |
| Combined persistence + fusion + L2 | 4-6x for small matrices | Stacking all benefits |

The RTX 5090's 96 MB L2 is larger than the A100's 40 MB L2, so the "fits in L2"
threshold is higher (12M nnz vs 5M nnz). More matrices will benefit from the
persistent approach on RTX 5090 than on A100.
