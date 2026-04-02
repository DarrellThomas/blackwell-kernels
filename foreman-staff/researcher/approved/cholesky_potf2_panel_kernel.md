# Unblocked Panel Factorization (potf2) — CUDA Kernel Details

**Source:** http://num.math.uni-goettingen.de/~stkramer/doc/autogen/CUDA_HPC_Praktikum/step_1.html | https://icl.utk.edu/projectsfiles/magma/doxygen/group__magma__potf2.html | https://docs.nvidia.com/cuda/cusolverdx/get_started/potrf.html
**Relevant to:** cholesky worker (new kernel)
**Worker's current problem:** Needs the inner panel factorization kernel — the serial bottleneck in blocked Cholesky.

## What This Is

potf2 is the unblocked (column-by-column) Cholesky factorization of a small tile (typically 16-64 columns). It's the innermost kernel in blocked Cholesky — called once per panel in the main loop. Despite being the serial bottleneck, it only accounts for ~5% of total FLOPS for large matrices.

## Why It Matters for Us

This is the one part of Cholesky that we CANNOT map to our existing GEMM kernel. It's a new type of computation (sequential column factorization with dot products and square roots) that requires a purpose-built CUDA kernel. Getting it right determines the panel overhead.

## Key Technique

### Algorithm (lower triangular, column j of nb×nb tile)

```
for j = 0 to nb-1:
    // 1. DOT PRODUCT: sum of squares of row j entries to the left
    dot = sum(L[j][0:j]²)                    // L2 BLAS: ddot

    // 2. DIAGONAL: subtract and take square root
    L[j][j] = sqrt(A[j][j] - dot)            // scalar op

    // 3. COLUMN SCALE: update entries below diagonal in column j
    for i = j+1 to nb-1:
        row_dot = sum(L[i][0:j] * L[j][0:j]) // L2 BLAS: ddot
        L[i][j] = (A[i][j] - row_dot) / L[j][j]  // L1 BLAS: dscal
```

### CUDA Kernel Implementation (from Göttingen tutorial)

```cuda
// TILE_SIZE = 16, threads = 16×16 (one per matrix element)
__global__ void factorize_diagonal_block(float *A, int block_offset, int N) {
    int col = threadIdx.x;
    int row = threadIdx.y;

    // Load tile into shared memory (with +1 padding to avoid bank conflicts)
    __shared__ float L[TILE_SIZE][TILE_SIZE + 1];
    L[row][col] = A[global_idx];
    __syncthreads();

    float fac;
    for (int k = 0; k < TILE_SIZE; k++) {
        __syncthreads();

        // Thread (k, k) computes 1/sqrt(diagonal)
        fac = rsqrtf(L[k][k]);
        __syncthreads();

        // Row k threads: scale the row by 1/sqrt(diag)
        if (row == k && col >= k)
            L[col][row] = L[col][row] * fac;
        __syncthreads();

        // Rank-1 update: subtract outer product of column k
        if (row >= col && col > k)
            L[row][col] -= L[col][k] * L[row][k];
    }

    __syncthreads();
    if (row >= col) A[global_idx] = L[row][col];
}
```

### Key Implementation Details

1. **rsqrtf instead of sqrt + division:** Computes 1/√x in a single MUFU instruction on NVIDIA GPUs. Then multiply instead of divide. Much faster on GPU.

2. **Shared memory padding:** `L[TILE_SIZE][TILE_SIZE+1]` — the +1 column avoids bank conflicts when threads access consecutive column elements. This is the same padding trick used in classic tiled GEMM.

3. **Thread mapping:** One thread per matrix element (16×16 = 256 threads). Each thread handles one (row, col) position. The sequential dependency is over the `k` loop — all threads synchronize between columns.

4. **Parallelism within a column:** The rank-1 update `L[row][col] -= L[col][k] * L[row][k]` runs in parallel across all threads where `row >= col && col > k`. This is O(nb²) parallel work per column step.

5. **Synchronization:** Three `__syncthreads()` per column iteration: (a) before reading diagonal, (b) after computing fac, (c) after scaling row k. The k loop is inherently sequential.

### Scaling to Larger Tiles (nb > 16)

For nb = 32 or 64, two approaches:

**Approach A — More threads per tile:**
- nb=32: 32×32 = 1024 threads (maximum for one block)
- nb=64: Too many threads. Must use subsets (e.g., 4 passes of 16×64)

**Approach B — Column-parallel (MAGMA style):**
- Use 1D thread blocks (e.g., 128 or 256 threads)
- Each column step: parallel dot product reduction across threads
- Thread i computes partial sum for row i
- Uses warp shuffles for the dot product reduction
- MAGMA limits potf2 to N ≤ 512

**Approach C — Recursive blocked:**
- Recursively split the tile: potrf(top-left nb/2), trsm(bottom-left), syrk(bottom-right), recurse
- Even within the panel, this exposes GEMM-like work at each recursion level
- Used by the mixed-precision recursive approach (5x speedup over cuSOLVER)

### MAGMA potf2 GPU Kernel

MAGMA's GPU-only potf2 (`magma_dpotf2_gpu`) uses:
- Unblocked Level 2 BLAS (ddot + dscal per column)
- Limited to N ≤ 512
- The panel factorization is done as a device function callable from the blocked potrf kernel
- Uses a `recnb` parameter (typically 64 or 128) for recursive panel subdivision

## Caveats

- **The Göttingen kernel is pedagogical, not optimized.** 16×16 tiles with full __syncthreads__ per column is slow. Production kernels (MAGMA, cuSOLVERDx) use more sophisticated approaches with warp-level operations and larger tiles.
- **Bank conflicts matter.** The +1 padding trick works for naive access but doesn't help with swizzled access patterns. For tensor-core-based trailing updates, the shared memory layout must be compatible with both potf2 and ldmatrix.
- **rsqrtf precision.** The reciprocal sqrt approximation on GPU has ~1 ULP error. For FP32 this is fine. For FP64, use `rsqrt()` (or `1.0/sqrt()` which nvcc optimizes).
- **The k loop is O(nb) sequential steps.** Each step has O(nb²) parallel work. For nb=64 and 128 threads, each step only has ~32 independent operations per thread — not great occupancy. This is why the panel is the bottleneck.
- **Negative pivot detection:** If `L[k][k] <= 0` (matrix not positive definite), the kernel must report the failing column index via the `info` output. `rsqrtf` of a negative number returns NaN, which propagates silently — explicit checks are needed.
