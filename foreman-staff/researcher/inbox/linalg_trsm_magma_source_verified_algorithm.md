# TRSM: MAGMA Source-Verified Algorithm Details and Small-N Strategy

**Sources:**
- [MAGMA strsm.cu source code (maxhutch fork)](https://github.com/maxhutch/magma/blob/master/magmablas/strsm.cu)
- [MAGMA strtri.cuh header](https://github.com/maxhutch/magma/blob/master/magmablas/strtri.cuh)
- [MAGMA strtri_diag.cu kernel](https://github.com/maxhutch/magma/blob/master/magmablas/strtri_diag.cu)
- [MAGMA trtri_diag API docs](https://icl.utk.edu/projectsfiles/magma/doxygen/group__magma__trtri__diag__batched.html)
- [Carrica et al., "Hierarchical Precision and Recursion" (Jan 2026)](https://arxiv.org/html/2601.08082v1)
- [Carrica & Onyango, "Julia Recursive TRMM/TRSM" (Apr 2025)](https://arxiv.org/html/2504.13821v1)

**Relevant to:** linalg worker / numerical worker (TRSM is a dependency for LU, Cholesky, QR)
**Worker's current problem:** TRSM at 0.82x cuBLAS F32 via torch.linalg.solve_triangular. For small-to-medium N (64-256), kernel launch overhead may dominate custom recursive approaches.

**Note:** This brief supplements the three existing TRSM briefs with verified source-level details from the actual MAGMA CUDA code. The existing briefs describe the algorithms conceptually; this one shows exactly what the code does.

---

## 1. MAGMA TRSM: Verified Source-Level Algorithm

I read the actual MAGMA `strsm.cu` source code. Here is exactly how it works (for the case: Left, Lower, NoTrans -- the most common case for Cholesky/LU):

### Constants (from strtri.cuh)

```
#define IB  16     // Inner block size for diagonal inversion
#define NB  128    // Outer block size for TRSM loop
```

IB=16 is the size of the elemental triangular inversion kernel. NB=128 is the block size for the outer TRSM loop.

### Diagonal Block Inversion (strtri_diag.cu)

The `trtri_diag` kernel builds up NB=128 block inversions recursively from IB=16 blocks:

1. **Base:** Invert each 16x16 diagonal block with a single-warp kernel (`strtri_diag_lower_kernel`, launched with `<<<nblocks, IB>>>` = one thread per row per block).

2. **Build-up via "triple GEMM" kernels:** Combine 16x16 inverses into 32x32, then 64x64, then 128x128 inverses. Each doubling step uses two kernel launches (`part1` and `part2`), performing the block triangular inversion formula:
   ```
   inv([A11  0  ]) = [inv(A11)              0       ]
       [A21  A22]    [-inv(A22)*A21*inv(A11) inv(A22)]
   ```
   The off-diagonal block `-inv(A22)*A21*inv(A11)` is computed as a triple GEMM via shared memory, with specialized kernels for each size (16, 32, 64, and >64).

3. **Thread configurations for the build-up:**
   - jb=16: threads(4, 4), grid depends on page count
   - jb=32: threads(8, 4)
   - jb=64: threads(16, 4)
   - jb>64: threads(16, 4), with 3-part kernel

4. **Total kernel launches for trtri_diag:** 1 (base) + 2*log2(NB/IB) = 1 + 2*3 = 7 kernel launches to invert all diagonal blocks.

### Main TRSM Loop (strsm.cu) -- Left, Lower, NoTrans

```c
// Step 0: Invert all diagonal blocks (if flag==true)
magmablas_strtri_diag(uplo, diag, m, dA, ldda, d_dinvA, queue);

// Step 1: Handle first NB-wide block with alpha scaling
jb = min(NB, m);   // = 128 if m >= 128
sgemm(NoTrans, NoTrans, jb, n, jb,
      alpha, d_dinvA(0), NB, dB, lddb,
      zero,  dX,         lddx);           // X[0:NB] = alpha * dinvA[0] @ B[0:NB]

if (NB < m) {
    sgemm(NoTrans, NoTrans, m-NB, n, NB,
          neg_one, dA(NB,0), ldda, dX, lddx,
          alpha,   dB(NB,0), lddb);        // B[NB:] = alpha*B[NB:] - A[NB:,0:NB] @ X[0:NB]
}

// Step 2: Remaining blocks (no alpha, already folded in)
for (i = NB; i < m; i += NB) {
    jb = min(m - i, NB);
    sgemm(NoTrans, NoTrans, jb, n, jb,
          one,  d_dinvA(i), NB, dB(i,0), lddb,
          zero, dX(i,0),    lddx);         // X[i:i+NB] = dinvA[i/NB] @ B[i:i+NB]

    if (i + NB >= m) break;

    sgemm(NoTrans, NoTrans, m-i-NB, n, NB,
          neg_one, dA(i+NB,i), ldda, dX(i,0), lddx,
          one,     dB(i+NB,0), lddb);      // B[i+NB:] -= A[i+NB:, i:i+NB] @ X[i:i+NB]
}
```

### Key Observations from the Source

1. **Every operation is a GEMM.** There is zero scalar triangular solve code in the main loop. The diagonal blocks are pre-inverted, so "solving" with a diagonal block is just a GEMM with the inverse.

2. **Two GEMMs per block:** One (NB x NB) @ (NB x NRHS) for the panel solve, one (M_remaining x NB) @ (NB x NRHS) for the trailing update.

3. **Out-of-place:** Solution goes to `dX`, then is copied back to `dB`. This requires NB*NRHS workspace.

4. **The alpha scaling is folded into the first GEMM.** Subsequent GEMMs use `alpha=1`.

5. **Total GEMM calls for N=256, NB=128:** trtri_diag (7 launches) + 2 panel GEMMs + 1 trailing GEMM + 1 copy = 11 launches total.

6. **Total GEMM calls for N=64, NB=128:** The entire matrix fits in one NB block. Only trtri_diag (7 launches) + 1 panel GEMM + 1 copy = 9 launches. The trailing GEMM is skipped.

---

## 2. Critical Insight for Small N (64-256)

For the worker's target sizes (N=64 to 256), the MAGMA algorithm has a structural problem:

### N=64: The matrix fits in a single NB=128 block
- trtri_diag inverts the 64x64 block (7 kernel launches for the recursive inversion!)
- Then ONE GEMM call: `X = dinvA @ B` (64x64 @ 64xNRHS)
- Total: 8-9 kernel launches for what is essentially a single matrix multiply
- At 3-5 us per launch: 24-45 us of launch overhead

### N=128: Also fits in a single NB block
- Same as N=64: trtri_diag + one GEMM
- Total: 8-9 kernel launches

### N=256: Two NB=128 blocks
- trtri_diag: 7 launches for 2 diagonal blocks
- Panel GEMM (first block): 1 launch
- Trailing GEMM: 1 launch
- Panel GEMM (second block): 1 launch
- Copy: 1 launch
- Total: 11 kernel launches

### The Opportunity

cuBLAS uses this same algorithm (derived from MAGMA). At small N, launch overhead dominates. A custom kernel that does the entire TRSM in ONE launch would eliminate 7-10 launches worth of overhead.

**For N <= 128:** The entire triangular matrix fits in shared memory (128*128*2 = 32 KB in BF16, or 64 KB in F32 -- both under the 99 KB limit). A monolithic kernel can:
1. Load A into shared memory
2. Load B into shared memory or registers
3. Do column-by-column forward substitution (warp-shuffle for 32-wide, 4 warps for 128-wide)
4. Write X out
5. Total: ONE kernel launch

**For N=256:** Two-block approach within a single kernel:
1. Load top-left 128x128 block of A into shared memory
2. Solve the top 128 rows (forward substitution in shared memory)
3. Load off-diagonal 128x128 block and do GEMM update in shared memory
4. Load bottom-right 128x128 block and solve the bottom 128 rows
5. Total: ONE kernel launch (uses cooperative groups or warp-level sync between phases)

This approach targets the dominant cost at small N: kernel launch overhead, not arithmetic throughput.

---

## 3. Concrete Implementation: Single-Kernel TRSM for N<=128

### Architecture

```
Thread block: (32, 4) = 128 threads = 4 warps
Each warp handles one set of 32 rows
Process WPB columns of B simultaneously (WPB=8-16, each thread handles multiple B columns)
```

### Algorithm (lower triangular, N=128)

```cuda
__global__ void trsm_small(const half* A, half* B, int N, int NRHS) {
    __shared__ half smem_A[128][128+PAD];  // ~33 KB in BF16 with padding

    // Phase 1: Load A into shared memory (all threads cooperate)
    // 128*128 = 16384 elements / 128 threads = 128 loads per thread
    cooperative_load(A, smem_A, N);
    __syncthreads();

    // Phase 2: Column-by-column forward substitution
    // 4 warps, each handling 32 rows
    int warp_id = threadIdx.y;
    int lane = threadIdx.x;
    int row = warp_id * 32 + lane;

    for (int col_b = 0; col_b < NRHS; col_b += WPB) {
        // Load WPB columns of B into registers
        float b_regs[WPB];
        for (int w = 0; w < WPB; w++)
            b_regs[w] = B[row * NRHS + col_b + w];

        // Forward substitution, column by column
        for (int j = 0; j < N; j++) {
            int j_warp = j / 32;       // which warp owns row j
            int j_lane = j % 32;       // which lane within that warp

            // The warp that owns row j divides by diagonal
            if (warp_id == j_warp && lane == j_lane) {
                float diag = __half2float(smem_A[j][j]);
                for (int w = 0; w < WPB; w++)
                    b_regs[w] /= diag;
            }

            // Broadcast solved values from the owning warp to all warps
            // This requires inter-warp communication via shared memory
            __shared__ float solved_vals[WPB];
            if (warp_id == j_warp && lane == j_lane) {
                for (int w = 0; w < WPB; w++)
                    solved_vals[w] = b_regs[w];
            }
            __syncthreads();

            // All threads with row > j update
            if (row > j) {
                float a_val = __half2float(smem_A[row][j]);
                for (int w = 0; w < WPB; w++)
                    b_regs[w] -= a_val * solved_vals[w];
            }
            __syncthreads();
        }

        // Write results back
        for (int w = 0; w < WPB; w++)
            B[row * NRHS + col_b + w] = __float2half(b_regs[w]);
    }
}
```

### Optimization Notes

1. **Inter-warp sync is the bottleneck.** Each column j requires an `__syncthreads()` to broadcast the solved value. For N=128, that's 128 sync points per batch of WPB columns. This is expensive but still cheaper than 9 kernel launches.

2. **Use FP32 accumulation.** Keep B values in FP32 registers throughout the solve, only convert to BF16 on final write. This matches cuBLAS F32 precision.

3. **Pad shared memory.** Use 128+4 stride instead of 128 to avoid bank conflicts on the column loads.

4. **Increase WPB for throughput.** More B columns per pass = more arithmetic per sync point. WPB=16 means each thread holds 16 FP32 values = 64 bytes = 16 registers. With 128 threads, that's 2048 registers for B alone, plus ~128 for A accesses. Well within the 64K register file limit.

5. **For N=64:** Only 2 warps needed. Thread block = (32, 2). Shared memory = 64*64*2 = 8 KB. Much smaller, allows higher occupancy.

---

## 4. For N=256: Blocked Single-Kernel Approach

At N=256, the triangular matrix is 256*256*2 = 128 KB in BF16, which exceeds the 99 KB shared memory limit. So the monolithic approach needs blocking:

### Two-Phase Single Kernel

```
Phase 1: Solve top half (rows 0-127)
  - Load A[0:128, 0:128] into shared memory (32 KB)
  - Forward substitution for 128 rows (as above)
  - Store X[0:128] to global memory

Phase 2: Update and solve bottom half (rows 128-255)
  - Load A[128:256, 0:128] into shared memory (off-diagonal block)
  - GEMM: B[128:256] -= A[128:256, 0:128] @ X[0:128]
    (This is a 128x128 @ 128xNRHS GEMM -- can use mma.sync BF16 m16n8k16!)
  - Load A[128:256, 128:256] into shared memory (bottom-right triangular block)
  - Forward substitution for 128 rows
  - Store X[128:256]
```

### The GEMM Update Can Use Tensor Cores

The off-diagonal GEMM in Phase 2 is a standard dense GEMM. Within the same kernel, you can call mma.sync instructions to accelerate this. This is where the tensor core advantage manifests -- not in the triangular solve itself, but in the off-diagonal update.

For N=256, the GEMM is 128x128 @ 128xNRHS. For NRHS=64: this is a 128x128 @ 128x64 GEMM, well suited to 4-warp mma.sync with m16n8k16 tiles.

### Shared Memory Budget

```
Phase 1: A[128x128] in BF16 = 32 KB + B working set
Phase 2 GEMM: A[128x128] offdiag in BF16 = 32 KB + B accum
Phase 2 solve: A[128x128] diagonal in BF16 = 32 KB + B working set
```

Each phase uses at most ~40 KB. Fits comfortably in 99 KB.

---

## 5. Why This Approach Beats cuBLAS at Small N

| N | cuBLAS launches | Single-kernel launches | Overhead saved |
|---|----------------|----------------------|----------------|
| 64 | 9 | 1 | ~24-40 us |
| 128 | 9 | 1 | ~24-40 us |
| 256 | 11 | 1 | ~30-50 us |

At 47-57 us total cuBLAS time, saving 24-50 us of launch overhead is a 40-90% improvement. This is the primary lever at small N.

For N >= 512, the GEMM compute dominates and the launch overhead matters less. At that point, the recursive approach with tensor-core GEMM (existing briefs) is the right strategy. But for N=64-256, the monolithic single-kernel approach is the better bet.

---

## Caveats

1. **N=128 monolithic with 128 syncthreads is expensive.** Each `__syncthreads()` costs ~20-30 cycles on sm_120. 128 syncs = ~3000-4000 cycles = ~1.5-2 us. This is small relative to the launch overhead saved.

2. **Inter-warp communication pattern.** The solved-value broadcast from one lane to all threads is the critical path. An alternative: use cooperative groups' `this_grid().sync()` for grid-level sync between phases, and `__shfl_sync` within warps. For N <= 32, no inter-warp communication is needed (pure warp-shuffle).

3. **The GEMM portion of the N=256 approach benefits from tensor cores.** But the forward-substitution portion does not. If the forward substitution dominates at N=128, tensor cores provide no benefit. The arithmetic is N^2 * NRHS / 2 for substitution, vs N^2/4 * NRHS for the GEMM update. For a 2-block decomposition (N=256, NB=128), the GEMM is exactly equal to one substitution in FLOPs. So tensor cores help for ~50% of the work.

4. **The MAGMA trtri_diag recursive inversion (IB=16 -> NB=128) is a reference for how to build up block inversions.** If the worker chooses the diagonal-inversion approach (instead of direct solve), this exact recursive build-up pattern (16->32->64->128 via triple GEMMs) could be replicated with mma.sync for the GEMM portions.

5. **Numerical precision.** The single-kernel forward substitution approach is numerically identical to standard TRSM (no inversion, no reordering). This is strictly more stable than the diagonal-inversion approach. For arbitrary triangular matrices (not just from Cholesky/LU), this is the safer choice.

---

## Sources

- MAGMA strsm.cu: https://github.com/maxhutch/magma/blob/master/magmablas/strsm.cu
- MAGMA strtri.cuh (NB=128, IB=16): https://github.com/maxhutch/magma/blob/master/magmablas/strtri.cuh
- MAGMA strtri_diag.cu: https://github.com/maxhutch/magma/blob/master/magmablas/strtri_diag.cu
- MAGMA trtri_diag API: https://icl.utk.edu/projectsfiles/magma/doxygen/group__magma__trtri__diag__batched.html
- MAGMA TRSM API: https://icl.utk.edu/projectsfiles/magma/doxygen/group__magma__trsm.html
- Carrica et al. "Hierarchical Precision" (2026): https://arxiv.org/html/2601.08082v1
- Carrica & Onyango "Julia Recursive TRSM" (2025): https://arxiv.org/html/2504.13821v1
- KBLAS GPU library: https://github.com/ecrc/kblas-gpu
- Charara et al. "Redesigning Triangular Dense Matrix" (2016): https://link.springer.com/chapter/10.1007/978-3-319-43659-3_35
