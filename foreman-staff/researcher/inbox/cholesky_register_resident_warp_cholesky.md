# Register-Resident Warp Cholesky for N<=32

**Source:** https://par.nsf.gov/servlets/purl/10065613 (MAGMA Guide for Small Matrices, IEEE TPDS 2018)
**Source:** https://www.sciencedirect.com/science/article/abs/pii/S1877750316305154 (Fast Cholesky on GPUs, MAGMA 2017)
**Source:** https://researchgate.net/publication/286681434 (Fast Batched Cholesky on GPU, IEEE 2014)
**Source:** https://dl.acm.org/doi/abs/10.1145/3038228.3038237 (High-Performance Cholesky GPU-Only, 2017)
**Relevant to:** Cholesky worker
**Worker's current problem:** Monolithic large Cholesky blocked by TF32 MMA defect. Pivoting to batched small Cholesky (N=32-64). Need the fastest possible N<=32 kernel.

## What This Is

For N<=32, the entire matrix fits in a single warp's registers. This enables
a Cholesky kernel with zero shared memory accesses and zero synchronization
barriers -- everything happens via warp shuffles.

## Architecture: One Warp Per Matrix

```
Grid: ceil(batch_size / warps_per_block) blocks
Block: multiple warps (e.g., 8 warps = 256 threads for 8 matrices per block)
Each warp: 32 threads, one matrix

Thread t (lane 0-31) owns row t of the N×N matrix
For N<32: threads with lane >= N are idle but participate in shuffles
```

### Register Budget Per Thread

Each thread holds one row of the matrix (N FP32 values):
- N=16: 16 data regs + ~10 control = 26 regs/thread -> high occupancy
- N=24: 24 data regs + ~10 control = 34 regs/thread -> high occupancy
- N=32: 32 data regs + ~10 control = 42 regs/thread -> good occupancy
- 255 regs/thread limit is far from reached

### Occupancy for N=32, 256 threads/block

42 regs/thread * 256 threads = 10,752 regs/block
sm_120 has 65,536 regs/SM -> 6 blocks/SM (limited by other factors before regs)
With 0 bytes shared memory: 48 warps/SM limit -> 6 blocks * 8 warps = 48 -> exactly at limit

## Algorithm: Column-by-Column Cholesky in Registers

```
for j = 0 to N-1:
    // STEP 1: Compute dot product for diagonal
    // Thread j computes: sum = A[j][0]^2 + A[j][1]^2 + ... + A[j][j-1]^2
    // This is a local computation using values already in thread j's registers

    // STEP 2: Compute L[j][j] = sqrt(A[j][j] - sum)
    // Thread j computes the square root
    float diag = sqrtf(my_row[j] - sum_of_squares);

    // STEP 3: Broadcast L[j][j] to all threads
    float diag_broadcast = __shfl_sync(0xFFFFFFFF, diag, j);

    // STEP 4: Compute L[i][j] for i > j
    // Each thread i>j: L[i][j] = (A[i][j] - dot(L[i][0:j], L[j][0:j])) / L[j][j]
    // The dot product needs L[j][0:j] which is in thread j's registers
    // Broadcast each element via shuffle:
    for k = 0 to j-1:
        float L_jk = __shfl_sync(0xFFFFFFFF, my_row[k], j);
        my_row[j] -= my_row[k] * L_jk;  // Accumulate dot product
    my_row[j] /= diag_broadcast;

    // STEP 5: No trailing update needed -- implicit in the column loop
    // (This is unblocked Cholesky, O(N^3/3) flops, done column by column)
```

### Warp Shuffle Count

For each column j:
- 1 shuffle to broadcast diagonal
- j shuffles to broadcast L[j, 0:j] for dot product
- Total shuffles: sum(j, j=0..N-1) + N = N*(N-1)/2 + N = N*(N+1)/2

For N=32: 32*33/2 = 528 shuffles per matrix.
At ~4 cycles per shuffle: 528 * 4 = 2112 cycles.
At 2.41 GHz (RTX 5090 boost): ~0.88 us per matrix.

With 170 SMs * 6 blocks/SM * 8 warps/block = 8160 concurrent matrices:
Throughput: 8160 matrices / 0.88 us = ~9.3 million matrices/second.

## Key Implementation Details

### Handling N < 32

For N=16 or N=24, threads with lane >= N hold zeros. The algorithm works
unchanged -- those threads just don't contribute meaningful values, and their
results are never read. This wastes some warp lanes but avoids code complexity.

### SPD Check

If the matrix is not SPD, `sqrtf` of a negative number produces NaN. Each
thread can check `if (my_row[j] < 0 && lane_id == j) { info = j+1; }` and
write a per-matrix error flag.

### Output

After the column loop, each thread's `my_row[0:N]` contains the j-th row of L.
Write back to global memory with coalesced stores:
```
// Thread t writes row t of matrix m
L[m * N * N + t * N + 0..N-1] = my_row[0..N-1]
```

For best coalescing, use a column-major layout (all threads write column j
simultaneously).

### Double-Buffered Loading

While one warp is computing Cholesky, the next matrix can be loaded into
registers. This hides global memory latency. However, for N=32 this doubles
register usage to ~84 regs/thread, reducing occupancy.

## Performance Targets

| Configuration | Estimated Throughput | vs cuSOLVER Batched |
|---------------|---------------------|---------------------|
| N=16, batch=10000 | ~15M matrices/sec | Likely 5-10x |
| N=24, batch=10000 | ~12M matrices/sec | Likely 3-5x |
| N=32, batch=10000 | ~9M matrices/sec | Likely 3-5x |

These estimates assume pure compute bound. Actual performance depends on
global memory bandwidth for loading/storing matrices.

### Memory Bandwidth Check

N=32, FP32: each matrix is 32*32*4 = 4096 bytes.
At 9.3M matrices/sec: 9.3M * 4096 * 2 (read+write) = 76 GB/s.
RTX 5090 bandwidth: 1792 GB/s. So we are heavily compute bound, not memory
bound. Good -- the register-resident approach wins here.

## Comparison with MAGMA

MAGMA's batched potrf for N<=32 uses the same basic approach (one thread per
row, register-resident data, warp shuffles for communication). Their published
results (2017):
- 6x speedup over cuBLAS batched on Pascal (P100)
- 11.8x speedup on Volta (V100)

MAGMA 2.9.0 (January 2025) improved potrf_batched performance and added sm_120
support. Our custom kernel should benchmark against MAGMA 2.9 as the primary
competitor, not cuSOLVER.

## What We Can Do Better Than MAGMA

1. **Specialize for fixed N:** MAGMA handles arbitrary N with runtime branching.
   We can template on N for zero-overhead dispatch.

2. **Sub-warp utilization for N<32:** For N=16, MAGMA wastes 16 lanes per warp.
   We could pack 2 matrices per warp (thread 0-15 = matrix A, thread 16-31 =
   matrix B). This doubles throughput for N<=16.

3. **FP32 fused dot product:** Use `__fmaf_rn` for each accumulation step
   to maximize precision with zero extra cost.

4. **Batched solve (potrs):** After factorization, the same register-resident
   data can be used for forward/backward substitution without a second global
   memory round trip.

## Caveats

1. **N>32 requires shared memory.** The warp-shuffle approach only works for
   N<=32 (warp size). For N=33-64, need shared memory for cross-warp
   communication, adding syncthreads and bank conflict concerns.

2. **Precision.** FP32 Cholesky is numerically stable for well-conditioned
   SPD matrices. For condition numbers > 10^6, consider mixed-precision
   (FP32 factorization + FP64 iterative refinement).

3. **No tensor cores.** For N<=32, tensor cores are not useful -- the matrices
   are too small for MMA to provide benefit. Pure FP32 CUDA core arithmetic
   with warp shuffles is optimal.
