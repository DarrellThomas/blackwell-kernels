# LU Factorization: Mixed-Precision Tensor Core Strategy and Batched Opportunity

**Source:** Multiple (see per-section citations)
**Relevant to:** numerical/ worker (LU factorization / getrf)
**Worker's current problem:** cuSOLVER baseline is 9.4ms at N=4096. Need to identify where tensor cores (BF16 mma.sync m16n8k16) can accelerate the factorization, and whether batched small LU offers an easier win.

---

## 1. Where Tensor Cores Apply in LU Factorization

### The LU Algorithm Has Three Distinct Compute Patterns

| Phase | Operation | Compute Type | Tensor Core? | Time Share (N=4096) |
|-------|-----------|-------------|-------------|---------------------|
| Panel (GETF2) | argmax, swap, scale, rank-1 update | Scalar FP32 | **NO** | ~5-10% |
| TRSM | L \ A = U (triangular solve) | GEMM-like | **YES** (via recursive GEMM) | ~5-10% |
| Trailing GEMM | A -= L * U update | Pure GEMM | **YES** | ~80-85% |
| LASWP | Row permutations | Memory copies | NO | ~3-5% |

**The trailing GEMM is where MMA provides benefit.** This is 80%+ of compute for N=4096.

### Using BF16 MMA for Trailing GEMM

The trailing update is: `A_trailing -= L_column_block * U_row_block`

This is a standard GEMM. Our existing BF16 GEMM kernel (0.97x cuBLAS) can be adapted:
1. Convert FP32 tiles to BF16 before loading into MMA fragments
2. MMA accumulates in FP32
3. Subtract result from FP32 trailing matrix

**Precision analysis:**
- BF16 has 7-bit mantissa = ~2 decimal digits
- For LU, the trailing update accumulates errors over N/NB iterations
- With NB=64, N=4096: 64 iterations of rank-64 updates
- Error: O(N * eps_BF16) = O(4096 * 2^-8) = O(16) per element
- **This is TOO MUCH for direct use.** Need mixed-precision approach.

### Mixed-Precision Approach (Haidar et al., SC18)

**Source:** Haidar et al., "Harnessing GPU Tensor Cores for Fast FP16 Arithmetic" (SC18, IEEE)

The proven approach:
1. Factor A in lower precision (FP16/BF16) using tensor cores — FAST
2. Use the low-precision factors as a preconditioner
3. Iterative refinement in FP32/FP64 converges to full precision

**Performance:** Up to 4x speedup on V100 using FP16 tensor cores.

**For sm_120 with BF16 mma.sync:**
- Step 1: sgetrf in BF16 (tensor-core accelerated trailing GEMM) = ~2-3ms
- Step 2: Iterative refinement (3-5 GEMV iterations) = ~0.5ms
- Total: ~3ms, vs cuSOLVER's 9.4ms = potential **3x speedup**

**MAGMA provides this:** `magma_dsgesv_gpu()` (single→double) and `magma_dhgesv_gpu()` (half→double). These do exactly LU-in-low-precision + iterative refinement.

### cuSOLVER's Internal Tensor Core Usage

From the Cholesky profiling, cuSOLVER uses a kernel named `getrf_wo_pivot_params_<float, 0, 256, 1, 64, 64, 68>`. The fact that it achieves ~15 TFLOPS on one SM strongly suggests internal tensor core usage for SYRK/GEMM updates. For LU with pivoting, cuSOLVER likely:
1. Uses FP32 for the panel factorization (pivoting needs precision)
2. Uses TF32 or BF16 tensor cores for trailing GEMM within the monolithic kernel
3. Achieves accuracy by keeping accumulators in FP32

**We know TF32 MMA has a B fragment defect on sm_120** (diagonal broadcasting). cuSOLVER may use a workaround (decomposed MMA, or BF16 MMA, or a proprietary instruction variant).

---

## 2. Using MMA for TRSM Within the Monolithic Kernel

### TRSM as Recursive GEMM

Triangular solve L * X = B can be decomposed:
```
Partition L into [L11, 0; L21, L22] and X into [X1; X2]:
  X1 = L11 \ B1      (small triangular solve)
  B2 -= L21 * X1     (GEMM — tensor core!)
  X2 = L22 \ B2      (recurse)
```

At the leaf level (tile size = MMA tile = 16), the triangular solve is scalar.
At higher levels, the GEMM dominates and benefits from tensor cores.

### Within the Monolithic Kernel

After panel factorization produces L (NB x NB lower triangular) and the row of U needs updating:
```
U_row = L_panel \ A_row
```

This TRSM operates on NB x (N - k*NB) elements. For NB=64 and k=0: 64 x 4032 elements.

Recursive decomposition with BF16 MMA at each level:
- Level 0: solve 64x64 triangular (scalar in FP32)
- The GEMM updates at each recursion level use mma.sync
- Most of the work is in the GEMM portion

---

## 3. Batched Small LU — The Easier Win

### Why Batched Small Matrices Are Different

For batched LU of many small matrices (N=32-256, batch=100-10000):
- cuSOLVER batched getrf has kernel launch overhead per batch item
- cuSOLVER doesn't fully optimize for small N (panel overhead dominates)
- MAGMA achieves 8.72x faster than cuBLAS for N=32 batched

### Our Advantages for Batched Small LU

We already have the building blocks:
1. **MAGMA's rowid trick** eliminates row swap overhead (register-only pivoting)
2. **Register-resident factorization** for N <= 60 (entire matrix in registers)
3. **Warp-level synchronization** for N <= 32 (no __syncthreads needed)
4. **Our fast batched GEMM** (1.34x cuBLAS) for trailing updates in larger batched LU

### Concrete Opportunity

| N | Batch | cuSOLVER batched (us) | Achievable (us) | Expected Speedup |
|---|-------|----------------------|-----------------|------------------|
| 32 | 1000 | ~200 (launch overhead dominated) | ~25 | **~8x** |
| 64 | 1000 | ~500 | ~100 | **~5x** |
| 128 | 1000 | ~2000 | ~500 | **~4x** |
| 256 | 100 | ~1500 | ~500 | **~3x** |

(Estimates based on MAGMA's published speedups on V100, scaled for sm_120)

### Implementation Path

For N <= 32 (register-only, single-warp):
```
__global__ void batched_lu_small(float** A_batch, int** ipiv_batch, int N, int batch_count) {
    int batch_id = blockIdx.x;
    if (batch_id >= batch_count) return;

    float* A = A_batch[batch_id];
    int* ipiv = ipiv_batch[batch_id];

    // Each thread holds one row of the matrix (N values in registers)
    float rA[N];  // N is compile-time template parameter
    int rowid = threadIdx.x;

    // Load row
    for (int j = 0; j < N; j++) rA[j] = A[rowid * N + j];

    // Factorize
    for (int i = 0; i < N; i++) {
        // Warp-level argmax (no shared memory needed for N <= 32)
        float max_val = (rowid >= i) ? fabsf(rA[i]) : 0.0f;
        int max_id = rowid;
        // Butterfly reduction via __shfl_xor_sync
        for (int mask = 16; mask > 0; mask >>= 1) {
            float other_val = __shfl_xor_sync(0xFFFFFFFF, max_val, mask);
            int other_id = __shfl_xor_sync(0xFFFFFFFF, max_id, mask);
            if (other_val > max_val) { max_val = other_val; max_id = other_id; }
        }

        // Virtual row swap (rowid trick)
        if (rowid == max_id) rowid = i;
        else if (rowid == i) rowid = max_id;
        ipiv[i] = max_id;

        // Scale
        float pivot = __shfl_sync(0xFFFFFFFF, rA[i], i);
        if (rowid > i) rA[i] /= pivot;

        // Rank-1 update
        float li = rA[i];
        for (int j = i + 1; j < N; j++) {
            float uij = __shfl_sync(0xFFFFFFFF, rA[j], i);
            if (rowid > i) rA[j] -= li * uij;
        }
    }

    // Write back with permutation applied
    for (int j = 0; j < N; j++) A[rowid * N + j] = rA[j];
}
```

This kernel would be extremely fast: zero global memory traffic during factorization, warp-level synchronization only, all shuffles.

### Batched LU is Independent of Single-Matrix LU

The batched small LU kernel can be developed independently. It doesn't need the monolithic blocked algorithm. It's a separate, simpler kernel that can be shipped quickly while the larger N=4096 monolithic kernel is developed.

---

## 4. cuSOLVER vs Monolithic Custom: Expected Performance Breakdown

### N=4096 FP32 Compute Budget

Total FLOPs for LU: 2/3 * N^3 = 2/3 * 4096^3 = 45.8 GFLOP

| Phase | FLOPs | % | Bottleneck |
|-------|-------|---|-----------|
| Panel (GETF2, all iterations) | ~0.5 GFLOP | 1% | Serial pivoting |
| TRSM (all iterations) | ~2.3 GFLOP | 5% | Memory bandwidth |
| Trailing GEMM (all iterations) | ~43 GFLOP | 94% | **Compute** |
| LASWP (all iterations) | ~0 (memory moves) | N/A | Bandwidth |

### Roofline for Trailing GEMM

RTX 5090 BF16 tensor core throughput: ~330 TFLOPS (mma.sync m16n8k16)
FP32-equivalent throughput (BF16 MMA with FP32 accum): ~165 TFLOPS

At 50% efficiency (realistic for variable-size GEMMs in LU): ~82 TFLOPS

43 GFLOP / 82 TFLOPS = **0.52 ms** for all trailing GEMMs combined

### cuSOLVER's 9.4ms = Very Conservative

If the trailing GEMM can be done in 0.5ms, cuSOLVER's 9.4ms means it spends 8.9ms on overhead + panel + LASWP. This suggests:
1. cuSOLVER may NOT use tensor cores for the trailing GEMM at FP32 precision
2. Or cuSOLVER uses a single SM (as seen in profiling), limiting GEMM parallelism
3. Or the 9.4ms includes host-side overhead

**Profile cuSOLVER with nsys** to determine which: is it one kernel? Multiple? How many SMs? This is the single most important diagnostic before starting implementation.

---

## Sources

- Haidar et al., "Harnessing GPU Tensor Cores for Fast FP16 Arithmetic to Speed Up Mixed-Precision Iterative Refinement Solvers" (SC18, IEEE, 2018)
- MAGMA mixed-precision solvers: dsgesv_gpu, dhgesv_gpu
- MAGMA batched GETRF: magma_sgetrf_batched, magma_sgesv_batched_small
- Cholesky agent state (experiment 22: cuSOLVER nsys profiling)
- cuSOLVERDx getrf_partial_pivot example
