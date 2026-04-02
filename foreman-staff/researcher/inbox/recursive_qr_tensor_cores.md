# Recursive QR Factorization with Tensor Cores — Deep Dive

**Sources:**
- Leng, Zou, Wang, Wu, Zhang. "High Performance Householder QR Factorization on Emerging GPU Architectures Using Tensor Cores." IEEE TPDS, Vol. 36, Issue 3, pp. 422-436, 2025. DOI: 10.1109/TPDS.2024.3522776
- Zhang, Wu. "High Accuracy Low Precision QR Factorization and Least Square Solver on GPU with TensorCore." arXiv:1912.05508, 2019.
- Elmroth, Gustavson. "Applying Recursion to Serial and Parallel QR Factorization Leads to Better Performance." IBM J. Research & Development, Vol. 44, No. 4, pp. 605-624, 2000.
- Elmroth, Gustavson. "A Faster and Simpler Recursive Algorithm for the LAPACK Routine DGELS." BIT, Vol. 41, No. 5, pp. 936-949, 2001.
- LAPACK reference implementations: DGEQRF, DGEQR2, DLARFT, DLARFB (netlib.org)

**Relevant to:** QR worker
**Worker's current problem:** Need efficient QR factorization that can beat cuSOLVER on sm_120 (RTX 5090, mma.sync m16n8k16 BF16)

---

## 1. What This Is

The IEEE TPDS paper by Leng et al. demonstrates that recursive QR factorization, combined with tensor core acceleration, achieves dramatic speedups over cuSOLVER: up to 8.67x (FP32), 6.22x (FP64), and 4.03x (FP16), tested on A100 and RTX 3090. The more realistic cuSOLVER-vs-cuSOLVER comparison shows 1.4x at 32768x16384 in FP32. The core insight is that standard blocked QR produces tall-skinny GEMMs that tensor cores handle poorly, while recursive QR restructures these into progressively squarer GEMMs that tensor cores can accelerate.

---

## 2. The Blocked QR Algorithm (DGEQRF) — What We're Improving On

LAPACK's DGEQRF implements blocked Householder QR. Understanding this is essential before understanding the recursive improvement.

### 2.1 The Algorithm (from LAPACK source)

```
Given m x n matrix A, k = min(m, n), block size nb:

for i = 1 to k-nx step nb:          // Process panels of width nb
    ib = min(k - i + 1, nb)

    // Step 1: PANEL FACTORIZATION (DGEQR2)
    // Factor A(i:m, i:i+ib-1) into ib Householder reflectors
    // Each reflector H(j) = I - tau(j) * v(j) * v(j)^T
    // Applied column by column: compute v(j), apply to remaining columns j+1..ib
    call DGEQR2(m-i+1, ib, A(i,i), lda, tau(i), work, info)

    // Step 2: FORM T MATRIX (DLARFT)
    // Build upper triangular T such that H(1)*H(2)*...*H(ib) = I - V*T*V^T
    // This is the "compact WY representation"
    call DLARFT('Forward', 'Columnwise', m-i+1, ib, A(i,i), lda, tau(i), T, ldt)

    // Step 3: TRAILING MATRIX UPDATE (DLARFB)
    // Apply block reflector: A(i:m, i+ib:n) = (I - V*T*V^T)^T * A(i:m, i+ib:n)
    // This decomposes into BLAS-3 operations (GEMM + TRMM)
    call DLARFB('Left', 'Transpose', 'Forward', 'Columnwise',
                m-i+1, n-i-ib+1, ib, A(i,i), lda, T, ldt, A(i,i+ib), lda, work, ldwork)

// Handle remaining columns with unblocked DGEQR2
if (i <= k) call DGEQR2(m-i+1, k-i+1, A(i,i), lda, tau(i), work, info)
```

### 2.2 DGEQR2 — Unblocked Panel Factorization

DGEQR2 processes columns one at a time:

```
for j = 1 to min(m, n):
    // Step A: Generate Householder reflector H(j) for column j
    // DLARFG computes scalar tau and vector v such that
    // H(j) = I - tau * v * v^T zeros out A(j+1:m, j)
    call DLARFG(m-j+1, A(j,j), A(j+1,j), 1, tau(j))

    // Step B: Apply H(j) to remaining columns A(j:m, j+1:n)
    // This is a rank-1 update: A := (I - tau * v * v^T) * A
    // = A - tau * v * (v^T * A)
    // First: w = v^T * A(j:m, j+1:n)     (BLAS-2: matrix-vector multiply)
    // Then:  A -= tau * v * w^T            (BLAS-2: rank-1 update)
    call DLARF('Left', m-j+1, n-j, A(j+1,j), 1, tau(j), A(j,j+1), lda, work)
```

**GPU challenge:** Each column depends on the previous one (sequential). The DLARF operations (GEMV + rank-1 update) are BLAS-2 and memory-bound. This is why the panel is the bottleneck on GPUs.

### 2.3 DLARFT — Building the T Matrix

DLARFT forms the upper triangular T such that the block reflector is H = I - V*T*V^T. The modern LAPACK implementation uses a recursive divide-and-conquer approach:

```
Given k reflectors stored in columns of V with scalars tau:

Base case (k=1): T(1,1) = tau(1)

Recursive case:
    l = k/2
    // Recursively form T11 for first l reflectors
    T11 = DLARFT(V(:,1:l), tau(1:l))

    // Recursively form T22 for last k-l reflectors
    T22 = DLARFT(V(:,l+1:k), tau(l+1:k))

    // Form off-diagonal: T12 = -T11 * V1^T * V2 * T22
    //   (This is the key cross-term)
    T12 = V1^T * V2         // DGEMM: (l x m) * (m x (k-l))
    T12 = -T11 * T12 * T22  // DTRMM twice

Result: T = [T11  T12]
            [ 0   T22]
```

**Key insight for GPU:** The off-diagonal block T12 computation involves a GEMM of V1^T * V2. For nb=64, this is a 32x32 GEMM at the top level — small but structured. The recursive formulation lets you build T entirely from BLAS-3 operations.

**Size considerations:** T is always nb x nb (e.g., 64x64). This is small enough to fit in registers or shared memory. On sm_120, a 64x64 FP32 T matrix is 16 KB — fits easily in the 99 KB shared memory budget.

### 2.4 DLARFB — Applying the Block Reflector

DLARFB applies H^T = (I - V*T^T*V^T) to the trailing matrix C. For the QR case (Left, Transpose, Forward, Column-stored):

```
// C = (C1)  where C1 is the top nb rows
//     (C2)

// V = (V1)  where V1 is nb x nb lower triangular (unit diagonal)
//     (V2)

Step 1: W = C1^T                                    // Copy
Step 2: W = W * V1        (DTRMM: nb x n_trail)     // Triangular multiply
Step 3: W = W + C2^T * V2  (DGEMM: nb x n_trail)    // *** MAIN GEMM ***
Step 4: W = W * T^T        (DTRMM: nb x n_trail)    // Triangular multiply
Step 5: C2 -= V2 * W^T     (DGEMM: update bottom)   // *** MAIN GEMM ***
Step 6: W = W * V1^T       (DTRMM: nb x n_trail)    // Triangular multiply
Step 7: C1 -= W^T           (element-wise update)
```

**The two dominant GEMMs (Steps 3 and 5):**
- Step 3: (n_trail x nb) = (n_trail x m_remain) * (m_remain x nb) -- TALL-SKINNY
- Step 5: (m_remain x n_trail) -= (m_remain x nb) * (nb x n_trail) -- TALL-SKINNY

Both have one dimension = nb. With nb=64, these are 64-wide GEMMs. **This is the fundamental problem that recursive QR solves.**

---

## 3. The Recursive QR Algorithm — Elmroth & Gustavson

### 3.1 Core Idea

The Elmroth-Gustavson recursive QR algorithm (RGEQR3) replaces the fixed-block-size loop of DGEQRF with a recursive splitting strategy. Instead of processing panels of width nb, it recursively divides the columns into two halves.

From the 2001 Elmroth-Gustavson paper:

> "RGEQR3 factors a matrix by recursively dividing it into two approximately equal halves. For each recursive step, the first half is recursively factorized, the second half is updated and also recursively factorized."

### 3.2 The RGEQR3 Algorithm

```
function [Y, R, T] = RGEQR3(A[j:m, j:j+jb-1])
    // A is m_remaining x jb (jb columns to factor)

    if jb == 1:
        // BASE CASE: single column
        // Generate one Householder reflector
        [v, tau] = DLARFG(A(:, 1))
        T = tau  // 1x1 T matrix
        R = A(1,1)  // scalar R value
        return

    // RECURSIVE CASE: split columns in half
    jb1 = jb / 2
    jb2 = jb - jb1

    // Step 1: Recursively factor left half
    [Y1, R1, T1] = RGEQR3(A[:, 1:jb1])

    // Step 2: Apply reflectors to right half (TRAILING UPDATE)
    // A[:, jb1+1:jb] = (I - Y1 * T1^T * Y1^T) * A[:, jb1+1:jb]
    // This is DLARFB applied to the RIGHT HALF (not the entire trailing matrix!)
    W = T1 * Y1^T * A[:, jb1+1:jb]     // GEMM: (jb1 x m) * (m x jb2)
    A[:, jb1+1:jb] -= Y1 * W           // GEMM: (m x jb1) * (jb1 x jb2)

    // Step 3: Recursively factor updated right half
    [Y2, R2, T2] = RGEQR3(A[jb1+1:m, jb1+1:jb])

    // Step 4: Merge T matrices
    // T = [T1   -T1 * Y1^T * Y2 * T2]
    //     [ 0                     T2 ]
    T12 = -T1 * (Y1^T * Y2) * T2       // GEMM + 2x TRMM

    // Assemble outputs
    Y = [Y1, [0; Y2]]
    R = [R1, R12; 0, R2]
    T = [T1, T12; 0, T2]
```

### 3.3 Why This Converts Tall-Skinny to Square GEMMs

Consider factoring an m x n matrix (m >= n) with the recursion:

**Standard blocked QR (DGEQRF with block size nb=64):**
- Every trailing update GEMM has shape: (nb x m_remain) * (m_remain x n_remain)
- The inner dimension is m_remain (large), but one outer dimension is always nb=64
- For m=n=4096, nb=64: each GEMM is 64 x ~4000 x ~4000 -- terribly tall-skinny

**Recursive QR (RGEQR3):**
- Level 0 (top): Split n into n/2 + n/2. The Step 2 GEMM is (n/2 x m) * (m x n/2)
- Level 1: Split n/2 into n/4 + n/4. GEMMs are (n/4 x m) * (m x n/4)
- Level k: GEMMs are (n/2^k x m) * (m x n/2^k)
- Base case: n/2^k = 1 (single column)

For n=4096:
- Top-level GEMM: 2048 x m x 2048 -- **SQUARE!** (perfect for tensor cores)
- Next level: 1024 x m x 1024 -- still large and square
- ...down to small base cases

**The bottom levels (small GEMMs) are handled efficiently by the recursion itself.** The key is that the MOST EXPENSIVE GEMMs (at the top of the recursion tree) are the LARGEST and MOST SQUARE.

### 3.4 Operation Count

From the Zhang/Wu paper: Recursive Householder QR costs approximately 2mn^2 - (2/3)n^3 flops, same as standard Householder QR. The operation count is identical -- only the GEMM shapes change.

Modified Gram-Schmidt recursive QR costs 2mn^2 (slightly more, ~n^3/3 extra), but is simpler and has better numerical stability properties for the backward error.

### 3.5 The Outer Loop: RGEQRF

The Elmroth-Gustavson paper describes two levels:

1. **RGEQRF (outer):** Loops over block columns of width nb, calling RGEQR3 for the panel and DLARFB for the trailing update (entire remaining columns). This provides the standard blocked structure.

2. **RGEQR3 (inner):** Recursively factors a panel of width nb. This converts the panel's BLAS-2 operations (DGEQR2's column-by-column approach) into BLAS-3 operations (GEMM calls at each recursion level).

The paper also describes a **fully recursive HRQR** (Hybrid Recursive QR) that eliminates the outer loop entirely, making the recursion handle both panel and trailing update. This is the preferred approach for tensor core implementations because it maximizes GEMM sizes throughout.

```
Algorithm 2.1 (RGEQRF wrapper):
    do j = 1, n, nb
        jb = min(n - j + 1, nb)
        call RGEQR3(A(j:m, j:j+jb-1)) = (Y, R, T)
        if (j + jb <= n) then
            // Apply to trailing matrix -- BIG GEMM, but still nb-wide
            A(j:m, j+jb:n) = (I - Y*T^T*Y^T) * A(j:m, j+jb:n)
        endif
    enddo
```

The fully recursive variant (Section 6 of the paper, HRQR) integrates the outer blocking into the recursion, so the top-level GEMMs are n/2-wide instead of nb-wide.

---

## 4. Mixed-Precision Strategy for Tensor Cores

### 4.1 From the IEEE TPDS Paper (Leng et al.)

The paper extends tensor core support to three precision levels:
- **FP16 native:** Direct tensor core GEMM in FP16, fastest but lowest accuracy
- **FP32 via FP16 tensor cores:** Cast operands to FP16 for the GEMM, accumulate in FP32. The GEMM portion (trailing update) runs on tensor cores while panel factorization uses FP32 scalar math
- **FP64 via INT8 tensor cores:** Novel technique mapping FP64 arithmetic onto INT8 tensor cores

The speedup numbers (vs "state-of-the-art" implementations, not cuSOLVER directly):
| Precision | Speedup | Notes |
|-----------|---------|-------|
| FP64 | 6.22x | Via INT8 tensor cores |
| FP32 | 8.67x | Via FP16 tensor cores, FP32 accumulate |
| FP16 | 4.03x | Native FP16 tensor cores |

The realistic cuSOLVER comparison: **1.4x at 32768 x 16384 in FP32**.

### 4.2 From the Zhang/Wu Paper (arXiv:1912.05508)

This earlier paper by two of the same authors uses a different QR variant — recursive Modified Gram-Schmidt (RMGSQR) instead of Householder. Key implementation details:

**Panel factorization (CAQR — Communication Avoiding QR):**
- Divide the tall matrix into 256x32 subblocks
- Each subblock fits in shared memory on the GPU
- Factor each subblock independently using Modified Gram-Schmidt (Algorithm 4)
- Stack the R factors vertically and recurse until small enough for a single threadblock
- The inter-block communication happens via cuBLAS batched SGEMM (tensor core accelerated)

**The 256x32 panel kernel (Algorithm 4):**
```
// Modified Gram-Schmidt on a 256x32 tile in shared memory
for k = 1 to n:
    R(k,k) = norm(Q(:,k))       // Reduction across 256 threads
    Q(:,k) = Q(:,k) / R(k,k)   // Normalize
    R(k,k+1:n) = Q(:,k)^T * Q(:,k+1:n)   // Inner products (reduction)
    Q(:,k+1:n) -= Q(:,k) * R(k,k+1:n)    // Rank-1 updates
```

This runs entirely in shared memory with CUB reductions. They launch 256 threads per block, manually unroll the loop 4 ways, and achieve minimal global memory traffic.

**Performance:** 2.9x to 14.7x faster than cuSOLVER SGEQRF, depending on matrix shape. Taller/skinnier matrices get higher speedups because they have more GEMM content relative to panel.

**Accuracy tradeoff:** The half-precision GEMM introduces ~1e-4 orthogonality loss. For QR factorization alone, backward error remains small. For least squares problems, they add iterative refinement (CGLS with R as preconditioner) to recover double precision accuracy in 4-5 iterations.

### 4.3 Mapping to sm_120 / BF16 mma.sync

For our RTX 5090 implementation:

**Trailing update GEMMs:** Use our existing BF16 GEMM kernel (0.97x cuBLAS). The mma.sync.m16n8k16.bf16 instruction with FP32 accumulators gives us:
- FP32 accumulation precision (important for QR stability)
- ~660 TFLOPS BF16 tensor core throughput
- Well-suited to the square GEMMs produced by recursive QR

**Panel factorization:** Keep in FP32. The panel is BLAS-2 dominated (sequential column operations) and doesn't benefit from tensor cores. Use scalar FP32 in shared memory or registers.

**T matrix computation:** Always FP32. T is nb x nb (small). The V1^T * V2 GEMM inside LARFT can use tensor cores if nb is large enough (64x64 BF16 GEMM is viable on our kernel), but the TRMM portions are small enough for scalar.

**Key difference from the paper:** The paper uses FP16 on Volta/Ampere. We use BF16 on Blackwell sm_120. BF16 has:
- Same range as FP32 (8-bit exponent) -- no overflow risk
- Lower precision than FP16 (7-bit vs 10-bit mantissa) -- but with FP32 accumulators, intermediate results are FP32
- The orthogonality loss may be slightly worse than FP16 due to BF16's lower mantissa precision

---

## 5. Panel Factorization — GPU Implementation Approaches

### 5.1 MAGMA's Approach (Hybrid CPU-GPU)

MAGMA's `magma_dgeqrf_gpu` uses a hybrid strategy:
- **Panel on CPU:** Transfer the panel column to CPU, run LAPACK's DGEQR2, transfer back. This avoids the GPU's inefficiency at sequential BLAS-2 operations.
- **Trailing update on GPU:** Use GPU GEMM for the large trailing matrix update.
- **Overlap:** Pipeline panel factorization on CPU with trailing update on GPU via CUDA streams.

Available MAGMA variants:
- `dgeqrf_gpu` — standard hybrid, stores T matrices for later use
- `dgeqrf2_gpu` — LAPACK-compliant output format
- `dgeqrf3_gpu` — enhanced with pre-computed triangular matrices
- `dgeqr2x_gpu` (v1-v3) — optimized unblocked panel variants for GPU

### 5.2 Zhang/Wu's CAQR Panel (All-GPU)

The CAQR (Communication Avoiding QR) approach avoids CPU-GPU transfers entirely:
1. Divide the m x nb panel into tiles of 256 x nb
2. Factor each tile independently in shared memory using Modified Gram-Schmidt
3. Stack the R factors and recurse
4. The cross-tile communication uses batched SGEMM (tensor cores!)

This is attractive for sm_120 because:
- No CPU-GPU transfer latency
- The 256x32 tiles fit in our 99 KB shared memory (256 x 32 x 4 bytes = 32 KB for FP32)
- CUB-style reductions are well-understood on our GPU
- The recursive reduction step uses tensor-core-accelerated batched GEMM

### 5.3 cuSOLVERDx Device-Side GEQRF

cuSOLVERDx v0.2.0+ provides `geqrf` as a device-callable function on sm_120. This handles the panel factorization entirely on-device in shared memory. Suitable as a drop-in panel solver in a blocked or recursive QR implementation.

### 5.4 Recommended Panel Approach

For the QR worker, the recommended progression:
1. **Start:** cuSOLVERDx `geqrf` for panel (proven on sm_120)
2. **Baseline:** cuSOLVER `cusolverDnSgeqrf` for full QR (measure what we need to beat)
3. **Optimize:** Custom CAQR panel if cuSOLVERDx panel becomes the bottleneck
4. **Advanced:** Register-tiled panel factorization (like MAGMA's `dgeqr2x_gpu` fused variants)

---

## 6. LARFT: Forming the T Matrix

### 6.1 What T Represents

The compact WY representation stores a block of nb Householder reflectors as:

    H(1) * H(2) * ... * H(nb) = I - V * T * V^T

where:
- V is m x nb lower trapezoidal (unit diagonal), columns are the Householder vectors
- T is nb x nb upper triangular
- This representation enables applying nb reflectors via BLAS-3 operations (GEMM + TRMM)

### 6.2 Computing T (Recursive DLARFT)

Modern LAPACK computes T recursively (divide and conquer):

```
For nb reflectors with vectors V(:,1:nb) and scalars tau(1:nb):

    Split into two halves: first l = nb/2, second nb-l

    T11 = DLARFT(V(:,1:l), tau(1:l))           // Recurse on first half
    T22 = DLARFT(V(:,l+1:nb), tau(l+1:nb))     // Recurse on second half

    // Cross-term (BLAS-3):
    T12 = V(:,1:l)^T * V(:,l+1:nb)             // DGEMM: l x (nb-l) output
    T12 = -T11 * T12                            // DTRMM
    T12 = T12 * T22                             // DTRMM

    T = [T11  T12]
        [ 0   T22]

    Base case: T(1,1) = tau(1)
```

### 6.3 Size and GPU Considerations

- T is always nb x nb. For nb=64: 64x64 = 4096 elements = 16 KB in FP32
- The V^T * V GEMM inside LARFT: for nb=64 with m=4096, this is (32 x 4096) * (4096 x 32) = 32x32 output. Small enough for shared memory computation, but large enough inner dimension that a tensor core GEMM could help.
- In practice, T computation is NOT the bottleneck. It's O(nb^2 * m) while the trailing update is O(nb * m * n). T matters only when nb is large or m is small.

### 6.4 MAGMA's GPU LARFT

MAGMA provides batched LARFT kernels (`slarft_batched_fused_sm.cu`) that compute T entirely on GPU in shared memory. These use register-based computation for the small matrix operations and shared memory for the V^T * V inner products.

---

## 7. Open Source Implementations

### 7.1 MAGMA (icl-utk-edu/magma on GitHub)

Most comprehensive GPU QR implementation. Key files:
- `src/dgeqrf_gpu.cpp` — main GPU QR driver (hybrid CPU-GPU)
- `src/dgeqrf2_gpu.cpp` — LAPACK-compliant variant
- `src/dgeqr2x_gpu.cpp` (v1-v3) — optimized GPU panel factorization
- `magmablas/dlarft_batched_fused_sm.cu` — GPU LARFT kernel
- `magmablas/dlarfb_gpu_gemm.cpp` — GPU LARFB using GEMM
- `src/dgeqrf_batched.cpp` — batched QR for many small matrices

MAGMA uses fixed block sizes tuned per GPU via `magma_get_dgeqrf_nb(m, n)`. No recursive QR variant in the public release.

### 7.2 CUTLASS

No QR factorization in CUTLASS. CUTLASS provides the GEMM building blocks but not factorization algorithms.

### 7.3 cuSOLVER / cuSOLVERDx

- `cusolverDnSgeqrf()` — blocked QR, closed source, our benchmark target
- cuSOLVERDx `geqrf` — device-callable panel factorization for sm_120

### 7.4 The Leng et al. Paper Code

Not publicly available as of this writing. The paper mentions testing on A100 and RTX 3090 but does not reference a code repository. The UH dissertation by Shaoshuai Zhang may contain more implementation details (https://uh-ir.tdl.org/).

### 7.5 PLASMA (icl-utk-edu/plasma)

Task-based parallel QR for multicore CPUs. Uses tile algorithms and dynamic scheduling. Not directly useful for GPU but the tile QR algorithmic structure (tile GEQRF + TSQRT + SSRFB) could inform a multi-SM GPU approach.

---

## 8. Additional Relevant Papers

### 8.1 Mixed-Precision Linear Algebra on Tensor Cores

- Haidar et al. "Harnessing GPU Tensor Cores for Fast FP16 Arithmetic to Speed up Mixed-Precision Iterative Refinement Solvers." SC 2018.
  - Demonstrates iterative refinement pattern: factor in FP16, refine in FP64
  - Achieves FP64 accuracy at FP16 speed for well-conditioned systems

- Haidar et al. "Investigating Half Precision Arithmetic to Accelerate Dense Linear System Solvers." 2017.
  - Panruo Wu is a co-author (same group as the QR tensor core paper)
  - Early exploration of FP16 tensor cores for linear algebra

### 8.2 Communication-Avoiding QR

- Demmel et al. "Communication-Avoiding QR Decomposition for GPUs." IPDPS 2011.
  - The CAQR algorithm that Zhang/Wu adapted for tensor cores
  - Key idea: factor tiles independently, then combine R factors recursively
  - Reduces global memory traffic significantly

### 8.3 Compact WY Representation

- Schreiber, Van Loan. "A Storage-Efficient $WYS$ Representation for Products of Householder Transformations." SIAM J. Sci. Stat. Comput. 10(1), pp. 53-57, 1989.
  - The foundational paper for the I - V*T*V^T representation
  - Makes blocked Householder QR possible

---

## 9. Recommended Implementation Path for sm_120

### Phase 1: Baseline and Infrastructure (Week 1)

1. **Measure cuSOLVER SGEQRF baseline** for target sizes (1024-8192 square, plus tall-skinny)
2. **Implement blocked QR driver** using cuSOLVERDx `geqrf` for panel + our BF16 GEMM for trailing update
3. **Profile:** What fraction of time is panel vs trailing update vs TRMM vs T computation?

### Phase 2: Recursive QR (Week 2)

4. **Implement RGEQR3** (recursive panel factorization) using cuSOLVERDx at the base case
5. **Implement recursive DLARFT** (T matrix assembly via divide-and-conquer GEMM)
6. **Implement DLARFB** using our BF16 GEMM for the two main GEMMs (Steps 3 and 5)
7. **Benchmark recursive QR** vs standard blocked QR -- measure GEMM utilization improvement

### Phase 3: Full Recursive / Optimization (Week 3+)

8. **Implement fully recursive HRQR** (eliminate outer loop, let recursion handle everything)
9. **Custom CAQR panel** if panel becomes bottleneck (256x32 tiles, shared memory MGS)
10. **CUDA Graph capture** for the recursive call tree (reduce launch overhead)
11. **Monolithic kernel** if launch overhead dominates (based on Cholesky/LU learnings)

### Key Decision Points

- **nb (block size):** Start with nb=64 (matches our 64x64 GEMM tile). Try 32 and 128.
- **Recursion cutoff:** When to stop recursing and use DGEQR2. Start with cutoff=1 (full recursion to single columns), then try cutoff=nb for the panel.
- **BF16 vs FP32 panel:** Always FP32 for the panel. The Householder reflector computation involves norms and divisions that are sensitive to precision.
- **Iterative refinement:** If BF16 GEMM orthogonality loss exceeds tolerance (~1e-3), add 1-2 steps of iterative refinement. Use the R factor as preconditioner for CGLS.

---

## 10. Caveats and sm_120 Considerations

1. **TF32 MMA is broken on sm_120.** The B operand broadcasting defect means we must use BF16 mma.sync, not TF32. This gives ~1e-3 precision per GEMM, worse than TF32's ~1e-4. Orthogonality drift may be worse than the paper's FP16 results (FP16 has higher mantissa precision than BF16).

2. **No wgmma / TMA on sm_120.** We use mma.sync + cp.async, not the datacenter Blackwell instructions. Performance ceiling is lower than A100/H100 for GEMM, but our existing GEMM at 0.97x cuBLAS is already excellent.

3. **Panel factorization is the serial bottleneck.** Recursive QR improves trailing update GEMMs but does NOT help the panel. The panel remains sequential column-by-column Householder reflections. For large matrices (m,n > 4096), trailing update dominates and recursive QR wins big. For small matrices, the panel fraction is higher and gains are smaller.

4. **Orthogonality matters more than for LU.** QR's purpose is often orthogonalization (eigensolvers, least squares). If ||I - Q^T*Q|| is too large, downstream algorithms suffer. BF16 GEMM introduces per-GEMM error that accumulates through the recursion. Monitor orthogonality loss and be prepared to add refinement.

5. **The 8.67x number needs context.** That's FP32-via-FP16-tensor-cores vs a non-tensor-core FP32 implementation (likely MAGMA or vendor BLAS without tensor cores). The more relevant comparison for us is vs cuSOLVER, which already uses whatever acceleration is available. The 1.4x vs cuSOLVER is the realistic target.

6. **Recursive QR has higher launch overhead.** For n=4096, nb=64: standard blocked QR has 64 GEMM launches. Fully recursive QR has O(n) GEMM launches across all recursion levels. CUDA Graph capture is essential to amortize this.
