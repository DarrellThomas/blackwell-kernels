# GPU-Native TRSM Optimization: Arithmetic Intensity, Roofline Analysis, and Implementation Depth

**Source:** Multiple academic papers + NVIDIA docs (see Sources at bottom)
**Relevant to:** linalg worker
**Worker's current problem:** TRSM at 0.82x cuBLAS F32 (57 us) via torch.linalg.solve_triangular delegation. Worker has 0.97x BF16 GEMM and 2.02x FP8 batched GEMM. Next direction: "Native BF16 TRSM -- multi-SM blocked approach (significant complexity)."

**Note:** This brief supplements the existing `linalg_trsm_gpu_techniques.md` and `linalg_recursive_trsm_to_gemm.md`. It focuses on questions those briefs left open: Is TRSM fundamentally memory-bound at small N? What does cuBLAS actually do internally? And what are the quantitative bounds on how much a native kernel can gain?

---

## What This Is

A quantitative analysis of TRSM's performance characteristics on GPU at the matrix sizes relevant to this worker (N=256 to 4096), answering whether a custom kernel is worth the complexity cost, and providing specific implementation-level details not covered in the existing briefs.

---

## Why It Matters for Us

The existing briefs describe *what* to build (recursive decomposition + base-case kernel + GEMM updates) but leave three critical questions open:

1. **Is TRSM memory-bound or compute-bound at our sizes?** This determines whether a tensor-core approach can help at all.
2. **What does cuBLAS do internally?** If cuBLAS already uses the optimal algorithm, beating it requires a better implementation, not a better algorithm.
3. **What is the realistic ceiling?** The existing brief estimates 1.0-1.1x for Phase 1 and 1.2-1.5x for Phase 2. Can we tighten those bounds?

---

## Key Findings

### 1. TRSM Arithmetic Intensity: The Fundamental Constraint

TRSM solving L*X = B where L is NxN lower-triangular and B is NxNRHS:

- **FLOPs:** N^2 * NRHS (NOT N^3 -- this is critical)
- **Memory traffic:** 2 * N^2 (read L) + 2 * N * NRHS (read/write B) bytes (F32)
- **Arithmetic intensity:** N^2 * NRHS / (2*N^2 + 2*N*NRHS) = NRHS / (2 + 2*NRHS/N) flops/byte

For a square system (NRHS = N):
- AI = N / 4 flops/byte

For RTX 5090 (1792 GB/s bandwidth, ~209 TFLOPS BF16 tensor, ~105 TFLOPS F32):
- **F32 compute/bandwidth crossover:** N = 4 * 105000 / 1792 = ~234
- **BF16 TC crossover:** N = 4 * 209000 / 1792 = ~467

**This means:**
- **N < 234 (F32):** TRSM is memory-bandwidth-bound. Tensor cores cannot help.
- **N = 256:** Barely compute-bound in F32. Tensor cores provide marginal benefit.
- **N >= 512:** Solidly compute-bound. Tensor cores (via GEMM decomposition) provide real benefit.
- **N = 4096:** Very compute-bound. GEMM dominates, tensor cores provide maximum benefit.

**BUT** this is for the *monolithic* TRSM. The blocked/recursive approach changes the picture because the GEMM updates have *higher* arithmetic intensity than the base-case solves. The GEMM portions (which are (N/2 x N/2) * (N/2 x NRHS) at the top recursion level) have AI proportional to the tile dimensions, not the solve size. So even at moderate N, the GEMM portions are compute-bound while the base cases are memory-bound.

### 2. What cuBLAS Does Internally

Based on MAGMA documentation and NVIDIA's own approach (cuBLAS TRSM is derived from MAGMA's approach):

**cuBLAS uses the diagonal-block-inversion + GEMM approach:**
1. Pre-compute inverses of all NB x NB diagonal blocks (trtri_diag kernel)
2. For each block column: multiply B-panel by inverted diagonal block (GEMM), then update trailing B with off-diagonal GEMM

**cuBLAS TRSM block size:** NB = 128 (from MAGMA get_nb.cpp defaults)

**Critical insight:** cuBLAS's TRSM already converts most work to GEMM calls internally. The diagonal inversion kernel is F32 scalar math. The GEMM calls use cuBLAS GEMM (which on sm_120 uses tensor cores for BF16/FP16 but F32 TRSM uses F32 GEMM without tensor cores).

**This is the key opportunity:** cuBLAS F32 TRSM uses F32 GEMM (no tensor cores) for the off-diagonal updates. A BF16 recursive TRSM using our BF16 tensor-core GEMM would use tensor cores for 75-85% of the FLOPs. This is where the speedup comes from -- not a better algorithm, but lower-precision tensor-core acceleration of the same algorithm.

### 3. GEMM Fraction of Total FLOPs (Quantitative)

For the recursive decomposition with base case size NB:

At each recursion level for an NxN system with NRHS right-hand sides:
- GEMM work: (N/2) * (N/2) * NRHS = N^2 * NRHS / 4 FLOPs
- Base case work: 2 * NB^2 * NRHS FLOPs (two half-size triangular solves, recursed further)

Total GEMM fraction for N=4096, NB=64, NRHS=4096:
- Recursion levels: log2(4096/64) = 6 levels
- GEMM work accumulates: ~87% of total FLOPs
- Base case: ~13% of total FLOPs

For N=256, NB=32, NRHS=256:
- Recursion levels: 3
- GEMM fraction: ~75%
- Base case: ~25%

**Conclusion:** Even at N=256, three-quarters of the work is GEMM. At N=4096, it's nearly 90%.

### 4. Left-Looking vs Right-Looking: Which to Choose

**Right-looking (KBLAS choice):**
```
For each diagonal block k:
  Solve: X_k = base_trsm(A_kk, B_k)
  Update ALL blocks below: B_j -= A_jk * X_k, for j > k
```
- Pro: Each diagonal solve sees the latest data immediately
- Pro: The trailing update is one large GEMM (better GPU utilization)
- Con: More writes to global memory (intermediate B updates)

**Left-looking:**
```
For each diagonal block k:
  Accumulate: B_k -= sum(A_kj * X_j) for j < k    ← one GEMM per prior block
  Solve: X_k = base_trsm(A_kk, B_k)
```
- Pro: Fewer global memory writes
- Con: Multiple smaller GEMMs (less GPU utilization per call)

**Recommendation for sm_120:** Use the recursive approach (which is neither pure left nor right looking -- it's a divide-and-conquer that naturally produces one large GEMM at each level). This gives the best GEMM sizes for tensor core utilization. If implementing the blocked (non-recursive) variant, right-looking is better because the single large trailing GEMM maps well to our 64x64-tile GEMM kernel.

### 5. The 57 us Baseline: What's Actually Happening

The worker's current path: `torch.linalg.solve_triangular(A_f32, B_f32) -> cast to BF16`

At 57 us for (presumably) N=256 or similar:
- This calls cuBLAS dtrsm/strsm under the hood
- cuBLAS uses F32 arithmetic throughout (no tensor cores)
- The cast back to BF16 adds negligible time

A custom BF16 recursive TRSM with tensor-core GEMM would:
- Run the GEMM updates ~2x faster (tensor core BF16 vs F32 CUDA cores)
- Run the base cases at similar speed (scalar F32 either way)
- Have ~75-87% of FLOPs in the GEMM path

**Rough estimate:** If GEMM is 80% of work and runs 2x faster, overall speedup = 1 / (0.2 + 0.8/2) = 1 / 0.6 = 1.67x. This suggests 1.4-1.7x over cuBLAS F32 is realistic for moderate-to-large N.

### 6. Practical Implementation Notes Not in Existing Briefs

**Workspace management:** The recursive approach needs no extra workspace beyond the input/output matrices. The diagonal-inversion approach (MAGMA style) needs ceil(N/NB) * NB^2 elements of workspace for the inverted blocks. For N=4096, NB=128: 32 * 128^2 * 2 bytes = 1 MB. Trivial.

**Kernel launch overhead:** The recursive approach with NB=64 on N=4096 generates 2*log2(64) = 12 recursion levels, but not 12 sequential launches -- the recursion tree has parallelism. However, the sequential dependency (solve top half -> GEMM update -> solve bottom half) means you cannot parallelize within a single recursion level. Total sequential launches: O(N/NB) for the diagonal chain. For N=4096, NB=64: 64 sequential kernel launches. At ~5 us per launch, that's 320 us of launch overhead alone -- this could dominate!

**Mitigation:** Use CUDA graphs to batch launches, or use a single persistent kernel with grid-level synchronization (cooperative launch). Alternatively, use a larger base case (NB=128 or 256) to reduce recursion depth. With NB=256 on N=4096: only 4 sequential stages, and each GEMM is large enough for good tensor core utilization.

**The MAGMA blocked approach avoids this:** The diagonal inversions can be computed in one batched kernel launch, then the loop is just alternating GEMM calls. Total launches: 2 * N/NB (one GEMM for diagonal multiply, one for trailing update, per block). For N=4096, NB=128: 64 launches but they're all GEMMs on warm hardware.

---

## Key Technique

**The optimal strategy for our setup, given existing infrastructure:**

### Step 1: Blocked TRSM with Diagonal Inversion (MAGMA-style, simpler than recursive)

This is actually simpler than the recursive approach and avoids the kernel-launch-depth problem:

```
// Pre-compute: invert all NB x NB diagonal blocks
// This is a batched small-matrix inversion: ceil(N/NB) independent NB x NB inversions
// Each inversion is ~NB^3/3 FLOPs -- trivial for NB=64
trtri_diag_batched(A, dinvA, N, NB);  // one kernel launch

// Main loop (sequential in k, but each step is a large GEMM)
for (k = 0; k < N; k += NB):
    jb = min(NB, N - k)

    // Step 1: X[k:k+jb, :] = dinvA[k/NB] @ B[k:k+jb, :]
    // This is a GEMM: (jb x jb) @ (jb x NRHS) -> use our BF16 GEMM with tensor cores
    gemm_bf16(dinvA_block_k, B_panel_k, X_panel_k)

    // Step 2: B[k+jb:, :] -= A[k+jb:, k:k+jb] @ X[k:k+jb, :]
    // This is a GEMM: (N-k-jb x jb) @ (jb x NRHS) -> use our BF16 GEMM
    if (k + jb < N):
        gemm_bf16(A_offdiag, X_panel_k, B_trailing, alpha=-1, beta=1)
```

**Why MAGMA-style over recursive for us:**
- Simpler implementation (~100 lines vs ~200 for recursive)
- Predictable launch pattern (no recursion overhead)
- The diagonal inversion kernel is a standard batched operation
- The GEMMs at each step are rectangular -- our GEMM kernel handles these
- cuBLAS uses this exact approach, so we're comparing apples to apples but with faster GEMM

### Step 2: Write the trtri_diag Kernel

Small batched triangular inversion of NB x NB blocks:
- Each block is independent (embarrassingly parallel)
- NB=64: each warp handles one block, column-by-column back-substitution
- Load the NB x NB block into shared memory
- Forward substitution to compute the inverse column by column
- Store the inverted block to workspace
- This is numerically stable for well-conditioned triangular matrices (from LU/Cholesky)

### Step 3: Wire It Together

```python
def trsm_native(A, B):
    N = A.shape[0]
    NRHS = B.shape[1]
    NB = 64  # tune this

    # Pre-invert diagonal blocks (one kernel launch)
    dinvA = trtri_diag_batched(A, NB)

    # Blocked solve
    X = B.clone()  # work in-place on copy of B
    for k in range(0, N, NB):
        jb = min(NB, N - k)
        # Panel solve via GEMM with inverted diagonal
        X[k:k+jb] = our_gemm(dinvA[k//NB], X[k:k+jb])
        # Trailing update
        if k + jb < N:
            X[k+jb:] -= our_gemm(A[k+jb:, k:k+jb], X[k:k+jb])
    return X
```

---

## Caveats

1. **The F32 vs BF16 precision gap.** cuBLAS TRSM runs in F32. Our BF16 GEMM accumulates in F32 but the inputs/outputs are BF16. For the GEMM updates (B -= A*X), the accumulation is in F32 which helps, but the truncation to BF16 at each step introduces error. For well-conditioned triangular matrices (condition number < 10^3), this is fine. For ill-conditioned systems, the BF16 GEMM path will lose accuracy. The base-case diagonal inversion can stay in F32 to preserve precision where it matters most.

2. **Kernel launch overhead at small N.** For N=256 with NB=64: 4 steps, each with 2 GEMM launches = 8 launches. At ~3-5 us per launch, that's 24-40 us of just launch overhead, against a 57 us baseline. This is why the blocked approach (with pre-computed inversions) is better than recursive: fewer total launches.

3. **The existing 0.82x baseline changed.** The agent state shows 57 us at 0.82x, which means cuBLAS is around 47 us. To beat 47 us, the custom kernel needs to be genuinely faster, not just algorithmically better. The GEMM speedup (tensor core BF16 vs F32 scalar) is the primary lever.

4. **NRHS matters enormously.** If the benchmark is N=256 solving against N=256 right-hand sides (square system), the GEMM portions are large and tensor cores help. If it's N=256 with NRHS=1, the problem is TRSV (triangular solve with a vector), which is memory-bandwidth-bound and tensor cores cannot help. Check the benchmark specification before investing in this.

5. **Rectangular GEMM performance.** The GEMM calls in blocked TRSM are rectangular: (NB x NB) @ (NB x NRHS) for the diagonal multiply, and (M x NB) @ (NB x NRHS) for the trailing update. The worker's GEMM kernel is tuned for square matrices. Rectangular shapes (especially tall-skinny or short-wide) may not hit the same 0.97x. This needs benchmarking before committing.

6. **Numerical stability of diagonal inversion.** Inverting triangular blocks is conditionally stable: safe for matrices from Cholesky (positive definite) or LU with pivoting, but potentially problematic for arbitrary triangular matrices. The recursive approach (which never inverts) is unconditionally stable. If the worker needs to support arbitrary triangular matrices, the recursive approach is safer.

---

## Sources

- Carrica & Onyango, "Toward Portable GPU Performance: Julia Recursive Implementation of TRMM and TRSM" (2025): https://arxiv.org/html/2504.13821v1
- Charara, Ltaief, Keyes, "Redesigning Triangular Dense Matrix Computations on GPUs" (Euro-Par 2016): https://link.springer.com/chapter/10.1007/978-3-319-43659-3_35
- Charara et al., "A framework for dense triangular matrix kernels on various manycore architectures" (2017): https://onlinelibrary.wiley.com/doi/full/10.1002/cpe.4187
- Carrica et al., "Hierarchical Precision and Recursion for Accelerating Symmetric Linear Solves on MXUs" (2025): https://arxiv.org/html/2601.08082v1
- KBLAS GPU library: https://github.com/ecrc/kblas-gpu
- MAGMA TRSM documentation: https://icl.utk.edu/projectsfiles/magma/doxygen/group__magma__trsm.html
- MAGMA library: https://icl.utk.edu/magma/
- MAGMA exascale BLAS paper (2024): https://journals.sagepub.com/doi/10.1177/10943420241261960
- cuSolverDx documentation: https://docs.nvidia.com/cuda/cusolverdx/
- NVIDIA CUTLASS: https://github.com/NVIDIA/cutlass
- Zhang, "Matrix Computations on TensorCore GPU" (UH thesis): https://uh-ir.tdl.org/server/api/core/bitstreams/718e26c5-4ae9-4fe9-a37b-623ecdf1538b/content
