# LU Pivoting Avoidance: Random Butterfly Transforms and remifa Library

**Source:** Multiple (see per-section citations)
**Relevant to:** numerical/ worker (LU factorization / getrf)
**Worker's current problem:** cuSOLVER baseline is 1645 us (N=1024) and 9400 us (N=4096). Pivoting is the serial bottleneck in GPU panel factorization -- argmax + row swap per column creates sequential dependencies. Need alternatives that eliminate or minimize pivoting overhead in a monolithic kernel.

---

## 1. Random Butterfly Transformation (RBT) -- Avoid Pivoting Entirely

**Source:** https://icl.utk.edu/files/publications/2017/icl-utk-948-2017.pdf (Baboulin, Dongarra, Tomov et al., Concurrency & Computation 2017)
**Source:** https://inria.hal.science/hal-01223018 (HeteroPar 2015)
**Source:** https://arxiv.org/pdf/2312.09376 (Generalized RBT, ACM TOMS 2024)

### What It Is

Random Butterfly Transforms (RBT) are a preprocessing technique that replaces pivoting. Instead of finding pivots during factorization (serial bottleneck), you pre-multiply the matrix by random orthogonal butterfly matrices, then factorize WITHOUT pivoting. The butterfly transform "randomizes" the matrix so that small pivots are statistically unlikely.

### The Complete Algorithm

```
1. Generate random butterfly matrices W, V    // O(n) -- trivial
2. Compute A' = W * A * V                      // O(n^2 log n) -- butterfly structure
3. LU factorize A' WITHOUT pivoting            // O(n^3) -- NO argmax, NO row swaps
4. Solve: x = V * (U \ (L \ (W * b)))         // Back-substitution
5. Iterative refinement (2-3 steps)            // O(n^2) per step -- recover accuracy
```

### Butterfly Matrix Structure

A depth-d butterfly matrix B_d for an n x n matrix (n = 2^d) has the recursive form:
```
B_d = [ R_1   S_1 ] * [ B_{d-1}     0     ]
      [ R_2   S_2 ]   [    0      B_{d-1}  ]
```
where R_i, S_i are random diagonal matrices. This gives O(n log n) application cost (like FFT).

### Why This Matters for Our Monolithic Kernel

**The pivoting serial bottleneck disappears entirely.** Without pivoting, the panel factorization becomes:
```
for j = 0 to NB-1:
    // NO argmax (no pivot search)
    // NO row swap
    scale: A[j+1:M, j] /= A[j, j]
    rank-1 update: A[j+1:M, j+1:NB] -= A[j+1:M, j] * A[j, j+1:NB]
```

This is fully parallel -- no sequential dependency between columns for pivot selection. All threads can proceed without waiting for a reduction result. The panel factorization becomes a simple triangular update that can be aggressively parallelized.

### Performance Results

- **1.2x to 1.8x faster** than partial pivoting LU on GPUs (Baboulin et al. 2017)
- RBT preprocessing is negligible (O(n^2 log n) vs O(n^3) factorization)
- For N >= 6000, **20-30% faster** than GEPP (Gaussian Elimination with Partial Pivoting)
- Available in MAGMA: `magma_dgesv_rbt()` -- RBT + LU no pivot + iterative refinement

### MAGMA RBT Implementation Details

The MAGMA library includes the complete RBT solver pipeline:
- `magma_dgesv_rbt()` -- high-level solver
- `magma_dgetrf_nopiv()` -- LU factorization without pivoting
- GPU kernels for butterfly matrix application using shared memory for butterfly stages
- Mixed-precision iterative refinement to recover full FP64 accuracy

The CUDA kernel for RBT application uses shared memory arrays for each block to store butterfly transformation elements, improving memory access efficiency.

### Generalized RBT for Arbitrary Sizes (2024)

**Source:** https://dl.acm.org/doi/10.1145/3699714 (ACM TOMS, 2024)

Classical RBT requires matrix dimension n = 2^d. The 2024 generalized RBT paper extends this to arbitrary dimensions:
- Introduces padding-free butterfly structures for non-power-of-2 sizes
- Maintains O(n^2 log n) transformation cost
- Numerical stability verified for arbitrary sizes
- **Directly applicable to N=1024 (already 2^10) and N=4096 (already 2^12)** -- both our target sizes are powers of 2, so the classical RBT works without generalization

### Numerical Stability Caveat

RBT relies on probabilistic stability -- the transform makes pathological pivot patterns statistically improbable, but not impossible. The iterative refinement step is essential to recover accuracy. For well-conditioned matrices (condition number < 10^8), RBT is reliable. For ill-conditioned matrices, partial pivoting may still be needed.

### Concrete Approach for sm_120

```
Phase 1: RBT preprocessing (cheap)
  - Generate random diagonal matrices (host-side)
  - Apply W * A * V on GPU: O(N^2 log N) = O(4096^2 * 12) ~ 200M ops ~ 0.01ms
  - Negligible cost

Phase 2: Monolithic LU without pivoting
  - Panel factorization: NO argmax, NO swap. Pure scale + rank-1 update.
  - All threads proceed in lockstep -- no serial pivot dependency
  - Trailing GEMM: same BF16 mma.sync approach as with pivoting
  - Expected speedup vs pivoted panel: 1.5-2x on panel portion
  - Overall: panel is ~5-10% of compute, so ~2-5% total speedup from no-pivot
  - BUT the implementation simplicity is massive -- no pivot infrastructure needed

Phase 3: Iterative refinement (2-3 iterations)
  - Each iteration: residual r = b - A*x (GEMV), solve L*U*dx = r, x += dx
  - O(N^2) per iteration = ~34M ops per iteration
  - 3 iterations ~ 100M ops ~ negligible vs 46 GFLOP factorization
```

---

## 2. remifa -- Open-Source Mixed-Precision Tensor Core LU Implementation

**Source:** https://github.com/flipflapflop/remifa (BSD-3-Clause)
**Source:** Lopez & Mary, "Mixed Precision LU Factorization on GPU Tensor Cores" (2020)

### What It Is

remifa (REduced and MIxed precision Factorization Algorithms) is an open-source C++/CUDA library implementing mixed-precision LU factorization exploiting Volta/Turing tensor cores. It is the reference implementation for the Lopez-Mary paper.

### Technical Details

- **Language:** C++ (82.7%), CUDA (14.5%)
- **License:** BSD-3-Clause
- **Architecture:** Left-looking blocked LU factorization
- **Tensor core usage:** FP16 tensor cores (wmma) for trailing matrix updates
- **Key innovation:** Matrix stored in FP16, panel factorization in FP32 buffers of controlled size
- **Result:** 2x faster than FP32 LU, half the memory footprint, comparable accuracy

### Why It Matters

remifa is directly inspectable source code for a working tensor-core-accelerated LU factorization. The worker can study:
1. How the left-looking blocked algorithm streams tiles through shared memory
2. How FP16 tensor core GEMM is used for trailing updates
3. How FP32 panel buffers maintain accuracy during pivoting
4. How doubly-partitioned updates balance accuracy vs performance

### Architecture Relevance

remifa targets Volta/Turing (sm_70, sm_75) using wmma. Our sm_120 uses mma.sync m16n8k16 (BF16) instead of wmma. The algorithmic patterns (left-looking blocked LU, mixed precision buffers, tensor core trailing GEMM) transfer directly -- only the MMA instruction interface changes.

### Key Algorithmic Pattern from remifa

```
Left-looking blocked LU:
for k = 0 to N/NB - 1:
    // Update panel k using ALL previously factored panels (left-looking)
    for j = 0 to k-1:
        load L[k,j] and U[j,k] tiles (FP16 from global memory)
        GEMM update panel k using tensor cores (FP16*FP16 + FP32 accum)

    // Panel factorization in FP32 (partial pivoting)
    GETF2(panel[k])   // FP32, standard argmax + swap + scale + rank-1

    // Convert factored panel to FP16 and store
    store L[k,k] and U[k,k] in FP16
```

vs the more common right-looking approach:
```
for k = 0 to N/NB - 1:
    GETF2(panel[k])         // panel factorization
    TRSM(solve for U row)   // triangular solve
    GEMM(update trailing)   // ONE large trailing GEMM
```

**Left-looking advantage for tensor cores:** Each update involves small GEMMs (NB x NB tiles) that fit in shared memory. No need for a large trailing GEMM kernel. Better for a single-block monolithic kernel that streams tiles through shared memory.

**Left-looking disadvantage:** More total GEMM calls (k GEMMs at step k, vs 1 large GEMM). Less parallelism per GEMM.

**Recommendation for N=4096:** Right-looking is better because the trailing GEMM is large enough to saturate the GPU. Left-looking is better for smaller matrices or single-block monolithic kernels.

---

## 3. MAGMA Lazy Swap -- Deferred Row Interchanges (Reminder)

**Source:** https://www.netlib.org/utk/people/JackDongarra/PAPERS/icl-utk-1237-2018.pdf

This is already documented in the existing `lu_magma_batched_panel_techniques.md` brief, but worth reiterating in the context of the monolithic kernel:

### The Key Insight

Standard partial pivoting does `NB` row swaps during panel factorization, each touching the ENTIRE row (all N columns). In a monolithic kernel, this means N*NB global memory reads+writes just for swaps.

Lazy pivoting defers ALL row interchanges to a single batch write at the end of the panel kernel. During factorization, only register/shared-memory swaps happen (within the panel's NB columns). The global memory swap (LASWP) happens once, AFTER the panel is fully factored.

**For N=4096, NB=64:** This saves 64 individual row swap passes across 4096 columns, replacing them with one batched LASWP pass.

### Combined with rowid Trick

The MAGMA rowid trick (from `lu_magma_native_kernel_internals.md`) means even the in-register swaps are eliminated -- threads just track which logical row they own via a `rowid` integer. Zero data movement during panel factorization. Permutation applied only during final write-back.

---

## 4. Synthesis: Pivoting Strategy Decision Tree for Monolithic Kernel

```
Is the input matrix well-conditioned (condition number < 10^8)?
  YES --> Use RBT preprocessing + LU without pivoting
          (simplest monolithic kernel, no pivot infrastructure)
  NO/UNKNOWN --> Use partial pivoting with these optimizations:
                 1. Lazy pivoting (defer all row swaps to end of panel)
                 2. rowid trick (virtual pivoting in registers)
                 3. Single-warp reduction for argmax (N <= 32 panel width)

Is maximum performance the goal (not just beating cuSOLVER)?
  YES --> Consider Pre-Pivoted LU (PRP):
          - Pre-compute pivot order in BF16 using tensor cores (fast)
          - Apply pivots to FP32 matrix
          - LU without pivoting on pre-pivoted matrix (no serial dependency)
          (see existing lu_monolithic_gpu_factorization_research.md section 4)
```

### Recommended Implementation Order

1. **v1:** Cooperative groups monolithic kernel with standard partial pivoting + lazy swap + rowid trick
2. **v2:** If panel factorization is bottleneck, try RBT (pivot-free) or PRP (pre-pivoted)
3. **v3:** Mixed precision (BF16 trailing GEMM with FP32 panel) -- study remifa for patterns

---

## Sources

- [RBT GPU Implementation (ICL UTK, 2017)](https://icl.utk.edu/files/publications/2017/icl-utk-948-2017.pdf)
- [Randomized LU on GPU/Xeon Phi (HeteroPar 2015)](https://inria.hal.science/hal-01223018)
- [Generalized RBT for Arbitrary Sizes (ACM TOMS, 2024)](https://dl.acm.org/doi/10.1145/3699714)
- [RBT arxiv preprint](https://arxiv.org/pdf/2312.09376)
- [remifa: Mixed Precision Factorization Library](https://github.com/flipflapflop/remifa)
- [Lopez & Mary: Mixed Precision LU on Tensor Cores (2020)](https://eprints.maths.manchester.ac.uk/2782/)
- [MAGMA Library](https://icl.utk.edu/magma/) -- includes magma_dgesv_rbt, magma_dgetrf_nopiv
- [Progressive Optimization of Batched LU (ICL UTK, 2018)](https://www.netlib.org/utk/people/JackDongarra/PAPERS/icl-utk-1237-2018.pdf)
