# IA-Chol: Input-Aware Operator Fusion for GPU Factorization (ICS 2025)

**Source:** Deng & Wang, "IA-Chol: Input-Aware Cholesky Decomposition on CPU and GPU," ICS 2025 (39th ACM International Conference on Supercomputing)
**URL:** https://dl.acm.org/doi/10.1145/3721145.3725756
**Relevant to:** LU worker (directly applicable technique), QR worker (indirectly)
**Worker's current problem:** Building monolithic blocked LU kernel for N=4096 on sm_120. Need optimal tile/block size and operator fusion strategy.

---

## What This Is

A new approach to blocked Cholesky factorization that achieves **85.1% of peak efficiency on A100**, substantially beating cuSOLVER's 75.8%. The key innovations are:

1. **Operator fusion** within each blocking step (POTRF + TRSM + SYRK fused into a single kernel)
2. **Input-aware tile size prediction** that automatically selects the optimal tile size (NB) based on matrix dimension N

While the paper targets Cholesky, both innovations directly transfer to blocked LU factorization (GETF2 + LASWP + TRSM + GEMM).

---

## Why It Matters for Us

Our Cholesky experience showed that multi-kernel approaches (separate POTRF, TRSM, GEMM launches) cannot beat cuSOLVER's monolithic kernel due to launch overhead. IA-Chol confirms this diagnosis AND provides a concrete solution: fuse all operations within each blocking step into a single kernel, and choose the right tile size.

For LU at N=4096:
- The operator fusion pattern (GETF2 + LASWP + TRSM + GEMM in one kernel per step) is exactly what our v3 monolithic kernel needs
- The tile size prediction model tells us which NB to use (it is NOT always 64 or 128 -- the optimal NB changes with N and with the fusion pattern)

---

## Key Technique 1: Operator Fusion

### Standard Blocked Factorization (many kernels)
```
For each step k = 0..N/NB:
    launch kernel: panel_factor(A[k:, k:k+NB])       // POTRF or GETF2
    launch kernel: trsm(L[k], A[k, k+NB:])           // TRSM
    launch kernel: gemm(A[k+NB:, k:k+NB], ...)       // SYRK or GEMM trailing update
```
Each iteration = 3 kernel launches. For N=4096, NB=64: 64 iterations x 3 = 192 kernel launches.

### IA-Chol Fused Approach (one kernel per iteration)
```
For each step k = 0..N/NB:
    launch kernel: fused_step(A, k, NB)
        // Inside single kernel:
        // 1. Load panel to shared memory
        // 2. Panel factorization (in shared memory)
        // 3. TRSM (in shared memory)
        // 4. Trailing matrix update (from shared memory)
        // 5. Write results back to global memory
```
Each iteration = 1 kernel launch. For N=4096, NB=64: 64 kernel launches total.

**The fusion saves ~128 kernel launches and eliminates all intermediate global memory writes between sub-operations within each step.** The data flows through shared memory without ever hitting global memory between GETF2, TRSM, and GEMM.

---

## Key Technique 2: Input-Aware Tile Size Prediction

### The Problem
Traditional tile size selection uses a fixed NB (e.g., MAGMA uses NB=64 for most cases). But the optimal NB depends on:
- Matrix size N (smaller N benefits from smaller NB)
- Whether operators are fused (fusion changes the compute-to-memory ratio)
- GPU architecture (shared memory size, SM count, register file)

### Their Solution
A lightweight prediction model that takes N as input and outputs the optimal NB. The model is based on two observations:

1. **Panel cost scales as O(NB^2 * N)** -- larger NB makes the panel phase more expensive
2. **Trailing update cost scales as O(N^3 / NB)** -- larger NB amortizes the trailing update better
3. **With fusion**, the crossover point shifts because the fusion benefit grows with NB (more work per fused kernel = better amortization of launch overhead)

The result: optimal NB is typically 128-256 for N >= 4096 with fusion, vs 64 without fusion. This is a significant finding -- most implementations default to NB=64.

---

## Key Technique 3: Performance Results

| Matrix Size | cuSOLVER | IA-Chol | Speedup |
|-------------|----------|---------|---------|
| N=4096 | 75.8% eff | 85.1% eff | 1.12x |
| N=8192 | similar | higher | >1.1x |
| N=20000+ | near peak | near peak | comparable |

The biggest wins are for medium-sized matrices (N=2048 to 16384) where launch overhead and tile selection matter most. For very large N, both approaches converge to near-peak GEMM efficiency since the trailing update dominates.

---

## Direct Application to Our LU Kernel

### For v1 (blocked with cuBLAS):
- Profile with different NB values (32, 64, 128, 256) to find the sweet spot
- Even without fusion, the right NB could save 10-20% vs default NB=64

### For v3 (monolithic kernel):
- The operator fusion pattern is our target architecture
- Load panel to shared memory -> GETF2 (in shmem) -> LASWP (in shmem) -> TRSM (in shmem) -> GEMM (shmem + registers) -> write back
- Tile size 128-256 may be better than 64 once fusion eliminates launch overhead

---

## Caveats

1. **Tested on A100, not sm_120** -- shared memory (128KB on sm_120 vs 192KB on A100 with opt-in) and SM count differ. May need to adjust NB predictions.
2. **Cholesky, not LU** -- LU has pivoting and row swaps (LASWP), which add complexity to the fusion. Row swaps touch the ENTIRE matrix width, not just the current panel.
3. **Their shared memory budget is generous** -- A100 can opt-in to 164KB shared/block. sm_120 has 128KB hard limit but our kernels use 99KB. Panel NB x NB float = NB^2 * 4 bytes. NB=128 -> 64KB just for the panel.
4. **Single thread block per step** is limiting for the trailing GEMM at large N. Need multiple CTAs for the trailing update while the panel stays in one CTA.
