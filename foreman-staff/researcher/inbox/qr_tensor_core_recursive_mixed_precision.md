# Tensor Core Recursive QR: Mixed Precision Approach

**Source:** IEEE TPDS 2024 — "High Performance Householder QR Factorization on Emerging GPU Architectures Using Tensor Cores" (https://ieeexplore.ieee.org/document/10816084/)
**Also:** UH dissertation by Shaoshuai Zhang (https://uh-ir.tdl.org/server/api/core/bitstreams/718e26c5-4ae9-4fe9-a37b-623ecdf1538b/content)
**Relevant to:** QR worker
**Worker's current problem:** QR not yet started. This provides the algorithmic strategy for beating cuSOLVER SGEQRF.

## What This Is

A tensor core-accelerated Householder QR factorization that achieves up to **8.67x speedup** (FP32) and **6.22x** (FP64) over state-of-the-art implementations. The key innovation: **recursive QR** converts the tall-skinny GEMMs that dominate standard blocked QR into square/near-square GEMMs that tensor cores handle efficiently.

## Why It Matters for Us

Standard blocked QR has a critical bottleneck: the trailing matrix update uses tall-skinny GEMMs (V^T * A where V is n_remaining x nb). These GEMMs are memory-bound on tensor cores because one operand dimension is only nb (typically 32-64). Tensor cores need square-ish tiles to reach peak throughput.

The recursive approach restructures the computation so that GEMMs grow progressively larger (nb, 2*nb, 4*nb, ..., n/2) — exactly the same insight as recursive TRSM. This matches how our BF16 GEMM kernel performs: excellent at square shapes, worse at tall-skinny.

## Key Technique

### Standard Blocked QR (the problem):
```
For each panel k = 0..n/nb:
    GEQR2(A[k:, k:k+nb])           // Panel: produces nb reflectors
    LARFT(V, tau, T)                 // Build compact WY representation
    // Trailing update (TWO GEMMs):
    W = T * V^T * A[k:, k+nb:]      // Tall-skinny GEMM: (nb × m_remain) × (m_remain × n_remain)
    A -= V * W                       // Tall-skinny GEMM: (m_remain × nb) × (nb × n_remain)
```

Both GEMMs have one dimension = nb. With nb=64, this is a 64×m×n GEMM — very memory-bound.

### Recursive QR (the solution):
```
function recursive_qr(A, m, n):
    if n <= nb:
        return panel_qr(A)           // Base case: standard GEQR2

    n1 = n/2
    // Step 1: Factor left half (recursive)
    [V1, T1] = recursive_qr(A[:, :n1], m, n1)

    // Step 2: Apply to right half (SQUARE GEMM!)
    // A[:, n1:] = (I - V1 * T1 * V1^T) * A[:, n1:]
    W = T1 * V1^T * A[:, n1:]       // ← NOW this is (n1 × m) × (m × n1) — SQUARE!
    A[:, n1:] -= V1 * W             // ← ALSO (m × n1) × (n1 × n1) — better ratio!

    // Step 3: Factor right half (recursive)
    [V2, T2] = recursive_qr(A[n1:, n1:], m-n1, n-n1)
```

**The key insight:** At each recursion level, the GEMM dimensions are n/2 × m × n/2 (at the top level), then n/4 × m × n/4, etc. The top-level GEMMs are near-square when m ≈ n. Even for tall-skinny input (m >> n), the GEMMs at the top levels are larger in both dimensions than standard blocked QR's fixed nb-width GEMMs.

### Performance numbers from paper:
| Method | vs State-of-Art | Precision |
|--------|----------------|-----------|
| TC Recursive QR (FP32 via FP16 TC) | **8.67x** | FP32 accuracy via mixed precision |
| TC Recursive QR (FP64 via INT8 TC) | **6.22x** | FP64 accuracy via mixed precision |
| TC Recursive QR (FP16 native) | **4.03x** | FP16 accuracy |
| TC Recursive QR vs cuSOLVER SGEQRF | **1.4x** | FP32 @ 32768x16384 |

### Mixed precision strategy:
- Use FP16/BF16 tensor cores for the GEMM portions (trailing update)
- Keep panel factorization in FP32 (for reflector accuracy)
- FP32 accumulators in MMA preserve intermediate precision
- Iterative refinement recovers full FP32/FP64 accuracy if needed
- On sm_120: use `mma.sync.m16n8k16.bf16` with FP32 accumulators (proven in our GEMM kernel)

## Application to Our sm_120 Codebase

### What we already have:
- BF16 GEMM at 0.97x cuBLAS (64x64 tiles, 4 warps, 6 blocks/SM)
- FP8 GEMM at 1.34x cuBLAS (for aggressive precision trade-off)
- TRMM at 1.02x reference (needed for T*V^T computation)
- Shared memory infrastructure (cp.async, XOR swizzle, ldmatrix_x4_mma)

### What we need to build:
1. **GEQR2 panel kernel** — Householder reflector generation. Column-by-column in shared memory. Needs: vector norm (reduction), reflector application (rank-1 update). More complex than Cholesky's potf2 or LU's getf2.

2. **LARFT kernel** — Build T matrix from reflectors. Small (nb×nb), pure FP32. Sequential inner products.

3. **Recursive driver** — Host-side recursion calling our existing GEMM for trailing updates. Similar to recursive TRSM structure.

### Expected performance path:
1. Start with cuSOLVER baseline (like Cholesky/LU)
2. Blocked QR with cuBLAS GEMM calls → establishes launch overhead baseline
3. CUDA Graph capture → reduces launch overhead
4. Recursive QR with our GEMM → the winning configuration
5. Monolithic kernel → if needed (based on Cholesky/LU learnings)

## Caveats

1. **TF32 MMA is broken on sm_120.** The B operand broadcasting defect (verified by Cholesky worker) means TF32 MMA cannot be used for general GEMM. Use BF16 MMA. See `lu_tf32_mma_defect_cross_pollination.md` in the LU worker docs for details.

2. **Panel factorization is inherently sequential.** Each reflector depends on the previous one. This is the same serial bottleneck as LU's panel. The recursive approach doesn't help the panel — it helps the trailing update.

3. **Recursive depth.** For n=4096 with nb=64: log2(4096/64) = 6 levels. Each level generates GEMM calls. With CUDA Graphs, the launch overhead is manageable.

4. **Orthogonality matters more for QR than LU.** BF16 GEMM's ~1e-3 precision loss accumulates differently in QR (orthogonality drift) vs LU (factorization error). May need iterative refinement or higher precision for the panel.

5. **The 8.67x number is vs "state-of-the-art," not vs cuSOLVER.** The cuSOLVER comparison shows 1.4x — still valuable, but more realistic.
