# QR Algorithm Selection: Blocked Householder vs TSQR vs CAQR for sm_120

**Sources:**
- Anderson, Ballard, Demmel, Keutzer. "Communication-Avoiding QR Decomposition for GPUs." IPDPS 2011 / LAWN 240 (https://www.netlib.org/lapack/lawnspdf/lawn240.pdf)
- Haidar, Tomov, Dongarra, Luszczek. "Batch QR Factorization on GPUs: Design, Optimization, and Tuning." ICCS 2022 (https://www.netlib.org/utk/people/JackDongarra/PAPERS/batchqr-gpu-2022.pdf)
- Zou, Leng, Wang, Wu, Zhang. "Efficient GPU-Centered SVD Using Divide-and-Conquer." arXiv:2508.11467, 2025 (https://arxiv.org/html/2508.11467v1)
- Ootomo, Yokota. "TSQR on TensorCores." SC'19 Poster (https://sc19.supercomputing.org/proceedings/tech_poster/poster_files/rpost147s2-file2.pdf)
- MAGMA geqrf documentation (https://icl.utk.edu/projectsfiles/magma/doxygen/group__magma__geqrf.html)
- cuSOLVERDx geqrf (https://docs.nvidia.com/cuda/cusolverdx/get_started/functions/geqrf.html)

**Relevant to:** QR worker
**Worker's current problem:** QR just started (0 iterations). Must choose the right algorithmic approach before building. The key question: Householder-based (blocked or recursive) vs TSQR vs CAQR, specifically for square matrices N=1024-4096 on sm_120.

---

## TL;DR Recommendation

**Use blocked Householder QR with recursive trailing updates. Not TSQR. Not CAQR.**

For square matrices (N=1024-4096), blocked Householder is the right choice because:
1. TSQR is designed for tall-skinny (M >> N) and provides no benefit for square matrices
2. CAQR is an extension of TSQR that performs comparably to blocked Householder on square matrices
3. Recursive Householder QR converts the tall-skinny trailing GEMMs into square GEMMs -- this is the specific optimization that matters for tensor cores
4. The monolithic kernel pattern (cuSOLVERDx panel + cuBLASDx/custom GEMM trailing update) matches what Cholesky/LU workers learned: single-kernel approaches beat multi-launch

---

## 1. Algorithm Comparison for Square Matrices

### 1.1 Blocked Householder QR (LAPACK dgeqrf)

**How it works:** Process the matrix in panels of width NB. For each panel: (1) factor NB columns via Householder reflectors (GEQR2), (2) form compact WY representation (LARFT), (3) apply block reflector to trailing matrix via two GEMMs (LARFB).

**GPU parallelism for N=4096, NB=64:**
- Panel: sequential column-by-column. BLAS-2 (memory-bound). ~40% of total time for well-optimized implementations.
- Trailing update: two GEMMs, each with one dimension = NB=64. These are tall-skinny and tensor-core-unfriendly.
- 64 iterations (N/NB = 4096/64).

**Strengths:** Simple, well-understood, LAPACK-compatible output. Good numerical stability.
**Weakness:** Fixed NB-wide GEMMs in trailing update are memory-bound on tensor cores.

### 1.2 TSQR (Tall-Skinny QR)

**How it works:** Partition rows into tiles. Factor each tile independently via local QR. Stack the R factors and recurse (binary tree reduction) until one R remains.

**Why it fails for square matrices:**
- TSQR's parallelism comes from independent row tiles. For a 4096x4096 matrix with NB=64, you get tiles of 64x4096. Each tile's local QR is itself a large QR problem (64 rows x 4096 cols is wider than tall -- wrong orientation).
- The tree reduction makes sense when each tile produces a small R (e.g., 64x64 from a 512x64 tile). For square matrices, R is the same size as the input -- no reduction benefit.
- Performance data confirms this: TSQR speedups are 18x for 4096x50 but negligible for square matrices. One benchmark showed multi-core achieves 90 GFLOPs (58.8% efficiency) for square matrices but only 2 GFLOPs (1.2% efficiency) for tall-skinny -- TSQR inverts this, but at no gain for the square case.
- The tensor-core TSQR implementation (enp1s0/tsqr-gpu) explicitly targets "tall-skinny matrices" and demonstrates on 9211x51 -- not square.

**Verdict for N=1024-4096 square: Do not use.** TSQR provides no structural advantage and adds complexity.

### 1.3 CAQR (Communication-Avoiding QR)

**How it works:** CAQR extends TSQR to general matrices. The matrix is divided into a grid of blocks. The panel uses TSQR (factor tiles independently, reduce). The trailing update can start before the entire panel is factored because TSQR works on blocks. This removes the synchronization barrier in standard blocked Householder.

**For square matrices on a single GPU:**
- The CAQR paper (Anderson et al., IPDPS 2011) reports performance "very similar to" standard blocked Householder for square matrices. The 17x speedup over CULA and 12x over MKL is for tall-skinny matrices.
- The communication-avoiding benefit matters for distributed systems (multi-GPU, multi-node) where memory transfers dominate. On a single GPU, global memory bandwidth is uniform -- there is no "communication" to avoid between SMs.
- CAQR's panel factorization is more complex than standard GEQR2 (tile-based TSQR + tree reduction), but for square matrices the panel is a small fraction of total time anyway.

**Verdict for N=1024-4096 square on single GPU: No significant benefit over blocked Householder.** The algorithm adds complexity without performance gain for this use case.

### 1.4 Recursive Householder QR (Elmroth-Gustavson RGEQR3)

**How it works:** Instead of processing fixed-width NB panels, recursively split columns in half. Factor left half, apply to right half (producing a GEMM with both dimensions = N/2), factor right half. Recurse.

**Why this wins for square matrices with tensor cores:**
- Top-level GEMM: 2048 x 4096 x 2048 (for N=4096) -- near-square, excellent for mma.sync
- Standard blocked QR: 64 x 4096 x ~4000 -- terribly tall-skinny, tensor cores waste bandwidth
- The operation count is identical (2MN^2 - 2N^3/3). Only the GEMM shapes change.
- Same flop count, but tensor cores see 10-50x better utilization on square vs 64-wide GEMMs.

**Verdict: This is the approach.** Already documented in existing briefs. The recursive structure is the key innovation.

### 1.5 GPU-Centered Modified CWY (NEW -- from Zou et al. 2025)

**What's new:** A recent paper (arXiv:2508.11467, May 2025) on GPU-centered SVD describes a modified CWY approach for QR that eliminates ALL BLAS-2 operations from the T matrix construction:

Standard LARFT builds T using: `T12 = -T11 * V1^T * V2 * T22` (involves TRMM -- BLAS-2-like for small T).

Modified CWY computes T^{-1} = Y^T * Y (a single GEMM), then uses T^{-1} in the trailing update via TRSM instead of TRMM. The key insight: computing T^{-1} via `Y^T * Y` is a BLAS-3 GEMM, while standard LARFT uses sequential BLAS-2 TRMV operations to build T column by column.

**Why this matters for sm_120:**
- LARFT is typically not the bottleneck, but for the monolithic kernel pattern (single thread block), every BLAS-2 operation is a serialization point.
- The modified CWY trailing update becomes: `Z = Y^T * A_trailing` (GEMM), `Z = (T^{-1})^{-1} * Z` (TRSM on NB x N_trail), `A_trailing -= Y * Z` (GEMM).
- The TRSM on NB x N_trail replaces TRMM on NB x N_trail. For NB=64, TRSM has the same complexity as TRMM but the T^{-1} computation is purely BLAS-3.
- **The GPU-centered paper reports outperforming both cuSOLVER and MAGMA for geqrf across all tested sizes on V100.**
- Panel factorization is done entirely on GPU (no CPU-GPU transfers), using the modified CWY formulation to keep everything BLAS-3.

**Verdict: This is a meaningful refinement to investigate.** The paper's approach of keeping T^{-1} construction as GEMM (instead of sequential TRMV in LARFT) could matter when the panel fraction is high (smaller matrices or large NB).

---

## 2. Answers to Specific Questions

### Q: For N=1024-4096, which approach has better GPU parallelism?

**Recursive blocked Householder** has the best parallelism for square matrices on a single GPU. The parallelism comes from the trailing matrix update GEMMs, which dominate for N >= 1024. Recursive QR makes these GEMMs square (N/2 x M x N/2 at top level), maximizing tensor core utilization.

TSQR has better parallelism for tall-skinny (M >> N) because the independent row-tile factorizations can run in parallel. But for square matrices, there's nothing to parallelize in the row dimension that blocked Householder doesn't already exploit.

Panel factorization time breakdown for well-optimized GPU QR at N=4096:
- Panel (GEQR2): ~40% of total time (memory-bound, sequential columns)
- Trailing GEMMs (LARFB): ~55% of total time (compute-bound, parallelizable)
- LARFT (T construction): ~5% (small)

### Q: Can the trailing matrix update use mma.sync BF16?

**Yes.** The trailing update decomposes into:
1. `W = V^T * A_trailing` -- GEMM, can use BF16 mma.sync.m16n8k16 with FP32 accumulators
2. `W = T * W` -- TRMM on NB x N_trail, too small for tensor cores (NB=64), use scalar FP32
3. `A_trailing -= V * W` -- GEMM, can use BF16 mma.sync.m16n8k16 with FP32 accumulators

The two dominant GEMMs in LARFB are excellent candidates for our existing BF16 GEMM kernel (0.97x cuBLAS). With recursive QR, these become near-square at the top levels.

**Precision concern:** BF16 has 7-bit mantissa (vs FP16's 10-bit). The orthogonality loss per GEMM is ~1e-3. For QR factorization of N=4096 with recursive depth log2(4096/64) = 6, the accumulated orthogonality error is roughly 6 * 1e-3 = 6e-3 in the worst case. This is adequate for many applications. If higher accuracy is needed, the panel stays in FP32, and iterative refinement can recover full FP32 accuracy in 4-5 iterations.

### Q: How does cuSOLVER implement geqrf on sm_120?

**Best available evidence:** cuSOLVER's geqrf is closed-source, but:
- cuSOLVERDx's device-side geqrf uses Householder reflections (confirmed in documentation)
- MAGMA's GPU-native variant (`geqrf_native`) performs all computation on GPU without CPU involvement
- The GPU-centered paper (Zou et al. 2025) criticizes the MAGMA hybrid approach (panel on CPU) as bottlenecked by CPU-GPU transfers, suggesting cuSOLVER likely does panel-on-GPU for modern architectures
- cuSOLVER Release 13.1 (Jan 2026) supports sm_120, and the DnSgeqrf function is listed with deterministic mode support
- The single-block monolithic pattern is consistent with what we've observed for cuSOLVER's potrf and getrf on sm_120

**Most likely implementation:** Single thread block, blocked Householder with device-side panel factorization, possibly with cuBLASDx-style tensor core trailing updates. Not recursive QR (cuSOLVER tends to use conservative, well-tested algorithms).

### Q: Is there a TSQR variant that works for square matrices?

**No practical variant exists.** TSQR's fundamental design splits rows for independent processing, which only helps when M >> N. For square matrices:
- Each row tile is N-wide, so its local QR is itself expensive
- The tree reduction produces R factors that are N x N -- no compression
- CAQR extends TSQR to general matrices by combining it with blocked Householder for the trailing update, but for square matrices this degenerates to standard blocked Householder performance

The recursive Householder approach (RGEQR3/HRQR) is the correct "generalization" of TSQR ideas to square matrices: it uses divide-and-conquer to restructure GEMMs, but splits **columns** (not rows) and works directly with Householder reflectors.

---

## 3. Recommended Algorithm Strategy

```
Phase 1: Baseline
  - Measure cuSOLVER sgeqrf at N=1024, 2048, 4096
  - Profile panel vs trailing update fraction

Phase 2: Monolithic Blocked QR
  - Single kernel using cuSOLVERDx geqrf (panel) + cuBLASDx GEMM (trailing)
  - Match the pattern from factorization_monolithic_kernel_approaches.md
  - NB=64 (matches our GEMM tile size)
  - This should approach cuSOLVER performance by eliminating launch overhead

Phase 3: Recursive Trailing Updates
  - Replace fixed-NB trailing GEMMs with recursive QR structure
  - Top-level GEMMs become N/2-wide (square), ideal for BF16 mma.sync
  - Use cuSOLVERDx geqrf at recursion base case
  - Expected: 1.0-1.4x cuSOLVER based on IEEE TPDS results

Phase 4: Optimizations
  - Modified CWY (T^{-1} = Y^T * Y) to eliminate BLAS-2 from T construction
  - Custom CAQR panel if panel becomes bottleneck (tile-based, smem-resident)
  - Cooperative groups for multi-SM trailing updates if single-block is too slow
```

---

## 4. What Transfers from Cholesky/LU

| Learning | Applicability to QR |
|----------|-------------------|
| Monolithic kernels beat multi-launch | Directly applicable -- same pattern |
| cuSOLVERDx for panel factorization | Use geqrf + unmqr device-side calls |
| BF16 MMA for trailing GEMM | The two LARFB GEMMs are perfect candidates |
| TF32 MMA broken on sm_120 | Must use BF16, same as Cholesky/LU |
| 99KB shared memory limit | Constrains NB for panel: max ~157x157 FP32 |
| CUDA Graph for launch overhead | Useful for recursive QR's many small GEMMs |
| Cooperative groups for multi-SM | Option for Phase 4 if trailing GEMM dominates |

---

## Sources

- [CAQR for GPUs (LAWN 240)](https://www.netlib.org/lapack/lawnspdf/lawn240.pdf) -- TSQR/CAQR algorithm, square vs tall-skinny performance comparison
- [MAGMA Batch QR (ICCS 2022)](https://www.netlib.org/utk/people/JackDongarra/PAPERS/batchqr-gpu-2022.pdf) -- Panel fraction analysis (40% of time), fused kernel strategies, register-tiled panels
- [GPU-Centered SVD (Zou et al. 2025)](https://arxiv.org/html/2508.11467v1) -- Modified CWY with T^{-1} = Y^T*Y, all-BLAS-3 T construction, GPU-only panel, outperforms cuSOLVER+MAGMA
- [TSQR on TensorCores (Ootomo/Yokota, SC'19)](https://sc19.supercomputing.org/proceedings/tech_poster/poster_files/rpost147s2-file2.pdf) -- TSQR with FP16/TF32 tensor cores, tall-skinny only
- [TSQR GPU repo](https://github.com/enp1s0/tsqr-gpu) -- Reference implementation, Volta+, archived 2021
- [IEEE TPDS Recursive QR (Leng et al. 2024)](https://ieeexplore.ieee.org/document/10816084/) -- 1.4x cuSOLVER via recursive Householder + tensor cores
- [MAGMA geqrf variants](https://icl.utk.edu/projectsfiles/magma/doxygen/group__magma__geqrf.html) -- Hybrid vs native GPU-only modes
- [cuSOLVERDx geqrf](https://docs.nvidia.com/cuda/cusolverdx/get_started/functions/geqrf.html) -- Device-side Householder QR for sm_120
- [MAGMA batched QR framework (ICL 2015)](https://icl.utk.edu/files/publications/2015/icl-utk-798-2015.pdf) -- Three-strategy approach (fused/panel+update/LAPACK-style)
- [Gram-Schmidt vs Householder GPU](https://link.springer.com/article/10.1140/epjst/e2012-01638-7) -- Householder preferred for GPU due to stability + BLAS-3 structure
