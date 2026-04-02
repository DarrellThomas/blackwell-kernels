# Mixed-Precision Pre-Pivoting for LU: PRP and MPF Algorithms

**Source:** "Mixed-Precision Pre-Pivoting Strategy for the LU Factorization" (Higham, Mary, Pranesh, Zounon), The Journal of Supercomputing, October 2024
**URL:** https://link.springer.com/article/10.1007/s11227-024-06523-w
**Also:** https://www.researchgate.net/publication/381913182
**Relevant to:** LU worker
**Worker's current problem:** Building v1 blocked LU. Pivoting is the serial bottleneck in GPU panel factorization — argmax + row swap per column creates sequential dependencies. Need approaches that reduce or eliminate pivoting overhead.
**Date:** 2026-03-15

---

## What This Is

A 2024 paper introducing two novel algorithms that use reduced precision (FP16/BF16)
to accelerate LU factorization while maintaining full-precision (FP64/FP32) accuracy.
The key insight: **pivot selection doesn't require full precision** — you can identify
good pivot orderings cheaply in half precision, then apply that ordering to the full-
precision factorization.

## Why It Matters for Us

Our LU worker faces the classic GPU pivoting bottleneck: argmax + row swap per column
is inherently sequential. The existing `lu_rbt_pivoting_avoidance_and_remifa.md` brief
covers Random Butterfly Transforms (RBT) which avoids pivoting entirely but requires
pre/post-processing and iterative refinement. PRP/MPF is a different approach that
**keeps partial pivoting** (maintaining numerical stability guarantees) but does the
expensive pivot search in BF16, which is 2x faster on our tensor cores.

This is directly relevant to the v1 blocked LU strategy — the panel factorization
(where pivoting happens) is the bottleneck, and PRP can accelerate it.

## Key Techniques

### Algorithm 1: Pre-Pivoted LU (PRP)

```
1. Compute PA' = L'U' in reduced precision (FP16/BF16)
   — This gives us the pivot permutation P
2. Apply P to the original matrix: A_reordered = P * A
3. Compute LU of A_reordered WITHOUT pivoting (full precision FP32)
4. The factorization is: P*A = L*U
```

**Why this works:** The pivot ordering from step 1 is "good enough" — the reduced-
precision factorization identifies which rows have large elements in the right
positions. Even with FP16 rounding errors, the relative magnitudes are preserved
well enough to produce a valid pivot sequence. The actual factorization in step 3
uses full precision, so accuracy is maintained.

### Two PRP Variants

**hPRP (half-precision PRP):**
- Compute the entire LU in FP16/BF16
- Extract pivot list P
- Apply P to FP32 matrix, factorize without pivoting
- Fastest, but limited to matrices where κ(A) < ~10^4

**xPRP (mixed-precision PRP):**
- Compute LU using mixed FP16+FP32 (panel in FP32, trailing update in FP16)
- More robust pivot selection for ill-conditioned matrices
- Slightly slower but works for κ(A) up to ~10^8

### Algorithm 2: Mixed-Precision Panel Factorization (MPF)

```
For each panel block in the blocked LU:
  1. Factor the panel using hPRP internally:
     a. Compute panel LU in FP16 → get pivot list
     b. Apply pivots to FP32 panel
     c. Factor FP32 panel without pivoting
  2. Apply pivots to trailing matrix (FP32)
  3. Trailing matrix update via GEMM (FP32)
```

MPF is the practical variant — it integrates PRP into the standard blocked LU
algorithm at the panel level. This is exactly where our worker's bottleneck is.

## Performance Results

The paper reports results on NVIDIA V100 GPUs:

| Matrix Size | DGETRF (baseline) | MPF | Speedup |
|-------------|-------------------|-----|---------|
| Small (N<1024) | Reference | ~1.1-1.2x | Modest |
| Medium (N=2048-4096) | Reference | ~1.2-1.5x | Significant |
| Large (N>8192) | Reference | ~1.3-1.5x | Panel becomes smaller fraction |

**Accuracy:** "on par with standard DGETRF" — the FP32 factorization step ensures
full accuracy. The FP16 step only affects pivot ordering, not the final result.

## Applicability to Our Setup

### Advantages for RTX 5090 (sm_120):
- BF16 MMA (m16n8k16) is 2x throughput of FP32 — BF16 pre-pivoting is FAST
- Panel factorization is the serial bottleneck — PRP directly attacks it
- No need for iterative refinement (unlike RBT)
- Maintains partial pivoting guarantees (numerical stability)

### How to Implement:
1. For each panel in blocked LU:
   a. Convert panel to BF16 (cheap, vectorized)
   b. Run small LU in BF16 with pivoting (use cuSOLVERDx `getrf_partial_pivot`)
   c. Extract pivot list P
   d. Apply P to FP32 panel
   e. Run FP32 panel LU WITHOUT pivoting (use cuSOLVERDx `getrf_no_pivot`)
   f. Continue with standard trailing update

2. Step (b) can use BF16 MMA for the panel GEMM updates
3. Step (e) is faster than standard `getrf` because no argmax/swap

### Key Question:
Does the BF16 pre-pivoting produce accurate enough pivot lists for our N=4096
target? The paper suggests yes for well-conditioned matrices, but our numerical
accuracy requirements may need testing. For FP32 accuracy (not FP64), the BF16
pivot selection should be very robust since BF16 has ~3 decimal digits of precision,
which is plenty for identifying relative magnitudes.

## Comparison with Other Pivoting Strategies

| Strategy | Accuracy | Speed | Complexity |
|----------|----------|-------|------------|
| Standard partial pivoting | Best | Slowest (sequential argmax) | Simple |
| RBT (no pivoting) | Good + refinement | Fast | Moderate (butterfly pre-multiply) |
| Threshold pivoting (SLATE) | Good | Moderate | Simple |
| **PRP/MPF** | **Good** | **Fast** | **Moderate** |
| No pivoting | Poor for ill-conditioned | Fastest | Simplest |

PRP/MPF sits between RBT and threshold pivoting in the tradeoff space. It's simpler
than RBT (no butterfly matrices) and more robust than threshold pivoting.

## Caveats

- Paper tested on V100 (Volta), not Blackwell — FP16 tensor core throughput ratios differ
- For our FP32-target LU (not FP64), the benefit of BF16 pre-pivoting is smaller
  because FP32 panel factorization is already fast on GPU
- The cuSOLVERDx `getrf_partial_pivot` in BF16 may have the same sequential overhead
  as FP32 — the speedup comes from faster trailing updates during pre-pivoting, not
  from faster pivoting itself
- If the panel is small (NB=32-64), the pre-pivoting LU is also small and the overhead
  of converting to BF16 + running twice might not pay off
- **Best case:** Large panels (NB=128+) where BF16 trailing GEMM during pre-pivot is
  significantly faster than FP32 trailing GEMM during standard pivot
