# Threshold Pivoting for Dense LU: Reducing Pivot Data Movement

**Source:** https://ieeexplore.ieee.org/document/10024579/ (Lindquist, Gates, Luszczek, Dongarra, IEEE HPEC 2022)
**Source:** https://zenodo.org/records/6972268 (Software implementation in SLATE)
**Source:** https://icl.utk.edu/publications/threshold-pivoting-dense-lu-factorization
**Relevant to:** LU worker
**Worker's current problem:** Pivoting is the serial bottleneck in GPU LU panel factorization. Each column requires argmax + row swap, creating sequential dependencies. Need ways to reduce or eliminate pivoting overhead without sacrificing numerical stability.

---

## What This Is

Threshold pivoting is a modification to partial pivoting that avoids row swaps
when the current diagonal element is "close enough" to the maximum. Instead of
always swapping with the row containing the largest absolute value, it only swaps
when the current pivot is significantly smaller than the best candidate.

This reduces row interchange data movement by up to 44% with minimal accuracy loss.

---

## The Algorithm

### Standard Partial Pivoting (for comparison)

```
For each column k:
  1. Find max_val = max(|A[i,k]|) for i >= k      // ALWAYS do full argmax
  2. pivot_row = argmax(|A[i,k]|) for i >= k
  3. IF pivot_row != k: swap(row k, row pivot_row)  // ALWAYS swap if different
```

### Threshold Pivoting

```
For each column k:
  1. IF |A[k,k]| >= threshold * max(|A[i,k]|) for i >= k:
       // Current diagonal is "good enough" -- NO SWAP needed
       pivot_row = k
  2. ELSE:
       // Current diagonal is too small -- swap as usual
       Find pivot_row = argmax(|A[i,k]|) for i >= k
       swap(row k, row pivot_row)
```

### The Threshold Parameter

- `threshold = 1.0` -> standard partial pivoting (always swap with max)
- `threshold = 0.0` -> no pivoting (never swap, numerically unsafe)
- `threshold = 0.5` -> typical threshold (swap only when diagonal < 50% of max)
- Typical values: 0.25 to 0.75

The key insight: in many matrices, the diagonal element is already close to the
maximum in most columns. The argmax is still computed, but the expensive row swap
(touching ALL N columns of the row) is often avoided.

---

## Two Levels of Communication Avoidance

### Level 1: Avoid Inter-Block Row Swaps (Conservative)

When the pivot is within the current thread block's rows, the swap stays local
(shared memory or register swap). Only when a remote block owns the pivot row
does a global memory swap occur.

```
threshold_local = 0.5  // swap with remote only if local diag < 50% of global max
```

This avoids the expensive global-memory row interchange in many cases.

### Level 2: Avoid All Row Swaps (Aggressive)

Apply threshold to ALL swaps -- even local ones. If the current diagonal is
"good enough," skip the swap entirely.

```
threshold_any = 0.5   // skip ALL swaps when diag is adequate
```

---

## Performance Results (Summit, NB=896)

| Configuration | Speedup vs Partial Pivoting | Accuracy Impact |
|---------------|---------------------------|-----------------|
| Threshold 0.5 (inter-process only) | **up to 32%** | Negligible |
| Threshold 0.5 (all swaps) | **up to 44%** | ~1 digit lost |
| Threshold 0.25 | Moderate | Very small |

"Threshold pivoting improved performance by up to 32% without a significant effect
on accuracy."

---

## Growth Factor Analysis

The authors proved that element growth with threshold pivoting cannot be bounded
by partial pivoting's growth factor and vice versa -- they are incomparable.
However, empirically the growth is comparable for practical matrices.

---

## Why This Matters for Our Monolithic Kernel

### For the Panel Factorization

In our cooperative-groups monolithic kernel, the panel factorization is done by
a single block (or small set of blocks). Each column's argmax requires either:
- Warp-level reduction (if all data in registers)
- Shared memory reduction (if data spans warps)
- Global memory reduction (if data spans blocks)

With threshold pivoting:
1. **Always compute the argmax** (needed to check the threshold)
2. **Skip the row swap** ~50-70% of the time (empirically)
3. **Still maintain numerical stability** for most practical matrices

The row swap is the expensive part in a monolithic kernel -- it requires touching
all N columns across the full matrix width. Avoiding even half the swaps saves
significant global memory bandwidth.

### Concrete Savings for N=4096, NB=64

Each row swap touches N * sizeof(float) = 16 KB. Over 64 columns per panel:
- Partial pivoting: ~64 swaps * 16 KB = ~1 MB per panel
- Threshold (skip 50%): ~32 swaps * 16 KB = ~512 KB per panel
- Over 64 panels: saves ~32 MB of global memory traffic

At 1.8 TB/s bandwidth: 32 MB / 1.8 TB/s = ~18 us saved. Modest but real.

### Combined with rowid Trick

Threshold pivoting composes well with MAGMA's rowid trick:
1. Within the panel (NB columns), use rowid trick (zero-cost virtual swaps)
2. For the LASWP across the full row width, use threshold pivoting to skip
   swaps where the diagonal is adequate
3. Result: panel has zero swap cost, LASWP has reduced swap count

### Combined with RBT

For matrices known to be well-conditioned, RBT preprocessing can eliminate pivoting
entirely. Threshold pivoting is the middle ground: it works for ALL matrices
(including ill-conditioned) with a tunable accuracy-performance tradeoff.

---

## Implementation in SLATE

SLATE (Software for Linear Algebra Targeting Exascale) contains the reference
implementation of threshold pivoting:

- Source code: https://zenodo.org/records/6972268
- Two variants: inter-process only vs all swaps
- Tuning parameters: block size (nb=896), inner block size (ib=32)

The SLATE implementation modifies `getrf` with minimal changes -- the threshold
check is a single conditional wrapping the existing swap logic.

---

## Caveats

1. **Argmax is still required.** Threshold pivoting does NOT eliminate the argmax
   reduction (unlike RBT or tournament pivoting). It only eliminates the row swap
   when the threshold is met.

2. **Not deterministic.** With threshold pivoting, different runs may produce
   different pivot sequences (depending on initial matrix state). This affects
   reproducibility but not correctness.

3. **Potential for growth.** While empirically safe, threshold pivoting's growth
   factor cannot be bounded by partial pivoting's. For safety-critical applications,
   monitor the growth factor and fall back to partial pivoting if it exceeds bounds.

4. **GPU-specific benefit is in LASWP.** The argmax cost is the same. The benefit
   is entirely in avoided row swap memory traffic. For bandwidth-bound kernels,
   this matters; for compute-bound kernels, it may not.

---

## Sources

- [Threshold Pivoting for Dense LU (IEEE HPEC 2022)](https://ieeexplore.ieee.org/document/10024579/)
- [ICL Publication Page](https://icl.utk.edu/publications/threshold-pivoting-dense-lu-factorization)
- [SLATE Software Implementation](https://zenodo.org/records/6972268)
- [Threshold Pivoting Paper PDF](https://www.netlib.org/utk/people/JackDongarra/PAPERS/Threshold_Pivoting_for_Dense_LU_Factorization.pdf)
