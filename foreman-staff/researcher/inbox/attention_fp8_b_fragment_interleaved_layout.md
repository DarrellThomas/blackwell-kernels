# FP8 B-Fragment Interleaved Layout — Cross-Pollination from fused-mlp Worker

**Source:** Empirically verified by fused-mlp worker (for_foreman-claude/fp8_native_b_fragment_layout.md)
**Relevant to:** attention worker (FP8 kernel, ldmatrix reinterpret approach)
**Worker's current problem:** The ldmatrix.b16 reinterpret brief assumed FP8 bytes would be consecutive in registers. The fused-mlp worker proved the B-fragment has an INTERLEAVED k-pattern.

## What This Is

The fused-mlp worker empirically verified the exact FP8 m16n8k32 B-fragment register
layout on sm_120. The critical finding: **bytes within each register are NOT consecutive
K values — they interleave K-first-half with K-second-half.**

## The Verified B-Fragment Layout (m16n8k32 FP8)

```
n_col = lane_id / 4   (lanes 0-3→n=0, 4-7→n=1, ..., 28-31→n=7)
k_base = (lane_id % 4) * 2

b0 = {B[k_base,    n],    // byte 0: K = k_base     (first half)
      B[k_base+1,  n],    // byte 1: K = k_base + 1 (first half)
      B[k_base+16, n],    // byte 2: K = k_base + 16 (SECOND half!)
      B[k_base+17, n]}    // byte 3: K = k_base + 17 (SECOND half!)

b1 = {B[k_base+8,  n],    // byte 0: K = k_base + 8
      B[k_base+9,  n],    // byte 1: K = k_base + 9
      B[k_base+24, n],    // byte 2: K = k_base + 24
      B[k_base+25, n]}    // byte 3: K = k_base + 25
```

**Pattern:** Each 4-byte register holds {k, k+1, k+16, k+17} — two pairs separated
by stride 16. The hardware groups the first and second K-halves together.

## Impact on ldmatrix.b16 Reinterpret Approach

### A Operand (Q loading): Still Works as Described

The A operand is distributed across 4 sub-tiles via ldmatrix_x4. Each sub-tile covers
either K=0-15 or K=16-31 separately. After the a1/a2 swap, the registers contain:
- a0: rows 0-7, K=0-15 (consecutive FP8 pairs)
- a1: rows 8-15, K=0-15
- a2: rows 0-7, K=16-31
- a3: rows 8-15, K=16-31

**No interleaving needed for A.** ldmatrix.b16 reinterpret works directly.

### B Operand (K/V loading): Needs 2 PRMT Instructions

ldmatrix_x2_trans with FP8 data gives:
```
r0 = {K=k0, K=k0+1, K=k1, K=k1+1}  (from sub-tile 0, K=0-15)
r1 = {K=k2, K=k2+1, K=k3, K=k3+1}  (from sub-tile 1, K=16-31)
```

But the MMA expects:
```
b0 = {K=k_base, K=k_base+1, K=k_base+16, K=k_base+17}  (interleaved!)
b1 = {K=k_base+8, K=k_base+9, K=k_base+24, K=k_base+25}
```

**Fix: 2 PRMT instructions** to merge bytes from r0 and r1:
```ptx
// r0 = {k, k+1, k+2, k+3} from K=0-15 sub-tile
// r1 = {k+16, k+17, k+18, k+19} from K=16-31 sub-tile

// b0 = {r0[0], r0[1], r1[0], r1[1]}
prmt.b32 b0, r0, r1, 0x5410;

// b1 = {r0[2], r0[3], r1[2], r1[3]}
prmt.b32 b1, r0, r1, 0x7632;
```

PRMT selects 4 bytes from the concatenation of two 32-bit values. Selector nibbles:
- 0-3 select from first operand (r0 bytes 0-3)
- 4-7 select from second operand (r1 bytes 0-3, addressed as 4-7)

Selector 0x5410: byte 0→r0[0], byte 1→r0[1], byte 2→r1[0], byte 3→r1[1]
Selector 0x7632: byte 0→r0[2], byte 1→r0[3], byte 2→r1[2], byte 3→r1[3]

### Cost Analysis

| Path | Instructions per KV block |
|------|---------------------------|
| Current (BF16 load + CVT) | ~448 CVT + pack instructions |
| ldmatrix reinterpret + PRMT | ~16-32 PRMT instructions |
| **Savings** | **~420 instructions (93% reduction)** |

The 2 PRMT per B operand load × (number of ldmatrix_x2_trans calls per KV block) ≈
16-32 PRMT total. This is negligible compared to 448 CVT instructions.

## Implementation Update

The ldmatrix.b16 reinterpret approach from the previous brief needs this modification:

1. **A operand (Q):** ldmatrix_x4_mma → direct to MMA (no change needed)
2. **B operand (K, V):** ldmatrix_x2_trans → **2 PRMT** → MMA
3. **P→FP8 conversion:** Still needs CVT + possible shuffles (unchanged)

The PRMT approach for B adds minimal overhead and preserves the ldmatrix throughput
advantage over scalar loads.

## Caveats

1. **The PRMT selector values (0x5410, 0x7632) assume specific byte ordering from
   ldmatrix_x2_trans.** The exact values need empirical verification. The principle
   (merge low bytes from r0 with low bytes from r1) is correct, but the specific
   nibble mapping depends on how ldmatrix_x2_trans distributes data.

2. **The A operand layout has NOT been separately verified for FP8.** The analysis
   assumes consecutive K pairs based on structural correspondence with BF16.
   Worth writing a minimal test to confirm.

3. **The fused-mlp worker verified this for GEMM B (not attention K/V).** The MMA
   instruction is the same (m16n8k32), so the fragment layout should be identical.
   But verify independently.
