# UPDATE: FP8 ldmatrix Reinterpret — New External Validation

**Supplements:** attention_fp8_ldmatrix_reinterpret_technique.md (existing brief)
**Sources:**
- https://hazyresearch.stanford.edu/blog/2024-11-27-tk-fp8
- https://forums.developer.nvidia.com/t/how-to-load-fp8-using-ldmatrix-on-sm120-sm120a/330254
- https://research.colfax-intl.com/adding-fp8-to-flashattention/
**Relevant to:** attention worker (FP8 kernel)

## New Findings

### 1. ThunderKittens Confirms the ldmatrix.b16 Reinterpret Approach

ThunderKittens (Hazy Research, Nov 2024) independently uses exactly this technique:
"let the instruction load in 16-bits and use this to fill 2 fp8 values per load."
They describe this as "one of several hacks the hardware requires."

This is EXTERNAL VALIDATION that the ldmatrix.b16 reinterpret approach works. It is
not speculative -- a production framework achieving 1500 TFLOPS on H100 uses it.

### 2. NVIDIA Forum Thread on sm_120 FP8 ldmatrix

A user on the NVIDIA Developer Forums (thread 330254) attempted to use ldmatrix
on sm_120/sm_120a to load an 8x32 FP8 matrix for m16n8k32 MMA. Key findings:

- The user found NO suitable ldmatrix tile size in the PTX ISA for directly loading
  an 8x32 FP8 matrix.
- Curefab (forum member) suggested two approaches:
  (a) "You do not have to use ldmatrix, you can also directly load the elements"
  (b) "use it and then reshuffle the elements"

**Interpretation:** Option (b) is exactly our approach -- use ldmatrix.m8n8.x4.b16
to load the data (treating FP8 pairs as 16-bit elements), then reshuffle if needed.
The fact that this was suggested by an experienced CUDA developer reinforces viability.
Our structural analysis suggests reshuffling may NOT be needed if the smem layout
matches the fragment layout expectations.

### 3. Colfax FP8 FlashAttention: Fragment Layout Mismatch is Real

Colfax's FP8 FlashAttention-2 implementation documents that FP32 accumulator layout
and FP8 operand A layout are structurally different (PTX ISA Figures 118 vs 122).
For BF16, they happen to be identical (no data movement needed). For FP8, they differ,
requiring byte_perm + shfl_sync for the P->FP8 conversion path.

**Impact on our kernel:** The P->FP8 conversion path (~112 instructions/block) may
need the additional shfl_sync to correctly build FP8 A fragments from FP32 accumulators.
This doesn't affect K/V loading (which goes smem -> ldmatrix -> registers directly),
but it means the P conversion is slightly more complex than just `cvt.e4m3x2.f32` + pack.

### 4. B Operand Fragment for m16n8k32

From PTX ISA documentation and MLIR references, the B operand for m16n8k32 MMA
requires 2 x uint32 registers (= 8 FP8 bytes = 8 FP8 values). This matches
ldmatrix_x2 output (2 registers), confirming that:

- K (B operand for QK^T): ldmatrix_x2.b16 loads 2 registers of "16-bit pairs" =
  4 FP8 bytes each = 8 FP8 total. This is the correct count.
- V (B operand for PV): ldmatrix_x2_trans.b16 should work the same way.

### 5. MMA Register Counts Confirmed

| Operand | m16n8k16 BF16 | m16n8k32 FP8 | Same count? |
|---------|---------------|--------------|-------------|
| A | 4 x uint32 (8 BF16) | 4 x uint32 (16 FP8) | YES |
| B | 2 x uint32 (4 BF16) | 2 x uint32 (8 FP8) | YES |
| C/D | 4 x float32 | 4 x float32 | YES |

The register count is IDENTICAL between BF16 m16n8k16 and FP8 m16n8k32. This is the
strongest evidence that ldmatrix_x4.b16 (for A) and ldmatrix_x2.b16 (for B) produce
the correct register layout for FP8 MMA when FP8 data is stored as b16 pairs.

## Updated Confidence Assessment

| Aspect | Previous | Updated |
|--------|----------|---------|
| ldmatrix.b16 reinterpret works | Structural analysis only | **Externally validated (ThunderKittens)** |
| A operand (Q, K as A) | High confidence | **Very high** (register counts match) |
| B operand (K as B, V as B) | Uncertain | **High** (register counts match, x2 variant confirmed) |
| ldmatrix_x2_trans for FP8 B | Low confidence | **Medium** (needs empirical test, but count is right) |
| P->FP8 path | Assumed simple | **Needs shfl_sync** (Colfax confirmed layout mismatch) |
| Byte ordering | Unknown | Still unknown (test needed) |

## Revised Recommendation

The ldmatrix.b16 reinterpret approach for K and V loading has strong external
validation and should be attempted. The main risk is now concentrated in two areas:

1. **ldmatrix_x2_trans for V loading** -- test this first in isolation
2. **P->FP8 conversion with correct fragment layout** -- may need shfl_sync
   in addition to cvt.e4m3x2.f32

The worker should build the test_mma verification mentioned in the original brief
as the FIRST step, before any kernel changes.
