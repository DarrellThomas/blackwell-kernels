# FP8 Native Inputs via ldmatrix.b16 Reinterpret

**Source:** Derived from PTX ISA analysis + empirical fragment layout data from this project
**Relevant to:** attention worker (FP8 kernel)
**Worker's current problem:** BF16→FP8 conversion overhead is 14:1 vs MMA count (~448 ALU instructions/KV block), adding ~7.5 μs to theoretical minimum. Experiment 66 proved scalar loads cannot replace ldmatrix.

## What This Is

A technique for loading pre-quantized FP8 (e4m3) data from shared memory using
ldmatrix.b16, completely eliminating the BF16→FP8 conversion path. The key insight:
ldmatrix.b16 treats data as 16-bit elements, and each 16-bit "element" can hold
exactly 2 packed FP8 bytes. The m16n8k32 fragment layout is structurally identical
to m16n8k16 when viewed through 16-bit ldmatrix loads.

## Why It Matters for Us

The FP8 attention kernel is at 52 μs (2.33x SDPA). The conversion overhead accounts
for ~7.5 μs of this. Eliminating it would target ~44.5 μs — a potential 15% speedup
that would push to ~2.7x SDPA. Critically, this approach preserves ldmatrix throughput
(the lesson from experiments 65-66: ldmatrix CANNOT be replaced with scalar loads).

## Key Technique

### The Fragment Layout Correspondence

For BF16 m16n8k16 MMA:
- A fragment: 4 × uint32, each holding 2 BF16 values (16-bit × 2 = 32 bits)
- K dimension = 16 BF16 values

For FP8 m16n8k32 MMA:
- A fragment: 4 × uint32, each holding 4 FP8 values (8-bit × 4 = 32 bits)
- K dimension = 32 FP8 values

**The correspondence:** K=32 FP8 values = K=16 pairs of FP8 bytes. Each pair occupies
exactly one 16-bit position. Therefore the m16n8k32 fragment layout, when viewed at
16-bit granularity, has the SAME structure as m16n8k16.

### ldmatrix.b16 with FP8 Data

ldmatrix.x4.b16 loads an 8×8 tile of 16-bit elements (128 bits per thread row).
If each "16-bit element" is actually {fp8[row][2k+1], fp8[row][2k]}, then:

```
ldmatrix loads "16×16 of 16-bit" → actually 16×32 of FP8
Sub-tile 0 (threads 0-7):   rows 0-7,  K_pairs 0-7  → FP8 K=0-15
Sub-tile 1 (threads 8-15):  rows 0-7,  K_pairs 8-15 → FP8 K=16-31
Sub-tile 2 (threads 16-23): rows 8-15, K_pairs 0-7  → FP8 K=0-15
Sub-tile 3 (threads 24-31): rows 8-15, K_pairs 8-15 → FP8 K=16-31
```

Output registers:
```
r0: rows 0-7,  K=0-15  (4 FP8 bytes per thread)
r1: rows 0-7,  K=16-31
r2: rows 8-15, K=0-15
r3: rows 8-15, K=16-31
```

### The a1/a2 Swap Still Applies

m16n8k32 expects: a0=rows 0-7 K-first-half, a1=rows 8-15 K-first-half,
a2=rows 0-7 K-second-half, a3=rows 8-15 K-second-half.

So: `(r0, r2, r1, r3)` → `(a0, a1, a2, a3)` — same swap as BF16.
`ldmatrix_x4_mma()` with `{%0, %2, %1, %3}` works unchanged.

### Shared Memory Layout

Store FP8 data in smem as row-major bytes:
```
smem_fp8[row][k] = fp8_value_at(row, k)    // byte-addressed
// Equivalent to treating as 16-bit:
smem_b16[row][k_pair] = {fp8[row][2*k_pair+1], fp8[row][2*k_pair]}  // little-endian
```

For an 8×32 tile (8 rows × 32 FP8 columns = 256 bytes = 128-bit aligned rows):
- Each row is 32 bytes = 256 bits = 2 × 128-bit loads
- ldmatrix loads 128 bits (16 FP8 values) per thread, covering 8 K-pairs

**XOR swizzle:** The same swizzle pattern used for BF16 applies because we're still
accessing 16-bit elements from 128-bit aligned rows. The swizzle XOR mask operates
on the 16-bit element index, which maps to K_pair positions.

### B Operand (K and V)

For QK^T: K is the B operand. Currently loaded via ldmatrix_x4 and used as B fragment
for A*B^T. With FP8 K in smem, the same ldmatrix call loads pairs of FP8 bytes.

For PV: V is the B operand, loaded via ldmatrix_x2_trans. Same approach — each "16-bit
element" holds 2 FP8 bytes at consecutive K positions.

B fragment for m16n8k32: 2 × uint32 = 8 FP8 values per thread. ldmatrix_x2_trans
loads 2 × 32-bit = 8 "16-bit values" (via transpose), which are 16 FP8 bytes —
but only 8 needed per thread. **Need to verify empirically whether ldmatrix_x2_trans
produces correct B fragments for m16n8k32 with FP8 reinterpret.**

### What Gets Eliminated

| Component | Current (BF16 input) | With FP8 native input |
|-----------|---------------------|----------------------|
| K load | ldmatrix_x4 (BF16) | ldmatrix_x4 (FP8 reinterpret) |
| K convert | ~224 ALU/block | **ZERO** |
| V load | ldmatrix_x4_trans (BF16) | ldmatrix_x4_trans (FP8 reinterpret) |
| V convert | ~224 ALU/block | **ZERO** |
| Q load | ldmatrix_x4_mma (once) | ldmatrix_x4_mma (FP8 reinterpret, once) |
| Q convert | ~14 ALU (once) | **ZERO** |
| P convert | ~7 ALU/MMA pair | ~7 ALU/MMA pair (still needed) |
| smem/block | 32 KB (BF16 K+V) | **16 KB** (FP8 K+V, half size) |

Total savings: ~448 ALU/block → ~7.5 μs saved → target 44.5 μs (was 52 μs)

### cp.async Considerations

cp.async loads from global to shared memory. Currently loads BF16 (16-byte = 8 BF16
per cp.async.cg). With FP8 inputs, cp.async loads FP8 bytes (16-byte = 16 FP8 per
cp.async.cg). This HALVES the number of cp.async calls needed for K and V, reducing
memory pipeline pressure.

## Implementation Steps

1. **Accept FP8 tensors from host** (pre-quantized Q, K, V in e4m3 format)
2. **cp.async FP8 data** to shared memory (half the bandwidth of BF16)
3. **Use ldmatrix_x4_mma** unchanged — loads "16-bit elements" that are FP8 pairs
4. **Feed registers directly to m16n8k32 MMA** — no conversion needed
5. **P conversion remains:** FP32 softmax output → FP8 for PV MMA (unavoidable)
6. **Verify with test_mma**: write a minimal test loading FP8 from smem via ldmatrix,
   feeding to m16n8k32 MMA, and comparing output against known-good result

## Caveats

1. **Byte ordering needs empirical verification.** The analysis assumes little-endian
   packing where consecutive FP8 bytes at K=2k and K=2k+1 form a valid 16-bit ldmatrix
   element. If the hardware swaps bytes, a PRMT instruction can fix it (1 instruction
   per register, negligible overhead).

2. **API change required.** The kernel would accept `torch.float8_e4m3fn` tensors instead
   of `torch.bfloat16`. Host-side quantization adds ~5 μs (measured in exp 66), but this
   can be amortized across attention heads or done once per layer.

3. **The m16n8k32 fragment layout has NOT been empirically verified against ldmatrix
   output on sm_120.** The analysis is based on structural correspondence with m16n8k16.
   The worker MUST run a minimal test (similar to test_mma4.cu for the a1/a2 swap) before
   building a full kernel.

4. **ldmatrix_x2_trans with FP8 reinterpret for B operand is the least certain part.**
   The transpose operation re-arranges elements, and the 16-bit→2×FP8 correspondence
   may not survive transposition. Test this independently.

5. **Q can be quantized once and reused across all KV blocks.** This amortizes Q's
   quantization cost to near-zero.

6. **P→FP8 conversion still needed (~7 instr/pair).** This is the softmax output path
   and cannot be eliminated with pre-quantization. But it's only ~112 instructions/block
   (vs 448 for K+V conversion), so the net savings are ~336 instructions/block.
