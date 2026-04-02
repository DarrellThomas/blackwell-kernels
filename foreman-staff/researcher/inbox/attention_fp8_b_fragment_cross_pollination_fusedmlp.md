# FP8 B Fragment Layout — Cross-Pollination from Fused-MLP Worker

**Source:** `/data/src/bwk/fused-mlp/for_foreman-claude/fp8_native_b_fragment_layout.md`
**Relevant to:** attention worker (FP8 kernel, exp 66 followup)
**Worker's current problem:** Experiment 66 confirmed scalar uint16 loads are too slow for FP8 B fragments. Needs ldmatrix-based loading. The exact byte interleaving pattern was unknown — fused-mlp worker has now empirically verified it.

## What This Is

The fused-mlp worker independently tackled the same FP8 B fragment loading problem and
EMPIRICALLY VERIFIED the exact byte layout for `mma.sync.aligned.m16n8k32` B^T fragments
on sm_120. This is the missing piece for the attention worker's ldmatrix.b16 reinterpret
approach.

## Verified B Fragment Byte Layout (m16n8k32, e4m3)

For each thread with `lane_id` (0-31):

```
n_col  = lane_id / 4      (lanes 0-3 → n=0, 4-7 → n=1, ..., 28-31 → n=7)
k_base = (lane_id % 4) * 2

Register b0 = {B[k_base,    n_col], B[k_base+1,  n_col],
               B[k_base+16, n_col], B[k_base+17, n_col]}

Register b1 = {B[k_base+8,  n_col], B[k_base+9,  n_col],
               B[k_base+24, n_col], B[k_base+25, n_col]}
```

**Byte ordering is INTERLEAVED:** `{k, k+1, k+16, k+17}` — NOT consecutive `{k, k+1, k+2, k+3}`.

Each uint32 register holds 4 FP8 bytes arranged as two 16-bit pairs:
- LOW  16 bits: `{B[k_base, n], B[k_base+1, n]}`
- HIGH 16 bits: `{B[k_base+16, n], B[k_base+17, n]}`

## Why This Matters for ldmatrix.b16 Reinterpret

The 16-bit pair structure `{B[k, n], B[k+1, n]}` in the LOW half means:

1. **If FP8 data is stored column-major** (K-major) in smem, then B[k,n] and B[k+1,n]
   ARE adjacent bytes in memory.

2. **ldmatrix.b16 loads 16-bit values** from smem. If smem contains column-major FP8 data,
   each 16-bit load naturally picks up `{B[k, n], B[k+1, n]}` — the exact pair needed.

3. **The k+16 stride** in the HIGH half matches ldmatrix's cross-warp distribution pattern.
   ldmatrix distributes values across thread groups with 16-element strides (since the
   M-dimension is 16). For 8-bit types with k=32, this creates the k/k+16 interleaving
   that the fused-mlp worker observed.

## Connection to Existing Briefs

The `attention_fp8_ldmatrix_new_findings_2026_03_14.md` brief established that:
- SM120 FP8 inherits SM80 INT8 fragment layout (from CUTLASS source)
- gau-nernst achieves 692 TFLOPS on RTX 5090 using ldmatrix.b16 for FP8
- SageAttention has SM89 FP8 attention using mma.sync

The fused-mlp worker's verified byte layout **confirms the theoretical prediction**.
The interleaved `{k, k+1, k+16, k+17}` pattern is exactly what you'd get from ldmatrix.b16
loading column-major 8-bit data, because ldmatrix distributes 16-bit values across the
warp with the standard 16-row stride.

## Concrete Recommendation for Attention Worker

**For K (used in QK^T, B^T position):**

1. Store pre-quantized FP8 K in smem in column-major order:
   - K_smem layout: `K_smem[k][n]` where consecutive k values are adjacent
   - For BKV=64, D=64: each K tile is 32×8 FP8 bytes = 256 bytes (vs 512 bytes BF16)

2. Use `ldmatrix_x2_trans` on the FP8 smem, treating it as 16-bit data:
   ```cuda
   // addr points to column-major FP8 data
   // ldmatrix.b16 loads 2 x 16-bit values per thread
   // Each 16-bit value = 2 adjacent FP8 bytes = {B[k, n], B[k+1, n]}
   uint32_t b0, b1;
   asm volatile(
       "ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];\n"
       : "=r"(b0), "=r"(b1)
       : "r"(addr)
   );
   ```

3. The resulting b0, b1 should be directly usable as m16n8k32 B fragments — no byte
   shuffle needed — because the column-major storage + ldmatrix.b16 distribution produces
   the exact `{k, k+1, k+16, k+17}` interleaving the MMA expects.

4. **XOR swizzle the FP8 smem** using the same pattern as BF16, but with byte-level
   addressing (multiply swizzle constants by sizeof(fp8)=1 instead of sizeof(bf16)=2).

**For V (used in PV, B^T position):**

Same approach. V is already BF16 in the current kernel; if keeping BF16, no change.
If converting V to FP8, apply the same column-major + ldmatrix.b16 technique.

## Fused-MLP Worker's Alternative (Column-Major Transpose)

The fused-mlp worker is pursuing a different path: transpose FP8 data from row-major
to column-major IN SMEM (cp.async row-major → sync → transpose → sync → compute).
This works but adds 2 extra __syncthreads and a transpose pass.

For the attention kernel, a simpler approach: accept FP8 K pre-quantized in column-major
layout from the host. This avoids the in-kernel transpose entirely. The quantization
can be done once during model loading / KV cache construction, not per-forward-pass.

## Caveats

1. **Empirically verify** the ldmatrix output matches the fused-mlp worker's mapping
   by writing a minimal test kernel that loads FP8 data via ldmatrix.b16 and prints
   per-thread register contents. Compare against the verified mapping above.

2. **The k+16 stride** means the smem layout is not pure column-major for >16 k-values.
   The ldmatrix hardware interleaves across the 16-row boundary. This should work
   automatically if the column-major data covers the full k=32 range, but verify.

3. **Register count:** b0 + b1 = 2 registers per B fragment load (same as BF16).
   No increase in register pressure from the FP8 path.
