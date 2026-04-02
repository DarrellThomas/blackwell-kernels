# FP8 B Fragment Byte Mapping — Cross-Pollinated from fused-mlp Worker

**Source:** fused-mlp/for_foreman-claude/fp8_native_b_fragment_layout.md (empirically verified)
**Relevant to:** attention worker (FP8 kernel, native FP8 K/V inputs)
**Worker's current problem:** FP8 native K loading via scalar uint16 loads is 19% SLOWER than ldmatrix+CVT (experiment 66). The missing piece was the exact B fragment byte mapping.

## What This Is

The fused-mlp worker has **empirically verified** the FP8 m16n8k32 B operand fragment
layout. This is the thread-to-byte mapping that was listed as "still unknown" in our
previous research briefs. This mapping is required for ANY approach to native FP8 B
loading (column-major transpose, ldmatrix reinterpret, or register-level PRMT).

## The B Fragment Layout (VERIFIED)

For `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`, the B operand has
2 x uint32 registers per thread (b0, b1). Each register holds 4 FP8 bytes.

```
n_col  = lane_id / 4
k_base = (lane_id % 4) * 2

b0 = { B[k_base,    n_col],     // byte 0
       B[k_base+1,  n_col],     // byte 1
       B[k_base+16, n_col],     // byte 2
       B[k_base+17, n_col] }    // byte 3

b1 = { B[k_base+8,  n_col],     // byte 0
       B[k_base+9,  n_col],     // byte 1
       B[k_base+24, n_col],     // byte 2
       B[k_base+25, n_col] }    // byte 3
```

**CRITICAL: byte order is INTERLEAVED between K-halves.**

Each 32-bit register packs values from K-first-half AND K-second-half:
- b0 bytes 0,1 from K=0..15 (k_base, k_base+1)
- b0 bytes 2,3 from K=16..31 (k_base+16, k_base+17)

This is NOT what naive FP8 packing would produce. A simple byte-by-byte fill
gives {k, k+1, k+2, k+3} — the actual hardware needs {k, k+1, k+16, k+17}.

## Why Experiment 66 Failed

The attention worker's experiment 66 used scalar uint16 loads to build B fragments
from FP8 smem. Two problems:
1. **Scalar loads are slow** — 16-bit loads with stride-64 cause bank conflicts
2. **The byte ordering was probably wrong** — without the interleaved mapping, the
   packed bytes wouldn't match what the MMA hardware expects

## How to Use This for the Attention Kernel

### Option 1: ldmatrix.b16 Reinterpret + PRMT (RECOMMENDED)

Store K (or V) as BF16 in smem (same as now). Use ldmatrix to load as "16-bit pairs"
of FP8 bytes. Then use PRMT to rearrange the bytes into the interleaved pattern.

```
// Load K via ldmatrix_x2_trans.b16 (same as BF16 path)
// r0 = K-first-half (k=0..15 packed as 16-bit pairs)
// r1 = K-second-half (k=16..31 packed as 16-bit pairs)

// PRMT to get interleaved layout:
prmt.b32 b0, r0, r1, selector0;  // {k, k+1, k+16, k+17}
prmt.b32 b1, r0, r1, selector1;  // {k+8, k+9, k+24, k+25}
```

**The PRMT selectors need empirical derivation** — they depend on exactly which bytes
of r0/r1 correspond to which K-indices after ldmatrix_x2_trans. Run a test kernel
that fills smem with known FP8 values, loads via ldmatrix_x2_trans, and reads the
register bytes to determine the mapping.

### Option 2: Pre-quantized FP8 in Column-Major Smem

Store K/V as FP8 in column-major format in smem. Then 32-bit aligned loads get 4
consecutive k-values for the same n-column. The interleaved pattern {k, k+1, k+16, k+17}
can be achieved with a single 32-bit load per register IF the column is padded to
align the k+16 stride.

This requires a transpose during loading (cp.async + register-mediated transpose),
which adds 2 syncthreads and ~64 smem ops per thread per K-tile.

### Option 3: Direct Global Loads (Probably Too Slow)

Each thread does 8 global byte loads of FP8 values. With L2 hitting (K/V reused
across Q-blocks), this avoids smem entirely. But 64 global loads per B load vs
16 ldmatrix loads is likely too many.

## For K Loading in Attention (QK^T phase)

K is the B operand for QK^T. In our kernel, K is loaded via:
- `cp.async` from global to smem (double-buffered)
- `ldmatrix_x4` or `ldmatrix_x2_trans` from smem to registers
- FP8 conversion via `cvt.rn.satfinite.e4m3x2.f32`

With the ldmatrix+PRMT approach, the conversion is replaced by 2 PRMT instructions.
This saves ~224 conversion instructions per KV block (half the total ~448 conversion
overhead). Combined with the same approach for V, this could save ~7-8 us, bringing
FP8 attention from 52 us toward ~44 us (~2.7x SDPA).

## For V Loading in Attention (PV phase)

V is also the B operand (for PV). Same mapping applies. The ldmatrix_x2_trans
approach should work identically.

## Recommendation

1. **Write a test kernel** that loads known FP8 data via ldmatrix_x2_trans.b16
   and inspects register bytes. This confirms the exact ldmatrix → PRMT mapping.
2. **Derive PRMT selectors** from the test results.
3. **Implement in the attention kernel** — replace CVT with PRMT for K and V.
4. **Benchmark** — expect ~7-8 us savings from eliminated conversion.

This is the highest-value FP8 optimization remaining for the attention kernel.
