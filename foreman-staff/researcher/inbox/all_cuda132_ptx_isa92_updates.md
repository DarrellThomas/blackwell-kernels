# CUDA 13.2 / PTX ISA 9.2 Updates for sm_120

**Sources:**
- [CUDA Toolkit 13.2 Release Notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)
- [PTX ISA 9.2 Documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)
**Relevant to:** all workers
**Date:** March 2026

## PTX ISA 9.2 New Instructions

### 1. FP8 → BF16 Packed Conversion (NEW)

```
cvt.rn.bf16x2.e4m3x2  dst, src;   // 2x FP8 E4M3 → 2x BF16
cvt.rn.bf16x2.e5m2x2  dst, src;   // 2x FP8 E5M2 → 2x BF16
cvt.rn.bf16x2.e3m2x2  dst, src;   // 2x FP8 → 2x BF16 (other formats)
cvt.rn.bf16x2.e2m3x2  dst, src;
cvt.rn.bf16x2.e2m1x2  dst, src;
```

**What it does:** Vectorized FP8 → BF16 dequantization. Converts 2 packed FP8 values
to 2 packed BF16 values in a single instruction.

**Relevance:** This is the DEQUANTIZATION direction (FP8→BF16). Our attention worker
converts in the opposite direction (BF16→FP8 via `cvt.rn.satfinite.e4m3x2.f32`).
However, this could be useful for:
- Any kernel reading pre-quantized FP8 data that needs BF16 intermediate values
- Mixed-precision paths where FP8 data needs widening before non-MMA operations
- MXFP8 kernels that need to inspect/debug quantized values

### 2. 128-bit Async Store (NEW)

```
st.async.b128 [addr], val;
```

**What it does:** 128-bit asynchronous store to shared/global memory. Previously
`.b128` was only supported for loads.

**Relevance:** Could enable more efficient write-back patterns in epilogues. For
GEMM-style kernels writing 128-bit (4x float) results, this avoids stalling the
warp on store completion. Minor optimization opportunity.

### 3. Out-of-Bounds Ignore for Bulk Copy (NEW)

```
cp.async.bulk ... .ignore_oob;
```

**What it does:** When a bulk copy encounters out-of-bounds source addresses, it
ignores them instead of faulting. Simplifies boundary handling in tiled kernels.

**Relevance:** Could simplify edge-tile handling in GEMM/attention kernels. Currently
we handle boundaries with conditional loads and padding. With `.ignore_oob`, the
hardware handles it. Minor ergonomic improvement.

### 4. Packed 8-bit Integer Vector Operations (NEW)

```
add.u8x4  dst, src1, src2;     // 4x uint8 parallel add
sub.s8x4  dst, src1, src2;     // 4x int8 parallel subtract
min.u8x4  dst, src1, src2;     // 4x uint8 parallel min
max.u8x4  dst, src1, src2;     // 4x uint8 parallel max
```

**What it does:** SIMD operations on 4 packed 8-bit integers in a single 32-bit register.

**Relevance:** Could be useful for manipulating packed FP8 values (treating them as
uint8). For example, computing block-wise max of FP8 exponent fields for MXFP8
quantization. The FP8 E4M3 exponent is in bits [6:3], so `max.u8x4` on the raw
bytes would give a quick approximation of block max (not exact, but fast).

### 5. Saturating Add (NEW)

```
add.sat.u16x2  dst, src1, src2;
add.sat.s16x2  dst, src1, src2;
add.sat.u32    dst, src1, src2;
```

**Relevance:** Minimal. Saturating arithmetic is not commonly needed in our kernels.

## cuBLAS 13.2 Updates

- Extended Grouped GEMM API to support **MXFP8 inputs on sm_100 and sm_110**.
  Note: This is datacenter Blackwell only — NOT sm_120. Our workers use custom
  kernels, not cuBLAS, for FP8/MXFP8 GEMM.
- Improved FP32 GEMM heuristics for shapes where M,N >> K on Blackwell.
- FP16/BF16 GEMM improvements on "Blackwell Thor" (datacenter variant).

## cuSPARSE 13.2 Updates

- No sm_120-specific changes noted in release notes.
- General stability improvements.

## Summary for Workers

| Feature | Impact | Workers |
|---------|--------|---------|
| `cvt.bf16x2.e4m3x2` | Low (wrong direction for attention) | attention, gemm |
| `st.async.b128` | Low (minor epilogue optimization) | all |
| `cp.async.bulk .ignore_oob` | Low (boundary simplification) | all |
| `u8x4/s8x4` vector ops | Medium (FP8 byte manipulation) | attention, gemm |
| Grouped GEMM MXFP8 (cuBLAS) | None (sm_100 only) | none |

**Bottom line:** PTX ISA 9.2 has incremental improvements. No game-changing
new instructions for sm_120. The packed u8x4 ops could help with FP8 byte
manipulation in MXFP8 quantization paths.
