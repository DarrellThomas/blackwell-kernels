# PTX ISA 9.1 New FP8 Conversion Instructions

**Sources:**
- [PTX ISA 9.2 Documentation (includes 9.1 changes)](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)
- [CUDA Toolkit 13.1 Release Notes](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-toolkit-release-notes/index.html)
- [ZLUDA FP8 cvt PR](https://github.com/vosen/ZLUDA/pull/468)

**Relevant to:** attention worker (FP8 conversion path), GEMM worker (FP8 quantization)
**Date:** 2026-03-14

---

## What Changed in PTX ISA 9.1

PTX ISA 9.1 (shipped with CUDA 13.1, January 2026) added the QUANTIZATION direction
for packed FP8 conversions:

```ptx
cvt.rn.satfinite.e4m3x2.f16x2  dst, src;   // 2x FP16 -> 2x FP8 E4M3
cvt.rn.satfinite.e5m2x2.f16x2  dst, src;   // 2x FP16 -> 2x FP8 E5M2
cvt.rn.satfinite.e4m3x2.bf16x2 dst, src;   // 2x BF16 -> 2x FP8 E4M3
cvt.rn.satfinite.e5m2x2.bf16x2 dst, src;   // 2x BF16 -> 2x FP8 E5M2
cvt.rn.satfinite.e3m2x2.f16x2  dst, src;   // 2x FP16 -> 2x FP8 E3M2
cvt.rn.satfinite.e3m2x2.bf16x2 dst, src;   // 2x BF16 -> 2x FP8 E3M2
cvt.rn.satfinite.e2m3x2.f16x2  dst, src;
cvt.rn.satfinite.e2m3x2.bf16x2 dst, src;
cvt.rn.satfinite.e2m1x2.f16x2  dst, src;
cvt.rn.satfinite.e2m1x2.bf16x2 dst, src;
```

Then PTX ISA 9.2 (CUDA 13.2, March 2026) added the DEQUANTIZATION direction:
```ptx
cvt.rn.bf16x2.e4m3x2  dst, src;   // 2x FP8 E4M3 -> 2x BF16
cvt.rn.bf16x2.e5m2x2  dst, src;   // 2x FP8 E5M2 -> 2x BF16
// ... and others (already documented in all_cuda132_ptx_isa92_updates.md)
```

## Why This Matters for Us

### Attention Worker (FP8 Path)

Our attention kernel currently converts BF16 KV data to FP8 using:
```ptx
cvt.rn.satfinite.e4m3x2.f32  dst, src1, src2;  // 2x FP32 -> 2x FP8 E4M3
```

This goes through FP32 intermediates. The new PTX 9.1 instruction allows:
```ptx
cvt.rn.satfinite.e4m3x2.bf16x2  dst, src;  // 2x BF16 -> 2x FP8 E4M3 (DIRECT)
```

**Potential benefit:** Eliminates the BF16->FP32 widening step before FP8 conversion.
If the source data is already in BF16 registers (as packed bf16x2), this is a single
instruction instead of the current unpack-to-f32 + convert-to-fp8 sequence.

**Caveat:** Our current code uses `cvt.rn.satfinite.e4m3x2.f32` which takes two
separate f32 values. The new `.bf16x2` variant takes one packed bf16x2 register.
The data layout may need to change to benefit. Specifically, if we can keep KV data
in packed BF16 format through the conversion pipeline (ldmatrix loads BF16 data as
packed pairs), the new instruction could shave instructions from the conversion path.

### GEMM Worker (FP8 Path)

If doing online quantization of BF16 inputs to FP8 within the kernel, the direct
`bf16x2 -> e4m3x2` path is more efficient than going through f32.

## Summary

| PTX ISA | Direction | Instruction Pattern | CUDA Version |
|---------|-----------|-------------------|--------------|
| 9.0 | f32 -> FP8x2 | `cvt.rn.satfinite.e4m3x2.f32` | 13.0 (existing) |
| 9.1 | bf16x2/f16x2 -> FP8x2 | `cvt.rn.satfinite.e4m3x2.bf16x2` | 13.1 (NEW) |
| 9.2 | FP8x2 -> bf16x2 | `cvt.rn.bf16x2.e4m3x2` | 13.2 (NEW) |

The complete BF16<->FP8 packed conversion pipeline is now available in hardware
as of CUDA 13.2.
