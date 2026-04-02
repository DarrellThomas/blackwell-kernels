# PTX ISA 9.2: New cvt Instruction -- FP8 to BF16x2 Direct Conversion

**Sources:**
- [PTX ISA 9.2 Documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [CUDA 13.2 Release Notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)
**Relevant to:** attention worker, gemm worker, fused-mlp worker
**Date:** 2026-03-15

---

## What This Is

PTX ISA 9.2 (shipped with CUDA 13.2) adds a new `cvt` instruction variant:

```
cvt.rn.bf16x2.e4m3x2  dest, src;
cvt.rn.bf16x2.e5m2x2  dest, src;
cvt.rn.bf16x2.e3m2x2  dest, src;
cvt.rn.bf16x2.e2m3x2  dest, src;
cvt.rn.bf16x2.e2m1x2  dest, src;
```

This converts a **pair of FP8 values directly to a pair of BF16 values** (packed bf16x2),
without going through FP32 as an intermediate.

---

## Why It Matters for Us

### The Reverse Direction

We already know about `cvt.rn.satfinite.e4m3x2.f32` (FP32 -> FP8 pair), which was our
"vectorized CVT discovery" that made FP8 attention viable.

This new instruction is the **reverse direction**: FP8 -> BF16. Use cases:

1. **FP8 native input path for attention:** If Q/K/V arrive as FP8 tensors (from a
   quantization layer or FP8 training), we could load them as FP8 and convert to BF16
   for MMA if needed. Currently we go BF16 -> FP8 -> MMA. With native FP8 inputs, we'd
   skip the BF16 -> FP8 conversion entirely, eliminating the 448 ALU instructions per
   KV block that are the attention kernel's bottleneck.

2. **Mixed-precision epilogue:** After FP8 MMA with FP32 accumulator, we store BF16
   output. If we switch to FP8 MMA with FP16 accumulator (the m16n8k16 f16 variant),
   the output is already f16x2 packed. Converting f16x2 -> bf16x2 is one instruction.
   But if we need to go FP8 -> BF16 for some other data path, this instruction helps.

3. **FP8 GEMM output conversion:** For GEMM that takes FP8 inputs and produces BF16
   output, the epilogue currently converts FP32 accumulator -> BF16. If a future path
   uses FP16 accumulators, the f16x2 -> bf16x2 conversion is already fast. The new
   cvt instruction helps in cases where intermediate FP8 results need BF16 promotion.

### Practical Impact

**For the attention worker specifically:** The current FP8 attention bottleneck is
BF16-to-FP8 conversion (forward direction). This new instruction is the reverse
direction (FP8-to-BF16). It doesn't directly solve the current bottleneck, but it
enables the **FP8 native input** optimization path:

- If PyTorch/the model provides Q, K, V as FP8 tensors natively, load them as FP8
- Feed directly to `mma.sync.m16n8k32.f32.e4m3.e4m3.f32` -- zero conversion overhead
- The cvt.bf16x2.e4m3x2 instruction is useful for any debugging/verification path
  that needs to inspect FP8 values in BF16 form

### Availability

This requires CUDA 13.2 (PTX ISA 9.2). We currently run CUDA 13.0 (PTX ISA 9.0/9.1).
To use this instruction, we would need to upgrade to CUDA 13.2.

**However:** We may not need this instruction at all if we're doing FP8 -> FP32
conversion (which is already supported via `cvt.f32.e4m3`). The new instruction's
value is specifically in the FP8 -> BF16 direction, which is a less common path for us.

---

## Caveats

1. **Requires CUDA 13.2 upgrade.** Not available in our current CUDA 13.0 toolchain.

2. **The operand packing and byte ordering need verification.** The bf16x2 destination
   packs two BF16 values. Whether the high/low byte ordering matches our existing
   bf16x2 conventions (which have the reversed operand order quirk) needs testing.

3. **Limited immediate use case.** Our current bottleneck is BF16->FP8 (forward
   conversion), not FP8->BF16 (reverse). This instruction becomes valuable only when
   we adopt FP8 native inputs, which requires upstream model support.

4. **The other new PTX 9.2 additions** (`add/sub/min/max` for u8x4/s8x4, `add.sat`
   for u16x2/s16x2/u32, `.b128` for st.async, `.ignore_oob` for cp.async.bulk) have
   limited relevance for our tensor-core-focused kernels.
