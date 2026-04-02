# PTX ISA 9.2: FP8 MMA with FP16 Accumulator (NEW for sm_120)

**Sources:**
- [PTX ISA 9.2 Documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [CUDA 13.2 Release Notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)
- [CUDA Features Archive 13.2](https://docs.nvidia.com/cuda/cuda-features-archive/index.html)
**Relevant to:** attention, gemm, fused-mlp workers (anyone using FP8 MMA)
**Date:** 2026-03-15

---

## What This Is

PTX ISA 9.2 (shipped with CUDA 13.2) extended the `mma.sync` instruction to support
**FP16 accumulators** with FP8 input types and shape m16n8k16:

```
mma.sync.aligned.m16n8k16.row.col.f16.e4m3.e4m3.f16  {d0,d1}, {a0}, {b0}, {c0,c1};
mma.sync.aligned.m16n8k16.row.col.f16.e5m2.e4m3.f16  {d0,d1}, {a0}, {b0}, {c0,c1};
```

Previously, FP8 MMA on sm_120 only supported FP32 accumulators:
```
mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32  {d0,d1,d2,d3}, {a0,a1}, {b0}, {c0,c1,c2,c3};
```

---

## Why It Matters for Us

### 1. Half the Accumulator Registers

FP32 accumulator uses 4 registers per MMA output (128 bits = 4x f32).
FP16 accumulator uses 2 registers per MMA output (64 bits = 2x f16x2).

For our attention kernel (165 registers, 3 blocks/SM), register pressure is a
primary constraint. Cutting accumulator registers in half could:
- Free registers for more aggressive data prefetching
- Enable higher occupancy (more warps per SM)
- Reduce register spilling to local memory

### 2. Potential Throughput Difference

On sm_89 (Ada), FP16-accumulator MMA instructions have the SAME throughput as FP32
accumulators. On sm_120, this needs verification -- there MAY be a throughput difference.
If FP16 accumulators are faster (like on some Hopper configs), this is a significant win.

**CAUTION:** FP16 accumulation has lower numerical precision than FP32. For attention,
the softmax scaling chain may lose accuracy. For GEMM, accumulated dot products of
K=4096+ elements in FP16 will have more rounding error than FP32. This needs careful
accuracy testing.

### 3. Smaller Shape (m16n8k16 vs m16n8k32)

Note the shape difference: FP8 with FP16 accumulator uses **m16n8k16** (not k32).
This means half the K-dimension per MMA instruction:
- k16 does 16 FP8 multiply-accumulates per instruction
- k32 does 32 FP8 multiply-accumulates per instruction

So the instruction count doubles for the same K dimension, but each instruction uses
fewer registers and may have lower latency. Net throughput depends on whether the
pipeline can sustain back-to-back k16 MMAs.

---

## Practical Investigation Steps

1. **Throughput test:** Write a microbenchmark comparing:
   - `mma.sync.m16n8k32.f32.e4m3.e4m3.f32` (current FP8 path)
   - `mma.sync.m16n8k16.f16.e4m3.e4m3.f16` (new FP16-accum path)
   Measure cycles per MMA instruction on sm_120.

2. **Accuracy test:** For attention:
   - Run FP8-input attention with FP16 accumulator vs FP32 accumulator
   - Compare against FP64 reference
   - Check if softmax intermediate values overflow FP16 range (max 65504)

3. **Register pressure test:** Profile register usage with ncu for both variants.
   If FP16 accum saves enough registers to bump occupancy from 3 to 4 blocks/SM,
   this could be significant for latency-bound kernels.

---

## Caveats

1. **m16n8k16 not m16n8k32:** The shape is DIFFERENT. This is not a drop-in
   replacement -- the fragment layout and register mapping are different from
   the k32 variant. The A matrix operand uses 1 register (not 2), and the C/D
   operands use 2 registers (not 4).

2. **FP16 overflow risk:** FP16 max value is 65504. If accumulated values exceed
   this (common in deep layers of large models), the result will be inf/nan. For
   attention, the pre-softmax logits typically stay in reasonable range, but this
   must be verified for the specific model configs we target.

3. **Two-step accumulation:** A practical middle ground is to use FP16 accumulators
   within a tile's K-loop iterations, then periodically convert and accumulate into
   FP32 registers. This gives most of the register savings while controlling
   precision loss. Similar to what CUTLASS calls "mixed accumulation."

4. **No ldmatrix change needed:** The A/B operand loading is the same (FP8 values
   loaded with ldmatrix or cp.async). Only the C/D accumulator handling changes.
