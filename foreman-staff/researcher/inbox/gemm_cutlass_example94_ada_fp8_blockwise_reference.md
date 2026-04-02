# CUTLASS Example 94: Ada FP8 Blockwise Dequantization GEMM

**Source:** [https://github.com/NVIDIA/cutlass/tree/main/examples/94_ada_fp8_blockwise](https://github.com/NVIDIA/cutlass/tree/main/examples/94_ada_fp8_blockwise)
**Relevant to:** GEMM worker (FP8 kernel)
**Worker's current problem:** FP8 GEMM at 1.29x cuBLAS with 64x128 tiles. BF16-to-FP8 conversion overhead is the primary remaining bottleneck (long_scoreboard 40% in smaller configs). Need native FP8 input path.
**Date:** 2026-03-15

---

## What This Is

CUTLASS example 94 is a reference FP8xFP8 -> BF16 GEMM with blockwise
dequantization, targeting Ada (sm_89). Since Ada uses the same mma.sync ISA as
sm_120 (same m16n8k32 FP8 MMA instruction), this example is directly
applicable as a reference implementation for our sm_120 FP8 GEMM.

This is DIFFERENT from example 79 (which uses block-scaled MMA with the
`.kind::mxf8f6f4.block_scale` qualifier). Example 94 uses the same regular
`mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32` that our kernel uses.

---

## Architecture Parameters

| Parameter | Value | Comparison to Our Kernel |
|-----------|-------|-------------------------|
| ThreadblockShape | 64x128x128 | Close (ours: 64x128x64) |
| WarpShape | 64x64x128 | Ours: similar per-warp |
| InstructionShape | 16x8x32 | **Same** (FP8 e4m3 mma.sync) |
| BlockScaleSize | 128 | We don't use block scaling |
| Stages | 3 | **WARNING: 3-stage kills L1 on sm_120** |
| Alignment | 16 | Same |
| Architecture | sm_89 | Ours: sm_120 (same ISA) |

---

## Key Implementation Patterns

### 1. Native FP8 Input Path

The most important difference from our kernel: **inputs are already FP8 e4m3**.
There is no BF16-to-FP8 conversion overhead. The data path is:
```
Global memory (FP8) -> cp.async -> Shared memory (FP8) -> ldmatrix -> Registers (FP8) -> MMA
```

This is exactly the "native FP8 inputs" path that our GEMM worker has identified
as the next optimization direction (and filed a foreman note about). Example 94
provides a working reference.

### 2. Blockwise Scaling

Scaling factors are stored separately as FP32 values with one scale per
BlockScaleSize=128 elements along the K dimension. The dequantization happens
inside the MMA loop:
```
accumulated_result += scale_a[block_k] * scale_b[block_k] * MMA(A_fp8, B_fp8)
```

The scaling is applied to the FP32 accumulator AFTER the MMA, not to the inputs
before MMA. This preserves full FP32 precision for the scale application.

### 3. K=128 Inner Tile

The K dimension tile is 128, which means 128/32 = 4 FP8 MMA instructions per
K iteration. This is 2x our current BLOCK_K=64. The larger K tile increases
arithmetic intensity but also increases register pressure and shared memory
usage.

**For our kernel:** We use BLOCK_K=64 (2 FP8 MMAs per K iteration). Moving to
BLOCK_K=128 would double shared memory usage per pipeline stage. With our
double-buffering, this would be 2 * (64*128 + 128*128) * 1 byte = 49 KB per
buffer pair. That's tight but within the 99 KB limit.

---

## Caveats for sm_120 Adaptation

1. **3-stage pipelining:** Example 94 uses 3-stage (triple-buffer) pipelining.
   Our hard-won lessons document that triple-buffering kills L1 cache on sm_120
   (unified 128 KB L1/smem). **Must use 2-stage pipelining on sm_120.**

2. **Architecture target:** Example 94 targets sm_89 (Ada). While the MMA ISA
   is identical, sm_120 has different cache behavior, warp scheduling, and
   shared memory sensitivity. Tile choices may need adjustment.

3. **BlockScaleSize=128:** This specific scaling pattern is for quantized models
   where inputs are pre-quantized with per-block scales. Our current use case
   (converting BF16 inputs to FP8 on the fly) doesn't need this. However, if
   we move to native FP8 inputs (where PyTorch provides pre-quantized FP8
   tensors with scale factors), this pattern becomes directly applicable.

4. **Our dual-dispatch strategy** (64x128 for small, 128x128 for large) is not
   used in example 94. Their fixed 64x128 tile may not be optimal for all
   matrix sizes on sm_120.

---

## Action Items for GEMM Worker

1. **Study `mma_multistage_blockwise.h`** in CUTLASS for the FP8 data loading
   pattern with native FP8 inputs. This header implements the MMA loop with
   blockwise dequantization.

2. **Consider native FP8 input path:** If PyTorch provides FP8 tensors with
   scale factors (via `torch.float8_e4m3fn`), we can eliminate the BF16->FP8
   conversion entirely and adopt the example 94 data flow.

3. **Do NOT adopt 3-stage pipelining.** Stick with 2-stage double-buffer.

4. **The BLOCK_K=128 choice is worth testing** on sm_120 with 2-stage
   pipelining. The higher arithmetic intensity (4 MMAs per K load) could
   improve compute/load ratio beyond our current 5.3 MMA/KB.
