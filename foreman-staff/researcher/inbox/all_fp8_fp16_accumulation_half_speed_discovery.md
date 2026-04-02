# URGENT: FP8 MMA Runs at HALF SPEED with FP32 Accumulation on sm_120

**Source:** CUDA 13.2 cuBLAS documentation, PTX ISA 9.2
**Relevant to:** gemm worker, attention worker, fused-mlp worker
**Worker's current problem:** FP8 kernels use FP32 accumulators. cuBLAS 13.2 confirms this runs at half tensor core throughput on consumer Blackwell.

## The Discovery

cuBLAS 13.2 documentation (March 5, 2026) states:

> "FP8 and FP16 matmuls run at full speed with FP16 accumulation but only
> half speed with FP32 accumulation on consumer Blackwell."

**This means ALL our FP8 kernels are running at 50% of peak tensor core throughput.**

Our current FP8 kernels use:
```
mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32
                                   ^^^             ^^^
                                   FP32 accumulator = HALF SPEED
```

The full-speed instruction would be:
```
mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16
                                   ^^^             ^^^
                                   FP16 accumulator = FULL SPEED
```

## Impact on Our Kernels

| Kernel | Current FP8 vs Ref | Potential with FP16 accum |
|--------|-------------------|--------------------------|
| GEMM FP8 | 1.29x cuBLAS | Could reach ~2x+ if cuBLAS also uses FP32 accum |
| Attention FP8 | 2.33x SDPA | Conversion overhead becomes even more dominant |
| Fused-MLP FP8 | 1.12x cuBLAS | GEMM1 compute doubles |

**For GEMM:** If cuBLAS 13.2 switched to FP16 accumulators for its FP8 path, our
1.29x advantage may have EVAPORATED (cuBLAS got 20% faster on sm_120 in 13.2).
Must re-benchmark immediately.

**For Attention:** The FP8 kernel is latency-bound (SM 43.8%), not compute-bound.
Doubling tensor throughput would help but the BF16→FP8 conversion overhead (~448
ALU per KV block) would become an even larger fraction of total time. The native
FP8 input path becomes MORE urgent.

## PTX ISA 9.2 Confirmation

PTX ISA 9.2 (CUDA 13.2) extends the MMA instruction to support `.f16` accumulator
type with FP8 operands:

```
mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16
```

This was NOT available in PTX ISA 9.1 (CUDA 13.0). The worker's current CUDA 13.0
installation may not have this instruction. **CUDA 13.2 upgrade may be required.**

## FP16 Accumulation: Precision Trade-off

FP16 has:
- 5-bit exponent (range ±65504)
- 10-bit mantissa (~3 decimal digits)

vs FP32:
- 8-bit exponent (range ±3.4e38)
- 23-bit mantissa (~7 decimal digits)

**For GEMM:** FP8 inputs have ~2 decimal digits of precision. Accumulating
in FP16 preserves the input precision. Accumulating in FP32 gives more
headroom but doesn't improve the OUTPUT precision. For training (SGD absorbs
noise), FP16 accumulation should be sufficient.

**For Attention:** The softmax computation uses FP32 for numerical stability
(log-sum-exp, exp2f). The QK^T and PV MMA results feed into FP32 softmax.
Using FP16 accumulators for MMA would require careful analysis of whether
the reduced accumulation precision causes softmax instability (underflow in
exp2f, or loss of relative order in attention weights).

**gau-nernst's results on RTX 5090:**
- FP8 e4m3 + FP32 accum: 465 TFLOPS
- FP8 e4m3 + FP16 accum: 692 TFLOPS
- Ratio: **1.49x** — confirms ~2x theoretical but real workloads have overhead

## Immediate Action Items

1. **GEMM worker:** Test `mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16`
   on sm_120 with CUDA 13.0. It MAY already work even if not documented in
   PTX ISA 9.1 (test empirically — sm_120 has surprised us before).

2. **GEMM worker:** If FP16 accum works, re-benchmark vs cuBLAS 13.2 FP8.
   The output will be FP16 — need to convert to BF16/FP32 for the return value.

3. **Attention worker:** Evaluate FP16 accumulators for QK^T MMA only (not
   softmax). The QK^T result feeds into FP32 softmax anyway, so FP16→FP32
   conversion after MMA is a single instruction. The PV MMA accumulates into
   the output, which needs FP32 for rescaling — FP16 accum may be acceptable
   if the final rescale is in FP32.

4. **All workers:** Consider upgrading from CUDA 13.0 to CUDA 13.2. The
   PTX ISA 9.2 features (FP16 accum, 256-bit loads, 3-input min/max) could
   benefit all kernels.

## Also New in PTX ISA 9.2

- **256-bit (32-byte) load/store:** `ld` and `st` extended to `.b256`.
  Wider memory operations could help bandwidth-bound kernels (RMSNorm, DOT,
  NRM2, AXPY, SCAL). Currently our widest loads are 128-bit (uint4/float4).

- **3-input min/max:** `min.f32 d, a, b, c` — useful for clamping without
  extra instructions.

## Caveats

1. **CUDA 13.0 may already support FP16 accum for FP8 MMA.** The hardware
   capability exists (gau-nernst uses it). The question is whether PTX ISA
   9.1 assembler accepts the instruction. Test before upgrading.

2. **cuBLAS 13.2 baseline shift:** If cuBLAS got 20% faster on sm_120,
   ALL our "vs cuBLAS" numbers need re-benchmarking. The 1.29x FP8 GEMM
   advantage may have shrunk to ~1.07x.

3. **FP16 accumulation overflow risk:** For large K dimensions (K > 4096),
   FP16 accumulators can overflow (max 65504). Need tile-level accumulation
   with periodic FP32 reduction for large problems.
