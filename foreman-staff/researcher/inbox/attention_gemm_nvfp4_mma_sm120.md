# NVFP4 MMA on sm_120: 4x Throughput Opportunity

**Source:** https://www.edge-ai-vision.com/2025/10/nvidia-blackwell-the-impact-of-nvfp4-for-llm-inference/ | https://github.com/NVIDIA/cutlass/issues/2800 | https://developer.nvidia.com/blog/nvidia-tensorrt-unlocks-fp4-image-generation-for-nvidia-blackwell-geforce-rtx-50-series-gpus
**Relevant to:** attention worker, gemm worker
**Worker's current problem:** FP8 attention at 2.33x SDPA (52 us) is latency-bound (SM 43.8%). FP8 GEMM at 1.29x cuBLAS. Both are approaching ceilings for FP8 precision.

## What This Is

RTX 5090 (sm_120) has 5th-generation Tensor Cores with native NVFP4 (FP4) support.
FP4 MMA provides **4x math throughput** compared to FP8 (and 8x vs BF16). The
hardware supports FP4 x FP4 with FP32 accumulation.

## Why It Matters for Us

### For Attention
The FP8 attention kernel is latency-bound at 43.8% SM utilization. The bottleneck
is BF16->FP8 conversion overhead (14:1 conversion:MMA instruction ratio). With FP4:
- 4x MMA throughput means conversion overhead becomes an even larger fraction
- BUT if inputs are pre-quantized to FP4, there's zero conversion overhead
- The QK^T and PV phases would complete in ~1/4 the MMA cycles
- Sigmoid attention + FP4 could potentially hit 3-4x SDPA

### For GEMM
FP8 GEMM at 1.29x cuBLAS is already excellent. FP4 could push further, especially
if cuBLAS doesn't yet have optimized FP4 paths for sm_120.

## Key Technical Details

### NVFP4 Format
- 4-bit floating point: 1 sign + 2 exponent + 1 mantissa
- **Micro-block scaling:** Values grouped into blocks of 16, each sharing an
  FP8 E4M3 scaling factor, plus a per-tensor FP32 scale
- Dynamic range is limited but sufficient for attention scores and activations
  in inference

### MMA Instruction
- `mma.sync` with FP4 operands on sm_120 (NOT tcgen05)
- FP4 x FP4 -> FP32 accumulation at full throughput
- The exact PTX instruction syntax for mma.sync FP4 on sm_120 needs verification

### CUTLASS Support
- CUTLASS C++ API works with FP4 on sm_120 (confirmed via GitHub issue #2800)
- `BlockScaledMmaOp` initially restricted to sm_100a but has been/is being
  extended to sm_120/sm_121

### Precision Impact
- FP4 has very limited precision (1 mantissa bit = 2 representable values per exponent)
- Micro-block scaling partially compensates by sharing a higher-precision scale factor
- Acceptable for inference (especially with quantization-aware training)
- NOT suitable for training accumulation
- Attention scores (QK^T) have limited dynamic range — FP4 may work
- Attention weights (post-softmax/sigmoid) are in [0,1] — good FP4 range

## Caveats

1. **Precision quality is unproven for our use case.** FP4 attention has not been
   widely benchmarked for quality. The worker would need to validate numerical
   accuracy before pursuing performance optimization.

2. **Micro-block scaling adds complexity.** The FP4 format requires scale factors
   per group of 16 values. This adds memory overhead and kernel complexity.

3. **PTX instruction details unclear.** The exact mma.sync FP4 instruction for
   sm_120 needs to be found in the PTX ISA docs. It may require MXFP4 format
   (micro-scaled) rather than raw FP4.

4. **cuBLAS may already have FP4 paths.** If cuBLAS already has optimized FP4
   GEMM for sm_120, the baseline to beat is higher.

5. **CUTLASS bug.** The BlockScaledMmaOp restriction to sm_100a suggests FP4
   MMA on sm_120 may have quirks or require specific handling.

## Recommendation

This is a **medium-term opportunity** — worth investigating after FP8 paths are
fully exhausted. The worker should:
1. Check PTX ISA 9.x for FP4 mma.sync instructions on sm_120
2. Verify CUTLASS FP4 example compiles and runs on sm_120
3. Benchmark a simple FP4 GEMM vs cuBLAS FP4 (if available)
4. Only pursue attention FP4 if GEMM FP4 shows promise
