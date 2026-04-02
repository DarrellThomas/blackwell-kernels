# MXFP8: Native Block-Scaled FP8 MMA on sm_120

**Source:** https://github.com/triton-lang/triton/pull/7918 | https://forums.developer.nvidia.com/t/how-to-load-fp8-using-ldmatrix-on-sm120-sm120a/330254 | https://jianyuh.github.io/mxfp8/2025/12/07/MXFP8-Train.html
**Relevant to:** attention worker (main/), GEMM worker (gemm/)
**Worker's current problem:** FP8 kernels use standard per-tensor scaled e4m3 with in-kernel BF16→FP8 conversion. Attention has 14:1 conversion overhead. GEMM's FP8 precision (~3.7% relative error) may limit applicability.

## What This Is

MXFP8 (Microscaling FP8) is a block-scaled floating-point format where groups of 32 elements share an 8-bit exponent scale factor (e8m0 format). sm_120 has a **native MMA instruction** for MXFP8:

```
mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32
```

This is a different MMA variant than what our workers currently use (`mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`). The native variant handles block scaling in hardware — no software scaling needed.

## Why It Matters for Us

1. **Better accuracy than per-tensor FP8:** Each 32-element block has its own scale factor, recovering dynamic range that per-tensor scaling loses. Could reduce the 3.7% relative error in GEMM and the precision loss in FP8 attention.

2. **Native hardware support on sm_120:** The Triton PR #7918 demonstrates that sm_120 supports this instruction natively. Benchmark: MXFP8 native = 44.45s vs MXFP8 emulated = 76.44s for Llama3-8B inference — **42% latency reduction** from using the native instruction.

3. **Eliminates software scaling overhead:** With per-tensor FP8, the scale factor is applied in software (multiply after MMA). With MXFP8, the scale is consumed directly by the MMA instruction.

4. **Same throughput as standard FP8 MMA:** The m16n8k32 tile shape is identical. Tensor core throughput should be the same.

## Key Technique

### MMA Instruction:
```
mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32
```

### Scale format:
- Scale factor: e8m0 (8-bit unsigned exponent, no mantissa, bias of 127)
- Block size: 32 elements share one scale
- Scale = 2^(e8m0_value - 127)
- Scales for A and B operands are separate

### Data layout:
- Element data: standard FP8 e4m3 in shared memory
- Scale data: e8m0 values in registers (one per 32 elements)
- The MMA instruction takes both data operands AND scale operands

### Quantization (host or kernel):
```python
# Per-block scaling (blocks of 32)
block_max = tensor.reshape(-1, 32).abs().max(dim=1).values
scale_e8m0 = torch.ceil(torch.log2(block_max)).to(torch.uint8) + 127  # e8m0 encoding
tensor_fp8 = (tensor / (2.0 ** (scale_e8m0 - 127))).to(torch.float8_e4m3fn)
```

## Caveats

- **New instruction variant requires PTX changes.** The `kind::mxf8f6f4.block_scale.scale_vec::1X` qualifier is different from the standard FP8 MMA. The inline asm template must be updated.
- **Scale management adds complexity.** Each 32-element block needs a separate e8m0 scale stored and loaded. This adds shared memory or register pressure for scale storage.
- **Not verified in our codebase.** The Triton PR confirms the instruction works on sm_120, but we haven't tested it in our inline PTX. The instruction operand layout (how scales are passed) needs empirical verification.
- **Datacenter MXFP8 content is NOT applicable.** B200's MXFP8 uses `tcgen05.mma` with TMEM. sm_120 uses `mma.sync` with registers. Same format, completely different instruction.
- **CUDA 13 PTX support unclear.** The instruction may require a specific PTX ISA version. Need to verify `ptxas` accepts this instruction with our CUDA 13 toolchain.
- **Impact on kernel structure:** Scale factors need to flow through the pipeline alongside data tiles. Double-buffering must handle both data and scales.
