# sm_120a Compilation Flag + CUTLASS Example 79 GeForce Blackwell MXFP8

**Sources:**
- https://github.com/NVIDIA/cutlass/blob/main/examples/79_blackwell_geforce_gemm/79c_blackwell_geforce_mixed_mxfp8_mxfp6_bf16_gemm.cu
- https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html
- https://forums.developer.nvidia.com/t/run-ptx-mma-sync-aligned-kind-mxf8f6f4-block-scale-scale-vec-1x-m16n8k32-on-sm-120a/329702
- https://github.com/ggml-org/llama.cpp/issues/19662
- https://docs.nvidia.com/cuda/parallel-thread-execution/contents.html (PTX ISA 9.2)

**Relevant to:** ALL workers (compilation flag), attention + GEMM workers (MXFP8)
**Worker's current problem:** All workers compile with `arch=compute_120,code=sm_120`, which excludes sm_120a-specific accelerated features. CUTLASS has new GeForce Blackwell examples targeting sm_120 specifically.

---

## FINDING 1: sm_120 vs sm_120a — What Workers Are Missing

All worker `setup.py` files use:
```python
"-gencode", "arch=compute_120,code=sm_120",
```

The RTX 5090 IS sm_120a hardware. The "a" suffix enables additional accelerated features:

| Feature | sm_120 | sm_120a |
|---------|--------|---------|
| `mma.sync.m16n8k16` BF16 | YES | YES |
| `mma.sync.m16n8k32` FP8 e4m3 | YES | YES |
| `mma.sync.block_scale` MXFP8 | **NO** | YES |
| `ldmatrix.m8n8.b16` (standard) | YES | YES |
| `ldmatrix.m16n16.b8` (native 8-bit) | **NO** | YES |
| `mma.sync FP8 with .f16 accumulator` | **NO** | YES |
| `cvt.rn.satfinite.e4m3x2.f32` | YES | YES |

**Impact assessment:**
- **Current kernels work fine** — all existing mma.sync and ldmatrix instructions are available on sm_120
- **FP8 native inputs Path B** (ldmatrix.m16n16.b8) requires sm_120a — affects attention worker only
- **MXFP8 block-scaled MMA** requires sm_120a — potential new optimization path for GEMM + attention
- **FP16 accumulator for FP8 MMA** requires sm_120a — could reduce register pressure

**To enable sm_120a:** Change setup.py gencode to:
```python
"-gencode", "arch=compute_120a,code=sm_120a",
```

**Caveat:** sm_120a code will NOT run on non-accelerated sm_120 hardware. Since all workers target RTX 5090 specifically, this is fine. But the foreman should decide whether to change the template.

---

## FINDING 2: CUTLASS Example 79 — Official GeForce Blackwell FP8 GEMM

CUTLASS 4.x now includes `examples/79_blackwell_geforce_gemm/` with three examples targeting sm_120 specifically:

| Example | Description |
|---------|-------------|
| 79a | BF16 GEMM on GeForce Blackwell |
| 79b | Mixed MXFP8/BF16 GEMM |
| 79c | Mixed MXFP8/MXFP6/BF16 GEMM |

Key details from the examples:
- Uses `mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32`
- Block scaling with ue8m0 (8-bit unsigned exponent) scale factors per vector of 32 elements
- Claims **2x throughput vs Ada (sm_89) FP8 MMA** — this is the same 2x from m16n8k32 vs m16n8k16, not an additional 2x beyond our current FP8 MMA
- The CUTLASS collective builder interface handles GeForce Blackwell tile scheduling

**What this means for our workers:**
- The standard `mma.sync.m16n8k32.f32.e4m3.e4m3.f32` our workers already use has the SAME tensor core throughput
- The block_scale variant adds **per-block precision scaling** without changing throughput
- Block scaling is useful when input data has varying dynamic range — the e8m0 scale adjusts per 32-element vector
- For our current use case (BF16 inputs converted to FP8 inline), block scaling adds overhead without benefit — the conversion already handles precision
- For **true native FP8 inputs** (data arrives as FP8 from outside), block scaling preserves precision better than flat FP8

---

## FINDING 3: MXFP8 Block-Scaled MMA Operand Layout

From the NVIDIA forum post, the register layout for the block-scaled MMA:

```
mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0
```

| Operand | Registers | Type | Count |
|---------|-----------|------|-------|
| D (output) | d0-d3 | float | 4 |
| A (matrix) | a0-a3 | uint32_t | 4 |
| B (matrix) | b0-b1 | uint32_t | 2 |
| C (accumulator) | c0-c3 | float | 4 |
| Scale A | sfa0 | uint8_t | 1 |
| Scale B | sfb0 | uint8_t | 1 |

**The A/B/C/D register counts are IDENTICAL to the standard FP8 MMA.** The only addition is the two uint8_t scale factor registers. The scale factors can be stored in regular registers (not tensor memory) on sm_120a.

Scale factor format: ue8m0 = unsigned 8-bit exponent with no mantissa. Each scale factor applies to a vector of 32 consecutive FP8 elements (scale_vec::1X = 1 vector of 32).

---

## RECOMMENDATION

**For now: no action needed.** The existing sm_120 compilation and standard FP8 MMA are optimal for current workloads. Consider sm_120a only when:

1. The attention worker pursues FP8 native inputs Path B (ldmatrix.m16n16.b8)
2. A worker wants to experiment with block-scaled MXFP8 for precision-sensitive computations
3. FP16 accumulator for FP8 MMA becomes desirable (reduces register pressure at the cost of precision)

The CUTLASS example 79 is worth studying as a reference implementation, but our custom kernels already achieve 1.29x cuBLAS (GEMM) and 2.33x SDPA (attention) with the standard FP8 path.
