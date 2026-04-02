# cuBLAS API Reference — GEMM Operations & Data Types

Source: https://docs.nvidia.com/cuda/cublas/
Fetched: 2026-03-27

## GEMM Functions

### cublasGemmEx() — Extended Precision GEMM
Mixed-precision matrix multiply with flexible input/output/compute types.
Supports automatic type conversions. Returns results in specified output type.

### cublasLtMatmul() — Lightweight GEMM
Programmable alternative with:
- Flexible matrix data layouts beyond column-major
- Parameterizable algorithmic implementations
- Heuristic-based algorithm selection with caching
- User-controlled workspace allocation

Design pattern: create a plan once, reuse for identical configurations.

### gemmStridedBatched()
Efficient batch processing with contiguous data and fixed strides. Preferred
when matrices are pre-allocated in contiguous GPU memory.

### gemmGroupedBatched() (cuBLASLt)
Heterogeneous batch groups with differing dimensions within a single launch.

## Supported Data Types

| Type | Code | Notes |
|------|------|-------|
| FP16 | `CUDA_R_16F` | Half-precision |
| BF16 | `CUDA_R_16BF` | Wider dynamic range |
| TF32 | (implicit) | 19-bit mantissa in 32-bit, tensor core acceleration |
| FP32 | `CUDA_R_32F` | Standard single precision |
| FP64 | `CUDA_R_64F` | Double precision (native or emulated) |
| FP8 E4M3 | `CUDA_R_8F_E4M3` | Extreme quantization for LLM inference |
| FP8 E5M2 | `CUDA_R_8F_E5M2` | Wider range FP8 |
| INT8 | `CUDA_R_8I` / `CUDA_R_8U` | Integer quantization |

## Compute Types

```
CUBLAS_COMPUTE_32F              — Standard FP32
CUBLAS_COMPUTE_32F_FAST_16F     — Tensor core with FP16 compute
CUBLAS_COMPUTE_32F_FAST_TF32    — TF32 tensor core acceleration
CUBLAS_COMPUTE_64F              — Double precision
CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT — Fixed-point FP64 emulation
```

## FP64 Emulation via BF16x9

Decomposes FP32 values into three BF16 components for faithful reconstruction.
Beneficial when peak BF16 throughput exceeds FP32 by 9x margin. Uses Ozaki
Scheme with shared power-of-two scaling.

**Workspace requirements scale quadratically with mantissa bits.**
`sliceCount = ceildiv(mantissaBitCount + 1, 8)`

## Algorithm Selection

On Ampere and newer: algorithm selection is automatic. No manual control needed.
`CUBLAS_GEMM_DEFAULT` lets the library choose. Heuristics are cached.

## Tensor Core Alignment Requirements

Maximum throughput requires 16-byte alignment:
- Matrix dimensions x element size → 16-byte aligned
- Leading dimensions x element size → 16-byte aligned
- All data pointers → 16-byte aligned

**FP8 strictly enforces alignment.** Other types relaxed since cuBLAS 11.0.

## Workspace Management

- Default: 4-32 MiB auto-allocated (GPU-dependent)
- User workspace: set via `cublasSetWorkspace()`, 256-byte alignment minimum
- Recommended: >= 4 MiB baseline, 32 MiB for Hopper/Blackwell
- Essential for reproducibility with multiple CUDA streams
- Fixed-point emulation requires 100+ MiB

## Math Modes

| Mode | Purpose |
|------|---------|
| `CUBLAS_DEFAULT_MATH` | Production: uses >= requested precision + Tensor Cores |
| `CUBLAS_PEDANTIC_MATH` | Testing: standardized arithmetic, no optimizations |
| `CUBLAS_TF32_TENSOR_OP_MATH` | Single-precision via TF32 |
| `CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION` | Forces accumulation in compute type |

## Environment Variables

- `CUBLAS_WORKSPACE_CONFIG=:4096:8` — for reproducibility
- `NVIDIA_TF32_OVERRIDE=0` — disable TF32 globally
- `CUBLAS_EMULATE_DOUBLE_PRECISION=1` — enable FP64 emulation

## Performance Tips

1. Ensure 16-byte alignment for Tensor Core kernels
2. Provide sufficient workspace (especially for fixed-point)
3. Use strided batched for contiguous data, grouped for heterogeneous
4. Limit concurrent streams to <= 32
5. cuBLAS auto-detects and uses Tensor Cores unless explicitly disabled
