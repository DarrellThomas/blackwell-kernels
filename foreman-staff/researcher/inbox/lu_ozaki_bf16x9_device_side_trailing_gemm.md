# Ozaki Scheme and BF16x9: Device-Side FP32-Accurate GEMM for Monolithic LU

**Source:** https://docs.nvidia.com/cuda/cublasdx/examples.html (cuBLASDx dgemm_emulation example)
**Source:** https://arxiv.org/abs/2511.13778 (Guaranteed DGEMM Accuracy via Ozaki Scheme, Nov 2025)
**Source:** https://arxiv.org/abs/2508.00441 (DGEMM with FP8 Tensor Cores via Ozaki, Aug 2025)
**Source:** https://developer.nvidia.com/blog/unlocking-tensor-core-performance-with-floating-point-emulation-in-cublas/
**Relevant to:** LU worker
**Worker's current problem:** Building a monolithic LU kernel requires a device-side trailing GEMM that is both fast (tensor cores) and accurate (FP32). Need to understand how to implement BF16x9 FP32 emulation inside a CUDA kernel without calling cuBLAS.

---

## What This Is

The Ozaki scheme is a general framework for emulating high-precision matrix
multiplication using multiple low-precision multiplications. cuBLAS uses a
specific instantiation -- BF16x9 -- for FP32 emulation. cuBLASDx provides a
device-side Ozaki example (`dgemm_emulation`) demonstrating the pattern.

This brief covers how to use these techniques INSIDE a monolithic kernel for
the LU trailing GEMM update.

---

## BF16x9 Decomposition for FP32 GEMM (Device-Side)

### Step 1: Split FP32 into 3 BF16 Components

Every FP32 value v can be exactly represented as three BF16 values:

```
v = v0 + v1 + v2

where:
  v0 = round_to_BF16(v)                    // top 8 mantissa bits
  r1 = v - float(v0)                       // residual
  v1 = round_to_BF16(r1)                   // next 8 mantissa bits
  r2 = r1 - float(v1)                      // residual
  v2 = round_to_BF16(r2)                   // remaining bits
```

BF16 has 7 explicit mantissa bits (8 with implicit leading 1). Three BF16 values
cover 3 * 8 = 24 bits, sufficient for FP32's 23-bit mantissa plus sign.

### Step 2: Compute 9 BF16 GEMMs

```
C = A * B
  = (A0 + A1 + A2) * (B0 + B1 + B2)
  = A0*B0                                    // dominant term
  + (A0*B1 + A1*B0)                          // first correction
  + (A0*B2 + A1*B1 + A2*B0)                  // second correction
  + (A1*B2 + A2*B1)                          // third correction
  + A2*B2                                     // negligible (may skip)
```

Each product Ai*Bj is a BF16 GEMM with FP32 accumulation using
`mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`.

### Step 3: Accumulate with Scaling

The terms are accumulated with implicit scaling (the BF16 rounding already
encoded the magnitude). Just sum all 9 (or 7-8 if skipping negligible terms)
FP32 accumulator results.

---

## Device-Side Implementation Pattern

### For the Monolithic LU Trailing GEMM

```cpp
// Inside the monolithic kernel's trailing GEMM phase:
// For each tile (i, j) of the trailing matrix:

__shared__ __align__(16) float A_tile_f32[TILE_M][TILE_K];  // from L column
__shared__ __align__(16) float B_tile_f32[TILE_K][TILE_N];  // from U row

// Split into BF16 components (can be done in registers)
__nv_bfloat16 A0[...], A1[...], A2[...];
__nv_bfloat16 B0[...], B1[...], B2[...];

split_fp32_to_bf16x3(A_tile_f32, A0, A1, A2);
split_fp32_to_bf16x3(B_tile_f32, B0, B1, B2);

// 9 MMA calls (or 7 if skipping A2*B2 and one cross-term)
float C_accum[...] = {0};  // FP32 accumulator

mma_bf16(A0, B0, C_accum);   // dominant
mma_bf16(A0, B1, C_accum);   // correction 1a
mma_bf16(A1, B0, C_accum);   // correction 1b
mma_bf16(A0, B2, C_accum);   // correction 2a
mma_bf16(A1, B1, C_accum);   // correction 2b
mma_bf16(A2, B0, C_accum);   // correction 2c
mma_bf16(A1, B2, C_accum);   // correction 3a
mma_bf16(A2, B1, C_accum);   // correction 3b
// mma_bf16(A2, B2, C_accum);  // negligible, can skip

// Write C_accum back to trailing matrix
trailing[i][j] -= C_accum;   // the LU update is C -= L*U
```

### Register Budget

Each BF16 component occupies half the registers of FP32:
- 3 BF16 components for A: 1.5x the A registers
- 3 BF16 components for B: 1.5x the B registers
- FP32 accumulator: same as normal GEMM
- Total: ~2.5x the register pressure of a single BF16 GEMM

For our typical 64x64 tile with 4 warps: feasible within the 255 reg/thread limit.

### Throughput Analysis

- 9 BF16 MMA calls vs 1 native FP32 computation
- BF16 MMA throughput on sm_120: ~330 TFLOPS
- Native FP32 scalar: ~20.6 TFLOPS (non-tensor)
- Effective FP32-equivalent throughput: 330/9 = ~36.7 TFLOPS
- Speedup vs scalar FP32: ~1.8x

**Wait -- this is lower than the 3-4x cuBLAS achieves.** The difference is that
cuBLAS uses Blackwell-specific hardware scaling features to reduce the overhead
of applying the 2^-8, 2^-16 scaling factors. On sm_120 (consumer Blackwell),
the hardware scaling support needs to be verified empirically.

---

## cuBLASDx dgemm_emulation Example (Reference)

cuBLASDx v0.5.1 provides `dgemm_emulation` demonstrating the Ozaki scheme for
FP64 emulation via INT8 GEMMs:

1. Decompose FP64 matrices into INT8 "slices"
2. Perform GEMM on each slice combination
3. Reconstruct the FP64 result

The same algorithmic pattern applies to FP32-via-BF16. The example provides
device-side reference code for:
- Matrix decomposition into lower-precision slices
- Loop over slice combinations
- Accumulation with appropriate scaling
- Result reconstruction

**The FP64-via-INT8 example uses more slices (7-11) than FP32-via-BF16 (3 slices,
9 GEMMs). Our case is simpler.**

---

## Ozaki Scheme Advances (2025)

### Guaranteed DGEMM Accuracy (Nov 2025)

The paper by Uchino et al. introduces unsigned slice encoding that reduces the
number of INT8 slices from 8 to 7 for FP64 accuracy (22% reduction). Key results:

- GB200: up to 2.3x speedup over native FP64
- RTX Pro 6000: up to 13.2x speedup (because native FP64 is very slow)
- QR factorization (end-to-end): up to 3.7x speedup

**Critical finding:** The authors successfully integrated Ozaki-scheme GEMM into
`cusolverDnGeqrf` (blocked Householder QR) for trailing matrix updates. The
approach integrates "transparently" with minimal code modifications. This proves
the Ozaki scheme works for factorization trailing updates, not just standalone GEMM.

### FP64 via FP8 Tensor Cores (Aug 2025, arXiv 2508.00441)

Mukunoki et al. used FP8 (E4M3) tensor cores with FP32 accumulation to emulate
FP64 DGEMM on GeForce RTX 5060 Ti. Key details:
- FP8 outperforms FP16 Ozaki at N >= 8192 (1.26x speedup at N=16384)
- FP8 tensor core throughput: 94.74 TFLOPS vs FP16: 47.37 TFLOPS
- **121 sub-GEMMs required** for FP64 accuracy regardless of K dimension
  (vs 36-81 for FP16, which scales with K)
- Inner-product-wise blocking in K dimension reduces memory footprint
- Requires 16-element alignment (matches our MMA alignment)

**Not relevant for our FP32 case** (we only need 9 BF16 GEMMs), but validates
the Ozaki approach on consumer Blackwell GPUs and shows FP8 tensor cores work
for precision emulation.

### GEMMul8: Open-Source Ozaki Implementation (RIKEN)

GitHub: https://github.com/RIKEN-RCCS/GEMMul8
Implements Ozaki Scheme II using INT8/FP8 matrix engines. GEMM-oriented
emulation using Chinese Remainder Theorem (integer modular technique). Could
serve as additional reference for the decomposition pattern.

---

## Practical Decision for LU Worker

### For v1 (Blocked LU with cuBLAS)

Use `CUBLAS_COMPUTE_32F_EMULATED_16BFX9` for the trailing GEMM call:
```cpp
cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
    &alpha, L_col, CUDA_R_32F, lda, U_row, CUDA_R_32F, ldb,
    &beta, trailing, CUDA_R_32F, ldc,
    CUBLAS_COMPUTE_32F_EMULATED_16BFX9, CUBLAS_GEMM_DEFAULT);
```

Or set the cuSOLVER math mode (see companion brief on cuSOLVER 13.2 BF16x9).

### For v3 (Monolithic Kernel)

Two options:

**Option A: cuBLASDx device-side BF16 GEMM (simpler)**
Use cuBLASDx's BF16 GEMM with FP32 accumulation for each of the 9 sub-products.
Compose with cuSOLVERDx getrf for the panel.

**Option B: Custom BF16 MMA (full control)**
Use our proven BF16 MMA infrastructure (0.97x cuBLAS) to implement the 9-GEMM
decomposition manually. More work but avoids MathDx dependency.

**Recommendation:** Start with Option A (cuBLASDx) for faster development. If
performance is insufficient, switch to Option B with custom MMA.

---

## Sources

- [cuBLASDx dgemm_emulation Example](https://docs.nvidia.com/cuda/cublasdx/examples.html)
- [Guaranteed DGEMM Accuracy via Ozaki (Nov 2025)](https://arxiv.org/abs/2511.13778)
- [DGEMM with FP8 via Ozaki (Aug 2025)](https://arxiv.org/abs/2508.00441)
- [cuBLAS FP Emulation Blog](https://developer.nvidia.com/blog/unlocking-tensor-core-performance-with-floating-point-emulation-in-cublas/)
- [GEMMul8: INT8/FP8 Ozaki Implementation](https://github.com/RIKEN-RCCS/GEMMul8)
