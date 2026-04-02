# Open-Source QR GPU Implementations: 2024-2025 Survey

**Sources:**
- MixedPrecisionBlockQR (https://github.com/jaidonlybbert/MixedPrecisionBlockQR)
- CholeskyQR2-IM (https://github.com/HybridScale/CholeskyQR2-IM)
- MAGMA (https://github.com/CEED/MAGMA)
- RandLAPACK (https://github.com/BallisticLA/RandLAPACK)
- enp1s0/tsqr-gpu (https://github.com/enp1s0/tsqr-gpu)
**Relevant to:** QR worker
**Worker's current problem:** Looking for reference implementations and code to study.

## 1. MixedPrecisionBlockQR (jaidonlybbert)

**URL:** https://github.com/jaidonlybbert/MixedPrecisionBlockQR
**Language:** C++ (96.5%) + CUDA (2%)
**Algorithm:** Blocked Householder QR with FP16 matrix-matrix multiplications
**Requirements:** NVIDIA GPU compute capability >= 7.5, CUDA 12.1+

### What it does:
- Block QR decomposition using FP16 for GEMM (trailing update) and FP32 for panel
- Targets "large, wide matrices" (~2000 x 2000)
- Explicitly notes: "Other QR algorithms are better suited for small or tall-and-skinny matrices"

### Relevance for us:
- Shows how to integrate FP16 GEMM into blocked Householder QR
- The mixed-precision pattern (FP16 trailing GEMM + FP32 panel) is exactly what we'll do with BF16
- Authors acknowledge "many things could be done more efficiently, especially with HtoD memory traffic"
- **Study their LARFB implementation** for how they handle the FP16/FP32 boundary

### Limitations:
- Not optimized for maximum performance (research code)
- No tensor core MMA usage (uses cuBLAS FP16 GEMM)
- Targets older CUDA (12.1), not CUDA 13

## 2. MAGMA (CEED/MAGMA)

**URL:** https://github.com/CEED/MAGMA
**Key QR files:**
```
src/sgeqrf_gpu.cpp              -- Standard hybrid CPU-GPU QR
src/sgeqrf2_gpu.cpp             -- LAPACK-compliant variant
src/sgeqrf3_gpu.cpp             -- Variant with pre-computed T matrices
src/sgeqr2x_gpu.cpp (v1-v3)    -- Optimized GPU panel variants
magmablas/sgeqr2_batched_fused_sm.cu  -- Fused panel kernel (register-tiled)
magmablas/slarft_batched_fused_sm.cu  -- Fused LARFT kernel
magmablas/slarfb_gpu_gemm.cpp         -- LARFB via GEMM
src/sgeqrf_batched.cpp          -- Batched QR
```

### What to study:
1. **sgeqr2_batched_fused_sm.cu**: The register-tiled panel kernel. Thread-per-row design, shared memory reductions, kernel fusion. This is the reference implementation for a custom panel kernel.
2. **slarft_batched_fused_sm.cu**: GPU LARFT kernel. How T is built on-device.
3. **slarfb_gpu_gemm.cpp**: How LARFB decomposes into GEMM calls.
4. **Block size tuning**: `magma_get_sgeqrf_nb(m, n)` -- returns optimal nb for given matrix size.

### Three strategies used:
1. Fully fused (sizes <= 32): Single kernel, everything in registers
2. Panel+Update fusion (medium): Fused panel, separate trailing update
3. LAPACK-style (large): Separate kernels for panel, LARFT, LARFB(GEMM)

### Relevance for us:
- MAGMA is the gold standard for GPU QR. Study their code before writing our own.
- The fused panel kernel design directly applies to our custom GEQR2.
- The three-strategy approach (fused / panel+update / LAPACK-style) is the right structure.
- **Note**: MAGMA uses hybrid CPU-GPU by default (panel on CPU). Their GPU-native variant eliminates this but may be less optimized.

## 3. RandLAPACK (BallisticLA)

**URL:** https://github.com/BallisticLA/RandLAPACK
**Algorithm:** BQRRP (randomized column-pivoted QR)
**GPU support:** Yes (CUDA)

### Relevance:
- Only relevant if we need column-pivoted QR (QRCP)
- 65% of cuSOLVER geqrf throughput on H100 for pivoted QR
- Open-source reference for randomized QR techniques

## 4. CholeskyQR2-IM (HybridScale)

**URL:** https://github.com/HybridScale/CholeskyQR2-IM
**Algorithm:** CholeskyQR2 with Gram-Schmidt stabilization
**GPU support:** CUDA + ROCm, distributed memory

### Relevance:
- For tall-skinny sub-problems in CAQR panel factorization
- All BLAS-3 operations (GEMM + Cholesky + TRSM)
- Good reference for CholeskyQR integration into a larger QR framework

## 5. tsqr-gpu (enp1s0)

**URL:** https://github.com/enp1s0/tsqr-gpu
**Algorithm:** TSQR with tensor cores (FP16/TF32)
**GPU support:** Volta+ (compute >= 7.0)

### Relevance:
- Already covered in existing docs (qr_householder_vs_tsqr_gpu.md)
- Targets tall-skinny only, not useful for square matrices
- Archived (2021), no recent updates

## Recommendation for the QR Worker

### Study order:
1. **MAGMA sgeqr2_batched_fused_sm.cu** -- understand the register-tiled panel design
2. **MAGMA slarfb_gpu_gemm.cpp** -- understand how LARFB maps to GEMM
3. **MixedPrecisionBlockQR** -- understand the FP16/FP32 boundary in blocked QR
4. **cuSOLVERDx blocked Cholesky example** -- understand the composition pattern

### What NOT to spend time on:
- tsqr-gpu: already evaluated, not applicable for square matrices
- RandLAPACK: only for pivoted QR, not our primary target
- Distributed implementations: we're single-GPU
