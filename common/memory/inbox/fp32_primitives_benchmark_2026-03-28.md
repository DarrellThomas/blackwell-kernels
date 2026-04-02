# FP32 SYRK and TRMM Custom MMA Kernels — Benchmark Results (2026-03-28)

## Summary

New FP32-input primitives benchmarked on RTX 5090 (sm_120a) at N=4096.
Both use BF16 MMA (m16n8k16) with FP32 accumulators, FP32 input/output.
Triangle-aware tiling skips zero blocks for ~2x less compute than full GEMM.

## SYRK FP32 (syrk_f32_sm120.cu) — 1.14x cuBLAS SSYRK

| Size | torch.mm(A,A.t()) | cuBLAS SSYRK | Custom MMA | vs cuBLAS |
|------|-------------------|--------------|------------|-----------|
| 1024 | 2306 us | 2317 us | 2300 us | 1.01x |
| 2048 | 2586 us | 2525 us | 2462 us | 1.03x |
| 4096 | 6845 us | 3873 us | 3398 us | **1.14x** |

Architecture: 64x64 tiles, BLOCK_K=32, 4 warps, 6 blocks/SM.
Lower-triangle block skip (`if (bn > bm) return`).
FP32→BF16 conversion in shared memory, cp.async not usable (register-mediated).
Max relative error vs FP32 reference: 3.6e-4 (acceptable for CholQR2).

## TRMM FP32 (trmm_f32_sm120.cu) — 1.25x cuBLAS STRMM

| Size | torch.mm(L,B) | cuBLAS STRMM | Custom MMA | vs cuBLAS |
|------|---------------|--------------|------------|-----------|
| 1024 | 2298 us | 2284 us | 2267 us | 1.01x |
| 2048 | 2510 us | 2503 us | 2354 us | 1.06x |
| 4096 | 4187 us | 3799 us | 3030 us | **1.25x** |

Architecture: same as SYRK but with triangle-aware K-loop.
For lower-triangular L, row block bm only iterates K-tiles 0..(bm+1)*2.
For upper-triangular, K-tiles bm*2..end. Both variants tested and working.
Max relative error vs FP32 reference: 2.4e-3 (BF16 MMA precision).

## Key Finding

cuBLAS STRMM on sm_120a does NOT fully exploit triangle structure at large sizes.
Our kernel's per-row-block K-loop truncation gives 25% speedup at N=4096.
The gain increases with matrix size (more blocks to skip).

cuBLAS SSYRK is better optimized (it does use triangle skip internally) but
our BF16 MMA path still beats it by 14% — likely due to higher MMA throughput
vs cuBLAS's FP32 or TF32 path on this consumer GPU.

## TRSM: cuBLAS is optimal, no custom kernel needed

TRSM is inherently sequential along the diagonal. At NRHS=64 (QR use case),
it's bandwidth-bound (65 FLOP/byte vs 117 roofline). cuBLAS STRSM is already
heavily optimized with multi-level blocking. No custom kernel warranted.

## Applicability

- **QR (CholQR2):** Replace cublasGemmEx SYRK → custom SYRK FP32 (1.14x).
  Replace cublasStrmm for R2@R1 → custom TRMM FP32 (1.25x). Keep cublasStrsm.
- **Cholesky:** Blocked v3 can use SYRK FP32 for trailing update.
  Monolithic path should inline the MMA pipeline directly (not call as separate kernel).
