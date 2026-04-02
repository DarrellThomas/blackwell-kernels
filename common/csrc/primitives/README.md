# Best-Known Primitives — sm_120

Shipped kernel source files for use by numerical method projects and the
Octave `.so` library. These are the **best known versions** from their
source worktrees, with version tracking.

## Version Policy

Every shipped primitive has a version number and content hash:
- **Version** increments on every reship (v1, v2, v3...)
- **Hash** is the first 8 chars of SHA256 of the file content
- **If the hash doesn't match, the shelf is stale — reship immediately**

Run `verify-primitives.sh` to check all hashes match.

## Reship Policy

When a source worktree (gemm/, linalg/) improves a primitive:
1. The worktree's `test_edge_cases.py` MUST pass (all tests green)
2. Copy the file to the shelf
3. Update this manifest (bump version, update hash, vs_ref, date)
4. Copy to ALL consumers (numerical/, qr/, etc.)
5. Verify with `verify-primitives.sh`

**Do NOT ship primitives without full BLAS-compatible signatures**
(alpha, beta, lda, ldb, ldc). See `04_HARD_WON_LESSONS.md`.

## Current Inventory

### GEMM (gemm/ + linalg/ worktrees)

| File | Op | vs_ref | Version | Hash | Tests | Date |
|------|----|--------|---------|------|-------|------|
| `gemm/bf16_gemm_sm120.cu` | GEMM BF16 | 0.97x | v1 | — | basic | 2026-03-14 |
| `gemm/fp8_gemm_sm120.cu` | GEMM FP8 | 1.34x | v1 | — | basic | 2026-03-14 |
| `gemm/dgemm_sm120.cu` | DGEMM FP64 | 1.14x | v2 | — | 31/31 edge | 2026-03-28 |
| `linalg/gemm_f32_sm120.cu` | GEMM FP32 (BF16 MMA) | 1.58x | v2 | a460d3d6 | 136/136 | 2026-03-29 |
| `linalg/dgemm_sm120.cu` | DGEMM FP64 (BLAS iface) | 1.12x | v3 | 1de9009b | 136/136 | 2026-03-29 |

### SYRK

| File | Op | vs_ref | Version | Hash | Tests | Date |
|------|----|--------|---------|------|-------|------|
| `linalg/syrk_sm120.cu` | SYRK BF16 | 1.70x | v2 | 207e9367 | 136/136 | 2026-03-29 |
| `linalg/syrk_f32_sm120.cu` | SYRK FP32 (BF16 MMA, ensure_psd) | 1.76x | v3 | 926a0b73 | 136/136 | 2026-03-29 |
| `linalg/dsyrk_sm120.cu` | DSYRK FP64 (+ TRANS_A for A^T@A) | 2.12x | v3 | 4ea35d77 | 136/136 + 46 edge | 2026-03-29 |

### TRSM / TRMM

| File | Op | vs_ref | Version | Hash | Tests | Date |
|------|----|--------|---------|------|-------|------|
| `linalg/trsm_sm120.cu` | TRSM BF16 | 1.00x | v1 | 63ad0faa | 136/136 | 2026-03-29 |
| `linalg/trsm_f32_sm120.cu` | TRSM FP32 | 1.00x | v2 | 6202acba | 136/136 | 2026-03-29 |
| `linalg/trmm_sm120.cu` | TRMM BF16 | 1.75x | v2 | 0379db94 | 136/136 | 2026-03-29 |
| `linalg/dtrmm_sm120.cu` | DTRMM FP64 (+ UPPER variant) | 2.11x | v3 | 35d22649 | 136/136 + 46 edge | 2026-03-29 |

### GEMV

| File | Op | vs_ref | Version | Hash | Tests | Date |
|------|----|--------|---------|------|-------|------|
| `linalg/gemv_sm120.cu` | GEMV BF16 | 1.75x | v2 | d0d03bfa | 136/136 | 2026-03-29 |
| `linalg/dgemv_sm120.cu` | DGEMV FP64 | 1.22x | v2 | 44db3e85 | 136/136 | 2026-03-29 |
| `linalg/batched_gemv_sm120.cu` | Batched GEMV BF16 | 1.00x | v1 | 360bfeec | 136/136 | 2026-03-29 |

### BLAS Level 1

| File | Op | vs_ref | Version | Hash | Tests | Date |
|------|----|--------|---------|------|-------|------|
| `linalg/blas1_sm120.cu` | DOT 2.00x, NRM2 1.49x, AXPY 1.00x, SCAL 1.00x | all >= 1.0x | v2 | 89afd997 | 136/136 | 2026-03-29 |
| `linalg/dblas1_sm120.cu` | DDOT 1.67x, DNRM2 1.33x, DAXPY 1.0x, DSCAL 1.0x, DASUM 5.63x, IDAMAX 6.49x, DCOPY 1.50x, DSWAP 2.25x | all >= 1.0x | v2 | 1436600b | 136/136 | 2026-03-29 |

### BLAS Level 2 Extra (FP64)

| File | Op | vs_ref | Version | Hash | Tests | Date |
|------|----|--------|---------|------|-------|------|
| `linalg/dblas2_extra_sm120.cu` | DSYR 7.17x, DROT 17.28x, DROTG (CPU) | all beating | v1 | 1d1040e0 | 136/136 | 2026-03-29 |

### Batched

| File | Op | vs_ref | Version | Hash | Tests | Date |
|------|----|--------|---------|------|-------|------|
| `linalg/batched_gemm_sm120.cu` | Batched GEMM BF16 + bindings (dsyr/drot/drotg) | 1.00x | v3 | 3aa83378 | 136/136 | 2026-03-29 |
| `linalg/batched_gemm_fp8_sm120.cu` | Batched GEMM FP8 | 2.14x | v1 | — | basic | 2026-03-15 |
| `linalg/dbatched_gemm_sm120.cu` | Batched GEMM FP64 | 1.00x | v1 | — | basic | 2026-03-28 |

### Permutation

| File | Op | vs_ref | Version | Hash | Tests | Date |
|------|----|--------|---------|------|-------|------|
| `linalg/permute_sm120.cu` | permute_rows 2.57x, swap_rows 7.12x | all beating | v2 | — | 31/31 edge | 2026-03-28 |

### Factorizations (from numerical/, qr/ worktrees)

| File | Op | vs_ref | Version | Hash | Tests | Date |
|------|----|--------|---------|------|-------|------|
| `qr/qr_sm120.cu` | QR (CholQR2) | FP32 2.17x, FP64 >=1.0x | v1 | — | 10/10 | 2026-03-28 |

### SpMV (from spmv/ worktree)

| File | Op | vs_ref | Version | Hash | Tests | Date |
|------|----|--------|---------|------|-------|------|
| `spmv/spmv_sm120.cu` | SpMV CSR+ELL+sorted-ELL+adaptive+indirect | FP32 1.87x, FP64 1.53x | v1 | — | basic | 2026-03-28 |
