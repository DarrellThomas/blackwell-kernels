# Test Report: Full Sweep — 2026-03-28

## Summary
- **Projects tested:** linalg, gemm, numerical
- **Total tests run:** 57
- **Passed:** 52
- **Failed:** 5
- **New issues filed:** 4

## Primitives Shelf Verification

Run: `/data/src/bwk/common/scripts/verify-primitives.sh`

| File | Status | Details |
|------|--------|---------|
| bf16_gemm_sm120.cu | SYNCED | a9bf4d8b |
| dgemm_sm120.cu | SYNCED | e4413a27 |
| fp8_gemm_sm120.cu | SYNCED | 5e9439d4 |
| batched_gemm_fp8_sm120.cu | SYNCED | 41657cef |
| **batched_gemm_sm120.cu** | **STALE** | shelf=aae0a911 worktree=f982ac1e (linalg) |
| batched_gemv_sm120.cu | SYNCED | 360bfeec |
| blas1_sm120.cu | SYNCED | a3398098 |
| dbatched_gemm_sm120.cu | SYNCED | 56bc6629 |
| dblas1_sm120.cu | SYNCED | 276347f0 |
| dgemv_sm120.cu | SYNCED | 578db033 |
| dsyrk_sm120.cu | SYNCED | 53dc106f |
| dtrmm_sm120.cu | SYNCED | e2c3d795 |
| **gemm_f32_sm120.cu** | **STALE** | shelf=0266d7c8 worktree=a460d3d6 (linalg) |
| gemv_sm120.cu | SYNCED | d7d3bb8a |
| permute_sm120.cu | SYNCED | 4a52bc6e |
| **syrk_f32_sm120.cu** | **STALE** | shelf=009e7bb3 worktree=926a0b73 (linalg) |
| syrk_sm120.cu | SYNCED | 84c23674 |
| **trmm_f32_sm120.cu** | **ORPHAN** | hash=6d48c69c (no worktree source) |
| trmm_sm120.cu | SYNCED | 23a52b8d |
| trsm_f32_sm120.cu | SYNCED | 6202acba |
| trsm_sm120.cu | SYNCED | 63ad0faa |

**Totals:** 18 synced, 3 stale, 1 orphan

## Test Results: linalg

**31 tests — 30 passed, 1 failed**

| Test | Status | Details |
|------|--------|---------|
| test_gemm_1x1 | PASS | |
| test_gemm_1xN | PASS | |
| test_gemm_Nx1 | PASS | |
| test_gemm_non_tile_multiple | PASS | |
| test_gemm_identity | PASS | |
| test_gemm_zeros | PASS | |
| test_gemm_large_values | PASS | |
| test_gemm_tiny_values | PASS | |
| test_syrk_1x1 | PASS | |
| test_syrk_tall_thin | PASS | |
| test_syrk_short_wide | PASS | |
| test_syrk_symmetry | PASS | |
| **test_syrk_positive_semidefinite** | **FAIL** | Negative eigenvalue: -0.000117 |
| test_trsm_identity | PASS | |
| test_trsm_diagonal | PASS | |
| test_trsm_verify_solution | PASS | |
| test_dot_orthogonal | PASS | |
| test_dot_self | PASS | |
| test_axpy_zero_alpha | PASS | |
| test_scal_zero | PASS | |
| test_scal_one | PASS | |
| test_nrm2_unit_vector | PASS | |
| test_nrm2_zeros | PASS | |
| test_dgemm_precision | PASS | |
| test_dsyrk_precision | PASS | |
| test_dgemm_ill_conditioned | PASS | |
| test_ddot_precision | PASS | |
| test_syrk_submatrix | PASS | |
| test_gemm_non_contiguous | PASS | |
| test_syrk_inplace_update | PASS | |
| test_gemm_alpha_beta | PASS | |

## Test Results: gemm

**13 tests — 10 passed, 3 failed**

| Test | Status | Details |
|------|--------|---------|
| test_dgemm_1x1 | PASS | |
| test_dgemm_identity | PASS | |
| test_dgemm_zeros | PASS | |
| test_dgemm_rectangular | PASS | |
| test_dgemm_non_tile_aligned | PASS | |
| test_dgemm_large_values | PASS | |
| test_dgemm_tiny_values | PASS | |
| test_dgemm_ill_conditioned | PASS | |
| test_dgemm_symmetric_result | PASS | |
| test_dgemm_associativity | PASS | |
| **test_bf16_gemm_zeros** | **FAIL** | `_C` has no attribute `gemm` |
| **test_bf16_gemm_identity** | **FAIL** | `_C` has no attribute `gemm` |
| **test_bf16_gemm_large** | **FAIL** | `_C` has no attribute `gemm` |

## Test Results: numerical

**13 tests — 12 passed, 1 failed**

| Test | Status | Details |
|------|--------|---------|
| test_chol_identity | PASS | |
| test_chol_diagonal | PASS | |
| test_chol_reconstruction | PASS | |
| test_chol_lower_triangular | PASS | |
| test_chol_positive_diagonal | PASS | |
| test_chol_1x1 | PASS | |
| test_chol_2x2 | PASS | |
| test_chol_non_tile_multiple | PASS | |
| test_chol_small_nb | PASS | |
| test_chol_well_conditioned | PASS | |
| **test_chol_ill_conditioned** | **FAIL** | Reconstruction error: NaN |
| test_chol_large_values | PASS | |
| test_chol_vs_torch | PASS | |

## Issues Filed

- `issues/linalg_syrk_f32_psd_violation.md` — SYRK F32 produces non-PSD output (negative eigenvalue)
- `issues/gemm_bf16_binding_missing.md` — BF16 GEMM not exposed in Python bindings
- `issues/numerical_cholesky_ill_conditioned_nan.md` — Cholesky produces NaN on ill-conditioned input
- `issues/shelf_3_stale_1_orphan.md` — 3 stale primitives + 1 orphan need reship
