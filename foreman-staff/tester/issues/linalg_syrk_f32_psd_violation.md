# SYRK F32 produces non-positive-semidefinite output

## What's broken

`bk.syrk_f32(A)` computes C = A @ A^T but the result has a negative eigenvalue
(-0.000117), violating the positive-semidefinite property that A @ A^T must satisfy.

## How to reproduce

```python
import torch
from blackwell_kernels import _C as bk

N, K = 256, 128
A = torch.randn(N, K, dtype=torch.float32, device="cuda")
C = bk.syrk_f32(A)
eigenvalues = torch.linalg.eigvalsh(C)
print(eigenvalues.min().item())  # -0.000117 (should be >= 0)
```

Run from: `cd /data/src/bwk/linalg && CUDA_VISIBLE_DEVICES=1 PYTHONPATH=python python3 tests/test_edge_cases.py`

## Expected vs actual

- **Expected:** All eigenvalues >= 0 (within tolerance -1e-4)
- **Actual:** Minimum eigenvalue = -0.00011747, which is just beyond the -1e-4 tolerance

## Severity

**Precision loss** — The magnitude is small but breaks the mathematical guarantee
that downstream consumers (Cholesky, numerical/) depend on. If the Cholesky
consumer calls `syrk_f32` to form the Gram matrix before factoring, a slightly
non-PSD result can cause NaN in the factorization.

## Which consumers are affected

- **numerical/** — Cholesky factorization relies on SYRK producing PSD output
- **qr/** — Gram matrix formation
- **octave .so** — Any BLAS user calling SSYRK expects PSD output

## Analysis

The error is likely accumulation-order dependent. FP32 SYRK using MMA (which
accumulates in FP32 but loads via BF16/TF32 conversion) can lose enough
precision to flip a near-zero eigenvalue negative. Possible fixes:
1. Symmetrize output: C = (C + C^T) / 2 (doesn't fix PSD but helps)
2. Use higher-precision accumulation for the diagonal
3. Accept the tolerance and loosen the test to -5e-4
