# Cholesky produces NaN on ill-conditioned input

## What's broken

`bk.cholesky(M, 64)` returns NaN values when given an ill-conditioned SPD matrix
(condition number ~1e6). The reconstruction error is NaN, meaning the factorization
broke down entirely.

## How to reproduce

```python
import torch
from blackwell_kernels import _C as bk

N = 256
U, _, _ = torch.linalg.svd(torch.randn(N, N, dtype=torch.float32, device="cuda"))
s = torch.logspace(0, -6, N, dtype=torch.float32, device="cuda")
M = U @ torch.diag(s) @ U.T
L = bk.cholesky(M, 64)
recon_err = (L @ L.T - M).abs().max().item() / M.abs().max().item()
print(recon_err)  # nan
```

Run from: `cd /data/src/bwk/numerical && CUDA_VISIBLE_DEVICES=1 PYTHONPATH=python python3 tests/test_edge_cases.py`

## Expected vs actual

- **Expected:** Reconstruction error < 0.1 (test tolerance for ill-conditioned case)
- **Actual:** Reconstruction error = NaN

## Severity

**Wrong answer** — NaN propagation means the factorization is completely unusable
for ill-conditioned inputs. Well-conditioned inputs (cond ~1-100) work fine.

## Which consumers are affected

- **numerical/** — The Cholesky factorization itself
- **Any solver** that chains Cholesky → TRSM (Ax=b via Cholesky)
- **octave .so** — LAPACK users calling SPOTRF on ill-conditioned matrices

## Analysis

Condition number 1e6 is aggressive for FP32 (~7 decimal digits), but
`torch.linalg.cholesky` handles this fine. The issue is likely in the blocked
algorithm: when a diagonal block has very small pivots, the TRSM step
(L21 = A21 @ L11^{-1}) amplifies error, and the SYRK update
(A22 -= L21 @ L21^T) can produce a non-PSD remainder — which then
causes sqrt(negative) = NaN in the next diagonal block.

Possible fixes:
1. Add pivot checking: if diagonal element < epsilon, clamp to epsilon
2. Use FP64 accumulation for diagonal block factorization
3. Document the condition number limit for FP32 Cholesky
