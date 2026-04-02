# BF16 GEMM not exposed in gemm/ Python bindings

## What's broken

The gemm/ project's `blackwell_kernels._C` module has no `gemm` attribute.
All 3 BF16 GEMM edge case tests fail with `AttributeError: module
'blackwell_kernels._C' has no attribute 'gemm'`.

FP64 DGEMM (`bk.dgemm`) works fine — only BF16 is missing.

## How to reproduce

```python
cd /data/src/bwk/gemm
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=python python3 -c "
from blackwell_kernels import _C as bk
print(dir(bk))  # 'gemm' not in list
"
```

## Expected vs actual

- **Expected:** `bk.gemm(A, B)` accepts BF16 tensors and returns BF16 GEMM result
- **Actual:** `AttributeError: module 'blackwell_kernels._C' has no attribute 'gemm'`

## Severity

**Build/binding issue** — The BF16 GEMM kernel likely exists in CUDA source but
is not registered in the PyBind11/torch extension module. This means BF16 GEMM
is untestable from Python.

## Which consumers are affected

- **gemm/** tests — All BF16 tests are dead
- Any Python-level consumer of the gemm project's BF16 kernel
- Does NOT affect linalg/ (which has its own bindings) or cuBLAS benchmarks (which use C++)
