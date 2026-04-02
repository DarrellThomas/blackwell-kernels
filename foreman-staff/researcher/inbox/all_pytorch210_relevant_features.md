# PyTorch 2.10 Features Relevant to Our Work

**Sources:**
- [PyTorch 2.10 Release Blog](https://pytorch.org/blog/pytorch-2-10-release-blog/)
- [PyTorch sm_120 Forum Discussion](https://discuss.pytorch.org/t/pytorch-support-for-sm120/216099)

**Relevant to:** all workers (testing infrastructure), numerical worker (eigenvalue)
**Date:** 2026-03-14

---

## sm_120 Support Status

PyTorch 2.10 ships with CUDA 12.8+ wheels that support sm_120 (RTX 5090). However:
- **NVFuser does NOT support sm_120.** torch.compile may fall back to other backends.
- Building from source with `TORCH_CUDA_ARCH_LIST="12.0"` works for custom extensions.
- Our setup (CUDA 13, PyTorch 2.10) should work for sm_120 custom kernel testing.

## Relevant New Features

### 1. varlen_attn() -- Variable Length Attention Op

New `torch.nn.attention.varlen_attn()` for ragged/packed sequences. Supports:
- Forward + backward pass
- torch.compile-able
- BF16 and FP16 dtypes
- Requires A100+ (sm_80+)

**Relevance:** This is a high-level API, not a kernel primitive. But it establishes
a reference interface for variable-length attention. Our attention kernel could be
benchmarked against this API for variable-length workloads.

### 2. Efficient Eigenvalue Decomposition

PyTorch linalg now uses `cusolverDnXgeev` for general eigenvalue decomposition.
This integrates the cuSOLVER 13.1 GEEV improvements (1.7x speedup for large matrices).

**Relevance:** For the future eigenvalue project in the pipeline, PyTorch now provides
a fast host-side reference. Our custom kernel would need to beat this.

### 3. Combo-Kernels Horizontal Fusion (torchinductor)

torch.compile now performs horizontal fusion -- combining independent parallel
operations into a single GPU kernel launch. Reduces kernel launch overhead.

**Relevance:** This is at the framework level, not kernel level. But it means
PyTorch-generated code may have fewer kernel launches, making the "fused" advantage
of our custom kernels smaller. We should ensure our benchmarks compare against
the latest torch.compile output.

### 4. FP8 Support: Intel GPU Only

PyTorch 2.10's FP8 improvements are Intel-GPU-focused (scaled_mm, channel-wise
scaling). NVIDIA FP8 support remains via `torch._scaled_mm` (unchanged from 2.9).

**No new NVIDIA FP8 features in PyTorch 2.10.**

### 5. Python 3.14 Support

torch.compile now supports Python 3.14, including experimental freethreaded builds.
Not directly relevant to kernel work but good for infrastructure.

## Summary

| Feature | Impact on Us | Workers |
|---------|-------------|---------|
| sm_120 wheels | Infrastructure (already working) | all |
| varlen_attn | Reference comparison | attention |
| cuSOLVER GEEV integration | Reference baseline | numerical (future) |
| Combo-kernel fusion | Benchmark methodology | all |
| NVIDIA FP8 | No change | none |
