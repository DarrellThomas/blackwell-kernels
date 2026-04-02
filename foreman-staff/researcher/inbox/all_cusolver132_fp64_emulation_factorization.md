# cuSOLVER 13.2: FP64 Emulation for Dense Factorizations

**Sources:**
- [CUDA 13.2 Blog Post](https://developer.nvidia.com/blog/cuda-13-2-introduces-enhanced-cuda-tile-support-and-new-python-features/)
- [CUDA 13.2 Release Notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)

**Relevant to:** numerical worker (LU/QR/Cholesky factorizations)
**Date:** 2026-03-14

---

## What This Is

cuSOLVER 13.2 (March 2026) adds FP64-emulated calculation APIs for dense linear
algebra routines. This uses lower-precision tensor cores (BF16 or TF32) to emulate
FP64 arithmetic, similar to how cuBLAS BF16x9 emulates FP32 GEMM.

## New APIs

- `cusolverDnSetMathMode()` / `cusolverDnGetMathMode()` -- control emulation mode
- `cusolverDnSetEmulationStrategy()` / `cusolverDnGetEmulationStrategy()` -- fine-tune
  mantissa precision and special value handling

## Performance Numbers

Benchmarks from the CUDA 13.2 blog post (on B200 datacenter):

| Operation | Speedup vs Native FP64 | Matrix Size |
|-----------|----------------------|-------------|
| QR (GEQRF) | Up to 2x | ~80K |
| LU (GETRF) | Similar | ~80K |
| Cholesky (POTRF) | Similar | ~80K |

The blog notes this is "particularly beneficial for platforms with high INT8-to-FP64
throughput ratios." Consumer Blackwell (sm_120) has the same tensor core types as
datacenter for BF16/TF32, so the emulation approach should work, but the speedup
numbers may differ.

## Relevance for Numerical Worker

This is the HOST-SIDE (cuSOLVER library) version of FP64 emulation. It means:

1. **Reference baseline moves:** If cuSOLVER's FP64-emulated LU is 2x faster than
   native FP64 LU, the bar for our custom kernel also moves. We should benchmark
   both native and emulated cuSOLVER as reference points.

2. **Same principle as our approach:** Our numerical worker is exploring BF16 tensor
   core GEMM for trailing matrix updates. cuSOLVER 13.2 validates this approach at
   the library level -- NVIDIA is officially endorsing mixed-precision emulation for
   dense factorizations.

3. **Not device-side:** These are cuSOLVER host APIs, not cuSOLVERDx device-side.
   Our monolithic kernel approach still needs custom or cuSOLVERDx device-side
   factorizations. But the emulation strategy guidance could inform our precision
   choices.

## Caveats

- Performance numbers are for B200 (datacenter), not RTX 5090 (consumer).
- The "mantissa control" APIs suggest tunable precision -- more mantissa bits =
  higher accuracy but lower speedup. The sweet spot may differ by application.
- No mention of sm_120 specifically in the release notes for this feature.
