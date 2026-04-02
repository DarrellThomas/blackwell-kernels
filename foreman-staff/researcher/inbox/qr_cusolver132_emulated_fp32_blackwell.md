# cuSOLVER 13.2: Emulated FP32 on Blackwell and QR Implications

**Source:** NVIDIA cuSOLVER 13.2 documentation (https://docs.nvidia.com/cuda/cusolver/index.html) and "Unlocking Tensor Core Performance with FP Emulation in cuBLAS" (https://developer.nvidia.com/blog/unlocking-tensor-core-performance-with-floating-point-emulation-in-cublas/)
**Relevant to:** QR worker
**Worker's current problem:** Must understand what cuSOLVER geqrf is doing on sm_120 so we know what we're competing against.

## What This Is

cuSOLVER 13.2 (shipping with CUDA 13.2) adds a new math mode `CUSOLVER_FP32_EMULATED_BF16X9_MATH` that uses BF16 tensor cores to emulate FP32 arithmetic with **bit-accurate FP32 results**. This is relevant because cuSOLVER's geqrf is listed as an "affected function" -- meaning cuSOLVER may already be using tensor-core-accelerated GEMM for the trailing matrix update.

## Why It Matters for Us

If cuSOLVER's geqrf already uses BF16x9 emulated FP32 GEMM for trailing updates on Blackwell, then:
1. **Our cuSOLVER baseline is already tensor-core-accelerated** -- we're not competing against naive FP32 SGEMM
2. **We need to beat tensor-core-accelerated FP32 GEMM**, not just match it
3. **Our advantage must come from algorithmic improvement** (recursive QR, fused kernels) rather than just "use tensor cores where cuSOLVER doesn't"

## Key Technical Details

### BF16x9 Emulation
The cuBLAS blog describes BF16x9 as a "static decomposition that can be used to performantly and safely emulate all normal and subnormal FP32 values using Blackwell BF16 tensor cores."

- **How it works**: Decomposes each FP32 value into 9 BF16 values. The GEMM is computed using BF16 tensor cores with multiple passes, and the partial products are summed to produce bit-accurate FP32 results.
- **Performance**: Up to **3x speedup** over native FP32 SGEMM on GB200. The speedup varies by matrix shape -- strongest for moderate to large problems.
- **Accuracy**: Bit-accurate FP32. Not approximate. The error distribution matches or betters native FP32.

### cuSOLVER Integration
- cuSOLVER 13.2 adds `cusolverDnSetMathMode()` with `CUSOLVER_FP32_EMULATED_BF16X9_MATH`
- `cusolverDnXgeqrf()` is listed as an "affected function"
- There's also `cusolverDnSetEmulationStrategy()` with EAGER mode
- Workspace sizes may depend on the math mode

### What This Means for cuSOLVER's geqrf on sm_120
cuSOLVER geqrf likely uses BF16x9 emulated FP32 for its trailing matrix update GEMMs when the emulated math mode is enabled. This means:
- Trailing GEMM runs at ~3x native FP32 speed using tensor cores
- Results are bit-accurate FP32 (no precision loss)
- The panel factorization likely remains native FP32 (sequential, not GEMM-shaped)

## Implications for Our Strategy

### What we can still beat:
1. **Recursive QR**: cuSOLVER almost certainly uses standard blocked QR (fixed nb-wide GEMMs). Even with BF16x9, a 64-wide GEMM is memory-bound. Recursive QR produces n/2-wide GEMMs that are compute-bound -- our BF16 mma.sync GEMMs will be faster per FLOP than BF16x9 emulated FP32 (because we use 1 BF16 MMA instead of 9).

2. **Monolithic kernel**: cuSOLVER launches separate kernels for panel, LARFT, and trailing update. A single-kernel approach eliminates launch overhead and enables shared memory reuse.

3. **GPU-only panel**: If cuSOLVER uses a hybrid CPU-GPU approach for the panel, we win by keeping everything on-device.

### What we should NOT do:
- Don't compete on GEMM accuracy alone. BF16x9 gives bit-exact FP32 for free. Our BF16 mma.sync gives ~1e-3 precision. For the trailing GEMM, cuSOLVER's approach is more accurate AND potentially faster (3x native FP32 on large problems).
- Instead, compete on **algorithm structure** (recursive QR), **kernel fusion** (monolithic), and **launch overhead elimination**.

### Revised competitive analysis:
| Component | cuSOLVER 13.2 (likely) | Our approach | Our advantage |
|-----------|----------------------|--------------|---------------|
| Panel (GEQR2) | GPU-native FP32 | cuSOLVERDx or custom register-tiled | Monolithic (no launch) |
| LARFT | FP32, possibly BLAS-2 | Modified CWY (BLAS-3) or skip entirely | BLAS-3 vs BLAS-2 |
| Trailing GEMM | BF16x9 emulated FP32 (~3x native) | BF16 mma.sync (~3x native, lower precision) | Recursive QR shape (square vs tall-skinny) |
| Algorithm | Standard blocked QR (nb=64?) | Recursive QR (n/2-wide GEMMs) | Better tensor core utilization |
| Kernel launches | Separate kernels | Monolithic | Zero launch overhead |

### Bottom line:
The BF16x9 emulation means cuSOLVER's trailing GEMMs on Blackwell are already tensor-core-accelerated with FP32 accuracy. **Our path to beating cuSOLVER is through recursive QR (better GEMM shapes) and monolithic kernels (zero launch overhead), not through tensor core usage alone.** The 1.4x speedup reported by Leng et al. was on older GPUs without emulated FP32 -- on Blackwell, the gap may be smaller unless we also exploit algorithmic advantages.

## Caveats

1. **BF16x9 is 9 MMA ops per FP32 GEMM element**: Despite 3x speedup over FP32, it's still ~3x slower than single BF16 MMA. If we accept BF16 precision (~1e-3) for the trailing GEMM, our approach is faster per FLOP. The question is whether the precision trade-off is acceptable.

2. **Default math mode unclear**: cuSOLVER may NOT enable BF16x9 by default. The user must call `cusolverDnSetMathMode()`. If the default is native FP32, then the baseline we measure is native FP32 and the comparison is simpler.

3. **GB200 vs RTX 5090**: The 3x speedup was measured on GB200 (datacenter). RTX 5090 (sm_120) has different tensor core throughput ratios, so the BF16x9 speedup may differ.
