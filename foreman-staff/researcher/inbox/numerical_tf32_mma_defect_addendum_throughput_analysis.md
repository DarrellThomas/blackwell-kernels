# TF32 MMA Defect Addendum: Throughput Parity and Alternative Strategies

**Source:** Multiple (see URLs below)
**Relevant to:** numerical worker (Cholesky, LU, all numerical methods)
**Worker's current problem:** TF32 MMA B fragment broadcasting is broken on sm_120. Worker needs to decide whether to invest in TF32 workarounds or move to BF16-based approaches.

## What This Is

Addendum to the existing `numerical_tf32_mma_b_fragment_defect_sm120_research.md` brief. Contains new findings that strengthen the case for abandoning TF32 on sm_120 entirely.

---

## Finding 1: TF32 Tensor Core TFLOPS = FP32 SIMT TFLOPS on Consumer GeForce

**This is the single most important finding for the numerical worker.**

On consumer GeForce GPUs, TF32 dense tensor core throughput equals FP32 CUDA core throughput:

| GPU | FP32 SIMT | TF32 Dense TC | TF32 Sparse TC | BF16 Dense TC |
|-----|-----------|---------------|----------------|---------------|
| RTX 5090 (sm_120) | 104.8 TFLOPS | 104.8 TFLOPS | 209.5 TFLOPS | 209.5 TFLOPS |
| RTX 4090 (sm_89)  | 82.6 TFLOPS  | 82.6 TFLOPS  | 165.2 TFLOPS  | 82.6 TFLOPS  |

**Implication:** Even if TF32 MMA worked correctly on sm_120, it would provide ZERO throughput advantage over regular FP32 CUDA cores for dense operations. The only TF32 advantage is with structured sparsity (2:4 pattern), which is inapplicable to numerical linear algebra (Cholesky, LU, etc.).

This is NOT a sm_120 anomaly -- it was also true on the RTX 4090 (sm_89). NVIDIA reserves the TF32 throughput advantage for datacenter GPUs (A100: 156 TFLOPS TF32 vs 19.5 TFLOPS FP32; B200: similar ratio).

**Conclusion: TF32 MMA is doubly useless on sm_120 for our purposes:**
1. The B fragment is broken (broadcasting defect)
2. Even if it worked, it provides no throughput gain over FP32 SIMT

**Source:** https://www.bestgpusforai.com/gpu-comparison/5090-vs-4080-super
**Source:** https://www.waredb.com/processor/nvidia-geforce-rtx-5090
**Source:** https://www.waredb.com/processor/nvidia-geforce-rtx-4090

---

## Finding 2: CUTLASS 3xTF32 is Slower Than SIMT on GeForce Cards

NVIDIA maintainer confirmed in CUTLASS Discussion #390:

> "On GeForce cards (like RTX 3070), FP32 without tensor cores performs identically to TF32 with tensor cores. Therefore, 3xTF32 provides no advantage on consumer GPUs."

Performance comparison:
- RTX 3070: 3xTF32 = 6,825 GFLOPS, SIMT FP32 = 11,233 GFLOPS (3xTF32 is 1.6x SLOWER)
- A100: 3xTF32 = 23,719 GFLOPS (well exceeds FP32 theoretical peak)

The NVIDIA maintainer also noted: "CUTLASS is a co-design between us and CUDA compiler team. Next CUDA compiler will significantly boost TF32x3 performance." -- but this likely refers to datacenter GPUs.

**Conclusion:** 3xTF32 is a loss on consumer GPUs. Do NOT pursue it.

**Source:** https://github.com/NVIDIA/cutlass/discussions/390

---

## Finding 3: BF16 MMA is the Only Viable Tensor Core Path

On the RTX 5090:
- BF16 dense tensor core: 209.5 TFLOPS (2x FP32 SIMT)
- BF16 sparse tensor core: 419.0 TFLOPS (4x FP32 SIMT)

BF16 MMA (m16n8k16) is the ONLY MMA instruction that provides a meaningful throughput advantage over FP32 SIMT on consumer Blackwell. This makes it the only viable path for tensor-core-accelerated numerical methods.

Precision impact per BF16 multiply: ~1e-3 relative error (7-bit mantissa). For Cholesky SYRK, errors accumulate across K iterations. Mitigation strategies:
- FP32 accumulators (already done in mma.sync)
- Iterative refinement (standard for mixed-precision factorizations)
- BF16x9 decomposition for full FP32 precision (see existing brief)

---

## Finding 4: Ozaki Scheme with FP8 Tensor Cores for FP64 Emulation

A recent paper (Mukunoki et al., 2025) demonstrated DGEMM without FP64 arithmetic using the Ozaki scheme with FP8 tensor cores on Blackwell:

- Achieves full FP64 accuracy
- Over 80 TFLOPS on Blackwell hardware (vs ~1 TFLOPS native FP64 on consumer GPUs)
- Uses FP64 emulation based on integer arithmetic, eliminating FP64 instructions entirely
- Decomposes FP64 operands into multiple FP8 slices, multiplies on tensor cores, reassembles

**Relevance for us:** FP8 MMA (m16n8k32) on the RTX 5090 delivers 419+ TFLOPS dense. If we need FP64-accurate GEMM for conditioning-sensitive numerical methods, the Ozaki scheme could provide massive speedups. However, the decomposition overhead is significant (many FP8 GEMMs per FP64 GEMM).

**Source:** https://arxiv.org/abs/2508.00441
**Source:** https://arxiv.org/html/2511.13778v1

---

## Finding 5: cuSolverDx -- Device-Side Factorization Library

NVIDIA now offers cuSolverDx (Device Extensions) that enables factorization routines to run entirely inside CUDA kernels:

- Supports: Cholesky, LU, QR, TRSM, eigenvalue, SVD
- Runs as device-side routines embeddable in a CUDA kernel
- Customizable: size, precision, type, fill mode, storage layout, target SM
- Available in CUDA 13.x

This is likely the mechanism behind cuSOLVER's monolithic `getrf_wo_pivot_params_` kernel. cuSolverDx may provide an API for building monolithic factorization kernels without hand-rolling device-side GEMM.

**Worth investigating:** Does cuSolverDx expose device-side SYRK with tensor core acceleration? If so, it could replace our manual BF16 MMA approach entirely.

**Source:** https://docs.nvidia.com/cuda/cusolverdx/
**Source:** https://docs.nvidia.com/cuda/cusolverdx/0.1.0/get_started/introduction.html

---

## Finding 6: TF32 B Fragment Layout (PTX ISA Documentation)

For completeness, the PTX ISA documents the TF32 m16n8k8 B fragment as:
- 2 registers (b0, b1) of type .b32 per thread
- B matrix is 8x8 (k=8, n=8)
- 32 threads share 2 registers each = 64 register values total = 64 matrix elements

The documented layout specifies column-major ordering within each register half. However, the documentation is the *specification* of intended behavior. The observed broadcasting defect (b0 -> B[k_even,n_even] AND B[k_even+1,n_even+1]) represents a deviation from spec on sm_120 hardware.

The m16n8k4 variant uses 1 register per thread. Whether it has the same defect is unknown.

---

## Updated Strategy Recommendation

Given these findings, the priority order for the numerical worker should be:

1. **BF16 MMA (m16n8k16)** for device-side SYRK/GEMM -- 2x FP32 throughput, proven working, acceptable precision with FP32 accumulators

2. **cuSolverDx** -- investigate whether it provides a ready-made device-side SYRK that handles the tensor core complexity internally

3. **BF16x9 decomposition** -- for cases requiring FP32-exact precision (only if iterative refinement is insufficient)

4. **Batched small Cholesky** -- pivot to problem sizes where cuSOLVER is weak (N=32-64)

**Definitively abandon:**
- TF32 MMA (broken AND no throughput advantage)
- 3xTF32 (slower than SIMT on consumer GPUs)
- Any TF32-based decomposition scheme

---

## Sources

- [RTX 5090 AI Specs](https://www.waredb.com/processor/nvidia-geforce-rtx-5090)
- [RTX 4090 AI Specs](https://www.waredb.com/processor/nvidia-geforce-rtx-4090)
- [RTX 5090 vs RTX 4080 Super Comparison](https://www.bestgpusforai.com/gpu-comparison/5090-vs-4080-super)
- [CUTLASS 3xTF32 Performance Discussion](https://github.com/NVIDIA/cutlass/discussions/390)
- [NVIDIA cuBLAS BF16x9 FP Emulation](https://developer.nvidia.com/blog/unlocking-tensor-core-performance-with-floating-point-emulation-in-cublas/)
- [Ozaki Scheme FP8 DGEMM (Mukunoki et al.)](https://arxiv.org/abs/2508.00441)
- [Extended Ozaki Scheme for Guaranteed DGEMM Accuracy](https://arxiv.org/html/2511.13778v1)
- [cuSolverDx Documentation](https://docs.nvidia.com/cuda/cusolverdx/)
- [cuSolverDx Introduction](https://docs.nvidia.com/cuda/cusolverdx/0.1.0/get_started/introduction.html)
- [PTX ISA 9.2 Documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)
- [Harnessing GPU Tensor Cores for Mixed-Precision Iterative Refinement](https://www.netlib.org/utk/people/JackDongarra/PAPERS/haidar_fp16_sc18.pdf)
