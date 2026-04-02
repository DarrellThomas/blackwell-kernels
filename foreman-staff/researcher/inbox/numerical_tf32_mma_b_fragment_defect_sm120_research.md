# TF32 MMA B Fragment Broadcasting Defect on sm_120 (RTX 5090) — Comprehensive Research Report

**Source:** Multiple (see URLs below)
**Relevant to:** numerical worker (Cholesky, LU), all kernel workers
**Worker's current problem:** TF32 MMA m16n8k8 produces incorrect results due to B operand broadcasting along diagonals on sm_120

## Executive Summary

After extensive searching, **no public documentation, errata, bug report, or community discussion was found describing the specific B fragment broadcasting defect** observed in `mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32` on sm_120. This appears to be an undocumented behavioral change or hardware limitation that we discovered empirically. However, several important contextual findings explain the broader landscape.

---

## Finding 1: SM120 Does NOT Use tcgen05 — It Uses Extended mma.sync

SM120 (consumer Blackwell / RTX 5090) is architecturally distinct from SM100 (datacenter Blackwell / B200). SM120 lacks TMEM hardware entirely and does NOT support the `tcgen05` instruction family. Instead, SM120 uses an **extended version of `mma.sync`**, the same instruction family dating back to Ampere (sm_80).

The "extended" part refers to new `.kind::f8f6f4` and `.kind::mxf8f6f4` variants supporting FP4, FP6, and FP8 block-scaled operations. The legacy mma.sync shapes (m16n8k4, m16n8k8 with tf32; m16n8k16 with bf16/f16; m16n8k32 with fp8) are available via backward compatibility.

**Source:** https://www.backend.ai/blog/2026-02-is-dgx-spark-actually-a-blackwell

---

## Finding 2: CUTLASS Has NO TF32 MMA Atoms for SM120

CUTLASS (as of v4.2.0) defines SM120-specific MMA atoms in `include/cute/arch/mma_sm120.hpp`. These atoms are **exclusively for sub-byte formats**: E2M1, E3M2, E2M3, E4M3, E5M2 with the `.kind::f8f6f4` instruction family. **There is NO TF32 MMA atom defined for SM120.**

TF32 MMA atoms exist only in `mma_sm80.hpp` as `SM80_16x8x4_F32TF32TF32F32_TN` (m16n8k4) and `SM80_16x8x8_F32TF32TF32F32_TN` (m16n8k8), guarded by `CUTE_ARCH_MMA_SM80_ENABLED`. These would compile for sm_120 via backward compatibility but are not tested or validated by NVIDIA for sm_120.

The CUTLASS changelog for SM120 support (v4.0.0 and v4.2.0) mentions ONLY blockscaled/narrow-precision GEMMs. TF32 is never mentioned in connection with SM120.

**Source:** https://raw.githubusercontent.com/NVIDIA/cutlass/main/include/cute/arch/mma_sm120.hpp
**Source:** https://docs.nvidia.com/cutlass/latest/CHANGELOG.html

---

## Finding 3: NVIDIA's Official SM120 GEMM Support Covers Only New Formats

The CUTLASS Blackwell functionality documentation lists SM120 MMA instructions as:
- `mma.sync.aligned.kind::f8f6f4` (FP8/FP6/FP4)
- `mma.sync.aligned.kind::mxf8f6f4.block_scale` (block-scaled FP8/FP6/FP4)
- `mma.sync.aligned.kind::mxf4.block_scale` (block-scaled FP4)
- `mma.sync.aligned.kind::mxf4nvf4.block_scale` (NVFP4)

Notably, `.kind::tf32` appears ONLY for SM100 (datacenter Blackwell with tcgen05). SM120 documentation does NOT list `.kind::tf32` or any legacy tf32 MMA as an officially supported path.

The page notes that "legacy types (tf32, f16, bf16, i8 and u8)" have the same alignment requirements as Hopper — but this appears to refer to SM100 (which supports `.kind::tf32` via tcgen05), NOT SM120.

**Source:** https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/blackwell_functionality.md

---

## Finding 4: cuBLAS Uses BF16x9 Algorithm for FP32 GEMM on Blackwell

NVIDIA's cuBLAS 12.9+ introduced a **BF16x9 FP32 emulation** algorithm that decomposes FP32 operands into 9 BF16 components (3x3 grid) and performs BF16 tensor core multiplications to recover full FP32 precision:

```
AB = sum(i=1..3) sum(j=1..3) 2^(-8(i+j-2)) * A_i * B_j
```

This achieves 3-4x speedup over native FP32 SGEMM with **equal or better numerical accuracy** than native FP32. The technique was specifically developed for Blackwell BF16 tensor cores.

**Critically:** The NVIDIA blog demonstrating BF16x9 shows that TF32 results for the ecTrans weather modeling application had "large error terms" and were excluded from velocity plots due to unacceptable accuracy. This suggests NVIDIA itself prefers BF16-based approaches over TF32 on Blackwell.

This is likely how cuSOLVER achieves its ~15 TFLOPS monolithic factorization performance on sm_120: BF16 tensor core operations with FP32 precision recovery, NOT TF32 MMA.

**Source:** https://developer.nvidia.com/blog/unlocking-tensor-core-performance-with-floating-point-emulation-in-cublas/
**Source:** https://developer.nvidia.com/blog/boosting-matrix-multiplication-speed-and-flexibility-with-nvidia-cublas-12-9

---

## Finding 5: The Microbenchmark Paper Lists TF32 as "Supported" but Does NOT Test It

The paper "Dissecting the NVIDIA Blackwell Architecture with Microbenchmarks" (2507.10789) lists TF32 in Table IV as a supported datatype on GB203 (sm_120), alongside FP4, FP6, FP8, INT8, FP16, BF16, and FP64. However, the paper **does not test TF32 MMA correctness or throughput on sm_120** — it only benchmarks FP4, FP6, and FP8.

The paper confirms that mma.sync on Hopper (GH100) compiles to HMMA SASS instructions. For Blackwell, it mentions QMMA (for FP8) and OMMA (for FP4), but does NOT specify which SASS instruction handles TF32 on sm_120.

**Source:** https://arxiv.org/html/2507.10789v1

---

## Finding 6: MMA-Sim Paper Covers Blackwell But Not This Specific Issue

The MMA-Sim paper (2511.10909) presents a bit-accurate reference model for tensor cores across 10 GPU architectures including Blackwell (B200) and RTX Blackwell (RTX PRO 6000). It documents several undocumented behaviors:
- TF32 truncation always zeros the 13 LSBs before computation
- NVIDIA uses rounding-to-zero (RZ) for intermediate products, then round-to-nearest-ties-to-even (RNE) for final output
- FP8 precision was increased from 13 to 25 fractional bits on Blackwell vs Hopper

However, the paper does NOT report B fragment broadcasting anomalies or tf32 MMA correctness issues on any architecture. The paper focuses on B200 (sm_100) datacenter Blackwell, not consumer sm_120.

**Source:** https://arxiv.org/abs/2511.10909

---

## Finding 7: No Community Reports of This Issue

Searches across NVIDIA Developer Forums, CUTLASS GitHub issues, PyTorch forums, and general web found **zero reports** of TF32 MMA producing incorrect results on sm_120. The most likely explanations:
1. Very few developers write hand-tuned PTX MMA kernels for sm_120
2. Most sm_120 work uses BF16 or FP8 (where CUTLASS has official support)
3. cuBLAS hides the issue by using BF16x9 instead of TF32 MMA
4. The ecosystem was slow to support sm_120 at all (PyTorch only recently added support)

---

## Finding 8: The m16n8k4 TF32 Variant — Unknown Status

The PTX ISA defines two TF32 MMA shapes:
- `mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32` (B uses 1 register)
- `mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32` (B uses 2 registers)

Our empirical testing confirmed the m16n8k8 defect. **The m16n8k4 variant has NOT been tested.** With only 1 B register per thread (vs 2 for m16n8k8), the broadcasting pattern may differ. However, given that both shapes share the same tensor core hardware, the m16n8k4 variant may have an analogous issue.

This is worth testing: if m16n8k4 works correctly, it could be used as a building block (two m16n8k4 calls = one m16n8k8 equivalent) at some throughput cost.

---

## Finding 9: Known PTX-to-SASS Lowering Issues

There is precedent for PTX MMA instructions not being properly lowered to tensor core SASS instructions. An NVIDIA forum thread (208808) documented `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` on A100 (sm_80) being compiled to scalar IMAD/IADD instead of HMMA, producing correct but extremely slow results.

A similar scenario could explain the sm_120 TF32 defect: if ptxas lowers `mma.sync...tf32.tf32...` to a non-tensor-core emulation path on sm_120, the emulation might have a different operand mapping that produces the broadcasting behavior. This would make it a **compiler/lowering bug** rather than a hardware defect.

To test this hypothesis: inspect the SASS output of the TF32 MMA kernel with `cuobjdump --dump-sass` and check whether HMMA instructions are actually generated. If the SASS shows scalar instructions instead, that confirms a lowering issue.

**Source:** https://forums.developer.nvidia.com/t/ptx-instruction-mma-not-lowered-to-tensor-core-related-sass-instruction/208808

---

## Finding 10: Alternative Approaches for TF32-Level Precision on sm_120

### 10a: BF16x9 FP32 Emulation (cuBLAS approach)
Decompose FP32 into 9 BF16 values, perform 9 BF16 MMA calls, sum results. Full FP32 precision, ~3-4x native FP32 speed. This is what cuBLAS does on Blackwell.

### 10b: "Recovering Single Precision Accuracy" (Ootomo & Yokota 2022)
Split FP32 operands into TF32 or FP16 components, compute corrections:
- `tf32tf32` method: uses TF32 MMA for main product + FP32 correction for residual
- On A100, achieves faster-than-cuBLAS-SGEMM with identical FP32 accuracy
- On sm_120: the base TF32 MMA is broken, but the FP16-based variant may work

**Source:** https://arxiv.org/abs/2203.03341

### 10c: BF16 MMA with FP32 Accumulation (our current approach)
`mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` — proven working on sm_120 at 0.97x cuBLAS. Precision is ~1e-3 per multiply. Acceptable for most applications, especially with iterative refinement.

### 10d: Mixed-Precision TF32/TF64 Frameworks (ORNL, SC'23)
Uses 3x TF32 GEMM calls to achieve SGEMM accuracy, or Ozaki-scheme decomposition for DGEMM. Only applicable if the base TF32 MMA is correct — which it isn't on sm_120.

**Source:** https://www.osti.gov/servlets/purl/2438716

---

## Conclusions and Recommendations

1. **The TF32 MMA B fragment broadcasting defect on sm_120 is NOT documented anywhere public.** This is a novel finding from our empirical testing.

2. **NVIDIA appears to have de-emphasized TF32 on sm_120.** CUTLASS has no SM120 tf32 atoms, cuBLAS uses BF16x9 instead, and the NVIDIA blog shows TF32 having "large error terms" on Blackwell.

3. **The most likely explanation is a hardware change in the sm_120 tensor core** that altered how B fragment registers are consumed. Since sm_120's tensor core is redesigned (extended mma.sync, not sm_80 clone), the internal wiring for the B operand path may have been simplified or broken for legacy tf32 shapes. NVIDIA may have intentionally deprioritized correctness for legacy tf32 shapes, since:
   - BF16 MMA provides similar throughput with better ecosystem support
   - BF16x9 provides full FP32 accuracy
   - New formats (FP8/FP6/FP4) are the focus for sm_120

4. **Action items:**
   - **Test m16n8k4 tf32** to determine if the smaller shape has the same issue
   - **Inspect SASS output** with `cuobjdump --dump-sass` to check if tf32 MMA is lowered to HMMA or scalar fallback
   - **Use BF16 MMA** for all general GEMM on sm_120 (already validated)
   - **Consider filing an NVIDIA developer forum post** to get official acknowledgment
   - **For monolithic kernels needing FP32 precision:** adopt the BF16x9 decomposition pattern from cuBLAS

5. **For cuSOLVER behavior:** cuSOLVER's monolithic `getrf_wo_pivot_params_` kernel almost certainly uses BF16 MMA with the BF16x9 or similar decomposition technique to achieve TF32-level performance with FP32 accuracy, NOT raw TF32 MMA instructions.

---

## All Sources

- [Backend.ai - Is DGX Spark Actually Blackwell? (SM120 vs SM100 architecture)](https://www.backend.ai/blog/2026-02-is-dgx-spark-actually-a-blackwell)
- [CUTLASS Blackwell Functionality Documentation](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/blackwell_functionality.md)
- [CUTLASS Changelog](https://docs.nvidia.com/cutlass/latest/CHANGELOG.html)
- [CUTLASS SM120 MMA Atoms (mma_sm120.hpp)](https://raw.githubusercontent.com/NVIDIA/cutlass/main/include/cute/arch/mma_sm120.hpp)
- [CUTLASS SM80 MMA Atoms (mma_sm80.hpp)](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/arch/mma_sm80.h)
- [cuBLAS 12.9 FP32 Emulation Blog](https://developer.nvidia.com/blog/boosting-matrix-multiplication-speed-and-flexibility-with-nvidia-cublas-12-9)
- [cuBLAS BF16x9 FP Emulation Technical Blog](https://developer.nvidia.com/blog/unlocking-tensor-core-performance-with-floating-point-emulation-in-cublas/)
- [Dissecting Blackwell Architecture with Microbenchmarks (2507.10789)](https://arxiv.org/html/2507.10789v1)
- [MMA-Sim: Bit-Accurate Tensor Core Reference Model (2511.10909)](https://arxiv.org/abs/2511.10909)
- [Microbenchmarking Blackwell Architecture (2512.02189)](https://arxiv.org/html/2512.02189v1)
- [Dissecting Tensor Cores via Microbenchmarks (2206.02874)](https://arxiv.org/abs/2206.02874)
- [Recovering Single Precision Accuracy from Tensor Cores (2203.03341)](https://arxiv.org/abs/2203.03341)
- [Mixed-Precision S/DGEMM Using TF32/TF64 Frameworks (ORNL)](https://www.osti.gov/servlets/purl/2438716)
- [PTX ISA 9.2 Documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)
- [NVIDIA Developer Forum: PTX mma not lowered to SASS](https://forums.developer.nvidia.com/t/ptx-instruction-mma-not-lowered-to-tensor-core-related-sass-instruction/208808)
- [NVIDIA Developer Forum: mma instruction understanding](https://forums.developer.nvidia.com/t/understand-the-mma-instruction-in-ptx/294471)
- [CUTLASS Issue #2186: GEMM arch 120 support](https://github.com/NVIDIA/cutlass/issues/2186)
- [gau-nernst: tcgen05 for dummies](https://gau-nernst.github.io/tcgen05/)
- [Lei Mao: Benchmarking Tensor Core MMA Peak Performances](https://leimao.github.io/blog/Benchmarking-NVIDIA-Tensor-Core-MMA-Peak-Performances/)
- [RTX Blackwell GPU Architecture Whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf)
