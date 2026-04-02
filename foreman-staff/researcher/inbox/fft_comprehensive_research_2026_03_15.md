# FFT Research Brief: Comprehensive Survey for sm_120 Custom Kernel

**Date:** 2026-03-15
**Researcher:** researcher-claude
**Relevant to:** future FFT kernel worker

---

## Finding 1: tcFFT — Tensor Core FFT via Radix-16 Decomposition

**Source:** https://arxiv.org/abs/2104.11471
**Relevant to:** FFT (tensor core path)
**Summary:** tcFFT maps FFT to WMMA tensor cores using radix-16 as the base radix, matching the 16x16 tensor core fragment. Uses fragment-level element manipulation for twiddle factor application. FP16 only.
**Key technique:** The N-point DFT is decomposed as `X = F_N1 * (T * X_in)` where F_N1 is a 16x16 DFT matrix computed as a single WMMA operation. Twiddle factors are applied by directly accessing elements within WMMA fragments using a reverse-engineered thread-to-element mapping. Multiple radix-16 stages compose larger FFTs (256 = two stages, 4096 = three stages).
**Applicability to sm_120:** The concept transfers but needs adaptation. sm_120's mma.sync m16n8k16 has asymmetric tiles (not 16x16), so the radix decomposition needs adjustment. BF16 inputs with FP32 accumulation gives slightly better precision than tcFFT's FP16. We already understand mma.sync fragment mappings from our GEMM work. Performance: 1.29x-3.24x over cuFFT on V100, 1.10x-3.03x on A100 (FP16 only).
**Caveats:** FP16/BF16 precision may be insufficient for applications requiring more than ~3 decimal digits of accuracy after O(log N) butterfly stages. Fragment mapping must be re-derived for sm_120's mma.sync layout.

---

## Finding 2: FlashFFTConv — Monarch Decomposition for Tensor Core FFT Convolution

**Source:** https://arxiv.org/abs/2311.05908 (ICLR 2024)
**Code:** https://github.com/HazyResearch/flash-fft-conv
**Relevant to:** FFT (fused convolution path)
**Summary:** Decomposes FFT into matrix-matrix multiplies using Monarch/Bailey's decomposition. Each FFT stage becomes a batch of matrix multiplies suitable for tensor cores. Fuses FFT+pointwise+IFFT into minimal kernels. Up to 7.93x over PyTorch FFT convolutions on H100.
**Key technique:** N-point FFT = reshape into N1xN2 matrix, column-wise FFT (matrix multiply by F_N1), twiddle multiply, row-wise FFT (matrix multiply by F_N2), transpose. Order-2 for N<4K, order-3 for N<32K, order-4 for N<4M. Each matrix multiply stage uses tensor cores. The trade-off: O(N^{3/2}) FLOPs instead of O(N log N), but tensor cores compensate. SRAM constraint: ~32K FP16 complex values fit in shared memory on A100/H100.
**Applicability to sm_120:** The algorithm is architecture-independent but needs mma.sync implementation. sm_120's 99KB shared memory fits ~12K FP32 complex values (less than H100's 228KB), pushing toward higher-order decompositions. The real win is for fused FFT convolution, not standalone FFT. BF16/FP16 precision required. No TMA on sm_120 — use cp.async instead.
**Caveats:** The O(N^{3/2}) FLOP overhead is significant for small N. For standalone FFT (no fusion), traditional Stockham may be faster. Code targets H100/A100.

---

## Finding 3: FFT Blitz + PACT Extension — Tensor Core + Warp Shuffle Hybrid

**Source:** https://dl.acm.org/doi/10.1145/3437801.3441623 (PPoPP 2021)
**Extended:** https://dl.acm.org/doi/10.1109/PACT52795.2021.00032 (PACT 2021)
**Relevant to:** FFT (tensor core path)
**Summary:** Maps Cooley-Tukey FFT to WMMA tensor cores with four optimization types. The PACT extension adds warp shuffle integration for a hybrid approach. 15%-250% over cuFFT (FFT), up to 4x for NTT.
**Key technique:** Four optimizations: (1) twiddle factor integration into fragment elements, (2) bank-conflict-free shared memory layout, (3) multi-stage butterfly fusion into fewer tensor core ops, (4) batch optimization for throughput. The PACT paper adds warp shuffles for small radix stages while tensor cores handle large radix stages.
**Applicability to sm_120:** The four optimizations are algorithm-level and apply to any mma.sync implementation. The tensor core + warp shuffle hybrid is particularly promising — use tensor cores for the "heavy" matrix-multiply stages and warp shuffles for "light" radix-2/4 stages. This maps well to sm_120 where both mma.sync and __shfl_xor_sync are fast.
**Caveats:** No public code found. Academic work from 2021, pre-dating Blackwell. FP16 precision only.

---

## Finding 4: TurboFFT — Near-cuFFT Performance with Template Code Generation

**Source:** https://arxiv.org/html/2405.02520v1
**Relevant to:** FFT (CUDA core path)
**Summary:** Template-based FFT framework achieving within 0.58% of cuFFT on A100 (FP32). Uses three-level tiling (thread/warp/block), XOR swizzling for bank conflicts, and warp shuffles for inter-thread exchange. Portable — no arch-specific PTX.
**Key technique:** Hierarchical decomposition: thread-level (radix 8/16/32 in registers), warp-level (warp shuffles, no sync), block-level (shared memory with XOR swizzle). XOR swizzle gives ~20% improvement over non-swizzled. Template-based code generation parameterized by (N1, N2, N3, n1, n2, n3, batch). Single kernel for N up to 8192.
**Applicability to sm_120:** Directly applicable — high-level CUDA C++ with no arch-specific code. The XOR swizzle is identical to our GEMM swizzle infrastructure in common/csrc/common/. Three-level tiling maps directly to sm_120's hierarchy. However, within 0.58% of cuFFT means minimal headroom for improvement in the traditional (non-tensor-core) CUDA core approach.
**Caveats:** Code not publicly available. Demonstrates that cuFFT is nearly optimal for FP32 FFT — hard to beat without tensor cores or fusion.

---

## Finding 5: TurboFNO — State-of-Art Fused FFT-GEMM-iFFT Kernel

**Source:** https://arxiv.org/html/2504.11681
**Relevant to:** FFT (fusion path)
**Summary:** First fully fused FFT-GEMM-iFFT kernel. Eliminates all intermediate global memory transfers. Up to 150% speedup, average 67% (2D) over PyTorch cuFFT+cuBLAS baseline on A100. Published April 2025.
**Key technique:** Restructures FFT iteration along the hidden dimension (matching GEMM's k-loop) so FFT output feeds directly into GEMM via shared memory. Two custom swizzle patterns achieve 100% bank utilization: FFT-to-GEMM (column-major write from FFT threads) and GEMM-to-iFFT (staggered offsets). Built-in frequency truncation prunes unnecessary butterfly operations — 25% truncation reduces compute to 37.5%.
**Applicability to sm_120:** Algorithm is general; our mma.sync GEMM primitives could replace the GEMM stage. The swizzle patterns are architecture-independent. Frequency pruning is a powerful optimization for any spectral method workload. sm_120's 99KB shared memory accommodates the 32x32x8 GEMM tile + FFT data.
**Caveats:** Designed for FNO (Fourier Neural Operators) specifically. Complex GEMM (not real). Small batch sizes don't benefit. The 150% speedup is vs the unfused baseline — the fusion is the source of improvement, not the FFT itself.

---

## Finding 6: cuFFTDx — sm_120 Officially Supported (v1.5.0+)

**Source:** https://docs.nvidia.com/cuda/cufftdx/release_notes.html
**Relevant to:** FFT (competitive benchmark)
**Summary:** NVIDIA's device-side FFT library. sm_120 support confirmed in v1.5.0 (2025). Enables FFT inside user CUDA kernels with fusion capability. 45% to 3x speedup over cuFFT for fused workloads. No tensor core usage. The real competitive bar for a custom fused FFT kernel.
**Key technique:** Compile-time expression templates generate optimized per-size FFT kernels. Key API: Size<N>, Precision<float>, SM<1200>. Provides block_dim, shared_memory_size, elements_per_thread, suggested_ffts_per_block. Register-based computation with shared memory staging.
**Applicability to sm_120:** Fully supported. This is the bar a custom kernel must beat. Openings for a custom kernel: (1) no tensor core path — BF16 TC FFT could be faster, (2) no runtime size selection, (3) no BF16 native FFT, (4) no frequency pruning/truncation, (5) no custom data layouts.
**Caveats:** Closed source. May have sm_120-specific optimizations we can't see. For non-fused FFT, cuFFTDx adds overhead vs cuFFT.

---

## Finding 7: VkFFT — Best Open-Source Reference (Updated)

**Source:** https://github.com/DTolm/VkFFT
**Relevant to:** FFT (reference implementation)
**Summary:** Most mature open-source GPU FFT. Stockham-based, radix-2/3/4/5/7/8/11/13. Runtime code generation. Average ~10% overhead vs cuFFT on A100, but beats cuFFT for large N and non-power-of-2 sizes. Bluestein's algorithm for arbitrary sizes is faster than cuFFT's.
**Key technique:** 1-thread-1-radix model with Stockham auto-sort. Runtime code generation via nvrtc allows per-size optimization. Supports FP16/FP32/FP64/FP128 (double-double). In-place transforms with zero performance loss. Groups nearby FFTs instead of transposing for strided batches.
**Applicability to sm_120:** CUDA backend works on sm_120. The codebase is the best reference for understanding GPU FFT internals. The runtime codegen approach is more flexible than cuFFTDx's compile-time templates. Does NOT use tensor cores — purely CUDA core + memory. For N > 1024, VkFFT shows cuFFT can be beaten.
**Caveats:** ~10% average overhead vs cuFFT for standard power-of-2 sizes. Runtime compilation adds startup latency. No sm_120 specific benchmarks.

---

## Finding 8: SMFFT + Overlap-and-Save — Shared Memory FFT Convolution

**Source:** https://github.com/KAdamek/SMFFT
**Related:** https://dl.acm.org/doi/10.1145/3394116
**Relevant to:** FFT (fused convolution reference)
**Summary:** Entire FFT in shared memory for sizes 32-4096. Cooley-Tukey DIT (no-reorder variant) achieves 40-60% speed advantage over cuFFT on V100 for batched small FFTs. The overlap-and-save paper demonstrates fused FFT+multiply+IFFT without any global memory intermediates.
**Key technique:** For convolution: load input segment to shared memory, FFT in shared memory (no global intermediates), pointwise multiply in registers, IFFT in shared memory, write output. CT-DIT without bit-reversal is fastest because the intermediate bit-reversed order cancels between FFT and IFFT. Shared memory padding: (N/32)*33 elements to avoid bank conflicts.
**Applicability to sm_120:** Directly applicable — simple CUDA code. The overlap-and-save fusion is the simplest path to a fused convolution kernel. Replace padding with XOR swizzle for better shared memory utilization. Use warp shuffles for first 5 stages. sm_120's 99KB shared memory handles FFTs up to 8192 in a single pass.
**Caveats:** Radix-2 only (radix-4/8 would be faster). V100 benchmarks. Padding wastes 3% shared memory.

---

## Finding 9: FlashFFTStencil — Recent (PPoPP 2025) Tensor Core FFT

**Source:** https://dl.acm.org/doi/10.1145/3710848.3710897
**Code:** https://github.com/HPHEX/FlashFFTStencil
**Relevant to:** FFT (tensor core reference code)
**Summary:** Most recent (March 2025) top-venue paper combining FFT with tensor cores. Restructures FFT into dense matrix multiplies for tensor cores, achieving 2.57x over state-of-art and 103x over cuFFT-based stencil implementations. Open-source.
**Key technique:** Three-part optimization: (1) kernel tailoring on HBM — fuse kernels to reduce memory transfers, (2) architecture aligning on SMEM — restructure FFT into dense GEMMs for shared memory tensor cores, (3) computation streamlining on TCU — minimize pipeline stalls, maximize register reuse. Supports Ampere/Hopper, probably works on sm_120 with recompilation.
**Applicability to sm_120:** Open-source code to study. The three-part optimization framework is a useful checklist. The "architecture aligning" technique of converting FFT butterfly stages into tensor-core-friendly dense GEMMs is the same core idea as tcFFT but more recent and better optimized. Supports RTX 3090/4090 (consumer GPUs like our RTX 5090).
**Caveats:** Designed for stencil computations, not general FFT. The 103x speedup is vs naive cuFFT-based stencil, not vs cuFFT standalone.

---

## Finding 10: cuFFT Internals — What We're Competing Against

**Source:** https://docs.nvidia.com/cuda/cufft/
**Relevant to:** FFT (understanding the bar)
**Summary:** cuFFT uses Stockham-based algorithms with Cooley-Tukey decomposition. Radix-2/3/5/7 building blocks for N = 2^a * 3^b * 5^c * 7^d. Bluestein's for non-composite sizes. For small N, the entire FFT is done in shared memory + registers. Extremely well-optimized — TurboFFT shows only 0.58% overhead on A100, meaning cuFFT is near-optimal for FP32.
**Key technique:** Plan-based optimization: selects algorithms, radix decomposition, and memory layout at plan creation time. Block-based multi-FFT algorithm reduces transpose overhead. Shared memory + registers for small N.
**Applicability to sm_120:** cuFFT is the bar. For FP32 FFT, beating it is very hard (only 0.58% headroom per TurboFFT). Realistic paths to beat cuFFT: (1) fused operations (FFT+op+IFFT), (2) tensor cores for BF16-tolerant workloads, (3) batched very small FFTs where launch overhead dominates, (4) non-power-of-2 sizes where VkFFT already wins.
**Caveats:** cuFFT is a black box. It may have sm_120-specific optimizations. It does NOT use tensor cores (confirmed by NVIDIA forum).

---

## Strategic Summary: Best Approaches for sm_120 FFT Kernel

### Path A: Traditional (CUDA Core + Shared Memory) — FP32 Quality
- **Algorithm:** Stockham auto-sort with radix-8 primary, radix-4 fallback
- **Inner stages:** Warp shuffles (__shfl_xor_sync) for stride < 32
- **Middle stages:** XOR-swizzled shared memory (reuse GEMM infrastructure)
- **Bank conflicts:** XOR swizzle (not padding)
- **Target sizes:** N = 64 to 8192 (single-pass), 16384-32768 (two-pass)
- **Win condition:** Fused operations (FFT+op+IFFT); standalone FFT is very hard to beat cuFFT
- **Bar:** cuFFTDx for fused, cuFFT for standalone
- **Expected speedup:** 1.5x-3x for fused convolution, ~0% for standalone FP32 FFT

### Path B: Tensor Core (BF16 Precision) — Where Allowed
- **Algorithm:** Radix-16 decomposition (matching mma.sync m16n8k16) or Monarch decomposition
- **Inner stages:** mma.sync BF16 matrix multiply for DFT matrix application
- **Twiddle factors:** Applied in-register via fragment-level element manipulation
- **Outer stages:** Warp shuffles for small radixes, shared memory for inter-warp
- **Target sizes:** N = 256 to 65536
- **Win condition:** Applications tolerating BF16 precision (~3 decimal digits)
- **Bar:** tcFFT achieved 1.1x-3.24x over cuFFT; our mma.sync should match or exceed
- **Expected speedup:** 1.5x-3x over cuFFT for BF16-tolerant workloads

### Path C: Fused FFT Convolution — The Biggest Win
- **Pattern:** FFT → pointwise multiply → IFFT in single kernel
- **Implementation:** Overlap-and-save with CT-DIF (no bit-reversal needed)
- **FFT in shared memory:** No global memory intermediates
- **Advanced:** TurboFNO-style FFT-GEMM-IFFT fusion for neural operator workloads
- **Bar:** cuFFTDx fused convolution (45%-3x over cuFFT)
- **Expected speedup:** 2x-4x over separate cuFFT+pointwise+cuFFT calls

### Recommendation
Start with **Path C** (fused convolution) — it has the largest performance gap
and the clearest win. Use **Path A** internals (Stockham + radix-8 + XOR swizzle +
warp shuffle) for the FFT/IFFT implementation within the fused kernel. Add
**Path B** (tensor core BF16) as an optional precision mode for ML workloads.
