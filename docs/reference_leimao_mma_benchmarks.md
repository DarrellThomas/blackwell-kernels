# Reference: Lei Mao — Benchmarking NVIDIA Tensor Core MMA Peak Performances

**Source:** https://leimao.github.io/blog/Benchmarking-NVIDIA-Tensor-Core-MMA-Peak-Performances/
**Author:** Lei Mao
**Code:** https://github.com/leimao/CUTLASS-Examples/tree/13c7307/examples/cute_mma_benchmark
**Attribution:** Techniques and findings from this work should be cited when referenced.

---

## Key Findings

Tested on RTX 5080 (sm_120, ~60 SMs). Uses CuTe MMA atoms from CUTLASS.

### Not All MMA Variants Achieve Peak

**Critical:** For FP4, `16x8x32` variants only achieve **26.5% of peak**, while `16x8x64`
variants achieve **103.6% of peak** (4x difference). The K-dimension of the MMA instruction
matters enormously.

### FP4 Dense Results

| MMA Atom | TFLOPS | % Peak |
|----------|--------|--------|
| SM120_16x8x32_TN | 238.5 | 26.5% |
| SM120_16x8x64_TN_VS (ue8m0) | **933.1** | **103.6%** |
| SM120_16x8x64_TN_VS (ue4m3) | **923.1** | **102.5%** |

### Theoretical Peaks (RTX 5080)

| Precision | Accum | Peak TFLOPS |
|-----------|-------|-------------|
| FP4 | FP32 | 900.4 |
| FP8 | FP32 | 225.1 |
| BF16 | FP32 | 112.6 |
| TF32 | — | 56.3 |

### Scaling to RTX 5090

RTX 5090 has ~170 SMs vs 5080's ~60 SMs (2.83x ratio):
- BF16 (FP32 accum) peak: ~112.6 × (170/60) ≈ **319 TFLOPS** ???

**Note:** This conflicts with the commonly cited 209.5 TFLOPS BF16 for RTX 5090.
The discrepancy likely comes from how NVIDIA counts "TFLOPS" — the 209.5 figure
may use a different clock speed or counting methodology. The blog's per-SM numbers
are measured, not theoretical.

**Practical implication:** Our 209.5 TFLOPS target may be conservative. The actual
achievable BF16 MMA throughput could be higher under boost clocks (>100% SOL is
possible, as demonstrated by FP4 results).

## Methodology

- **Dummy MMA kernel** — no memory ops, pure instruction throughput measurement
- Registers initialized with non-zero values (prevents compiler elimination)
- Results fed back as inputs (feedback loop prevents dead code removal)
- `#pragma unroll 1` prevents over-optimization

## Implications for Our Kernels

1. **BF16 mma.sync m16n8k16 should achieve peak if scheduled correctly.**
   The instruction itself is not the bottleneck — it's how we feed it data.

2. **>100% of advertised peak is possible** due to boost clocks exceeding base.
   Don't treat 209.5 TFLOPS as a hard ceiling.

3. **FP8 attention (Phase 3) would deliver ~2x MMA throughput** vs BF16.
   Strong motivation for that phase.

4. **K-dimension variants matter.** If sm_120 exposes larger-K BF16 MMA variants
   (e.g., m16n8k32), they might deliver significantly better throughput — worth
   investigating in CUTLASS CuTe atom definitions.
