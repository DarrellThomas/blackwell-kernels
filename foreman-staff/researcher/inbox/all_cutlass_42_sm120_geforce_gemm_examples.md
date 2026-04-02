# CUTLASS 4.2+ Official sm_120 GeForce GEMM Examples

**Sources:**
- [CUTLASS Example 79: Blackwell GeForce GEMM](https://github.com/NVIDIA/cutlass/blob/main/examples/79_blackwell_geforce_gemm/)
- [CUTLASS Changelog](https://docs.nvidia.com/cutlass/4.3.2/CHANGELOG.html)
- [CUTLASS sm_120 Block-Scaled MMA Issue](https://github.com/NVIDIA/cutlass/issues/2820)
- [CUTLASS Python DSL sm_120 FP4 Issue](https://github.com/NVIDIA/cutlass/issues/2800)

**Relevant to:** GEMM worker, attention worker, all workers using FP8
**Date:** 2026-03-14

---

## What This Is

CUTLASS 4.2+ now has official example kernels for consumer Blackwell (sm_120/sm_121)
in `examples/79_blackwell_geforce_gemm/`. These are reference implementations of
block-scaled narrow-precision GEMMs on the exact hardware we use.

## Example Variants

| Example | Input A | Input B | Output | Scale Type |
|---------|---------|---------|--------|------------|
| 79a | NVFP4 | NVFP4 | BF16 | Block-scaled |
| 79b | NVFP4 | NVFP4 | NVFP4 | Block-scaled + SF generation |
| 79c | Mixed MXFP8/MXFP6 | Mixed MXFP8/MXFP6 | BF16 | Block-scaled |

## Key Technical Details

**Programming model:** These use `mma.sync.aligned.kind::mxf8f6f4.block_scale`
-- the same extended mma.sync ISA (NOT tcgen05). This confirms the block-scaled
MMA path on sm_120a is the intended approach for narrow-precision GEMM on GeForce.

**Kernel architecture:** Warp-specialized persistent kernel with cooperative and
ping-pong scheduling. Uses SW-controlled dynamic scheduler based on cluster launch
control. This is a more advanced design than our current GEMM kernel.

**Throughput claim:** MXFP8 MMA has 2x throughput compared to Ada (sm_89) FP8 MMA.
This means `m16n8k32` with block_scale should be 2x the non-block-scaled FP8 MMA
on the same hardware.

**sm_120 vs sm_121:** Both are supported. sm_121 is DGX Spark (GB10).

## What's Still Broken

- **Python DSL (CuTe):** `BlockScaledMmaOp` restricts FP4 to sm_100a only. sm_120
  is blocked in the Python path. Only the C++ API works (issue #2800).
- **Runtime assertions:** Some users hit assertion failures with block-scaled MMA
  on sm_120 in CUTLASS 4.2 (issue #2820). May be fixed in 4.3+.

## Relevance for Our Workers

**GEMM worker:** The Example 79c (MXFP8/MXFP6 mixed) is directly relevant. Our
FP8 GEMM at 1.34x cuBLAS uses regular `mma.sync.m16n8k32.e4m3` -- switching to
block-scaled MXFP8 could yield 2x tensor core throughput. The CUTLASS example
provides a working reference implementation.

**Attention worker:** Less directly applicable (attention needs dynamic quantization
of KV data), but the block-scaled MMA infrastructure could help if pursuing MXFP8
attention paths.

**Key decision:** Do we want to pursue block-scaled MXFP8 GEMM? It doubles tensor
core throughput but adds scale factor management overhead. The CUTLASS examples
show the canonical implementation pattern.

## Caveats

- Our current custom kernels are hand-tuned PTX. Adopting CUTLASS patterns means
  either using CUTLASS directly or adapting their patterns into our framework.
- The warp-specialized persistent kernel design in Example 79 is significantly
  more complex than our current approach.
- Block-scaled MMA requires `compute_120a` compilation target (not plain `compute_120`).
