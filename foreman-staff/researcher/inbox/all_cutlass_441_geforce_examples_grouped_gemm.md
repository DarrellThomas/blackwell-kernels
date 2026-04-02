# CUTLASS 4.4.1: GeForce Blackwell Examples + Group GEMM Profiler Support

**Sources:**
- [CUTLASS examples/79_blackwell_geforce_gemm/](https://github.com/NVIDIA/cutlass/tree/main/examples/79_blackwell_geforce_gemm)
- [Group GEMM for GeForce PR #3092](https://github.com/NVIDIA/cutlass/pull/3092) (merged 2026-03-07)
- [CUTLASS v4.4.1 release](https://github.com/NVIDIA/cutlass/releases)
**Relevant to:** gemm worker, attention worker
**Date:** 2026-03-15

---

## What This Is

CUTLASS v4.4.1 (released February 27, 2026) includes new examples specifically for
Blackwell GeForce (sm_120) and DGX Spark (sm_121), plus a recently merged PR adding
Group GEMM profiler support for these architectures.

### New GeForce Examples (examples/79_blackwell_geforce_gemm/)

Four GEMM examples targeting sm_120:

| Example | Input Types | Output | Description |
|---------|------------|--------|-------------|
| 79a | NVFP4 x BF16 | BF16 | Block-scaled FP4 x BF16 GEMM |
| 79b | NVFP4 x NVFP4 | BF16 | Block-scaled FP4 x FP4 GEMM |
| 79c | MXFP8 x MXFP6 | BF16 | Mixed-precision block-scaled GEMM |
| 79d | NVFP4 grouped | BF16 | Grouped GEMM (MoE workloads) |

All use `mma.sync.aligned.block_scale` instructions (NOT standard mma.sync).

### Key Architecture Details

**Block-Scaled MMA (`mma.sync.aligned.block_scale`):**
- 2x throughput vs standard FP8 MMA (for NVFP4: 4x vs FP8)
- Tile config: 128x128x128, Cluster 1x1x1
- Uses `OpClassBlockScaledTensorOp` (different from standard `OpClassTensorOp`)

**GeForce-specific constraints:**
- No multicast TMA (datacenter feature)
- Cluster shape limited to 1x1x1
- No dynamic datatypes
- Requires `-arch=sm_120a` (the "a" suffix enables block_scale)

**Architecture designation `sm_120f`:**
- The profiler PR (#3092) adds `120f` as a valid arch for GeForce examples
- `sm_120f` is a CUTLASS-internal designation that enables GeForce-specific features
- For our hand-written kernels, use `-arch=sm_120a` in nvcc

---

## Why It Matters for Us

### For GEMM Worker

1. **Reference implementation of block-scaled MMA on sm_120.** The `79c` example
   (MXFP8 x MXFP6) is the closest to our FP8 GEMM workload. It shows the correct
   way to handle scale factors, tile sizes, and epilogue for block-scaled GEMM.

2. **cuBLAS baseline may shift.** cuBLAS 13.1/13.2 includes "experimental Grouped
   GEMM API supporting FP8 and MXFP8 on Blackwell." If cuBLAS adopts block-scaled
   GEMM, our standard FP8 baseline comparisons may change.

3. **Wider tile (128x128x128)** -- CUTLASS uses 128x128x128 tiles for block-scaled
   GEMM, much larger than our 64x128x64. This is feasible because:
   - block_scale MMA adds only +4 registers (2 SFA + 2 SFB) per MMA pair
   - With 1 block/SM, smem budget is generous
   - The 128x128 output tile is 4x our current area

### For Attention Worker

1. **Confirms block_scale MMA exists and works on sm_120a.** Our MXFP8 brief already
   covers the detailed instruction format. These CUTLASS examples validate the
   approach at scale.

2. **No attention example exists yet.** CUTLASS does not provide a block-scaled
   attention kernel for sm_120. We would be the first to implement this.

---

## Caveats

1. **CUTLASS examples use the CuTe abstraction layer**, not hand-written PTX. The
   tile scheduling, epilogue fusion, and memory management are handled by CUTLASS
   collective operations. Translating to our hand-written kernel style requires
   understanding the underlying PTX (documented in our MXFP8 brief).

2. **Group GEMM profiler support is infrastructure**, not a new kernel. It allows
   benchmarking existing CUTLASS kernels on GeForce/Spark via the profiler tool.

3. **cuBLAS 13.2 improvements:** "Up to 20% performance speedup on RTX PRO 6000
   for FP8, FP16/BF16, TF32, and INT8." RTX PRO 6000 is also sm_120. This suggests
   cuBLAS has already improved its sm_120 kernels. Our 1.29x FP8 advantage may have
   narrowed. Re-benchmark with CUDA 13.2 cuBLAS before claiming final numbers.
