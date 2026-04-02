# NVIDIA Blog: Tuning Flash Attention for Peak Performance in CUDA Tile

**Source:** [NVIDIA Technical Blog, March 5, 2026](https://developer.nvidia.com/blog/tuning-flash-attention-for-peak-performance-in-nvidia-cuda-tile/)
**Relevant to:** attention worker
**Worker's current problem:** BF16 attention at 1.76x SDPA (69 us), compute-bound with math_pipe_throttle ~48%. FP8 at 2.33x SDPA (52 us).
**Date:** 2026-03-15

---

## What This Is

An official NVIDIA blog post (published March 5, 2026) describing how to tune Flash
Attention using NVIDIA's CUDA Tile (cuTile) programming model. Benchmarked on B200
(datacenter Blackwell, sm_100), reaching 918 TFLOPS at 16K sequence length.

The blog explicitly mentions sm_120 (RTX 50 series) as a supported target architecture,
though all benchmarks are on B200.

---

## Why It Matters for Us

### Five Optimization Techniques (Applicable to Any Flash Attention)

**1. Large Tiles + Fast Math (34-72% improvement on B200)**

Simply increasing tile sizes from 64x64 to 256x128 degraded performance by 18-43% due
to a "compute bottleneck trap." The fix: enabling fast math flags:
- `flush_to_zero=True` -- denormal numbers become zero (avoids slow microcode path)
- `rounding_mode=APPROX` -- skips refinement iterations in exp/log/div

**Our kernel already uses `--use_fast_math`**, which enables both of these. So this
specific optimization is already baked in. But the lesson is important: **tile size
and math precision are coupled variables.** If we ever try larger tiles without fast
math, performance will degrade.

**2. K-Loop Split for Causal Attention (16-32% improvement)**

Instead of checking the mask on every iteration:
- **Fully unmasked blocks:** Skip masking entirely (just compute QK^T + softmax + PV)
- **Diagonal blocks:** Apply the causal mask
- **Fully masked blocks:** Skip entirely

**Our kernel already does this** ("Skip mask optimization for fully unmasked KV blocks"
in agent_state.md). The upstream flash-attention SM120 PR also implements this.

However, the NVIDIA blog reports this as the single largest optimization (31% at long
sequences). If our implementation is suboptimal -- e.g., checking the mask condition
per-element instead of per-block -- there may be room to improve.

**3. ProgramID Remapping (1-2.6% improvement)**

Reverses block processing order for causal attention so that blocks near the diagonal
(which do more work) are processed first. This improves SM load balancing -- early blocks
finish quickly (fully masked), creating stragglers.

**We don't currently do this.** With 170 SMs and typical grid sizes, the effect would be
small (~1-2%), but it's a free optimization: just reorder the grid launch.

**Implementation:** Instead of `block_idx = blockIdx.x`, use:
```
block_idx = total_blocks - 1 - blockIdx.x;
```
This maps the first CTAs to the hardest blocks (near diagonal) and last CTAs to the
easiest blocks (fully masked/fully unmasked).

**4. Autotuning Tile Sizes (10-45% improvement)**

Different sequence lengths benefit from different tile sizes:
- Short sequences (<=2K tokens): 64x64 tiles
- Medium (4K): 128x128 tiles
- Long (8K+): 256x128 tiles

**Our kernel uses dynamic BQ dispatch** (BQ=64 for small grids, BQ=128 for large grids),
which is a form of autotuning. The blog suggests this can be pushed further with
per-config tile selection.

**5. Interdependent Optimizations**

The blog emphasizes that optimizations interact non-linearly: "Large tiles fail without
fast math. K-loop split reduces work, making smaller tiles faster at short sequences
but larger tiles necessary at long sequences."

This matches our experience: attempts to change one variable (BKV, occupancy, PTX
scheduling) in isolation all converged to the same 69 us.

### What's New That We Should Try

**ProgramID remapping** is the only technique from this blog that we haven't tried.
It's trivial to implement and provides 1-2.6% on B200. On our RTX 5090 with 170 SMs,
the effect may be similar.

For the FP8 kernel (52 us), even 2% = ~1 us, bringing it to ~51 us.

### Tile Size Observations

The B200 benchmarks show:
| SeqLen | Best tile | TFLOPS |
|--------|----------|--------|
| 1K | 64x64 | 548 |
| 4K | 128x128 | 790 |
| 8K+ | 256x128 | 918 |

On our RTX 5090 (170 SMs vs B200's ~208 SMs, lower clocks, less power), the optimal
tiles may be smaller. But the principle holds: **tile size should scale with sequence
length.**

---

## Caveats

1. **Benchmarks are on B200 (sm_100), not RTX 5090 (sm_120).** B200 uses tcgen05/WGMMA,
   not mma.sync. The absolute TFLOPS numbers don't translate directly. The optimization
   principles (tile selection, causal split, program remapping) are architecture-agnostic.

2. **CUDA Tile is a high-level abstraction.** The blog uses cuTile Python DSL, not
   hand-written CUDA C++. Some optimizations (like autotuning) are features of the
   framework rather than kernel techniques.

3. **FP16 only.** No FP8 flash attention in this blog post. Our FP8 work remains
   unexplored territory from NVIDIA's public perspective.

4. **The techniques described are well-known** in the flash attention community. The
   value is in the quantified improvement numbers and the B200-specific tuning data,
   not in novel algorithms.
