# MLSys 2026 FlashInfer Contest: Blackwell Kernel Optimization Reference

**Sources:**
- [MLSys 2026 FlashInfer AI Kernel Generation Contest](https://mlsys26.flashinfer.ai/)
- [FlashInfer-Bench starter kit (GitHub)](https://github.com/flashinfer-ai/flashinfer-bench-starter-kit)
- [Contest dataset (HuggingFace)](https://huggingface.co/datasets/flashinfer-ai/mlsys26-contest)

**Relevant to:** all workers (especially linalg, rmsnorm)
**Worker's current problem:** Workers need reference implementations and benchmarks for Blackwell kernel optimization patterns.

## What This Is

The MLSys 2026 FlashInfer contest challenges participants to write optimized
CUDA/Triton kernels for LLM operations on NVIDIA Blackwell B200 GPUs.
Kernels are evaluated on correctness, speed, and win rate against FlashInfer
baselines.

## Why It Matters for Us

The contest provides:

1. **Kernel specifications** for standard LLM operations (attention, GEMM,
   normalization, sampling) that define the API contracts we should match.

2. **FlashInfer baseline implementations** that we can benchmark against
   and study for optimization patterns.

3. **B200 performance data** -- while B200 is datacenter Blackwell (sm_100,
   not sm_120), some optimization patterns transfer. Specifically:
   - Memory hierarchy optimization (shared memory, register pressure)
   - Warp-level programming patterns
   - Mixed-precision kernel design

4. **The starter kit** includes a complete CUDA/Triton workflow for
   benchmarking kernels, which could be adapted for our sm_120 testing.

## Caveats

- **B200 (sm_100) is NOT sm_120.** B200 has TMA, WGMMA, TMEM, and tcgen05
  that sm_120 lacks. Kernel code using these features won't work on RTX 5090.
  Filter for patterns that use `mma.sync` (which both architectures support).
- **Contest deadline and scoring** are likely tied to MLSys 2026 conference
  dates. The value for us is the reference material, not participating.
- The agent-baseline repository may contain useful kernel scaffolding and
  testing infrastructure that could be adapted for our eval.sh framework.
