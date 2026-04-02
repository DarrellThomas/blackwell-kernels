# RedFuser: Cascaded Reduction Fusion (ASPLOS '26)

**Source:** https://github.com/alibaba/redfuser (Apache 2.0)
**Paper:** ASPLOS '26 (22 pages, 13 figures)
**Relevant to:** attention worker (softmax+GEMM fusion), fused-mlp worker (reduction fusion)
**Worker's current problem:** Attention math_pipe_throttle 48% from softmax between MMA phases.
**Date:** 2026-03-15

---

## What This Is

RedFuser is a compiler framework from Alibaba (published at ASPLOS 2026) that
automatically fuses cascaded reduction operations into single kernels. Its key
contribution is a "formal theoretical methodology for analyzing cascaded reductions
which can fuse them into a single loop and introduce an incremental computation form."

The framework achieves 2x-5x speedup over state-of-the-art AI compilers, matching
hand-written kernels.

---

## Why It Matters for Us

### Softmax-GEMM Fusion (Attention Worker)

Our attention worker's #1 bottleneck is `math_pipe_throttle` at 48%, caused by
softmax operations between MMA phases. Softmax is a cascaded reduction:
1. Row-max reduction (for numerical stability)
2. Exponentiation (element-wise, depends on row-max)
3. Row-sum reduction (normalization denominator)
4. Division (element-wise, depends on row-sum)

RedFuser's "incremental computation form" for cascaded reductions is exactly the
kind of optimization that could reduce the softmax overhead between MMA phases.

The paper explicitly lists **Flash Attention** as a completed application in their
roadmap, meaning they have a working fused softmax+GEMM example.

### Planned Applications (from GitHub)

- Flash-attention: **completed**
- Flash-decoding: planned
- MoE routing: planned
- FP8 quantization with GEMM: planned

The FP8 quantization with GEMM entry is particularly relevant to our GEMM and
fused-mlp workers.

---

## Key Technique: Incremental Reduction Form

The core insight: cascaded reductions (like softmax) that traditionally require
multiple passes over data can be algebraically transformed into a single-pass
"incremental" form. This is essentially the theoretical foundation for what
online softmax does, but generalized to arbitrary cascaded reduction patterns.

For attention specifically, this means:
- Computing row-max, exp, row-sum, and normalization in a single pass
- Fusing these reductions with the subsequent PV matmul
- Eliminating intermediate materialization of the attention score matrix

**Our kernel already uses online softmax**, which is an instance of this pattern.
However, RedFuser may reveal additional algebraic transformations for other
reduction patterns in our pipeline.

---

## What to Study

The `python/tvm/redfuser/transform/` directory contains the core transformation
passes. The `example/flash_attention.py` workload shows how cascaded reductions
are identified and fused for attention.

Studying the transformation passes could reveal:
1. Whether our online softmax implementation leaves any fusion opportunities
2. How to better interleave reduction and MMA operations
3. Algebraic transformations we haven't considered for the rescaling step

---

## Caveats

1. **Compiler framework, not a hand-written kernel.** RedFuser generates code via
   TVM, not hand-written CUDA. The generated kernels match hand-written performance
   but may not exceed it.

2. **No sm_120 specifics.** The paper doesn't target specific GPU architectures.
   The transformations are algebraic (architecture-agnostic), but the generated
   code quality depends on TVM's backend for sm_120.

3. **"Matching hand-written kernels" means matching, not beating.** If our kernel
   is already well-optimized, RedFuser's techniques may already be incorporated.

4. **Apache 2.0 license.** Code is freely available for study.

---

## Recommendation

**Low-medium priority for direct use, high priority for study.** The RedFuser
paper provides a rigorous algebraic framework for understanding cascaded reduction
fusion. Even if we don't use TVM, understanding the formal transformation rules
could reveal missed optimization opportunities in our hand-written softmax
implementation.

Specifically, study the `flash_attention.py` example to see if their incremental
form differs from our online softmax implementation in any exploitable way.
