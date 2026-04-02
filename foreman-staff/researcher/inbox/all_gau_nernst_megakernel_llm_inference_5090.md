# gau-nernst: Megakernel LLM Inference on RTX 5090 (820 tok/s)

**Source:** https://github.com/gau-nernst/learn-cuda/tree/main/12_megakernel
**Relevant to:** all workers (architecture context), fused-mlp worker (fusion approach)
**Worker's current problem:** fused-mlp at 1.22x cuBLAS (epilogue-fused). Full fusion failed.
**Date:** 2026-03-15

---

## What This Is

gau-nernst's latest project (February-March 2026) implements end-to-end LLM
inference megakernels targeting Qwen3 models (0.6B, 4B params) on RTX 5090. The
approach fuses entire model layers (RMSnorm + projections + attention + MLP) into
single kernel launches, achieving **820 tokens/second** on Qwen3-0.6B decode --
outperforming vLLM's 620 tok/s by 32%.

---

## Why It Matters for Us

### Validation of Fused MLP Architecture

The project confirms that **epilogue-level fusion is the right approach** for MLP
on consumer GPUs. The megakernel fuses:
- RMS normalization into the first GEMM's prologue
- Gate activation (SiLU) into the up-projection's epilogue
- Element-wise multiply of gate and up outputs

This is exactly the Phase 1 approach our fused-mlp worker implemented successfully.
Full fusion (our Phase 2, which failed) is not attempted -- validating our finding
that full fusion has O(D_out/BLOCK_N) redundancy problems.

### RTX 5090 Decode Performance Data Points

**MLP kernel (M=1, single token decode):**

| Implementation | Time (us) | TFLOPS | GB/s |
|---------------|-----------|--------|------|
| HF Eager | 50.95 | 0.37 | 370.73 |
| torch.compile | 34.36 | 0.55 | 549.49 |
| Triton v2 | 18.28 | 1.03 | 1033.66 |

**Attention kernel (KV size=128):**

| Implementation | Time (us) | GB/s |
|---------------|-----------|------|
| HF Eager | 118.25 | 110.91 |
| Triton v1 | 22.70 | 577.97 |

**Key insight:** At M=1 (decode), MLP is memory-bandwidth bound (1034 GB/s vs
1792 GB/s theoretical). This matches our observation that batch-1 GEMM on RTX 5090
is bandwidth-limited, not compute-limited.

### Triton-to-CUDA Progression

The author is progressing from Triton implementations to CUDA C++, suggesting that
Triton's performance ceiling on sm_120 isn't satisfactory for the last mile of
optimization. This matches our experience.

---

## Key Techniques

### 1. Fused Output Norm (March 5)
RMS normalization of the final output is fused into the last attention layer's
epilogue. This eliminates a separate kernel launch and global memory round-trip
for normalization.

**Relevance to rmsnorm worker:** This is exactly the RMSnorm fusion approach our
rmsnorm worker is exploring. The megakernel validates it works in practice.

### 2. Full-Model Triton Kernel (March 5)
A single Triton kernel encompassing the entire model forward pass. At small batch
sizes, this reduces kernel launch overhead significantly.

### 3. vLLM as Reference Baseline
The project uses vLLM as the production reference, achieving 1.32x vLLM's throughput
at short contexts. This provides a higher bar than torch.compile.

---

## Caveats

1. **Small models only (0.6B, 4B).** Performance characteristics may differ at
   larger model sizes where compute-to-memory ratios change.

2. **Decode only (M=1).** Not benchmarked for prefill (large batch) which is
   compute-bound.

3. **Triton implementations.** The CUDA C++ versions are still WIP. Final
   performance numbers may improve.

4. **No FP8.** All implementations use BF16. No quantized inference attempted.

---

## Recommendation

**Low priority for direct adoption, high value for context.** The project validates
our architectural decisions:
- Epilogue fusion is the right MLP approach (not full fusion)
- RMSnorm fusion into GEMM epilogue is feasible and valuable
- M=1 decode is bandwidth-bound on RTX 5090
- vLLM at 620 tok/s is a reasonable production baseline

For the fused-mlp worker, this confirms that Phase 1 (epilogue-fused, 1.22x cuBLAS)
is the correct ceiling and further improvement should come from FP8 or quantization,
not deeper fusion.
