# vLLM PR #22602: Vectorized RMSNorm with Shared Memory Cache Pattern

**Source:** [vLLM PR #22602 - Vectorize RMSNorm CUDA kernel](https://github.com/vllm-project/vllm/pull/22602)
**Code:** [vllm/csrc/layernorm_kernels.cu](https://github.com/vllm-project/vllm/blob/main/csrc/layernorm_kernels.cu)
**Relevant to:** rmsnorm worker
**Worker's current problem:** Hit 4.1 us pipelining floor with 16 experiments. Standalone optimization exhausted. Next direction is fusion.

## What This Is

A recent (2026) refactor of vLLM's RMSNorm CUDA kernel that introduces three
optimizations our worker may not have tried in combination:

1. **Aligned vector loads** (uint4/float4) for coalesced global memory reads
2. **Shared memory caching of FP16/BF16 input** to avoid a second global read
   during the normalization pass
3. **Unified reduction logic** across fused/unfused/quantized variants

## Why It Matters for Us

The worker's v2 kernel already uses shared memory caching (their best architecture
at 1.35x). But the vLLM PR reveals a specific optimization combination worth
checking against our implementation:

**The key pattern:** Read input as aligned vectors (uint4 = 128-bit = 8 BF16),
cache the raw BF16 values in shared memory during the first pass (while computing
sum-of-squares), then read from shared memory for the second pass (normalization
+ output write). This avoids the second global memory read entirely.

The PR also revealed a **critical bug pattern**: when extending the kernel to
support FP8 quantized output with strided inputs, the original code used `int`
for index variables while the stride was `int64_t`, causing integer overflow
on large tensors. Worth noting if the worker adds FP8 output support to the
RMSNorm kernel.

## Key Technique

```
Pass 1 (reduction):
  - Load x[i] as uint4 (128-bit coalesced read from global)
  - Convert to FP32, accumulate sum-of-squares
  - Store raw BF16 values to shared memory
  - Warp reduction -> block reduction -> rsigma

Pass 2 (normalize + write):
  - Read from shared memory (NOT global) -- avoids second DRAM access
  - Normalize: y = x * gamma * rsigma
  - Write y as vectorized uint4 to global output
```

This is essentially what the worker's v2 already does, but the aligned vector
load/store may make a difference if the worker's current global reads are not
128-bit aligned. The worker should verify their load width.

## Caveats

- The worker already achieves 1.35x with smem caching (v2). The vLLM pattern
  may not yield additional improvement on top of what the worker already has.
- The worker's bottleneck is the 4.1 us pipelining floor, not kernel efficiency.
  No amount of standalone kernel optimization will break this floor.
- The real value is as a reference implementation for the fused kernel: the
  shared memory cache pattern is exactly what's needed for the Phase 1
  (normalize Q in smem) of the attention fusion approach.
- vLLM's kernel source is at `csrc/layernorm_kernels.cu` -- readable reference
  code in production-quality C++ CUDA.
