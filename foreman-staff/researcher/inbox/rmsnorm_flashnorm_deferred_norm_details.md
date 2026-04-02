# FlashNorm: Deferred Normalization Implementation Details

**Sources:**
- [FlashNorm: fast normalization for LLMs (arxiv 2407.09577, updated June 2025)](https://arxiv.org/abs/2407.09577)
- [FlashNorm HTML (v2)](https://arxiv.org/html/2407.09577v2)
- [Hugging Face paper page](https://huggingface.co/papers/2407.09577)

**Relevant to:** rmsnorm worker
**Worker's current problem:** Forward at 4.1 us floor. FlashNorm eliminates the standalone kernel entirely.
**Supplements:** `rmsnorm_fusion_strategies.md` (Section 1 already describes FlashNorm at a high level)

---

## What's New Here

The existing `rmsnorm_fusion_strategies.md` describes FlashNorm's weight-absorption trick. This brief adds:

1. **The backward pass for FlashNorm** -- how gradients flow when normalization is deferred
2. **The parallel execution model** -- how RMS computation and GEMM overlap on GPU
3. **Compatibility requirements** -- which model architectures this works with

---

## FlashNorm Backward Pass

Standard RMSNorm backward: `dx = rstd * (wdy - x_hat * (1/N) * dot(wdy, x_hat))`

FlashNorm changes the computation graph. Instead of:
```
x -> RMSNorm(x) -> y_norm -> Linear(y_norm) -> output
```

FlashNorm does:
```
x -> Linear'(x) -> z     (in parallel with:)
x -> RMS(x) -> rms_scalar
z / rms_scalar -> output
```

Where `Linear'` uses modified weights `W' = diag(gamma) @ W`.

### Backward through FlashNorm:

Given `output = z / rms_scalar` where `z = x @ W'` and `rms_scalar = sqrt(mean(x^2) + eps)`:

```
d_z = d_output / rms_scalar                    # gradient through the division
d_rms = -sum(d_output * z) / rms_scalar^2      # gradient through rms_scalar

d_x_from_linear = d_z @ W'^T                   # standard linear backward
d_x_from_rms = d_rms * x / (N * rms_scalar)    # gradient through RMS computation

d_x = d_x_from_linear + d_x_from_rms           # total gradient
d_W' = x^T @ d_z                               # weight gradient
```

**Key insight:** The backward through the linear layer (the big GEMM: `d_z @ W'^T`) and the backward through the RMS scalar (a small reduction + scale) are **independent** and can execute in parallel, just like the forward pass. The only serial dependency is the final addition `d_x = d_x_from_linear + d_x_from_rms`.

This means FlashNorm's backward is also faster than standard RMSNorm backward, because the normalization backward (a reduction) overlaps with the linear backward (a GEMM).

---

## Parallel Execution Model on GPU

The parallelism in FlashNorm works because:

1. **Forward:** `x @ W'` (GEMM, compute-bound) runs simultaneously with `sqrt(mean(x^2))` (reduction, memory-bound). They use different functional units.

2. **On sm_120:** The GEMM uses tensor cores (mma.sync). The RMS reduction uses FP32 ALU + shuffle. These are independent pipelines.

3. **Implementation options:**
   - **Option A: Separate kernels, different streams.** Launch GEMM on stream 1, RMS reduction on stream 2. Requires multi-stream setup but is the simplest.
   - **Option B: Fused kernel.** Warp specialization -- some warps run the GEMM tiles, one warp runs the RMS reduction. More complex but eliminates kernel launch overhead.
   - **Option C: RMS as GEMM prologue.** Compute RMS during the GEMM's global memory load phase (the loads are the bottleneck anyway). Each CTA contributes partial RMS from the tiles it loads, then a final reduction across CTAs.

For our worker's immediate goal (standalone backward), this fusion is not needed yet. But understanding the parallel structure helps design the backward kernel correctly.

---

## Model Compatibility

FlashNorm requires:
- **Bias-free linear layers:** The weight absorption `W' = diag(gamma) @ W` only works if there's no bias. With bias: `y = x @ W + b`, and `RMSNorm(x) @ W + b != (x @ W') / rms + b` in general. Actually this IS correct: `RMSNorm(x) @ W + b = (x * gamma / rms) @ W + b = (x @ diag(gamma) @ W) / rms + b = (x @ W') / rms + b`. So bias-free is NOT strictly required -- the bias just passes through unchanged.

  **Correction:** FlashNorm works even with biased linear layers. The division by rms_scalar only applies to the `x @ W'` term, not the bias.

- **All modern LLMs qualify:** Llama, Mistral, GPT-NeoX, Gemma, Qwen, DeepSeek all use RMSNorm + bias-free linears (or linears where bias can be handled separately).

- **Pre-norm architecture:** FlashNorm assumes `RMSNorm -> Linear` ordering (pre-norm). Post-norm (`Linear -> RMSNorm`) has a different structure and FlashNorm doesn't apply.

---

## Caveats

- FlashNorm modifies the weight matrices at initialization. This is a model-level change, not just a kernel-level optimization. The worker would need to provide utilities for weight conversion.
- The deferred division `output = z / rms_scalar` adds one division per output element. For large output dimensions, this is negligible compared to the GEMM. For small outputs, it's measurable.
- FlashNorm's accuracy is mathematically identical to standard RMSNorm. No approximation is involved.
- The paper (updated June 2025) does not include CUDA kernel source. Implementation is left to the reader. The algorithm is simple enough that the kernel is straightforward.
