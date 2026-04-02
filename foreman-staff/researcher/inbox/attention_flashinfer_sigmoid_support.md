# FlashInfer 0.2: Customizable Sigmoid Attention Support

**Source:** https://flashinfer.ai/2024/12/16/flashinfer-v02-release.html
**Relevant to:** attention worker
**Worker's current problem:** math_pipe_throttle ~48% from softmax. FlashSigmoid eliminates row-wise normalization but requires custom kernel implementation. FlashInfer provides a reference.

## What This Is

FlashInfer 0.2 (Dec 2024) introduced a modularized attention template where custom attention variants (including sigmoid) can be implemented by specifying functors. This provides a **reference implementation** of sigmoid attention in a production CUDA framework — useful for understanding the implementation pattern even though FlashInfer targets Hopper/sm_90.

## Why It Matters for Us

The existing FlashSigmoid brief describes the algorithm but doesn't provide implementation references beyond the Apple paper (which targets H100 WGMMA). FlashInfer's modular approach shows how sigmoid attention slots into a flash-attention-style kernel:

1. **No LogitsPostHook needed** — sigmoid is applied after QK^T dot products, replacing softmax
2. **No online correction** — the accumulator is simple addition: `O += sigmoid(S) * V`
3. **No max/sum tracking registers** — frees ~8-16 registers per warp

## Key Implementation Detail

FlashInfer's attention variant API:
```cpp
struct SigmoidAttentionVariant {
    // Replace softmax with sigmoid
    static __device__ float LogitsTransform(float score, ...) {
        return 1.0f / (1.0f + expf(-score - log_n_bias));  // sigmoid + bias
    }

    // No max tracking, no sum tracking
    // Output accumulation is just: O += P * V (no rescaling)
};
```

The key simplification vs softmax:
- **Softmax inner loop:** compute S → update max → exp(S-max) → update sum → rescale O → accumulate P*V
- **Sigmoid inner loop:** compute S → sigmoid(S+bias) → accumulate P*V

This eliminates the sequential dependency chain that causes math_pipe_throttle.

### GPU-Optimized Sigmoid Computation

The FlashSigmoid paper recommends: `sigmoid(x) = 0.5 * (1 + tanh(0.5 * x))`

On sm_120, this maps to:
```
MUL.F32  tmp, x, 0.5       // 1 instruction
MUFU.TANH tmp, tmp          // 1 hardware instruction (fast path)
FMA.F32  result, tmp, 0.5, 0.5  // 1 instruction
```
Total: 3 instructions per element.

Compare with current exp2f softmax path:
```
MUFU.EX2 + max reduction + sum reduction + rescale = many more instructions
```

### Bias Term

The sequence-length-dependent bias `b = -log(n)` is computed once on host and passed as a kernel parameter. It's added to each QK^T score before sigmoid. This prevents attention weight explosion for long sequences.

For our kernel: `scale = rsqrt(d) * LOG2E` already folds into Q scaling. The sigmoid bias can be similarly pre-computed and added during the QK^T accumulation or after.

## What to Take from This

1. **The inner loop simplification is real and substantial.** FlashInfer's modular design confirms that sigmoid attention is architecturally simpler — it's literally "remove the max/sum tracking and rescaling code."

2. **FP8 compatibility is natural for sigmoid.** Sigmoid output range is [0,1], which is perfectly representable in FP8 e4m3 (max value 448, so sigmoid values lose no dynamic range). No exp overflow risk. The P→FP8 conversion for PV MMA is simpler.

3. **The bias is a single float parameter.** No per-head or per-token complexity.

4. **Training compatibility requires retraining.** This is an inference-time optimization only if the model was trained with sigmoid attention. Cannot swap into a softmax-trained model.

## Caveats

1. **FlashInfer targets sm_90 (Hopper) with WGMMA.** The kernel template is not directly portable. But the algorithmic pattern (how sigmoid replaces softmax in the tiled computation) is architecture-independent.

2. **Sigmoid attention may have different convergence properties.** The Apple paper shows it works for training but may need LayerScale or QK normalization for stability. For benchmarking kernel speed, this doesn't matter — just use random inputs.

3. **The 17% speedup (H100) may differ on sm_120.** Our kernel's bottleneck distribution (math_pipe_throttle 48%) suggests potentially larger gains since we're more bottlenecked by softmax than H100 kernels are.

4. **Already covered in existing brief.** This supplements `attention_flash_sigmoid_alternative.md` with the FlashInfer implementation reference and GPU-optimized sigmoid computation details (tanh formulation = 3 instructions).
