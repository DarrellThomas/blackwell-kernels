# FlashSigmoid: Sigmoid Attention as Softmax Replacement

**Source:** https://arxiv.org/abs/2409.04431 | https://github.com/apple/ml-sigmoid-attention
**Relevant to:** attention worker (main/)
**Worker's current problem:** math_pipe_throttle 48% from softmax between QK^T and PV phases; softmax requires row-wise max reduction, row-wise sum reduction, and rescaling of previous outputs — all sequential scalar operations that starve tensor cores.

## What This Is

Apple's ICLR 2025 paper proposes replacing softmax with element-wise sigmoid in attention: `Attn(Q,K,V) = sigmoid(QK^T/sqrt(d) + b) * V` where `b = -log(n)` (sequence length bias for stability). FlashSigmoid is their hardware-aware CUDA kernel implementation achieving 17% inference speedup over FlashAttention-2 on H100.

## Why It Matters for Us

The attention worker's dominant bottleneck is the **softmax phase between QK^T and PV**. Softmax requires:
- Row-wise max reduction (4 `__shfl_xor_sync` calls)
- Row-wise sum reduction (4 `__shfl_xor_sync` calls)
- Rescaling of ALL previous O accumulator rows when a new max is found (FMA per element)
- Online softmax bookkeeping (m_prev, l_prev tracking across KV blocks)

Sigmoid eliminates ALL of these. It's element-wise: `sigmoid(x) = 0.5 * (1 + tanh(0.5 * x))` maps to a single MUFU.TANH instruction per element (vs exp2f + max + sum + rescale). No cross-element reductions needed. No online bookkeeping. The ~328 non-MMA softmax instructions between QK^T and PV phases would shrink dramatically.

For the attention worker at 94% of compiler ceiling with 48% math_throttle from softmax, this could break through the ceiling by fundamentally reducing the non-MMA instruction count.

## Key Technique

1. **Formula:** `sigmoid(QK^T / sqrt(d) + b) * V` where `b = -log(n)`
2. **GPU optimization:** Use `sigmoid(x) = 0.5 * (1 + tanh(0.5 * x))` — maps to MUFU.TANH in SASS
3. **No online algorithm needed:** Since sigmoid is element-wise, there's no "online softmax" — each P_ij depends only on S_ij, not on the row max/sum
4. **No rescaling:** Previous O accumulator doesn't need rescaling when processing new KV blocks (the dominant source of math_pipe_throttle stalls)
5. **Simpler backward pass:** No need to store row-max/row-sum for recomputation
6. **Bias term `b = -log(n)`** is critical for stability — prevents output explosion as sequence length grows
7. **LayerScale and QK normalization** recommended for stable training

## Caveats

- **Requires model retraining.** Sigmoid attention is NOT a drop-in for an already-trained softmax model. The model must be trained with sigmoid attention from scratch (or fine-tuned). This limits applicability to new models only.
- **FlashSigmoid targets H100** (sm_90, uses Hopper-specific optimizations). The kernel would need to be rewritten for sm_120 mma.sync, but the algorithm transfers directly.
- **The sequential QK^T → PV dependency remains** — you still compute QK^T, then apply sigmoid, then PV. The win is that the sigmoid phase is much simpler (element-wise vs row-wise), not that it's eliminated.
- **Accuracy difference:** Sigmoid attention may have different training dynamics. Apple reports it works well across vision, speech, and language but may need hyperparameter tuning.
- **MUFU.TANH availability on sm_120** should be verified empirically (likely available, but untested in our kernel).
