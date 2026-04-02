# FlashSigmoid: Sigmoid Attention as Softmax Alternative

**Source:** https://arxiv.org/abs/2409.04431 (Apple, Sep 2024), https://github.com/apple/ml-sigmoid-attention
**Relevant to:** attention worker
**Worker's current problem:** math_pipe_throttle ~48% from softmax between QK^T and PV phases. Softmax requires sequential row-wise max tracking + exp2f + sum reduction + rescaling, creating an irreducible dependency chain that prevents MMA/load overlap.

## What This Is

FlashSigmoid replaces softmax attention with element-wise sigmoid, eliminating the
row-wise normalization entirely. The paper proves sigmoid attention is a universal
function approximator and achieves 17% inference kernel speedup over FlashAttention2
on H100 GPUs (18.76% for causal attention).

## Why It Matters for Us

Our attention kernel's #1 bottleneck is the softmax phase between QK^T and PV:
- ~328 non-MMA instructions (exp2f, shuffle reductions, rescaling)
- math_pipe_throttle 48% from these sequential operations
- The compiler cannot overlap softmax with MMA because of data dependencies
- Online softmax requires: max tracking → exp2f → sum → normalize → scale output

Sigmoid attention replaces ALL of this with element-wise `sigmoid(x + b)`:
- No row-wise max tracking needed
- No row-wise sum needed
- No output rescaling between KV blocks needed (this is huge!)
- Each attention weight is independently computed

**The output rescaling elimination is the biggest win.** In online softmax flash
attention, when processing KV block k+1, the previous output must be rescaled by
`m_old/m_new` to account for the updated max. This creates a serial dependency
across KV blocks. Sigmoid has no such dependency — partial outputs simply accumulate.

## Key Technique

### Forward Pass Formula

```
SigmoidAttn(Q, K, V) = σ(QK^T / √d + b) · V
where σ(x) = 1 / (1 + exp(-x))    (element-wise sigmoid)
      b = -log(n)                   (sequence-length-dependent bias)
```

### Online Computation (Flash-style tiling)

For each KV block j:
```
S_j = Q_tile · K_j^T / √d + b          // QK^T + bias (standard)
P_j = sigmoid(S_j)                       // Element-wise! No row-wise reduction!
O += P_j · V_j                           // Simple accumulation, no rescaling!
```

Compare with softmax flash attention:
```
S_j = Q_tile · K_j^T / √d
m_new = max(m_old, rowmax(S_j))          // ← row-wise reduction (ELIMINATED)
P_j = exp(S_j - m_new)                   // ← exp with shift (REPLACED by sigmoid)
l_new = l_old * exp(m_old-m_new) + rowsum(P_j)  // ← row-wise sum (ELIMINATED)
O = O * (l_old/l_new) + P_j · V_j       // ← rescaling (ELIMINATED)
```

### GPU Implementation

The sigmoid can be computed as: `sigmoid(x) = 0.5 * (1 + tanh(0.5 * x))`
On sm_120, this maps to MUFU.TANH (hardware instruction), which should be faster
than the exp2f + reduction chain. Alternatively: `sigmoid(x) = 1.0 / (1.0 + exp(-x))`
using MUFU.EX2 + RCP.

**Register pressure improvement:** No need to maintain `m_i` (row max), `l_i` (row sum),
or the `exp(m_old - m_new)` correction factors across KV blocks. This frees ~8-16
registers per warp.

### What Gets Eliminated Per KV Block

| Softmax operation | Instructions (approx) | Sigmoid equivalent |
|-------------------|-----------------------|-------------------|
| Row max via shuffle | ~16 (8 exp2f + 8 max) | **ZERO** |
| exp2f(S - m_new) | ~32 (MUFU.EX2) | sigmoid: ~32 (MUFU.TANH or RCP+EX2) |
| Row sum via shuffle | ~16 | **ZERO** |
| Output rescaling (O *= l_old/l_new) | ~32 (MUL per accumulator) | **ZERO** |
| l update (l = l * exp(m-m') + sum) | ~8 | **ZERO** |
| **Total** | **~104 non-MMA instructions** | **~32 instructions** |

Net savings: ~72 instructions per KV block. For N=2048, D=64 with BKV=64, that's
32 KV blocks × 72 = ~2304 fewer instructions. At the attention kernel's IPC, this
could save ~5-10 μs.

## Performance Numbers (from paper)

| Config | FlashAttention2 | FlashSigmoid | Speedup |
|--------|-----------------|--------------|---------|
| Self-attention inference (H100) | baseline | — | **17.39%** |
| Causal attention inference (H100) | baseline | — | **18.76%** |
| Training forward+backward (H100) | baseline | — | **6.53%** |
| End-to-end inference | baseline | — | **~8%** |

## Caveats

1. **Model must be trained with sigmoid attention.** You cannot swap sigmoid into
   a model trained with softmax — the attention weight distributions differ. This
   limits applicability to new models or fine-tuned models.

2. **Requires bias term `b = -log(n)`.** Without this, sigmoid attention outputs
   grow unboundedly with sequence length. The bias ensures convergence.

3. **LayerScale or QK normalization needed for training stability.** The paper
   identifies these as essential for stable training with sigmoid attention.

4. **The Apple implementation targets H100 (sm_90) with wgmma.** We'd need to
   rewrite for sm_120's mma.sync. The algorithmic change is portable but the
   kernel implementation is not.

5. **FP8 compatibility unknown.** The paper uses BF16/FP16. Sigmoid's range [0,1]
   is well-suited to FP8 (no exp overflow risk), but hasn't been tested.

6. **The paper's speedup is on H100 with different memory hierarchy.** Our sm_120
   may see different relative gains due to different compute/memory balance.

## Recommendation

This is a **high-value experiment** if the worker is willing to modify the attention
formula. The implementation is simpler than softmax flash attention (no online correction),
and the potential speedup addresses the kernel's primary bottleneck (softmax overhead).

Start with: modify the existing BF16 kernel to use sigmoid instead of online softmax.
The inner loop becomes much simpler — no max tracking, no sum tracking, no rescaling.
Benchmark against the softmax kernel on the same inputs (correctness won't match
a softmax model, but timing is valid).
