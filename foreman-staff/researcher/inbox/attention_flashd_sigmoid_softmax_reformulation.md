# FLASH-D: Sigmoid Reformulation of Softmax (Mathematically Equivalent)

**Source:** https://arxiv.org/abs/2505.14201 | https://arxiv.org/html/2505.14201v1
**Relevant to:** attention worker
**Worker's current problem:** math_pipe_throttle 48% from softmax between QK^T and PV phases. Online softmax requires sequential max tracking, exp, sum, and output rescaling.

## What This Is

FLASH-D (May 2025) reformulates the standard FlashAttention softmax computation
using sigmoid functions. Unlike FlashSigmoid (which changes the attention mechanism),
FLASH-D produces **mathematically identical output** to standard softmax attention.
It's a computational trick, not a different model.

## Why It Matters (and Why It Probably Doesn't Help Us Much)

### The Reformulation

Standard FlashAttention output update:
```
o_i = o_{i-1} * (l_{i-1} * exp(m_{i-1} - m_i) / l_i) + v_i * (exp(s_i - m_i) / l_i)
```

FLASH-D reformulation:
```
w_i = sigmoid(s_i - s_{i-1} + ln(w_{i-1}))
o_i = o_{i-1} * (1 - w_i) + v_i * w_i
```

This eliminates:
- Division by l_i (hidden in sigmoid)
- Max tracking (m_i) — sigmoid operates on score DIFFERENCES
- Sum-of-exponents tracking (l_i)

### What It Replaces With

- Sigmoid function: hardware instruction (MUFU.TANH or MUFU.EX2 + RCP)
- Natural logarithm: `ln(w_{i-1})` — MUFU.LG2 + multiply by ln(2)
- Score difference: one subtraction

### Why It Probably Doesn't Help Our Kernel

**The sequential dependency REMAINS.** Each w_i depends on w_{i-1} through
`ln(w_{i-1})`. This means the computation is STILL serial across KV elements
within each block. The shuffle reductions for cross-thread max/sum are eliminated,
but replaced by a serial chain of sigmoid computations.

**The key bottleneck is the serialization, not the specific operations.** Our
attention worker's problem is that MMA cannot overlap with softmax because of
the data dependency chain. FLASH-D changes WHAT operations are in the chain
but doesn't break the chain itself.

### Where It COULD Help

1. **Fewer warp shuffles.** The max and sum reductions require cross-thread
   communication (shuffle instructions). FLASH-D's sigmoid chain is per-thread.
   This could reduce the shuffle stalls (~16 shuffles per KV block eliminated).

2. **Simpler output update.** `o * (1-w) + v * w` is 2 multiplies + 1 add per
   element vs `o * correction + v * weight` which is similar but requires
   computing the correction factor first.

3. **Numerical stability.** No risk of exp overflow — sigmoid is bounded [0,1].
   Eliminates the max subtraction needed for numerical safety.

## ASIC Focus (Not GPU)

FLASH-D was designed for ASIC hardware accelerators, not GPU kernels. The paper
shows 22.8% area and 20.3% power reductions at 28nm. No GPU kernel implementation
exists.

## Comparison with FlashSigmoid

| Aspect | FLASH-D | FlashSigmoid |
|--------|---------|-------------|
| Output | Identical to softmax | Different (sigmoid attention) |
| Model compatibility | Drop-in replacement | Requires retraining |
| Sequential dependency | REMAINS | **ELIMINATED** |
| Shuffle reductions | Eliminated | Eliminated |
| GPU kernel speedup | Unknown (no GPU impl) | 17% on H100 |
| Our bottleneck (math_pipe_throttle) | Probably unchanged | Should improve significantly |

## Recommendation

**Low priority for our attention worker.** FlashSigmoid (already briefed) is the
higher-value experiment because it eliminates the sequential dependency entirely.
FLASH-D is worth noting as a mathematical insight but unlikely to improve our
kernel's bottleneck metric.

If the worker wants softmax-compatible output with reduced complexity, FLASH-D
could be tried, but expect marginal gains at best.
