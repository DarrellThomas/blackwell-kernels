# FlashSigmoid: Sigmoid Attention as Softmax Alternative

**Source:** https://arxiv.org/abs/2409.04431 (ICLR 2025)
**Code:** https://github.com/apple/ml-sigmoid-attention (CUDA + C++)
**Relevant to:** attention worker (both BF16 and FP8 kernels)
**Worker's current problem:** Softmax creates a sequential dependency between QK^T and PV phases. math_pipe_throttle ~48% is the dominant stall, with softmax's exp2f + reduction + rescaling consuming ~328 non-MMA instructions between MMA phases. Register pressure from tracking row-max (m) and row-sum (l) is tight (165/170.7 regs).

## What This Is

Apple's FlashSigmoid replaces softmax with sigmoid in the attention computation,
achieving 17% inference kernel speedup over FlashAttention-2 on H100 GPUs. Published
at ICLR 2025 with open-source CUDA implementation. The key advantage: sigmoid is an
**element-wise** operation, eliminating the **row-wise reduction** that makes softmax
sequential.

## Why It Matters for Us

Our attention kernel's fundamental bottleneck is the softmax phase between QK^T and PV:
- ~328 non-MMA instructions (exp2f, shuffle reductions, rescaling)
- math_pipe_throttle 48% because tensor cores starve during softmax
- Compiler cannot overlap softmax with MMA across phase boundaries
- Row-max (m) and row-sum (l) tracking consume registers and add dependency chains

Sigmoid attention eliminates ALL of this:
- No row-max tracking (m) -- removed entirely
- No row-sum tracking (l) -- removed entirely
- No rescaling of previous blocks' partial sums
- No warp shuffle reductions for row-wise normalization
- Element-wise sigmoid is embarrassingly parallel

This is the most promising algorithmic alternative to softmax for our kernel architecture.

## Key Technique

### Computational Difference

**Softmax attention (current):**
```
For each KV block j:
  S_j = Q * K_j^T / sqrt(d)
  m_new = max(m_old, rowmax(S_j))           // row-wise reduction
  P_j = exp(S_j - m_new)                     // requires m_new from reduction
  l_new = exp(m_old - m_new) * l_old + rowsum(P_j)  // another reduction
  O = exp(m_old - m_new) * O + P_j * V_j     // rescale previous output
  m_old = m_new; l_old = l_new
Final: O = O / l_old                          // normalize
```

**Sigmoid attention (FlashSigmoid):**
```
For each KV block j:
  S_j = sigmoid(Q * K_j^T / sqrt(d) + b)    // element-wise, NO reduction
  O = O + S_j * V_j                          // simple accumulation, NO rescaling
```

### What Gets Eliminated

| Component | Softmax (current) | Sigmoid |
|-----------|-------------------|---------|
| Row-max m tracking | 2+ registers, shuffle reductions | **ELIMINATED** |
| Row-sum l tracking | 2+ registers, shuffle reductions | **ELIMINATED** |
| exp2f (MUFU.EX2) | ~64 per KV block (dominant ALU) | **ELIMINATED** |
| Rescaling previous O | 4 FMA per accumulator register | **ELIMINATED** |
| Warp shuffle reductions | 4-8 __shfl_xor per KV block | **ELIMINATED** |
| Normalization | Final division by l | Simple 1/n scaling (constant) |

### The Bias Term

Sigmoid attention uses `b = -log(n)` where n is sequence length, applied as an additive
bias to the attention logits. This ensures stable attention norms as sequence length grows.
Implementation: a single constant added to each S element, trivially cheap.

### Training Stability

The paper identifies that sigmoid attention requires stabilization of attention norms
during early training. Two techniques are needed:
- **LayerScale** (learnable per-channel scaling of attention output)
- **QK normalization** (RMSNorm on Q and K before computing attention)

These are training concerns, NOT inference kernel concerns. For inference, sigmoid
attention is a drop-in replacement once the model is trained with it.

### Register Pressure Reduction

Our current kernel uses 165 registers with only 5 spare. Sigmoid eliminates:
- `m_old` (float, ~1 register per accumulator row)
- `l_old` (float, ~1 register per accumulator row)
- Temporary registers for exp2f intermediate values
- Rescaling temporaries

Conservative estimate: **4-8 registers freed**, which could enable either more
accumulator state or a 4th block/SM (requires getting below ~128 registers).

### Performance Expectations on sm_120

The 17% speedup was measured on H100. On our sm_120 kernel:
- BF16 softmax overhead is ~328 instructions / ~48% of stalls
- Sigmoid replaces this with ~64 sigmoid instructions (MUFU.RCP + MUL + ADD per element)
- Net instruction savings: ~200+ per KV block
- The MMA:non-MMA ratio improves dramatically
- math_pipe_throttle should drop from 48% to much lower
- Estimated speedup: **15-25%** (more aggressive than H100 because our kernel is
  more softmax-bottlenecked)

For BF16: 69 us -> estimated 55-60 us
For FP8: 52 us -> estimated 40-45 us

With FP8 + sigmoid + ldmatrix reinterpret: potentially **35-40 us** (~3.0x SDPA).

## Implementation Steps

1. **Verify sigmoid MMA integration:** Replace softmax block with element-wise sigmoid.
   Each S element: `s = 1.0f / (1.0f + exp2f(-s * LOG2E + b_log2e))` where
   `b_log2e = -log(n) * LOG2E`. This is 3 instructions (MUFU.EX2, ADD, MUFU.RCP)
   per element.

2. **Remove online correction:** Delete m_old, l_old, all rescaling code, all
   shuffle reductions for row-max/row-sum.

3. **Simplify output:** `O = sum_j(sigmoid(S_j) * V_j)` -- pure accumulation.
   Final normalization: `O *= (1.0f / n)` -- a single constant multiply.

4. **Benchmark and profile:** The kernel should be significantly simpler.

## Caveats

1. **Requires models trained with sigmoid attention.** This is NOT a drop-in replacement
   for existing softmax-trained models. The model must be trained (or fine-tuned) with
   sigmoid attention + LayerScale + QK-norm. This limits applicability to new models
   or fine-tuned models.

2. **Apple's code targets H100 (sm_90).** The CUDA kernels use Hopper-specific features.
   The ALGORITHM is architecture-independent, but the code is not directly portable.
   We'd implement sigmoid attention using our existing mma.sync infrastructure.

3. **Numerical properties differ.** Softmax produces a probability distribution (sums
   to 1). Sigmoid produces values in (0,1) independently. The 1/n normalization is an
   approximation. For inference with a sigmoid-trained model, this is fine. For comparing
   against softmax-trained model outputs, results will differ.

4. **The 17% number is from H100.** Our kernel may see a larger speedup because our
   softmax overhead is proportionally larger (48% of stalls vs H100's more balanced
   pipeline). Or smaller, because our kernel is already highly optimized.

5. **Not yet mainstream.** As of 2026-03, most production models still use softmax.
   However, sigmoid attention is gaining traction after the Apple paper + ICLR 2025
   publication. Future models may adopt it.
