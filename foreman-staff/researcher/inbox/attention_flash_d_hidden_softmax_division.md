# FLASH-D: Softmax-Equivalent Attention Using Sigmoid (No Retraining)

**Source:** https://arxiv.org/abs/2505.14201
**Relevant to:** attention worker (BF16 and FP8 kernels)
**Worker's current problem:** math_pipe_throttle 48% from softmax's row-wise max/sum tracking and rescaling. Sigmoid attention eliminates this but requires model retraining. FLASH-D may offer a middle path.

## What This Is

FLASH-D ("FlashAttention with Hidden Softmax Division") is a mathematically equivalent reformulation of FlashAttention that hides the softmax division (normalization by sum-of-exponents) inside sigmoid function evaluations. It produces **exact softmax attention output** — bit-for-bit identical — but restructures the computation to eliminate explicit max tracking and sum accumulation.

Published May 2025, accepted at IEEE/ACM ISLPED 2025.

## Why It Matters for Us

This is a potential **third path** beyond our current two options:

| Path | Softmax compatible? | Register savings? | Instruction savings? |
|------|---------------------|-------------------|---------------------|
| Current softmax | Yes | No | No |
| FlashSigmoid (Apple) | NO (needs retraining) | Yes (~8 regs) | Yes (~200+ instr) |
| **FLASH-D** | **YES (exact equivalent)** | **Yes (m, l eliminated)** | **Needs testing** |

FLASH-D's key advantage: it works with **existing softmax-trained models** while still eliminating the row-max and row-sum state variables.

## Key Technique

### The Reformulation

Standard FlashAttention tracks per-row state across KV blocks:
```
m_i = max(m_{i-1}, rowmax(S_i))       // running maximum
l_i = exp(m_{i-1} - m_i) * l_{i-1} + rowsum(exp(S_i - m_i))  // running sum
O_i = exp(m_{i-1} - m_i) * O_{i-1} + exp(S_i - m_i) * V_i    // rescaled accumulation
```

FLASH-D reformulates so that:
1. **Maximum value subtraction is unnecessary** — numerical stability is maintained by ensuring attention score differences stay within sigmoid's active region [-6, 11]
2. **Sum-of-exponents is implicitly embedded** in the sigmoid function evaluations of the weights
3. **Output update saves one multiplier** — the reformulated update uses addition + sigmoid instead of multiplication + exp

### How Softmax Division Hides in Sigmoid

The insight: the ratio `exp(x) / (exp(x) + exp(y))` is equivalent to `sigmoid(x - y)`. FLASH-D exploits this to express the incremental normalization (which FlashAttention does via exp + division) as sigmoid evaluations of score differences. The division by l never appears explicitly — it's absorbed into the sigmoid.

### Hardware Implications (from ASIC results)

On a 28nm ASIC implementation:
- **22.8% area reduction** (fewer functional units needed)
- **20.3% power reduction**
- **Zero performance penalty** — same throughput

The savings come from:
- Removing the max-value logic entirely
- Removing the sum-of-exponents accumulator
- Replacing two multipliers + one adder with one adder + one subtractor + one multiplier in the output update

## Application to Our sm_120 Kernel

### What Gets Eliminated (Same as Sigmoid)

- `m_old` / `m_new` register tracking — GONE
- `l_old` / `l_new` register tracking — GONE
- Row-wise max reduction shuffles — GONE
- exp(m_old - m_new) rescaling multiplications — GONE

### What Replaces It

Instead of `exp(S - m)` followed by division by `l`, the computation uses:
- Sigmoid evaluations: `sigma(score_diff)` — each is MUFU.TANH (or MUFU.RCP + MUFU.EX2)
- Score differences between current block and accumulated state

### Open Question for GPU

The ASIC results show clear wins because the circuit area for max+sum+rescale is larger than the circuit area for sigmoid. On GPU, the question is whether:
- MUFU.TANH (1 special function unit instruction) is faster than the MUFU.EX2 + reduction + rescaling chain
- The register savings from eliminating m/l open up occupancy improvement
- The reduced dependency chain lengths reduce math_pipe_throttle

**The paper does not include GPU benchmarks.** This needs empirical testing.

### Estimated Impact

If FLASH-D's reformulation maps cleanly to sm_120:
- Register savings: ~4-8 (same as sigmoid, since m and l are eliminated)
- Instruction savings: potentially less than pure sigmoid (FLASH-D still computes sigmoid per element to hide the division)
- Dependency chain reduction: significant (no row-wise reductions)
- Compatibility: works with ALL existing models

### Risk

The reformulation may not be faster on GPU. GPUs hide latency through warp-level parallelism, and the softmax chain's latency may already be hidden well enough that restructuring it doesn't help. The ASIC wins come from area/power, not throughput — different optimization target than our kernel's math_pipe_throttle.

## Recommended Investigation

1. Read the full paper for the exact reformulated equations
2. Determine the instruction count of the FLASH-D inner loop on sm_120
3. If promising, prototype by modifying the softmax block in the existing kernel
4. Profile: does math_pipe_throttle decrease?

**If FLASH-D doesn't help on GPU, fall back to full sigmoid attention (which definitely helps but requires retraining).**

## Caveats

1. **Paper targets ASIC/FPGA, not GPU.** The area/power wins may not translate to GPU throughput wins.
2. **No GPU benchmark data available.** Must be tested empirically on sm_120.
3. **The sigmoid evaluations are per-element.** For BKV=64, D=64, that's 4096 sigmoid evaluations per KV block — same count as the current exp2f evaluations, just with different instruction mix.
4. **Numerical stability claims assume bounded score differences.** Need to verify this holds for real attention patterns.
