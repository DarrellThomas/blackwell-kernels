# FlashAttention-4 Portable Softmax Optimizations for sm_120

**Source:** https://modal.com/blog/reverse-engineer-flash-attention-4, https://arxiv.org/html/2603.05451v1
**Relevant to:** attention worker
**Worker's current problem:** math_pipe_throttle 48% from softmax between QK^T and PV phases. Online softmax requires ~328 non-MMA instructions per KV block (exp2f, shuffle reductions, rescaling). 94% of compiler ceiling reached.
**Date:** 2026-03-14

## What This Is

FlashAttention-4 (Dao et al., 2026) targets datacenter Blackwell (sm_100/tcgen05)
but introduces two algorithmic techniques that are **ISA-independent** and can be
ported to our mma.sync-based sm_120 kernel. These are separate from the warp
specialization and TMA pipeline that require tcgen05.

## Why It Matters for Us

Our BF16 attention kernel's #1 bottleneck is math_pipe_throttle at 48%. The softmax
phase between QK^T and PV has:
- exp2f (hardware SFU) for exponentials
- Warp shuffle reductions for max and sum
- Output rescaling every KV block to maintain numerical stability

FA4 optimizes the second and third of these. Combined, they could reduce the
instruction count in the softmax critical path by 30-50%.

## Key Technique 1: Selective Output Rescaling

### The Problem
In standard online softmax (FlashAttention-2 / our current kernel), when processing
KV block k+1:
```
m_new = max(m_old, rowmax(S_k+1))
O = O * exp2(m_old - m_new)        // RESCALE: always runs
O += exp2(S_k+1 - m_new) * V_k+1
l = l * exp2(m_old - m_new) + rowsum(exp2(S_k+1 - m_new))
```

The rescale `O = O * exp2(m_old - m_new)` runs **every KV block**, even when
`m_old == m_new` (max didn't change). When max is stable, `exp2(0) = 1.0` and
the multiply is a no-op — but the instructions still execute.

### FA4's Fix
FA4 only applies the rescaling correction "when the maximum has changed enough
to impact numerical stability." The paper reports this reduces correction operations
by approximately **10x**.

### Implementation for Our Kernel
```cuda
// Current (every KV block):
float scale = exp2f(m_old - m_new);
O[i] *= scale;                        // Always runs, often scale == 1.0

// FA4-style (conditional):
if (m_old - m_new > THRESHOLD) {      // e.g., THRESHOLD = 0.0f or a small epsilon
    float scale = exp2f(m_old - m_new);
    O[i] *= scale;
    l *= scale;                        // Also rescale denominator
}
```

**Why this works:** In causal attention with typical sequence lengths, the row-wise
maximum is established early and rarely changes for subsequent KV blocks. For our
primary config (B=2, H=8, N=2048, D=64, causal), with BLOCK_KV=64, there are
2048/64 = 32 KV blocks per query row. The maximum typically stabilizes within the
first 2-3 blocks, meaning ~29 out of 32 rescaling operations are multiplies-by-1.0.

**Cost of the branch:** One `setp` + `@p bra` per KV block. If the branch is
well-predicted (nearly always skipped after first few blocks), this costs ~2
instructions vs the ~16+ instructions for the full rescale.

**Savings estimate:** For 32 KV blocks, ~29 * 16 = 464 instructions saved, vs 2*32 =
64 instructions for the branch checks. Net savings: ~400 instructions per query row,
or ~12.5 instructions per KV block average. With 4 warps * 16 query rows * 32 KV
blocks, this could save thousands of instructions per kernel launch.

**Caveats:**
- Must handle the denominator `l` correctly (only rescale when O is rescaled)
- The final normalization `O /= l` at kernel end still happens unconditionally
- Threshold of 0.0f is safest (only skip when max didn't change at all)
- A small epsilon threshold (e.g., 2^-10) could catch more no-ops but risks
  accumulating small errors across many blocks

## Key Technique 2: Software Polynomial exp2 Approximation

### The Problem
Our kernel uses `exp2f()` which maps to the hardware SFU (Special Function Unit).
On sm_120, the SFU throughput is lower than CUDA core throughput, so heavy exp2f
usage can create an SFU bottleneck contributing to math_pipe_throttle.

### FA4's Fix
FA4 replaces hardware exp2 with a cubic polynomial approximation using Horner's
method (3 FMAs):

```cuda
// Hardware SFU (current):
float result = exp2f(x);

// Software cubic approximation (FA4):
// Coefficients fit to minimize error over typical attention score range
float result = ((c3 * x + c2) * x + c1) * x + c0;
```

This uses 3 FMA instructions on CUDA cores instead of 1 SFU instruction. The
advantage is that CUDA core FMA throughput is much higher than SFU throughput on
modern GPUs.

### Applicability to Our Kernel
The softmax in our kernel has two exp2f call sites:
1. `exp2f(m_old - m_new)` for output rescaling — **already addressed by technique 1**
2. `exp2f(S[i] - m_new)` for computing attention weights P

For site 2, the polynomial approximation could be beneficial if our SFU is saturated.
However:

**Pros:**
- 3 FMAs use CUDA core datapath, not SFU
- Compiler can interleave FMAs with other CUDA core operations
- Could reduce SFU queue stalls that contribute to math_pipe_throttle

**Cons:**
- 3 instructions vs 1 (more total instructions even if faster)
- Precision: cubic polynomial has ~2-3 ULP error in [−8, 0] range (typical
  attention scores after subtracting max). This may cause subtle accuracy changes.
- Our kernel already uses exp2f efficiently by folding LOG2E into Q scale
- The compiler may already pipeline SFU operations well

**Verdict:** Try it empirically. If ncu shows significant SFU stalls (check
`pipe_sfu_cycles_active` counter), the polynomial substitution is worth testing.
If SFU utilization is low, this won't help.

### Reference Coefficients
FA4 doesn't publish exact coefficients, but standard cubic Remez approximation
for exp2(x) on [-1, 0] (after range reduction):
```cuda
// exp2(x) ≈ c0 + c1*x + c2*x^2 + c3*x^3 for x in [-1, 0]
// After range reduction: x = n + r, where n = floor(x), r in [-1, 0]
// exp2(x) = exp2(n) * exp2(r), exp2(n) via bit manipulation
const float c0 = 1.0f;
const float c1 = 0.6931472f;   // ln(2)
const float c2 = 0.2402265f;
const float c3 = 0.0558016f;
// Horner: result = ((c3 * r + c2) * r + c1) * r + c0
// Then scale by 2^n via integer add to float exponent bits
```

## Suggested Experiments

### Experiment A: Selective Rescaling Only
1. Add `if (m_old != m_new)` guard around output rescaling block
2. Measure: kernel time, math_pipe_throttle, HMMA utilization
3. Verify: numerical accuracy vs reference (should be bit-exact if threshold=0.0f)
4. Expected: 2-5% speedup from eliminated no-op multiply chains

### Experiment B: Polynomial exp2f
1. Replace exp2f in P computation with cubic polynomial
2. Measure: kernel time, pipe_sfu_cycles, pipe_fma_cycles
3. Verify: output RMSE vs FP32 reference (should be <0.01% additional error)
4. Expected: helpful only if SFU is bottleneck (check ncu first)

### Experiment C: Both Combined
1. Selective rescaling + polynomial exp2f
2. This is the maximum softmax optimization before going to sigmoid attention

## Caveats

1. **FA4 targets sm_100 (tcgen05)** with very different pipeline structure. Only
   the algorithmic techniques above are portable. The warp specialization,
   asynchronous MMA, and TMA pipeline are NOT applicable to sm_120.

2. **Selective rescaling changes branch behavior.** If the compiler currently unrolls
   the KV loop without branches, adding a conditional could prevent unrolling or
   cause divergent execution within a warp. Test with `#pragma unroll` to ensure
   the compiler still unrolls.

3. **These are orthogonal to sigmoid attention.** If the worker switches to sigmoid
   attention (no softmax at all), these techniques become irrelevant. These are
   "make softmax faster" optimizations, not "replace softmax" alternatives.

4. **Compiler ceiling may already account for these.** The compiler at O3 may
   already optimize away multiply-by-1.0 in some cases. Check SASS to verify
   the rescaling instructions actually execute every iteration.
