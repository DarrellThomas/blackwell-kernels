# RedFuser Deep Dive: Algebraic Cascaded Reduction Fusion (ASPLOS '26)

**Source:** https://arxiv.org/abs/2603.10026 (full paper read)
**GitHub:** https://github.com/alibaba/redfuser (Apache 2.0)
**Authors:** Alibaba, published at ASPLOS 2026
**Relevant to:** fused-mlp worker, rmsnorm worker, attention worker
**Worker's current problem:** (fused-mlp) Phase 2 full fusion FAILED due to O(D_out/BLOCK_N) redundancy. (rmsnorm) 4.1 us pipelining floor, next step is fusion with attention.
**Date:** 2026-03-15 (deep dive, supersedes earlier overview brief)

---

## What This Is

RedFuser is a two-stage compiler framework that automatically identifies and
fuses cascaded reduction operations (reductions whose inputs depend on prior
reductions). It provides:

1. **Formal algebraic conditions** for when cascaded reductions can be fused
2. **Automatic derivation** of the incremental computation form (online algorithm)
3. **Hardware-aware code generation** via TVM/TileLang

The key insight: cascaded reductions that seem to require multiple passes can be
algebraically transformed into a single-pass incremental form, IF three conditions
hold (decomposability, commutativity, distributivity).

---

## The Three Fusibility Conditions

For a cascaded reduction to be fusible, RedFuser requires:

**1. Decomposability:** The reduction function must factor as:
```
F_i(x[l], d_i) = G_i(x[l]) (x) H_i(d_i)
```
where G_i depends only on the current input and H_i depends only on the
accumulated dependency. This separates "what comes from new data" from
"what comes from prior reductions."

**2. Commutative Monoid:** The operator (S, (x)_i) must be associative,
commutative, and have an identity element.

**3. Distributivity:** The reduction operator must distribute over the
combination operator:
```
(s1 (+) s2) (x) s3 = (s1 (x) s3) (+) (s2 (x) s3)
```

**Why this matters for us:** These are EXACTLY the conditions that make online
softmax work. The paper proves Flash Attention is a special case of their
framework. This means we can check whether OTHER fusion patterns (RMSNorm +
GEMM, activation + GEMM, etc.) satisfy these conditions.

---

## The Incremental Computation Form

When the three conditions hold, the running accumulator can be updated as:

```
d^k[L] = d^k[L-1] * H_i(old_max)^{-1} * H_i(new_max)  (+)  new_contribution * H_i(new_max)
```

For softmax specifically, this becomes online softmax:
```
m_new = max(m_old, max(current_tile))
c_new = c_old * exp(m_old - m_new) + sum(exp(current_tile - m_new))
```

**Our kernel already implements this.** The paper confirms that Flash Attention's
online softmax is an instance of their general framework. No new optimization
here for attention -- we already have the optimal incremental form.

---

## Performance Results

### Flash Attention (MHA)
| Platform | vs FlashAttention2 | vs PyTorch Dynamo | vs TVM |
|----------|-------------------|-------------------|--------|
| A10 (24GB) | **1.09x** | 2.8x | 2.6x |
| H800 (80GB, MLA) | **1.02x FlashMLA** | 2.4x | 8.7x |

### Other Workloads
| Workload | vs Dynamo | vs TVM | Notes |
|----------|-----------|--------|-------|
| MoE routing | 1.7x | 6.6x | Softmax + top-k fusion |
| FP8 Quant+GEMM | 3.4x | 12.1x | Dynamic scaling + quantized GEMM |

### Interpretation

RedFuser **matches but does not beat** hand-written kernels. It generates code
that is 1.02-1.09x of carefully tuned implementations. The 2-5x wins are over
compiler-generated code (Dynamo, TVM) which lack cascaded reduction fusion.

---

## Relevance to Fused MLP Worker

### The Full Fusion Failure (Context)

Our fused-mlp worker tried Phase 2: GEMM1 + activation + GEMM2 in one kernel.
It failed because tiling the output dimensions creates O(D_out/BLOCK_N) redundancy
in GEMM1 computation. At GPT-2 scale, this is 12x redundancy = 7.8x slower.

### Can RedFuser Help?

**No, for the same reason.** The fused-mlp redundancy problem is NOT a cascaded
reduction problem. It's a **data reuse** problem: the intermediate activation
matrix (M x D_ff) is consumed by multiple output tiles, each of which must
either recompute it (redundancy) or communicate it (synchronization).

RedFuser's algebraic framework applies to patterns like:
- softmax(QK^T) * V -- cascaded reduction (max, sum) feeding into GEMM
- FP8 quantize(X) * W -- reduction (absmax) feeding into GEMM

But GEMM1 + activation + GEMM2 is NOT a cascaded reduction. The activation
(ReLU^2) is element-wise, not a reduction. The intermediate is a full matrix,
not a scalar/vector statistic.

**The v2 full fusion failure is fundamental** -- it's not a compiler limitation
that RedFuser could fix. It's an algorithmic incompatibility between GEMM tiling
and intermediate reuse.

### What RedFuser COULD Help With

**FP8 Quant+GEMM fusion** is directly relevant. The paper shows 3.4x over
Dynamo for fusing:
1. Per-token absmax reduction (to compute FP8 scale factor)
2. Quantization (element-wise, depends on scale factor)
3. Quantized GEMM

Our fused-mlp worker's FP8 path already has `long_scoreboard 40%` due to
BF16->FP8 conversion overhead. If we pre-compute the FP8 scale factor and
fuse it with GEMM1, the RedFuser pattern applies. However, this would require
FP8 inputs from the preceding layer, which is an inference pipeline change.

---

## Relevance to RMSNorm Worker

### Standalone RMSNorm

RMSNorm is a single reduction (sum of squares), NOT a cascaded reduction.
RedFuser's framework requires inter-reduction dependencies to apply.

```
y = x * rsqrt(mean(x^2) + eps)  -- one reduction, not cascaded
```

**RedFuser cannot help with standalone RMSNorm optimization.**

### RMSNorm + Attention Fusion

The worker's next step is "fusion with attention." This IS a cascaded pattern:
1. RMSNorm: sum-of-squares reduction -> normalization (element-wise)
2. Attention QK^T: GEMM (depends on normalized input)
3. Softmax: cascaded reduction (max, sum)
4. PV: GEMM (depends on softmax output)

However, RedFuser's decomposability condition requires that the reduction
function factors into input-dependent and dependency-dependent parts. For
RMSNorm feeding into attention:
- RMSNorm output = x * rsqrt(sum(x^2)/N + eps)
- Q = RMSNorm(X) * W_Q
- The sum(x^2) reduction is independent of any prior reduction

This is NOT a cascaded reduction -- RMSNorm's reduction has no inter-dependency
with softmax's reduction. They are sequential but independent reductions.

**RedFuser's framework does not apply to RMSNorm + attention fusion.**

The practical approach to RMSNorm + attention fusion remains:
- Fuse RMSNorm into attention kernel's input loading path
- Compute normalization while loading Q (or K/V) tiles
- This is a standard epilogue/prologue fusion, not a cascaded reduction fusion

---

## Relevance to Attention Worker

### Already Implemented

Our attention kernel already uses online softmax, which the paper confirms is
an instance of their incremental computation form. There is no additional
algebraic optimization to extract.

### What to Study in the Code

The `flash_attention.py` example at `python/tvm/redfuser/example/flash_attention.py`
shows how RedFuser expresses the cascaded reduction for attention. Comparing
their symbolic representation to our implementation could reveal:

1. Whether their rescaling formula differs from ours (unlikely -- there's one
   correct online softmax)
2. How they handle the PV accumulator rescaling when max changes (we use
   exp2f with LOG2E folded into Q scale -- same thing, different notation)
3. Whether they found any algebraic simplification we missed (unlikely given
   our 66 experiments exhausted the space)

**Expected outcome: confirmation that our implementation matches the
theoretically optimal form. No new optimization.**

---

## Hardware Support Gap

RedFuser's code generation targets:
- **Ampere (A10):** cp.async + mma.sync
- **Hopper (H800):** TMA + WGMMA

**sm_120 (Blackwell consumer) is NOT supported.** Our GPU uses mma.sync (like
Ampere) but has Blackwell-specific features (wider FP8 MMA, different L2).
Using RedFuser's Ampere path might work but would miss sm_120 optimizations.

This is a framework limitation, not a technique limitation. The algebraic
transformations are architecture-agnostic. Only the code generation needs
sm_120 support.

---

## Actionable Takeaways

### For fused-mlp worker:
- **Phase 2 full fusion failure is NOT fixable** by RedFuser or any cascaded
  reduction technique. It's a data reuse problem, not a reduction fusion problem.
- **FP8 Quant+GEMM pattern** is worth studying if we move to native FP8 inputs.
  The 3.4x vs Dynamo suggests significant overhead in current quantize-then-GEMM
  pipelines.

### For rmsnorm worker:
- **RedFuser does not apply** to standalone RMSNorm or RMSNorm + attention fusion.
- Continue pursuing prologue fusion (fold normalization into attention input path).

### For attention worker:
- **No new optimization.** Our online softmax is already the incremental form
  that RedFuser derives algebraically.
- Study `flash_attention.py` only if you want theoretical confirmation.

### Overall:
- **Low priority for direct use.** We already implement the key technique (online
  softmax) and our other fusion challenges (MLP, RMSNorm) don't fit the cascaded
  reduction pattern.
- **High value as theoretical validation.** The paper confirms our approach is
  algebraically optimal for the patterns it covers.
