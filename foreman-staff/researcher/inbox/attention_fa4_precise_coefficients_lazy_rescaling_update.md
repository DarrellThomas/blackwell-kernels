# FA4 Polynomial exp2 -- Precise Coefficients & Lazy Rescaling Details

**Sources:**
- https://arxiv.org/html/2603.05451 (FlashAttention-4 paper, full text)
- https://research.colfax-intl.com/flashattention-4-algorithm-and-kernel-pipelining-co-design-for-asymmetric-hardware-scaling/
- https://modal.com/blog/reverse-engineer-flash-attention-4
**Relevant to:** attention worker
**Worker's current problem:** math_pipe_throttle 48% from softmax. FP8 kernel at 2.33x SDPA but latency-bound (SM 43.8%). Softmax sequential overhead limits both BF16 and FP8 paths.
**Supplements:** attention_fa4_portable_softmax_optimizations.md, attention_flashattention4_portable_algorithmic_innovations.md (existing briefs -- this adds precision data those briefs lacked)

## NEW DATA 1: FA4's Exact Polynomial Coefficients (from the paper)

The FA4 paper (arxiv 2603.05451) provides the EXACT coefficients for the degree-3
Horner polynomial approximation of 2^x on [0, 1):

```
p(x) = p0 + p1*x + p2*x^2 + p3*x^3
p0 = 1.0
p1 = 0.69514614
p2 = 0.22756439
p3 = 0.07711909
```

These were computed using the Sollya software package to minimize relative
approximation error over [0, 1). Evaluated via Horner's method:

```cuda
// 3 FMA instructions:
float result = fmaf(fmaf(fmaf(0.07711909f, r, 0.22756439f), r, 0.69514614f), r, 1.0f);
```

Where r = x - floor(x) is the fractional part after Cody-Waite range reduction.

The existing briefs used approximate coefficients (c3=0.0796, c2=0.2274, c1=0.6931).
These are WRONG -- use the paper's exact values above.

The reconstruction step combines the polynomial result with the integer part:
```cuda
// 2^floor(x) via IEEE 754 exponent bit manipulation
int n = (int)floorf(x);
int result_bits = __float_as_int(poly_result) + (n << 23);
float final = __int_as_float(result_bits);
```

Total instruction count: 3 FMA + 1 FLOOR + 1 INT_ADD + 1 SHIFT + 2 REINTERPRET
= ~7-8 instructions vs 1 MUFU.EX2.

## NEW DATA 2: Precision Analysis by Degree

The FA4 paper provides error measurements that were NOT in the existing briefs:

| Degree | FP32 max relative error | BF16 max relative error | Matches hardware? |
|--------|------------------------|------------------------|-------------------|
| 3      | 8.77 x 10^-5           | 3.90 x 10^-3          | Within 1 BF16 ULP on 99% of inputs |
| 5      | 1.44 x 10^-7           | same as degree-3       | Matches hardware exactly |

**Key insight for us:** For BF16 attention, degree-3 is sufficient because BF16
quantization error (~3.9e-3) dominates the polynomial error. Using degree-5 adds
2 extra FMAs per evaluation with no measurable accuracy benefit for BF16 outputs.

For FP8 attention (our primary focus), the situation is even better: FP8 e4m3 has
maximum representable value of 448 and ~6.25% step size at mid-range. The
polynomial's 8.77e-5 FP32 error is negligible compared to FP8 quantization error.

## NEW DATA 3: Partial Emulation Strategy (10-25% Software)

FA4 does NOT replace all exp2 calls with the polynomial. It uses a hybrid approach:
- 75-90% of exponential operations use hardware MUFU.EX2
- 10-25% use the software polynomial on FMA units

The exact fraction is tuned empirically based on the ratio of MMA throughput to
MUFU throughput for the specific tile configuration. The idea is to prevent MUFU
queue saturation without overwhelming the FMA units.

**For our kernel:** Our softmax has D=64 values per query row = 64 exp2f calls
per KV block. Testing 10% emulation = ~6 polynomial exp2 per block. This is
unlikely to make a measurable difference. FA4 gets benefit because their tiles
are 256x256 (much larger softmax). For our D=64, the SFU pressure is likely low
enough that replacing hardware exp2f is NOT the right optimization.

**However:** The polynomial IS useful if we switch to a strategy where we want
to overlap exp2 computation with MMA pipeline. The FMA polynomial can run
concurrently with tensor core MMA, whereas MUFU.EX2 and MMA compete for
scheduling slots. Even if FMA is slower per-instruction, the overlap could
reduce total latency. This is the FA4 insight.

## NEW DATA 4: Lazy Rescaling Threshold

FA4 uses tau = log2(256) = 8.0 as the rescaling threshold. This means:
- Rescaling is SKIPPED when abs(m_old - m_new) <= 8.0
- This means the new maximum can be up to 256x larger before rescaling triggers
- The paper reports this reduces correction operations by ~10x

**Important implementation detail:** When rescaling IS skipped, the accumulated
error must be corrected at the END of the kernel during final normalization.
FA4 ensures correctness by tracking the "true" maximum separately and applying
a final correction factor. The existing brief's implementation snippet is correct.

## Caveats

1. **The polynomial coefficients from the Modal reverse-engineering blog differ
   slightly** from the paper's coefficients: Modal reports 0.07711909, 0.22756439,
   0.69514614, 1.0 which MATCHES the paper. The existing brief had c3=0.0796,
   c2=0.2274, c1=0.6931 which are standard ln(2) Taylor coefficients, NOT the
   Sollya-optimized Remez approximation. Use the paper's values.

2. **The partial emulation strategy (10-25%) assumes datacenter hardware** with
   very high MMA:MUFU throughput ratios. On sm_120, the ratio may be different.
   Empirical tuning is required.

3. **For the FP8 attention kernel specifically**, the bottleneck is conversion
   overhead (448 ALU instructions), not softmax exp2f. The polynomial trick
   has lower priority than the ldmatrix reinterpret approach for eliminating
   conversion. But once conversion is eliminated, the softmax polynomial could
   become the next optimization target.
