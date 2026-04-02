# GTC 2026 Live Now + Blackwell Ultra SFU Doubling (Context)

**Sources:**
- https://www.nvidia.com/gtc/ (GTC 2026, March 16-19, San Jose)
- https://developer.nvidia.com/blog/making-softmax-more-efficient-with-nvidia-blackwell-ultra/
- https://www.nvidia.com/gtc/session-catalog/
**Relevant to:** attention worker, all workers
**Worker's current problem:** math_pipe_throttle 48% from softmax between MMA phases
**Date:** 2026-03-15

## GTC 2026 Is Live (March 16-19)

NVIDIA GTC 2026 starts TOMORROW (March 16) and runs through March 19. This is
the annual conference where NVIDIA announces new hardware, software, and
research. Relevant sessions to watch for:

- **Programming Blackwell tensor cores** -- CUTLASS team often presents new examples
- **FlashAttention updates** -- Tri Dao and team frequently present at GTC
- **FP8 / MXFP8 optimization** -- New techniques for narrow-precision compute
- **Consumer GPU / GeForce ML** -- sm_120 specific content (rare but possible)

The session catalog is at https://www.nvidia.com/gtc/session-catalog/ -- foreman
should check for relevant sessions as they become available and relay findings.

## Blackwell Ultra (GB300) Doubles SFU Throughput -- NOT Applicable to sm_120

NVIDIA's blog post "Making Softmax More Efficient with Blackwell Ultra" announces
that the GB300 (Blackwell Ultra, datacenter) doubles MUFU.EX2 throughput compared
to GB200:

- GB200: ~16 MUFU operations per clock per SM
- GB300: ~32 MUFU operations per clock per SM
- Result: ~35% increase in FP8 forward propagation throughput for attention

**This does NOT help sm_120 (RTX 5090).** The SFU doubling is a GB300-specific
hardware change. The RTX 5090 (GB202, sm_120) retains the standard SFU throughput.

**However, this validates our diagnosis:** NVIDIA's decision to double SFU in
Blackwell Ultra confirms that softmax/exp is the primary bottleneck in attention
on Blackwell-class hardware. Our worker's observation of 48% math_pipe_throttle
from softmax is consistent with what NVIDIA sees as the dominant bottleneck.
Their hardware solution (double SFU) is unavailable to us, so our software
solutions (polynomial exp2, conditional rescaling, sigmoid attention) are the
right approaches.

## Caveats

1. GTC sessions may not be immediately publicly available -- some are behind
   registration or delivered virtually with delays.
2. The Blackwell Ultra SFU info is context only. Do NOT attempt to use GB300
   techniques on sm_120.
