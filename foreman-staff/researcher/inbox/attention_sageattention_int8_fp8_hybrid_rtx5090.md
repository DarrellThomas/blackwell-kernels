# SageAttention: INT8 QK^T + FP8 PV Hybrid — 560 TOPS on RTX 5090

**Sources:**
- https://github.com/thu-ml/SageAttention
- ICLR 2025, ICML 2025, NeurIPS 2025 (Spotlight)
- SageAttention3 (2025): Microscaling FP4 Attention for Inference

**Relevant to:** attention worker
**Worker's current problem:** FP8 attention at 2.33x SDPA (52 us) is latency-bound (SM 43.8%). BF16→FP8 conversion overhead is the main bottleneck (~448 ALU instructions per KV block, 14:1 ratio vs MMA count). The worker's stated next directions are FP8 native inputs or algorithmic changes (sigmoid attention, etc.).

---

## What This Is

SageAttention is a quantized attention library from Tsinghua University that uses a **hybrid INT8/FP8 approach**:
- **QK^T MMA**: INT8 quantization with per-block smoothing
- **PV MMA**: FP8 (e4m3) quantization
- **Accumulator**: Mixed FP32+FP16 for PV (two-level accumulation)

It is accepted at three top venues (ICLR, ICML, NeurIPS 2025) and has production-quality CUDA + Triton implementations.

---

## Why It Matters for Us

### Performance on RTX 5090
SageAttention claims **560 TOPS on RTX 5090, 2.7x faster than FlashAttention2**.

For comparison, our current kernel:
- BF16: 1.76x SDPA (68 us, primary config)
- FP8: 2.33x SDPA (52 us)

If SageAttention achieves 2.7x vs FA2, and our BF16 kernel is already ~1.76x vs SDPA (which is roughly FA2-equivalent on sm_120), then SageAttention's approach may achieve similar or better throughput through a different quantization strategy.

### The INT8 QK^T Approach
Our FP8 kernel converts BF16→FP8 for BOTH QK^T and PV. SageAttention uses INT8 for QK^T instead. This is interesting because:

1. **INT8 m16n8k32 MMA** exists on sm_120: `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32`
2. INT8 has the same 2x throughput as FP8 (k=32 vs k=16)
3. BF16→INT8 conversion might be cheaper than BF16→FP8 conversion (direct truncation vs float conversion)
4. INT32 accumulator is exact (no rounding) — softmax precision improves

**However**, INT8 requires careful per-block scaling to avoid overflow/underflow. SageAttention adds "smoothing" (outlier redistribution) which adds overhead.

### The FP8 PV Approach
SageAttention uses FP8 for PV multiplication only. This is the same as what our kernel does. The innovation is the **two-level accumulation** strategy: accumulate partial PV results in FP16 (for speed), then merge into FP32 (for precision). This uses the FP8→FP16 accumulator path available on sm_120a.

### What We Can Learn
1. **INT8 QK^T is a viable alternative to FP8 QK^T** — potentially cheaper conversion
2. **Two-level accumulation** (FP16 inner + FP32 outer) could reduce register pressure
3. **Per-block quantization** is key for maintaining accuracy with low-precision attention
4. Their CUDA kernel implementation is available to study

---

## Key Techniques

### Per-block INT8 Quantization for Q, K
```
scale_Q = max(abs(Q_block)) / 127
Q_int8 = round(Q / scale_Q)
```
Applied per tile before MMA. The scale factors are carried through softmax.

### Smoothing (Outlier Redistribution)
Redistributes magnitude from outlier channels to non-outlier channels via a learned diagonal matrix. This reduces the dynamic range of Q/K before INT8 quantization, improving accuracy at the cost of some preprocessing.

### Two-Level PV Accumulation
```
# Inner loop (within KV blocks): FP16 accumulator for speed
pv_fp16 += P_fp8 @ V_fp8

# Outer loop (across KV block groups): accumulate to FP32
pv_fp32 += fp32(pv_fp16)
pv_fp16 = 0
```
This reduces register pressure because FP16 accumulators are 2 bytes vs FP32's 4 bytes.

---

## Caveats

1. **SageAttention targets inference**, not training. The accuracy tradeoffs may not be acceptable for training workloads.

2. **The 560 TOPS claim** needs verification on our hardware. Their benchmarks may use different configs (batch size, seq length, head dim).

3. **INT8 MMA on sm_120** produces INT32 accumulators, which are 4 bytes (same as FP32). The softmax computation on INT32 accumulator values requires conversion to float, which adds overhead.

4. **The smoothing step** is a per-head preprocessing operation that may add latency for small batch sizes where the preprocessing isn't amortized.

5. **sm_120a requirement**: The FP16 accumulator path for FP8 MMA (`mma.sync.m16n8k32.f16.e4m3.e4m3.f16`) requires compiling with `-arch=sm_120a`. Our workers currently compile with `sm_120` (see companion brief on sm_120a).

---

## Recommendation

**Study SageAttention's CUDA kernel implementation** for two specific techniques:

1. **INT8 QK^T** — if BF16→INT8 conversion is cheaper than BF16→FP8, this could be a faster path than our current approach
2. **Two-level accumulation** — mixing FP16 and FP32 accumulators to reduce register pressure

The INT8 approach is algorithmically different from our current FP8 path and represents a genuinely new optimization direction that doesn't require solving the FP8 native input problem.
