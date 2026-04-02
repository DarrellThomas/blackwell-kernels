# FP8 Precision Strategies for SwiGLU-Fused MLP

**Sources:**
- https://arxiv.org/abs/2409.12517 (Graphcore: Scaling FP8 training to trillion-token LLMs, ICLR 2025)
- https://developer.nvidia.com/blog/per-tensor-and-per-block-scaling-strategies-for-effective-fp8-training/
- https://arxiv.org/html/2505.20524 (FOG: Towards Fully FP8 GEMM LLM Training at Scale)
- https://arxiv.org/html/2511.05811v2 (MOSS: Efficient FP8 LLM Training with Microscaling)
- https://arxiv.org/html/2405.14428v1 (Mitigating Activation Spikes in GLU-Based LLMs)
- https://github.com/triton-lang/triton/pull/7918 (MXFP8 for SM120)
- https://github.com/triton-lang/triton/issues/7550 (dot_scaled on 5090)
- https://github.com/ggml-org/llama.cpp/issues/19662 (block_scale not supported sm_120)
- https://github.com/deepseek-ai/DeepGEMM (DeepSeek FP8 fine-grained scaling)
- https://docs.nvidia.com/deeplearning/cudnn/frontend/latest/fe-oss-apis/gemm_fusions/gemm_amax.html

**Relevant to:** fused-mlp worker (Phase 4: SwiGLU)
**Worker's current problem:** relu_sq has 9.5% FP8 saturation at GPT-2 scale, forcing BF16 GEMM2. SwiGLU has different dynamic range -- need to understand if FP8 GEMM2 is viable with SwiGLU.

---

## SwiGLU Dynamic Range Analysis

### The Mathematical Problem

SwiGLU computes `output = up * SiLU(gate)` where `SiLU(x) = x * sigmoid(x)`.

For large positive `gate`, `SiLU(gate) ~ gate` (linear growth). The product
`up * gate` grows **quadratically** when `up` and `gate` are correlated, because
both are linear projections of the same input x:
- `up = x^T * w_up`
- `gate = x^T * w_gate`

When w_up and w_gate align (which happens naturally during training via weight
decay/regularization), `up ~ gate ~ c`, and `SwiGLU(x) ~ c^2`. This is the
**quadratic amplification** unique to gated activations.

### Comparison to relu_sq

relu_sq: `output = max(0, x)^2` -- also quadratic, but in a single projection.
The worker already measured 9.5% saturation at GPT-2 scale (values > 448).

SwiGLU: `output = up * SiLU(gate)` -- quadratic through the product of two
projections. **The dynamic range is similar to or slightly better than relu_sq
in practice**, because:

1. **SiLU provides soft gating.** For negative gate values, SiLU(gate) -> 0,
   which suppresses the output regardless of `up`. relu_sq has no such gating --
   all positive values pass through squared.

2. **The product `up * SiLU(gate)` requires BOTH projections to be large.**
   relu_sq only needs one projection to be large. This makes extreme SwiGLU
   values rarer than extreme relu_sq values for random weights.

3. **But weight alignment changes this over training.** Graphcore (arXiv 2409.12517)
   showed that after ~200B tokens of training, w_up and w_gate spontaneously align
   in certain channels, producing sporadic extreme outliers that can exceed FP8
   e4m3 max (448). This is a **training-time** phenomenon that emerges in specific
   channels after prolonged training.

### What This Means for Our Inference Kernel

For **inference with pretrained LLaMA-style weights**, the SwiGLU output distribution
depends entirely on the specific model checkpoint:

- **Well-behaved models** (short training, or with Smooth-SwiGLU): Most activation
  values stay well within [-448, 448]. Per-tensor scaling with scale = amax/448
  would work fine.

- **Long-trained models** (>200B tokens with standard SwiGLU): Specific channels
  can have outlier activations exceeding 448. The FOG paper measured a 688-magnitude
  outlier in standard Llama vs 183 in their modified architecture.

- **Key empirical finding (FOG paper):** Standard Llama activations have a 90th
  percentile range of [-0.289, 0.289] but outliers can be 2000x+ larger. The
  distribution is extremely long-tailed in specific channels.

**Bottom line:** SwiGLU values CAN exceed 448 in pretrained models, but less
frequently than relu_sq because the gating mechanism suppresses many values.
The saturation rate will be model-dependent but likely lower than the 9.5%
measured for relu_sq.

---

## FP8 Quantization Approaches for SwiGLU

### Strategy 1: Per-Tensor Scaling (Simplest)

```
scale = max(abs(activation_tile)) / 448.0
fp8_val = clamp(activation / scale, -448, 448)
```

- Used by: Transformer Engine (delayed scaling mode), early FP8 implementations
- Pros: One scale factor per tensor, minimal overhead
- Cons: A single outlier in any channel destroys precision for all other values.
  With long-tailed SwiGLU distributions, this wastes most of FP8's dynamic range
  on the 99th percentile while the 90th percentile values get rounded to zero.

### Strategy 2: Per-Row (1x128) Scaling (DeepSeek V3 Style)

```
# For each 1x128 block of activation:
block = activation[row, col:col+128]
scale = max(abs(block)) / 448.0
fp8_block = clamp(block / scale, -448, 448)
```

- Used by: DeepSeek V3/DeepGEMM, MOSS
- Pros: Each row adapts to its own range. Outlier channels only affect their
  local block. Much better precision for the common case.
- Cons: Requires storing and applying per-block scale factors. GEMM must
  dequantize with per-block scales during accumulation.

### Strategy 3: Per-Channel Scaling (Smooth-SwiGLU)

```
# For each output channel i:
s_i = max(abs(activation[:, i]))  # over mini-batch
scaled_activation[:, i] = activation[:, i] / s_i
fp8_val = quantize(scaled_activation)
# After GEMM2, multiply result by s_i to undo scaling
```

- Used by: Graphcore Smooth-SwiGLU paper
- Pros: Directly targets the channel-specific outlier problem in SwiGLU.
  Eliminates the quadratic amplification in the quantized domain.
- Cons: Requires per-channel scale storage and an extra multiply in the GEMM2
  epilogue. The scale factors must be computed from a calibration pass or running
  statistics.

### Strategy 4: Just Keep GEMM2 in BF16 (Current Approach)

The worker's current relu_sq kernel uses BF16 for GEMM2 because 9.5% saturation
is too high. This is the safe default for SwiGLU too.

**Performance cost:** BF16 GEMM2 is the bottleneck. At GPT-2 scale, GEMM2
dominates total time. Making GEMM2 FP8 would give ~2x MMA throughput for that
portion.

---

## MXFP8 / Block Scaling on sm_120

### The Hardware Situation (Critical Finding)

**MXFP8 block_scale IS supported on sm_120a** (the "a" suffix matters), but
with important caveats:

1. **The instruction exists:** `mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32`
   This is a variant of the standard mma.sync that takes additional scale operands
   in e8m0 format, one per 32-element block.

2. **Triton PR #7918 confirmed it works on RTX 5090.** Benchmark: MXFP8 native
   = 44.45s vs MXFP8 emulated = 76.44s for Llama3-8B inference (42% faster).
   Standard FP8 was 42.83s -- so MXFP8 native is nearly the same throughput as
   standard FP8 while providing better precision.

3. **sm_120 vs sm_120a matters.** The "a" suffix indicates architecture-accelerated
   features. The block_scale instruction requires sm_120a. Without the "a" suffix,
   ptxas reports: `"Feature '.block_scale' not supported on .target 'sm_120'"`.
   Our build system must target sm_120a (which the RTX 5090 supports).

4. **MXFP4 is NOT supported on sm_120/sm_120a.** Only sm_120f (not our GPU)
   supports MXFP4. ptxas error: `"Feature '.kind::mxf4' not supported on .target 'sm_120'"`.

5. **Block size: 32 elements.** Each group of 32 consecutive elements along K
   shares one e8m0 scale factor (8-bit unsigned exponent, bias 127, power-of-two
   only: scale = 2^(e8m0_val - 127)).

6. **Scale computation:** `scale_e8m0 = ceil(log2(max(abs(block_32)))) + 127`

### What MXFP8 Gives Us

For SwiGLU output -> GEMM2 input: instead of one scale for the entire activation
tensor (per-tensor) or keeping everything in BF16, we can use MXFP8 with a
scale per 32 elements. This handles the channel-specific outlier problem
automatically -- a channel with a 688-magnitude outlier gets its own scale
factor, while neighboring channels with values in [-1, 1] keep their precision.

**However:** We have NOT verified this instruction in our inline PTX codebase.
The operand layout (how scale factors are passed alongside data) needs empirical
testing. The existing MXFP8 brief in `docs/mxfp8_native_mma_sm120.md` covers
the instruction but notes it is unverified.

---

## Framework Approaches

### Transformer Engine (NVIDIA)

- **Delayed scaling:** Scale factor for iteration N is computed from amax history
  of iterations N-1, N-2, ..., N-W (window of W iterations). This avoids the
  overhead of computing amax in the current iteration but can lag behind sudden
  distribution changes.
- **Current scaling (newer):** Computes amax during the current forward pass.
  Better accuracy but requires a two-pass approach or fused epilogue.
- **SwiGLU handling:** TE fuses GEMM + SwiGLU on Hopper (sm_90) but notes
  "slightly reduced accuracy in FP8 PTQ because one quantization scaling factor
  is discarded" in the fused path. Not available on sm_120.

### DeepSeek V3 / DeepGEMM

- **1x128 activation scaling + 128x128 weight scaling.** Fine-grained per-block.
- **scale = max(abs(block)) / 448.0** computed externally before the GEMM.
- **Two-level accumulation:** 4 consecutive WGMMA ops accumulate in low precision,
  then promote to FP32 accumulator. This is a Hopper-specific technique (WGMMA);
  on sm_120 with mma.sync, accumulation is always FP32.
- **Key insight for us:** Their scale computation is external to the GEMM kernel.
  The quantized data + scale factors are prepared before calling the GEMM.

### MOSS (Microscaling + Automatic Scaling)

- **Two-level quantization:** Global FP32 scale + local e8m0 per-32-element scales.
- **Automatic scaling:** Predicts scale factors from optimizer state, eliminating
  runtime amax computation entirely.
- **Epilogue-deferred dequant:** Main GEMM loop uses only tensor cores; all
  dequantization happens in the epilogue on CUDA cores.
- **34% throughput improvement** over BF16 at matched accuracy.

### FOG (Fully FP8 GEMM Training)

- **Per-tensor delayed scaling** for simplicity and throughput.
- **Architectural changes** to prevent outliers rather than managing them post-hoc:
  remove pre-normalization, add post-normalization, freeze QK RMSNorm gains.
- **Key finding:** "Modifying the FOG-max architecture to use SwiGLU resulted in
  stable FP8 training" -- the problem is not SwiGLU itself, but SwiGLU combined
  with outlier-prone architectures.

---

## Recommended Strategy for Our Kernel

### For Phase 4 SwiGLU Implementation: Start with BF16 GEMM2

**Rationale:** The SwiGLU epilogue (up * SiLU(gate)) has similar or slightly
better dynamic range than relu_sq. But since relu_sq already shows 9.5%
saturation forcing BF16 GEMM2, the safe path is to keep GEMM2 in BF16 for
SwiGLU too. This matches the current kernel architecture exactly -- only GEMM1
changes (doubled N dimension with column interleaving).

**Implementation:** The existing v1 architecture (FP8 GEMM1 + BF16 GEMM2)
applies directly. SwiGLU output in FP32 registers -> convert to BF16 -> store
to global -> BF16 GEMM2. No precision changes needed.

### Future Optimization: Per-Tile Scaled FP8 GEMM2

If GEMM2 performance matters (it dominates at GPT-2 scale), here is a path to
FP8 GEMM2 with SwiGLU:

**Option A: Software per-tile scaling (simplest, no new instructions)**

1. After SwiGLU epilogue, each thread has FP32 output values in registers.
2. Compute per-tile amax via warp reduction: `__shfl_xor_sync` across the warp,
   then `atomicMax` across warps in the block. Cost: ~10 instructions.
3. Compute `tile_scale = tile_amax / 448.0`.
4. Convert: `fp8_val = cvt.rn.satfinite.e4m3x2.f32(val / tile_scale)`.
5. Store FP8 values + per-tile scale factor to global memory.
6. GEMM2 reads FP8 data + per-tile scale, applies scale during accumulation
   or in the epilogue. Each K-iteration multiplies the partial result by the
   corresponding tile scale.

**Overhead:** One warp-level reduction per tile + one extra multiply per
K-iteration in GEMM2. Minimal compared to MMA cost.

**Precision:** Each 64x64 tile gets its own scale. Outlier channels only
affect their local tile. Much better than per-tensor scaling.

**Option B: MXFP8 native (hardware block scaling, best precision)**

1. After SwiGLU epilogue, compute per-32-element e8m0 scales.
2. Convert to FP8 with per-block scaling.
3. Store FP8 data + e8m0 scale array to global memory.
4. GEMM2 uses `mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32`
   which handles dequantization in hardware.

**Overhead:** Scale computation + storage. Hardware dequant means no extra
CUDA core work during GEMM2.

**Risk:** This instruction is unverified in our codebase. Requires sm_120a
target. Operand layout needs empirical testing.

**Option C: Smooth-SwiGLU per-channel scaling**

1. Pre-compute per-channel scale factors from calibration data.
2. Apply `s_i` before FP8 quantization in the SwiGLU epilogue.
3. Undo `s_i` in GEMM2's epilogue (multiply output by s_i).

**Overhead:** One extra multiply per element in SwiGLU epilogue + one in GEMM2
epilogue. Minimal. But requires calibration data and per-channel scale storage.

### Recommended Sequence

1. **Phase 4a:** SwiGLU with BF16 GEMM2 (safe, matches existing architecture)
2. **Phase 4b (if GEMM2 is the bottleneck):** Software per-tile scaled FP8 GEMM2
   (Option A). This is the most practical path -- no new instructions, just add
   a warp-level amax reduction in the SwiGLU epilogue and per-tile dequant in
   GEMM2.
3. **Phase 4c (optional):** MXFP8 native (Option B) for best precision with
   hardware support. Only pursue after verifying the block_scale instruction
   works in our inline PTX on sm_120a.

---

## Caveats

1. **sm_120 vs sm_120a:** The MXFP8 block_scale instruction requires the "a"
   suffix target. Verify our build flags use `-arch=sm_120a` (or
   `--gpu-architecture=compute_120a`). Using plain `sm_120` will cause ptxas
   errors for block_scale instructions.

2. **Inference vs training:** The catastrophic outlier amplification (Graphcore
   paper) is a training-time phenomenon emerging after 200B+ tokens. For inference
   with a fixed model, the activation distribution is deterministic. If the
   pretrained model's activations stay within FP8 range, per-tensor scaling
   suffices. The worker should test with actual pretrained LLaMA weights to
   measure empirical saturation rates.

3. **The 9.5% relu_sq saturation may not transfer to SwiGLU.** relu_sq squares
   a single projection (no gating), so all positive values pass through. SwiGLU's
   gating mechanism (sigmoid) suppresses many values, likely reducing the
   saturation rate. This needs empirical measurement with actual SwiGLU outputs.

4. **DeepGEMM's approach (external scale computation) won't work for our fused
   kernel.** In our design, the SwiGLU output is computed in-kernel and may never
   touch global memory in the fused path. Scale factors must be computed within
   the same kernel, not externally. This is why Option A (warp-level amax
   reduction) is the right approach for us.

5. **cuDNN's GEMM+amax fusion is SM100+ only.** We cannot use NVIDIA's fused
   GEMM+amax kernel. We must implement amax reduction ourselves.

6. **MXFP8 scale factors are power-of-two only (e8m0 format).** This means
   scales are coarser than arbitrary FP32 scales. For most practical value
   ranges this is fine, but extreme cases may lose an extra bit of precision
   compared to FP32 per-tile scales.
