# Training Kernel Strategy for RTX 5090

**Author:** Darrell Thomas & foreman-claude
**Date:** 2026-03-22
**Hardware:** 2× RTX 5090 (GB202, sm_120), Threadripper PRO 7995WX, 512 GB DDR5
**Status:** Reference — planning document for backward-pass kernel development

---

## 1. Executive Summary

The blackwell-kernels project has a mature set of forward-pass kernels that meet
or exceed reference implementations across GEMM, flash attention, fused MLP,
RMSNorm, and linear algebra operations. This document lays out the kernel
inventory needed for backpropagation training passes, analyzes the VRAM/RAM
offloading tradeoffs specific to our hardware, and proposes a build order.

**Key conclusions:**
- Most backward-pass linear layer work reuses our existing GEMM pipeline (NT/TN layouts)
- Flash attention backward is the single hardest new kernel (~50% of total effort)
- A fused Adam/AdamW optimizer kernel is the highest-ROI new kernel category
- 512 GB system RAM enables 13B+ training via optimizer state offloading, but
  PCIe 5.0 x16 (32 GB/s) imposes a 56× bandwidth penalty vs VRAM (1792 GB/s)
- For 7B and under, everything fits in 2×32 GB VRAM — no offloading needed
- Gradient checkpointing (recomputing forward) is cheaper than activation offloading
  because our forward kernels are fast and VRAM bandwidth is abundant

---

## 2. Current Forward-Pass Kernel Inventory

### 2.1 Shipped Primitives (common/csrc/primitives/)

| Operation | vs Reference | Source |
|-----------|-------------|--------|
| GEMM BF16 | 0.97× cuBLAS (4096³), 1.23× non-square | gemm/ |
| GEMM FP8 | 1.34× cuBLAS (4096³), 1.97× small configs | gemm/ |
| Batched GEMM BF16 | 1.44× cuBLAS | linalg/ |
| Batched GEMM FP8 | 1.97× cuBLAS | linalg/ |
| TRMM | 1.03× cuBLAS | linalg/ |
| GEMV | 1.75× cuBLAS | linalg/ |
| Batched GEMV | 1.00× cuBLAS | linalg/ |
| SYRK | 1.02× cuBLAS | linalg/ |
| TRSM | 1.00× cuBLAS | linalg/ |
| DOT | 1.98× cuBLAS | linalg/ |
| NRM2 | 1.60× cuBLAS | linalg/ |
| AXPY | 1.49× cuBLAS | linalg/ |
| SCAL | 1.70× cuBLAS | linalg/ |
| permute_rows | 2.19× cuBLAS | linalg/ |
| swap_rows | 6.97× cuBLAS | linalg/ |

### 2.2 Active Optimized Kernels

| Operation | vs Reference | Source |
|-----------|-------------|--------|
| Flash Attention BF16 | 1.76× cuDNN SDPA | main/ |
| Flash Attention FP8 | 2.33× cuDNN SDPA | main/ |
| Epilogue-fused MLP (GEMM + ReLU²) | 1.22× cuBLAS | fused-mlp/ |
| RMSNorm | 89.4% peak bandwidth (1605 GB/s) | rmsnorm/ |

### 2.3 Decompositions

| Operation | Status | Notes |
|-----------|--------|-------|
| LU | Archived | Completed, offboarded 2026-03-15 |
| QR | Testing | Uses TRMM primitive |
| Cholesky | Blocked | sm_120 TF32 diagonal broadcast defect |

---

## 3. Backward-Pass Kernel Requirements

### 3.1 Overview: What Backprop Computes

For a transformer layer, the backward pass computes gradients with respect to
inputs and parameters by reversing through each operation. Each forward operation
has a corresponding backward operation:

```
Forward:                          Backward (gradient):
─────────────────────────────     ─────────────────────────────────
Embedding lookup                  Scatter-add to embedding table
RMSNorm(x)                       dγ, dx via reduction
Q,K,V = x·Wq, x·Wk, x·Wv       dWq = x^T·dQ, dx += dQ·Wq^T (etc)
Attention(Q,K,V)                  dQ, dK, dV (flash attn backward)
O = attn·Wo                      dWo = attn^T·dO, dattn = dO·Wo^T
RMSNorm(x)                       (same as above)
FFN: x·W1 → act → ·W2           dW1, dW2, dx (fused MLP backward)
Residual: x + sublayer           dx += dsublayer (trivial addition)
Cross-entropy(logits, target)    softmax(logits) - one_hot(target)
```

### 3.2 Kernel-by-Kernel Analysis

#### 3.2.1 Linear Layer Backward (GEMM NT/TN)

**What it computes:**
For Y = X·W (forward GEMM in NN layout), backward needs:
- dX = dY · W^T  — gradient w.r.t. input (NT layout: dY is normal, W is transposed)
- dW = X^T · dY  — gradient w.r.t. weights (TN layout: X is transposed, dY is normal)

**Reuse potential: HIGH.** These are standard GEMMs with transposed operands. Our
existing GEMM pipeline (MMA, swizzle, double-buffer) transfers directly. The key
requirement is supporting NT and TN layouts efficiently:
- NT: B matrix loaded transposed → different shared memory layout, different ldmatrix pattern
- TN: A matrix loaded transposed → same consideration

**Work needed:**
- Verify existing GEMM handles NT/TN or add layout variants
- Tile dimensions may need adjustment (NT/TN have different optimal tile shapes
  due to different memory access patterns)
- FP8 backward GEMMs are typically done in BF16 (gradient magnitudes less stable)

**Estimated effort:** Low-moderate. Core infrastructure exists.

#### 3.2.2 Flash Attention Backward

**What it computes:**
Given saved outputs O and row-wise log-sum-exp (LSE) from the forward pass,
plus incoming gradient dO, compute dQ, dK, dV without materializing the full
N×N attention matrix.

**Algorithm (Dao et al.):**

```
# Precompute D_i = rowsum(dO_i ⊙ O_i) for each block i
# For each block of K,V (outer loop):
#   For each block of Q (inner loop):
#     Recompute S_ij = Q_i · K_j^T                    (GEMM)
#     Recompute P_ij = exp(S_ij - LSE_i)              (softmax recompute)
#     dV_j += P_ij^T · dO_i                           (GEMM)
#     dP_ij = dO_i · V_j^T                            (GEMM)
#     dS_ij = P_ij ⊙ (dP_ij - D_i)                   (element-wise)
#     dQ_i += dS_ij · K_j                             (GEMM, atomic add)
#     dK_j += dS_ij^T · Q_i                           (GEMM)
```

**Two tiling strategies:**
1. **dQ-accumulate (Q-outer, KV-inner):** Each thread block owns a Q block and
   iterates over K/V blocks. dQ is accumulated locally (no atomics), but dK/dV
   require atomic adds across thread blocks. Better parallelism for long sequences.
2. **dKV-accumulate (KV-outer, Q-inner):** Each thread block owns a K/V block and
   iterates over Q blocks. dK/dV accumulated locally, dQ requires atomics.
   Better for short sequences or large head dimensions.

**Complexity relative to forward:**
- Forward: 2 GEMMs per tile pair (QK^T, PV) + softmax
- Backward: 5 GEMMs per tile pair + softmax recompute + element-wise ops
- Roughly **2-2.5× the FLOPS** of forward
- More register pressure (must hold dQ, dK, dV, P, dP, dS simultaneously)
- More shared memory traffic (recompute S from Q,K each pass)

**Causal masking:** Same as forward — mask S_ij where j > i. But backward must
also mask dS_ij consistently, and skip tile pairs where the mask is all-zero.

**sm_120 considerations:**
- 128 KB shared memory (99 KB usable) must hold Q, K, V, dO tiles simultaneously
- Register pressure higher than forward — may need to reduce tile size
- Double-buffering is critical (same cp.async infrastructure as forward)
- Atomic adds for dQ or dK/dV accumulation — fp32 atomicAdd to global memory

**Estimated effort:** Very high. This is the hardest kernel in the training stack.
Expect 30-50+ experiments. Should be its own worktree project.

#### 3.2.3 Fused Adam/AdamW Optimizer

**What it computes (per parameter element):**

```python
# Standard Adam with weight decay (AdamW variant)
m = β₁ * m + (1 - β₁) * grad              # update first moment
v = β₂ * v + (1 - β₂) * grad²             # update second moment
m_hat = m / (1 - β₁^t)                     # bias correction
v_hat = v / (1 - β₂^t)                     # bias correction
param_fp32 -= lr * (m_hat / (√v_hat + ε) + wd * param_fp32)  # step + weight decay
param_bf16 = to_bf16(param_fp32)            # cast working copy
```

**Why fusion matters:**
Without fusion, this is 5+ separate global memory passes:
1. Read grad, read m, write m
2. Read grad, read v, write v
3. Read m, read v, compute step
4. Read param_fp32, write param_fp32
5. Read param_fp32, write param_bf16

Fused: **one read** (grad + m + v + param_fp32) → compute → **one write** (m + v + param_fp32 + param_bf16).

**Memory access per parameter (fused):**
- Read: 2 (grad, BF16) + 4 (m) + 4 (v) + 4 (param_fp32) = 14 bytes
- Write: 4 (m) + 4 (v) + 4 (param_fp32) + 2 (param_bf16) = 14 bytes
- Total: 28 bytes per parameter, purely bandwidth-bound

**At 1792 GB/s VRAM bandwidth:**
- 7B params × 28 bytes = 196 GB → **109 ms** at peak bandwidth
- With ~80% bandwidth efficiency: **~137 ms** per optimizer step
- This is fast enough that offloading is unnecessary for 7B

**Estimated effort:** Moderate. Straightforward bandwidth-bound kernel. The
interesting variant is the streaming version for RAM offloading (see Section 4).

#### 3.2.4 RMSNorm Backward

**What it computes:**

```
# Forward stored: x (input), rms = √(mean(x²) + ε), x_hat = x / rms
# Given: dy (incoming gradient), γ (learned scale)
dγ = sum_over_batch(dy ⊙ x_hat)            # weight gradient (reduction)
dx = (dy ⊙ γ - x_hat ⊙ mean(dy ⊙ γ ⊙ x_hat)) / rms   # input gradient
```

**Profile:** Two reductions (mean and dγ accumulation) + element-wise. Same
bandwidth-bound character as forward RMSNorm. Our forward kernel runs at 89.4%
of peak bandwidth; backward should achieve similar.

**Key detail:** Need to save either `x` or `x_hat` from the forward pass. If
using gradient checkpointing, recompute from `x` (adds one extra reduction).

**Estimated effort:** Low-moderate. Structurally similar to forward.

#### 3.2.5 Fused MLP Backward

**Forward:** x → GEMM(W1) → ReLU² → GEMM(W2) → out

**Backward:**

```
# Stage 1: backward through GEMM2
dact = dout · W2^T                          # NT GEMM
dW2  = act^T · dout                         # TN GEMM

# Stage 2: backward through activation
# d/dx(relu(x)²) = 2·relu(x)·(x > 0) = 2·max(0, x)
dgemm1_out = dact ⊙ 2·max(0, gemm1_out)    # element-wise (need saved gemm1_out or recompute)

# Stage 3: backward through GEMM1
dx  = dgemm1_out · W1^T                     # NT GEMM
dW1 = x^T · dgemm1_out                      # TN GEMM
```

**Fusion opportunity:** Fuse the activation backward (stage 2) into the epilogue
of the first backward GEMM (stage 1) — same approach as forward fusion. Load
saved `gemm1_out` from global memory, apply `2·max(0, ·)` mask, multiply into
the GEMM output before writing. This avoids a separate element-wise kernel launch.

**Memory optimization:** For ReLU², only need to save a 1-bit mask (x > 0) from
forward, not the full activation. The gradient is `2·max(0, x)` which requires `x`,
but if we save just the mask + the magnitude of `relu(x)`, that's cheaper than
saving full FP32/BF16 activations.

**Estimated effort:** Moderate. Builds on existing fused-mlp architecture.

#### 3.2.6 Cross-Entropy + Softmax Backward

**What it computes:**

```
# Forward: loss = -log(softmax(logits)[target])
# Backward: dlogits = softmax(logits) - one_hot(target)
```

This is deceptively simple in math but the softmax over vocabulary dimension
(32K-128K) needs tiling to avoid materializing the full softmax output.

**Fusion:** Combine forward softmax + cross-entropy loss + backward gradient into
a single kernel. Read logits once, compute softmax online (log-sum-exp trick),
produce gradient, write gradient. One read + one write over the vocab dimension.

**Estimated effort:** Low-moderate.

#### 3.2.7 Dropout Forward + Backward

**What it computes:**
- Forward: mask = (philox_rng(seed, offset) > p), output = input ⊙ mask / (1-p)
- Backward: dinput = doutput ⊙ mask / (1-p)

**Key insight:** Don't save the mask. Regenerate it from the same (seed, offset)
pair. This is standard practice and saves N bytes of memory per dropout layer.

**Best approach:** Don't implement standalone. Fuse into attention (after softmax,
before P·V GEMM) and/or MLP (after activation, before second GEMM). This means
the dropout kernels are part of the attention and MLP backward kernels, not
independent kernel launches.

**Estimated effort:** Low (fused into other kernels).

#### 3.2.8 Embedding Backward

**What it computes:**

```
# Forward: output[i] = embedding_table[input[i]]   (gather)
# Backward: dembedding_table[input[i]] += doutput[i]  (scatter-add)
```

**Challenges:**
- Irregular memory access (input tokens determine which rows get gradients)
- Multiple tokens may map to the same embedding row → atomic adds required
- Vocabulary is large (32K-128K rows × hidden_dim) but only a fraction of rows
  receive gradients per batch

**Approach:** Sort tokens by embedding index, then coalesce updates. Or use
fp32 atomicAdd (fast on sm_120). For small batch sizes the atomic approach is
simpler and sufficient.

**Estimated effort:** Low.

---

## 4. Memory Analysis and Offloading Strategy

### 4.1 Training Memory Budget

For mixed-precision training (BF16 compute, FP32 master weights, Adam optimizer):

| Component | Per Parameter | 1B | 3B | 7B | 13B |
|-----------|-------------|-----|-----|------|------|
| BF16 working weights | 2 B | 2 GB | 6 GB | 14 GB | 26 GB |
| FP32 master weights | 4 B | 4 GB | 12 GB | 28 GB | 52 GB |
| FP32 Adam m (momentum) | 4 B | 4 GB | 12 GB | 28 GB | 52 GB |
| FP32 Adam v (variance) | 4 B | 4 GB | 12 GB | 28 GB | 52 GB |
| BF16 gradients | 2 B | 2 GB | 6 GB | 14 GB | 26 GB |
| **Total parameter state** | **16 B** | **16 GB** | **48 GB** | **112 GB** | **208 GB** |
| Activations (w/ checkpointing) | varies | ~1 GB | ~3 GB | ~6 GB | ~12 GB |
| **Total training memory** | — | **~17 GB** | **~51 GB** | **~118 GB** | **~220 GB** |

### 4.2 What Fits Where

| Model Size | 1× 5090 (32 GB) | 2× 5090 (64 GB) | System RAM (512 GB) |
|-----------|-----------------|-----------------|-------------------|
| 1B (17 GB) | Fits entirely | — | — |
| 3B (51 GB) | No | Fits entirely | — |
| 7B (118 GB) | No | No — needs offload | Fits with offload |
| 13B (220 GB) | No | No — needs offload | Fits with offload |
| 30B (~480 GB) | No | No | Fits, barely |

### 4.3 Hardware Bandwidth Hierarchy

```
                    ┌─────────────────────────┐
                    │     SM Register File     │
                    │    ~20 TB/s effective     │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │    Shared Memory (SMEM)  │
                    │   ~12 TB/s (99 KB/SM)    │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   GDDR7 VRAM (32 GB)     │
                    │      1,792 GB/s          │
                    └────────────┬────────────┘
                                 │  56× slower
                    ┌────────────▼────────────┐
                    │   PCIe 5.0 x16 (DMA)    │
                    │       32 GB/s            │
                    └────────────┬────────────┘
                                 │  ~5× slower
                    ┌────────────▼────────────┐
                    │  DDR5 System RAM (512 GB) │
                    │     ~150 GB/s aggregate  │
                    └─────────────────────────┘
```

**The PCIe bottleneck is the central constraint.** System RAM itself is fast
enough, but everything must transit through the 32 GB/s PCIe pipe.

### 4.4 Offloading Strategy: What Lives Where

**Principle: keep the hot path in VRAM, offload the cold path to RAM.**

**Hot path (touched every forward/backward micro-step):**
- BF16 working weights for active layer(s): 2B/param
- Activations for active layer(s): ~1-2 GB
- Gradients for active layer(s): 2B/param
- Per-layer VRAM footprint: ~500 MB for a 7B model layer (~250M params/layer)
- With 2 GPUs: 20+ layers resident simultaneously

**Cold path (touched once per optimizer step):**
- FP32 master weights: 4B/param → offload to RAM
- Adam m state: 4B/param → offload to RAM
- Adam v state: 4B/param → offload to RAM
- Total offloaded: 12B/param = **84 GB for 7B**, **156 GB for 13B**

### 4.5 Offloading Pipeline

For each optimizer step, per layer:

```
Timeline for layer N:
├── PCIe: prefetch master_weights[N], m[N], v[N] from RAM → VRAM  (DMA, async)
├── GPU: run fused Adam kernel on layer N  (uses prefetched data)
├── PCIe: write back master_weights[N], m[N], v[N] to RAM  (DMA, async)
├── GPU: cast FP32 → BF16, keep BF16 weights in VRAM
└── PCIe: prefetch layer N+1 (overlapped with GPU compute on N)
```

**Transfer size per layer (7B / 32 layers ≈ 220M params/layer):**
- Read: 220M × 12 bytes = 2.64 GB → 82 ms at 32 GB/s
- Write: 220M × 12 bytes = 2.64 GB → 82 ms at 32 GB/s
- GPU Adam compute: ~4 ms (bandwidth-bound in VRAM, trivial)
- **Per-layer wall time: ~165 ms** (read + compute + write, partially overlapped)
- **Total optimizer step: 32 layers × ~165 ms ≈ 5.3 seconds**

**With double-buffered prefetch (overlapping layer N+1 read with layer N write):**
- Effective per-layer time: ~85 ms (one PCIe direction at a time)
- **Total optimizer step: ~2.7 seconds**

### 4.6 Offloading vs. Gradient Checkpointing

Two ways to reduce VRAM pressure. They solve different problems:

| Strategy | Saves | Costs | Best For |
|----------|-------|-------|----------|
| **Gradient checkpointing** | Activation memory (don't save intermediate activations, recompute in backward) | ~33% extra forward compute | When activations overflow VRAM but params + optimizer state fit |
| **Optimizer state offloading** | Parameter state memory (master weights + Adam states live in RAM) | PCIe transfer time (~3-5s/step) | When parameter state overflows VRAM |

**For 7B:** Gradient checkpointing alone is sufficient. Activations are the pressure
point, not parameter state. Our fast forward kernels (1.76× attention, 1.22× MLP)
make recompute cheap. No PCIe tax.

**For 13B+:** Need both. Offload optimizer state to RAM AND use gradient
checkpointing to keep activation memory bounded.

### 4.7 FP8 Training Considerations

FP8 training can halve the memory for some components:

| Component | BF16 | FP8 | Savings |
|-----------|------|-----|---------|
| Working weights | 2B/param | 1B/param | 50% |
| Activations | 2B/element | 1B/element | 50% |
| Gradients | 2B/param | Typically still BF16 | 0% |
| Master weights | FP32 always | FP32 always | 0% |
| Optimizer state | FP32 always | FP32 always | 0% |

**FP8 training requires per-tensor dynamic scaling:**
- Track amax (absolute maximum) of each tensor over a rolling window
- Compute scale factor = FP8_MAX / amax
- Apply scale before cast to FP8, apply inverse scale after
- This adds overhead but our FP8 GEMM infrastructure already handles scaling

**Gradient stability:** Forward activations work well in FP8. Backward gradients
are more sensitive to range — BF16 is typically safer for gradients. Mixed
approach: FP8 forward, BF16 backward, is the pragmatic choice.

**Impact on VRAM budget:** FP8 weights + activations saves ~50% on those
components, potentially making 7B fit on a single GPU with room to spare.

---

## 5. Build Order and Priority

### Phase 0: Foundation (before any backward kernels)

| Task | Effort | Dependency |
|------|--------|------------|
| Verify GEMM NT/TN layouts | Low | None — test existing kernel with transposed inputs |
| Add transpose-mode dispatch to GEMM | Low-mod | If NT/TN shows perf gap vs NN |
| Gradient checkpointing harness | Moderate | Need to integrate with forward kernels |

### Phase 1: Core Training Loop

| # | Kernel | Effort | Impact | Notes |
|---|--------|--------|--------|-------|
| 1 | Fused Adam/AdamW optimizer | Moderate | Very high | Bandwidth-bound, straightforward. Unlocks the training loop. |
| 2 | RMSNorm backward | Low-mod | High | Similar to forward, reductions + element-wise |
| 3 | Cross-entropy + softmax backward | Low-mod | High | Fused single-pass over vocab dimension |
| 4 | Embedding backward | Low | Medium | Scatter-add with atomics |

**Milestone: can train a transformer with unfused linear layers + naive attention backward.**

### Phase 2: Attention Backward

| # | Kernel | Effort | Impact | Notes |
|---|--------|--------|--------|-------|
| 5 | Flash attention backward | Very high | Critical | Dedicate a full worktree. 30-50+ experiments expected. |

**Milestone: full custom training stack, end to end.**

### Phase 3: Fusion and Optimization

| # | Kernel | Effort | Impact | Notes |
|---|--------|--------|--------|-------|
| 6 | Fused MLP backward | Moderate | High | Fuse activation gradient into GEMM epilogue |
| 7 | Dropout fusion (attn + MLP) | Low | Medium | Regenerate mask from Philox seed, no storage |
| 8 | Streaming Adam (RAM offload) | Moderate | Enables 13B+ | Fused Adam + integrated PCIe DMA prefetch |

### Phase 4: FP8 Training (if pursuing)

| # | Kernel | Effort | Impact | Notes |
|---|--------|--------|--------|-------|
| 9 | Dynamic scaling infrastructure | Moderate | Enables FP8 | Amax tracking, scale computation |
| 10 | FP8 forward + BF16 backward mixed pipeline | High | Memory savings | Integration across all kernels |

---

## 6. Model Size Recommendations

### For Initial Development: Start Small

**1B-3B parameter models** are the right target for building the training stack:
- 1B fits entirely on a single GPU (17 GB) — fast iteration, no offloading complexity
- 3B fits on 2 GPUs (51 GB) — tests multi-GPU but no offloading needed
- Fast experiment cycles: seconds per training step, not minutes
- Debug kernels at small scale before scaling up

### Production Targets

| Model | Training Memory | Strategy | Step Time (est.) |
|-------|----------------|----------|-----------------|
| 1B | 17 GB | Single GPU, no offload | ~50 ms |
| 3B | 51 GB | 2 GPUs, no offload | ~150 ms |
| 7B | 118 GB | 2 GPUs + grad checkpointing | ~500 ms |
| 7B offload | 118 GB | 2 GPUs + RAM offload | ~3-5 s |
| 13B | 220 GB | 2 GPUs + RAM offload + grad ckpt | ~8-12 s |

---

## 7. Architecture Decisions

### 7.1 Two-GPU Strategy

For training, the two GPUs can be used in two ways:

**Data parallel (same model, split batch):**
- Each GPU has full model copy, processes different batch elements
- Gradients averaged (all-reduce) after backward — but with 2 GPUs on same
  machine, this is just a PCIe transfer or even direct GPU-GPU via NVLink
- Note: RTX 5090 does NOT have NVLink. All inter-GPU goes through PCIe.
- Doubles effective batch size, doesn't reduce memory per GPU.

**Pipeline/model parallel (split model across GPUs):**
- Each GPU holds half the layers
- Forward: GPU0 computes layers 0-15, sends activations to GPU1 for layers 16-31
- Backward: reverse
- Halves per-GPU memory, but adds PCIe latency at the split point
- For 7B: each GPU holds ~59 GB of parameter state. Still doesn't fit.

**Hybrid (model parallel + offload):**
- Split model across 2 GPUs, offload optimizer state to RAM
- Each GPU holds BF16 weights for its layers (~7 GB for 7B/2)
- Activations + gradients for active layers (~3-5 GB)
- Optimizer state (84 GB) in RAM, streamed per-layer
- **This is the most practical approach for 7B+**

### 7.2 Gradient Accumulation

For effective batch sizes larger than what fits in memory:
1. Run multiple forward+backward passes (micro-batches), accumulating gradients
2. Run optimizer step once after N micro-batches
3. Amortizes the optimizer step (and any PCIe offloading) over N iterations
4. For RAM offloading, this is crucial: if optimizer step costs 3s but you
   accumulate over 8 micro-batches, the amortized cost is 375 ms/effective-step

### 7.3 What NOT to Build

- **All-reduce / NCCL replacement:** Not worth it for 2 GPUs. PCIe peer-to-peer
  or simple cudaMemcpyPeer is sufficient.
- **Pipeline parallelism scheduler:** Overkill for 2 GPUs. Simple sequential
  layer processing is fine.
- **Custom memory allocator:** PyTorch's caching allocator is good enough.
  Focus on kernel compute, not memory management.
- **Custom autograd:** Use PyTorch autograd with custom autograd.Function wrappers
  around our kernels. Don't reinvent the backward-pass scheduler.

---

## 8. Reference: Theoretical Compute Limits

### 8.1 Per-Step FLOPS (Transformer Layer, Backward)

For a transformer with hidden_dim H, seq_len S, batch B, num_heads N, head_dim D:

| Operation | Forward FLOPS | Backward FLOPS | Ratio |
|-----------|--------------|----------------|-------|
| QKV projection | 6BSH² | 12BSH² | 2× |
| Attention (QK^T + PV) | 4BS²HD | 10BS²HD | 2.5× |
| Output projection | 2BSH² | 4BSH² | 2× |
| FFN (2 linear layers) | 16BSH² | 32BSH² | 2× |
| **Total per layer** | **~24BSH² + 4BS²HD** | **~48BSH² + 10BS²HD** | **~2.2×** |

**Rule of thumb:** Backward pass is ~2-2.5× the compute of forward. For
attention-heavy configs (long sequence), the ratio increases toward 2.5×.

### 8.2 RTX 5090 Compute Budget

| Precision | Peak TFLOPS | Sustained (est.) |
|-----------|-------------|-----------------|
| FP32 | 104.8 | ~85 |
| BF16 (tensor) | 419.2 | ~350 |
| FP8 (tensor) | 838.4 | ~700 |

### 8.3 Sample Training Step Time Estimates

**7B model, B=4, S=2048, H=4096, 32 layers:**
- Forward FLOPS: ~1.2 TFLOPS → ~3.4 ms at 350 TFLOPS (BF16)
- Backward FLOPS: ~2.6 TFLOPS → ~7.4 ms at 350 TFLOPS
- Optimizer step: ~137 ms (bandwidth-bound, see Section 3.2.3)
- **Compute-only step time: ~148 ms** (dominated by optimizer bandwidth)
- With overhead (kernel launch, memory allocation): ~200-500 ms realistic

These estimates assume all data in VRAM. Add PCIe transfer time if offloading.

---

## Appendix A: Key References

- Dao et al., "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning" — backward pass algorithm
- Rajbhandari et al., "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models" — offloading strategy
- Micikevicius et al., "Mixed Precision Training" — FP16/BF16 training fundamentals
- NVIDIA, "Transformer Engine" — FP8 training with dynamic scaling
- Our own `common/docs/theoretical_limits.md` — sm_120 peak throughput calculations
- Our own `common/claude/04_HARD_WON_LESSONS.md` — empirical findings that apply to all kernels

## Appendix B: Glossary

| Term | Definition |
|------|------------|
| NN/NT/TN/TT | GEMM layout: Normal/Transposed for A and B matrices |
| LSE | Log-Sum-Exp: log(Σ exp(x_i)), saved from attention forward for backward recompute |
| Gradient checkpointing | Recomputing forward activations during backward instead of saving them |
| Micro-batch | A sub-batch processed in one forward+backward pass; gradients accumulated over multiple micro-batches before optimizer step |
| Amax | Absolute maximum of a tensor, used to compute FP8 scaling factors |
| DMA | Direct Memory Access: hardware-managed CPU↔GPU transfers via PCIe, overlappable with compute |
