# Sigmoid & Linear Attention Landscape — Comprehensive Update (March 2026)

**Sources:** Multiple (listed per finding below)
**Relevant to:** attention worker (BF16 and FP8 kernels)
**Worker's current problem:** BF16 at 1.78x SDPA, optimization exhausted. math_pipe_throttle 48% from irreducible softmax overhead. Agent state lists "sigmoid attention or other softmax alternatives" as next direction #3.

## Context: What We Already Have

Previous briefs delivered:
- `attention_flashsigmoid_alternative.md` — Apple's FlashSigmoid paper (arxiv 2409.04431), algorithm details, register savings, estimated 15-25% speedup
- `attention_flashinfer_sigmoid_support.md` — FlashInfer 0.2 sigmoid support, 3-instruction sigmoid via tanh formulation

This brief covers **everything new** found in the broader ecosystem since those briefs were written.

---

## 1. Sigmoid Attention — Theoretical Validation Strengthened

### 1.1 ICLR 2025 Acceptance (Apple Paper)

**Source:** https://openreview.net/forum?id=Zhdhg6n2OG

The FlashSigmoid paper (arxiv 2409.04431) was accepted at **ICLR 2025** as a conference paper. This is significant — it's no longer a preprint, it's peer-reviewed and accepted at a top venue. The community has validated:
- Sigmoid attention matches softmax quality across language, vision, and speech
- The b = -log(n) bias term enables sequence length generalization
- FlashSigmoid achieves 17% kernel speedup on H100

### 1.2 Sigmoid Has Lower Sample Complexity (Feb 2025)

**Source:** https://arxiv.org/abs/2502.00281

New theoretical result: "Sigmoid Self-Attention has Lower Sample Complexity than Softmax Self-Attention: A Mixture-of-Experts Perspective" (Feb 2025, updated May 2025). This paper proves that sigmoid attention is not just "as good as" softmax but is **theoretically more sample-efficient** when viewed through a mixture-of-experts lens.

**Why this matters for us:** Strengthens the case that sigmoid attention will become the default for new models. More models trained with sigmoid = more demand for a fast sigmoid attention kernel.

### 1.3 NeurIPS 2025 Best Paper — Qwen's Attention Gating

**Source:** https://towardsdatascience.com/neurips-2025-best-paper-review-qwens-systematic-exploration-of-attention-gating/

Qwen's systematic exploration of attention gating mechanisms won a NeurIPS 2025 Best Paper. The work explores various gating strategies including sigmoid-based mechanisms. This puts sigmoid attention firmly in the mainstream research agenda.

---

## 2. FLASH-D: Hidden Softmax Division via Sigmoid

**Source:** https://arxiv.org/abs/2505.14201
**Published:** May 2025, accepted at IEEE/ACM ISLPED 2025

### What It Is

FLASH-D reformulates the FlashAttention kernel so that the softmax division (normalization by sum-of-exponents) is **hidden inside sigmoid function evaluations**. It is mathematically equivalent to standard FlashAttention (produces identical outputs) but the computation is restructured.

### Key Technical Insight

The incremental division by the sum of exponents in baseline FlashAttention is effectively hidden within sigmoid function evaluations. Specifically:
- The maximum value tracking is **entirely removed** — numerical stability is maintained by ensuring attention score differences stay within sigmoid's active region [-6, 11]
- The running sum-of-exponents is **implicitly embedded** in the computation of each weight
- One vector multiplier is saved in the output update, and the max+sum logic is entirely removed

### Hardware Results

At 28nm ASIC: **22.8% area reduction and 20.3% power reduction** vs FlashAttention-2 parallel hardware, with zero performance penalty.

### Why This Matters for Us

This is different from FlashSigmoid (which changes the attention semantics). FLASH-D produces **exact softmax attention** but uses sigmoid as an implementation trick to avoid tracking m and l. This means:
- It could work with **existing softmax-trained models** (no retraining needed)
- It still eliminates row-max and row-sum accumulators
- The register savings and reduced instruction count are real

**Caveat:** The paper targets ASIC/FPGA, not GPU CUDA kernels. The reformulation may or may not be faster on GPU — the sigmoid evaluations replace the exp+division chain, and on GPU the relative cost of MUFU.EX2 vs MUFU.RCP+MUFU.TANH determines whether this wins. Needs empirical testing on sm_120.

**This is worth investigating as a potentially less disruptive path than full sigmoid attention** — same model compatibility, similar register savings.

---

## 3. Flash Linear Attention Ecosystem

### 3.1 fla-org/flash-linear-attention (Active, Major Library)

**Source:** https://github.com/fla-org/flash-linear-attention
**Stars:** Major library, actively maintained through 2025-2026

This is the canonical library for efficient linear attention models. Key facts:
- Provides Triton-based kernels (NOT raw CUDA) for linear attention variants
- Supports: RetNet, RWKV (v4-v7), Gated DeltaNet, Mamba-style models, GLA, HGRN, and more
- v0.3.2+ published as `fla-core` on PyPI
- January 2025: Added RWKV7 kernels and models
- December 2024: Added Gated DeltaNet, flash-bidirectional-attention

**Why this matters:** This library is the reference for understanding what linear attention kernels look like. However, it uses **Triton**, not raw CUDA with mma.sync. The algorithms are transferable; the kernel code is not directly usable on sm_120 without Triton support.

### 3.2 Tiled Flash Linear Attention (TFLA) — NeurIPS 2025

**Source:** https://arxiv.org/abs/2503.14376
**Code:** https://github.com/NX-AI/mlstm_kernels
**Published:** March 2025, accepted NeurIPS 2025

TFLA introduces a second level of sequence parallelization within chunks for linear RNN kernels. Key benchmark results on H100:

- **Faster than FlashAttention-3 for long sequences**
- **Over 2x faster than Mamba-2 kernels for all sequence lengths**
- Tested at standard 7B model embedding dim (4096), 65K total tokens

The mlstm_kernels library provides PyTorch, JAX, and Triton implementations including:
- Chunkwise kernels (TFLA-based, parallel over chunks)
- Parallel kernels (like standard attention)
- Recurrent step kernels (for inference/generation)

Also introduces **mLSTMsig** — an mLSTM variant using sigmoid input gate with reduced computation, enabling even faster kernels "at no performance drops on language modeling up to 1.4B parameter scale."

**Relevance:** TFLA's approach of increasing arithmetic intensity through tiling is conceptually similar to what we do with our attention kernel. The benchmark numbers (beating FA3 on long sequences) confirm that softmax-free approaches can be faster even on datacenter hardware.

### 3.3 Gated DeltaNet — ICLR 2025 (NVIDIA)

**Source:** https://arxiv.org/abs/2412.06464
**Code:** https://github.com/NVlabs/GatedDeltaNet
**Published:** December 2024, ICLR 2025

NVIDIA's linear attention variant that uses delta rule + gating. Key numbers:
- 45 Kt/s throughput for 1.3B model on H100
- Hybrid variants: 54 Kt/s
- Fused Triton/CUDA kernel keeps state in registers/shared memory

**Why this matters:** This is what Qwen 3.5 adopted (see below). NVIDIA's backing signals linear attention is going mainstream.

---

## 4. Production Adoption: Qwen 3.5 Uses Linear Attention

### 4.1 Qwen 3.5 Hybrid Architecture

**Sources:**
- https://huggingface.co/blog/mlabonne/qwen35
- https://blog.vllm.ai/2025/09/11/qwen3-next.html
- https://developer.nvidia.com/blog/new-open-source-qwen3-next-models-preview-hybrid-moe-architecture-delivering-improved-accuracy-and-accelerated-parallel-processing-across-nvidia-platform

Qwen 3.5 (released 2025-2026) uses a **hybrid attention architecture**:
- ~75% of layers use **Gated DeltaNet** (linear attention)
- ~25% of layers use **full attention** (softmax-style, gated)
- 3:1 ratio: linear attention is the dominant mechanism

The linear attention layers use SiLU (closely related to sigmoid) as the gating activation. The architecture supports 200K-1M context lengths thanks to linear attention's O(N) scaling.

**Why this matters for us:** This is the strongest evidence that softmax-free attention is going mainstream. The second-largest open-source LLM family has moved to predominantly linear attention. Demand for fast sigmoid/linear attention kernels on consumer GPUs is coming. A fast sigmoid attention kernel for sm_120 would serve Qwen 3.5 inference.

### 4.2 vLLM Support

**Source:** https://blog.vllm.ai/2025/09/11/qwen3-next.html

vLLM has added native support for Qwen 3.5's hybrid architecture, meaning the linear attention kernel path is now in the serving stack. Consumer GPU users running Qwen 3.5 via vLLM would benefit from optimized linear attention kernels.

---

## 5. Additional Linear Attention GPU Kernel Work

### 5.1 Optimized Linear Attention CUDA Implementation (Oct 2025)

**Source:** https://arxiv.org/abs/2510.21956

"Transformer Based Linear Attention with Optimized GPU Kernel Implementation" by Gerami and Duraiswami. Claims:
- **3.3x speedup** over state-of-the-art linear attention implementations
- **3.6x memory reduction**
- Novel forward and backward pass method
- Validated with 1.4B parameter language model training

The paper proposes O(ND^2) complexity (linear in N, quadratic in D). For our D=64 config, this means the inner loop is dominated by D^2=4096 operations, which should be heavily MMA-bound — ideal for tensor cores.

### 5.2 SageAttention (Quantized Attention, Complementary)

**Source:** https://github.com/thu-ml/SageAttention

Not sigmoid-specific, but relevant: SageAttention achieves 2-5x speedup over FlashAttention using INT8 QK^T + FP8 PV quantization. Won ICLR 2025, ICML 2025, NeurIPS 2025 Spotlight.

Their approach of separating QK^T quantization (INT8) from PV quantization (FP8) could combine with sigmoid attention: sigmoid output is in [0,1], which is perfectly representable in FP8 with no dynamic range concerns, potentially enabling more aggressive quantization of the P matrix.

---

## 6. Summary: What's Actionable for the Attention Worker

### Immediate (can prototype now):

1. **Sigmoid attention kernel** — the existing FlashSigmoid brief has full implementation details. The new evidence (ICLR acceptance, theoretical superiority proof, Qwen 3.5 adoption) strengthens the case to build this.

2. **FLASH-D reformulation** — potentially offers softmax-compatible sigmoid-based computation (no retraining needed). Worth reading the paper to see if the reformulated inner loop is implementable with mma.sync. Paper: https://arxiv.org/abs/2505.14201

### Medium-term (for future project planning):

3. **Linear attention kernel** (Gated DeltaNet style) — if Qwen 3.5 becomes a target model, a dedicated linear attention kernel using mma.sync could be a new project. The fla-org library provides algorithmic reference, but we'd need raw CUDA.

4. **Hybrid sigmoid + FP8** — sigmoid output in [0,1] is a natural fit for FP8 PV multiplication. A sigmoid+FP8 kernel could potentially reach 35-40 us (3.0x SDPA).

### Already known (no new information):

- FlashSigmoid algorithm details (covered in existing brief)
- FlashInfer sigmoid support (covered in existing brief)
- Register savings estimates (covered in existing brief)
