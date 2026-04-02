# RMSNorm Fusion: New Findings (2026-03-14 research cycle)

**Relevant to:** RMSNorm worker, attention worker
**Worker's current problem:** 4.1 us pipelining floor. Standalone kernel is optimal. Real win is fusion.
**Supplements:** `rmsnorm_fusion_strategies.md` and `rmsnorm_fusion_attention_approaches.md` (already delivered)

---

## Finding 1: FlashFormer -- Whole-Model Kernel with Atomic Sync (arxiv 2505.22758)

**Source:** https://arxiv.org/html/2505.22758v1

**What it is:** FlashFormer fuses an entire Llama-3 transformer forward pass into a single kernel. Unlike the "No Bubbles" megakernel (counter-based interpreter), FlashFormer uses per-thread global memory fences + atomic-based synchronization. Tested on H100, achieves 8-20% over GPTFast, up to 61% over vLLM for Llama-3.1 8B.

**How it handles RMSNorm:** FlashFormer caches the input activation in shared memory, then computes RMSNorm + RoPE + QKV projection as an integrated sequence. The key: QKV projection is implemented as matrix-vector operations (batch=1), so RMSNorm is computed on the cached shared memory input before the mat-vec begins. No global memory round-trip for the normalized intermediate.

**Synchronization between layers:** Uses `__threadfence()` (per-thread global memory fence) + atomics for cross-CTA ordering. Each sublayer writes results to global memory with a threadfence, then increments an atomic counter. The next sublayer's CTAs spin-wait on the counter. This is lighter than cooperative groups' `grid_sync()` (which stalls ALL CTAs), because only the dependent CTAs need to synchronize.

**Why it matters for us:** The atomic fence approach is a viable alternative to cooperative groups for our sm_120 kernel if we ever go full-megakernel. More importantly, the pattern of "cache input in smem -> compute RMSNorm -> immediately proceed with next operation" is exactly the fusion pattern our worker should implement.

**Caveat:** FlashFormer targets H100 only. The atomic-based sync pattern is architecture-agnostic, but the mat-vec kernels (batch=1 decode) are not directly applicable to our compute-bound attention kernel.

---

## Finding 2: FlashSigmoid -- Eliminate Softmax Row Reduction Entirely (Apple)

**Source:** https://github.com/apple/ml-sigmoid-attention
**Paper:** https://arxiv.org/abs/2409.04431

**What it is:** FlashSigmoid replaces softmax with element-wise sigmoid in attention. Because sigmoid is element-wise (no row reduction needed), it eliminates the online softmax tracking (row-max, row-sum) that FlashAttention requires. Result: 17% inference kernel speedup over FlashAttention-2 on H100.

**Why it matters for the fusion question:** The RMSNorm fusion challenge is fundamentally about handling a row reduction (sum-of-squares) before the main attention computation. Sigmoid attention eliminates a different row reduction (softmax). If the model uses sigmoid attention, the attention kernel's complexity drops significantly -- and there's more "room" in the kernel to absorb the RMSNorm reduction without performance penalty.

**Specific technical advantages:**
- No need to track running max and sum for online softmax correction
- No accumulation of intermediate rescaling variables
- Fewer register requirements (no m_i, l_i accumulators per row)
- Fewer kernel dispatches (no two-pass correction)

**Caveat:** Requires a model trained with sigmoid attention (not a drop-in replacement for softmax-trained models). Apple's implementation targets H100. The open-source code at github.com/apple/ml-sigmoid-attention could be adapted for sm_120.

---

## Finding 3: FlashInfer JIT Functor System -- Inject Norm into Attention

**Source:** https://flashinfer.ai/2024/12/16/flashinfer-v02-release.html
**Paper:** https://arxiv.org/abs/2501.01005

**What it is:** FlashInfer's JIT system lets you define custom attention variants via C++ functors (QueryTransform, KeyTransform, LogitsTransform) that get inlined into the attention kernel at compile time. The system uses Jinja templates + PyTorch JIT compilation. Example QueryTransform:

```cpp
template <typename T>
__device__ __forceinline__ T QueryTransform(const ParamsT& params, T q) {
    return float(q) * params.logits_scale * math::log2e;
}
```

**Relevance to RMSNorm fusion:** A QueryTransform functor could potentially apply per-element normalization to Q values as they're loaded. However, RMSNorm requires a full-row reduction BEFORE applying the per-element scaling, which doesn't fit cleanly into a per-element functor. You'd need a two-phase approach:

1. Phase 1: Custom prologue that computes RMS(row) via reduction
2. Phase 2: QueryTransform that divides by the pre-computed RMS

FlashInfer doesn't currently support this two-phase pattern natively. But the architecture shows that functor-based customization of attention kernels is a proven approach -- and our hand-written kernel can implement the same idea more directly.

**Caveat:** FlashInfer's JIT system is designed for inference serving, not raw kernel performance. Their attention kernel may not match our hand-tuned FA kernel. The value here is architectural insight, not code to copy.

---

## Finding 4: SGLang JIT QK Norm Kernel -- Fused Q/K RMSNorm Inplace

**Source:** https://lmsys.org/blog/2026-01-16-sglang-diffusion/

**What it is:** SGLang implements a JIT-compiled QK norm kernel that fuses Q/K RMSNorm into a single inplace kernel. This is applied to DiT (Diffusion Transformer) models where Q-norm and K-norm are standard. The kernel: (1) loads Q and K, (2) computes RMS for each head, (3) normalizes in-place, (4) writes back. Cuts launch count and memory traffic.

**Key pattern:** The inplace approach avoids allocating a normalized copy. The kernel reads Q, computes the row reduction, normalizes, and writes the result back to the same buffer. This is exactly the "Phase 1" pattern for fusing norm before attention -- except SGLang does it as a standalone kernel rather than fusing into attention itself.

**Why it matters:** Validates that Q/K normalization is a common enough pattern that production frameworks implement dedicated fused kernels. If we fuse this INTO our attention kernel's Q-load path, we save even the QK-norm kernel launch.

---

## Finding 5: RedFuser -- Automatic Cascaded Reduction Fusion (ASPLOS 2026)

**Source:** https://arxiv.org/abs/2603.10026
**Code:** https://github.com/alibaba/redfuser

**What it is:** RedFuser (Alibaba, ASPLOS 2026) automatically fuses cascaded reductions -- sequences of dependent reduction operations like those in softmax (max -> subtract -> sum -> divide) or normalization (sum-of-squares -> rsqrt -> scale).

**Key technique:** Decomposes reduction functions into separable components: F_i(x, d_i) = G_i(x) * H_i(d_i), where G operates on inputs and H on dependencies. This allows fusing the reductions into a single pass that computes all intermediate values incrementally with O(1) memory overhead.

**Performance:** Achieves 1.09x FlashAttention-2 for attention, 2.8x PyTorch Dynamo on LLaMA-65B, 1.7x on MoE routing. For FP8 Quant+GEMM: 3.4x over Dynamo.

**Relevance:** The mathematical decomposition framework could apply to fusing RMSNorm's sum-of-squares reduction with the subsequent attention computation's softmax reductions. If both reductions read the same input data, a single fused pass could compute both. However, in the pre-norm transformer pattern, RMSNorm and softmax operate on different tensors (input activations vs Q*K^T), so the data overlap is limited.

**Caveat:** Built on TVM, not raw CUDA. The mathematical framework is the valuable part -- understanding which reductions can be fused and which cannot.

---

## Finding 6: DeepFusionKernel -- Reductions Are NOT Good Fusion Targets (Negative Finding)

**Source:** https://arxiv.org/abs/2602.11808

**What it is:** DeepFusionKernel (Feb 2026) fuses SwiGLU MLP operations for 13.2% speedup on H100. But it explicitly states: **"true reductions (e.g., Softmax) introduce long-range dependencies that limit cross-SM streaming and therefore are not good fusion targets."**

**Why this matters:** This is a direct counterpoint to aggressive fusion of RMSNorm into other kernels. The argument: RMSNorm requires a full-row reduction (sum-of-squares across D elements), which creates a synchronization barrier. If the downstream operation (GEMM or attention) tiles the row across multiple CTAs, the reduction becomes a cross-CTA communication problem.

**BUT -- this doesn't apply to our case:** In our attention kernel, each CTA processes complete Q rows (D=768). The reduction is CTA-local (no cross-CTA communication needed). The DeepFusionKernel warning applies to cases where the reduction dimension is split across multiple CTAs, which happens in large GEMMs but NOT in our attention kernel's Q-loading path.

**Bottom line:** The "reductions are bad fusion targets" claim is valid for general cross-SM fusion but does NOT apply when each thread block handles the full reduction dimension. Our case (D=768, each attention CTA loads complete Q rows) is the exception.

---

## Finding 7: cuDNN v9.10+ RMSNorm Improvements for sm_120

**Source:** https://docs.nvidia.com/deeplearning/cudnn/backend/v9.18.1/release-notes.html

**What it is:** cuDNN v9.10+ has relaxed the 128-byte alignment restriction on LayerNorm and RMSNorm engines to 16-byte alignment, enabling more fusion opportunities on Blackwell GPUs. Also: improved performance of GEMM fusions with large gemm-K dimensions in the runtime fusion engine, and support for fusing GroupedRMSNorm (two overlapped RMS_Norms into one).

**Relevance:** If we ever use cuDNN for the attention or GEMM path, the built-in RMSNorm prologue fusion is available. Also validates that NVIDIA considers norm-GEMM fusion important enough to optimize at the library level for sm_120.

---

## Updated Recommendation

The existing fusion strategies doc covers the core approaches well. These new findings add:

1. **FlashFormer's atomic sync pattern** as an alternative to cooperative groups for megakernel sync
2. **FlashSigmoid** as a way to reduce kernel complexity (removing softmax reduction), making room for norm fusion
3. **FlashInfer's functor architecture** as a design pattern for norm-attention fusion
4. **DeepFusionKernel's "reductions are bad" claim** -- important to understand WHY it doesn't apply to our case (CTA-local reduction)
5. **RedFuser's decomposition framework** for understanding which reductions can be co-fused
6. **cuDNN sm_120 improvements** validating the norm-GEMM fusion direction at the library level

The recommended path remains the same: fuse RMSNorm into the attention kernel's Q-load prologue. The new findings reinforce that this is the right approach and provide additional architectural patterns.
