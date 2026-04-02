# Industry Consensus: Where Does RMSNorm Get Fused?

**Sources:**
- [FlashAttention repo (Dao-AILab)](https://github.com/Dao-AILab/flash-attention)
- [FlashAttention RMSNorm issue #570](https://github.com/Dao-AILab/flash-attention/issues/570)
- [vLLM Fusion Passes](https://docs.vllm.ai/en/latest/design/fusions/)
- [vLLM QK Norm RoPE Fusion](https://docs.vllm.ai/en/latest/api/vllm/compilation/qk_norm_rope_fusion/)
- [FlashInfer norm API](https://docs.flashinfer.ai/generated/flashinfer.norm.fused_add_rmsnorm.html)
- [FlashInfer v0.2 Release](https://flashinfer.ai/2024/12/16/flashinfer-v02-release.html)
- [Transformer Engine API](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/pytorch.html)
- [TensorRT-LLM FusedAddRMSNormQuant PR](https://github.com/NVIDIA/TensorRT-LLM/pull/9905)
- [FlashFormer (arxiv 2505.22758)](https://arxiv.org/html/2505.22758v1)
- [FlashNorm (arxiv 2407.09577)](https://arxiv.org/html/2407.09577v1)

**Relevant to:** rmsnorm worker
**Worker's current problem:** Deciding where to fuse RMSNorm -- into the attention kernel or into the QKV projection GEMM? The existing briefs cover individual strategies. This brief answers the meta-question: what does industry actually do?

---

## The Short Answer

**Nobody fuses RMSNorm into the attention kernel.** The universal industry pattern is to fuse RMSNorm with the QKV projection (linear layer) or with the residual add + quantization, NOT with the attention computation itself. The attention kernel receives already-normalized Q, K, V.

---

## Framework-by-Framework Evidence

### 1. FlashAttention (Dao-AILab)

FlashAttention does **NOT** fuse normalization into the attention kernel. The flash-attention repo includes standalone RMSNorm and LayerNorm kernels (as separate CUDA kernels) that are called before attention. The attention kernel's interface expects pre-normalized Q, K, V tensors.

Issue #570 on the repo discusses the memory efficiency of the standalone fused RMSNorm kernel (fusing residual add + norm), confirming it is treated as a separate operation from attention. Tri Dao's response confirms the RMSNorm kernel is standalone, optimized for memory efficiency through fusion of element-wise ops (residual + norm), not fused with attention.

### 2. vLLM

vLLM implements three RMSNorm fusion patterns, **none of which fuse into attention**:

1. **`fuse_allreduce_rms`**: AllReduce + RMSNorm + Quantization. Fuses the norm with the communication collective and downstream quantization.

2. **`fuse_norm_quant`**: RMSNorm + Quantization. Fuses normalization with FP8/INT8 quantization for the next layer.

3. **`fused_qk_norm_rope`**: Split QKV + Q/K RMSNorm + RoPE. This is a POST-projection fusion -- applied to Q and K AFTER the QKV GEMM, for models that use per-head QK normalization (e.g., Qwen). It fuses the norm with rotary embeddings, not with attention.

Key detail: `fused_qk_norm_rope` targets sm80+ but has "perf issues on H100" and is not enabled by default. Tested only on sm90 and sm100.

### 3. FlashInfer

FlashInfer provides standalone norm operations:
- `flashinfer.norm.rmsnorm()`
- `flashinfer.norm.fused_add_rmsnorm()` -- fuses residual add + RMSNorm

These are separate from the attention kernels (`flashinfer.prefill`, `flashinfer.decode`). The attention kernel does NOT include normalization.

FlashInfer's JIT functor system (QueryTransform, KeyTransform, LogitsTransform) allows injecting custom transformations into the attention kernel. In principle, a per-element scaling could be injected via QueryTransform. However, RMSNorm requires a full-row reduction BEFORE the per-element scaling, which does NOT fit into the per-element functor model. FlashInfer does not support a "prologue reduction" hook.

The MLA (Multi-Latent Attention) decode kernel applies RMSNorm to query latent representations, but this is done outside the main attention functor system as a model-specific customization, not as a general-purpose fusion.

### 4. NVIDIA Transformer Engine

Transformer Engine fuses RMSNorm with the **linear layer (GEMM)**, not with attention:

- **`LayerNormLinear`**: Fuses RMSNorm/LayerNorm with the subsequent linear transformation. Supports `normalization='RMSNorm'`. The module computes norm + GEMM, avoiding materializing the normalized intermediate in global memory.

- **`LayerNormMLP`**: Fuses RMSNorm/LayerNorm with the MLP (two GEMMs + activation).

- **No `LayerNormAttention` module exists.** The attention modules (`MultiheadAttention`, `DotProductAttention`) expect pre-normalized inputs.

Implementation detail: TE's fusion is NOT a single CUDA kernel in most cases. It launches the norm kernel, which writes FP8 output directly (via `with_quantized_norm=True`), then launches the GEMM kernel which reads the FP8 output. The "fusion" eliminates the BF16 intermediate and quantization kernel, but there are still two kernel launches. True single-kernel norm+GEMM fusion is available via CUTLASS EVT (Epilogue Visitor Tree) on sm_90+.

### 5. TensorRT-LLM

TensorRT-LLM implements:

- **`FusedAddRMSNormQuant`**: Residual add + RMSNorm + FP4 quantization in a single warp-specialized kernel (sm90+). This is a standalone kernel that runs between layers, NOT fused into attention.

- **AllReduce fusion**: AllReduce + residual add + RMSNorm in a single kernel for tensor-parallel inference.

- **Module9 triple fusion** (DeepSeek-R1 on B200): AllReduce + Add_RMSNorm + DynamicQuant (BF16->NVFP4).

All of these fuse RMSNorm with surrounding element-wise/communication ops. None fuse it into the attention kernel.

### 6. FlashFormer (Whole-Model Kernel)

FlashFormer is the ONLY system that fuses RMSNorm into the same kernel as attention, but it does so by fusing THE ENTIRE TRANSFORMER FORWARD PASS into one kernel:

- RMSNorm is computed on input cached in shared memory
- RMSNorm output feeds directly into QKV projection (mat-vec for batch=1)
- QKV projection output feeds into attention
- Attention output feeds into output projection
- Output projection feeds into MLP
- MLP output feeds into next layer's RMSNorm

The fusion is between ALL operations (not specifically RMSNorm+attention). Inter-operation synchronization uses per-thread `__threadfence()` + atomics.

FlashFormer is batch-1 decode only (mat-vec, not GEMM) and targets H100.

---

## Why the Industry Fuses Norm with GEMM, Not Attention

The architectural reason is clear:

1. **In the transformer computation graph**, RMSNorm feeds into Q/K/V projections (three GEMMs), not directly into attention. The data flow is:
   ```
   x -> RMSNorm(x) -> [W_Q, W_K, W_V] projections -> Attention(Q, K, V)
   ```
   Fusing norm into attention would skip over the projections entirely, which only makes sense if you also fuse the projections.

2. **RMSNorm + Linear is a natural fusion pair** because the norm output is the linear input -- one memory read eliminated. Fusing norm into attention would require also fusing the GEMM, which is a much larger engineering task.

3. **The GEMM already loads each input row once.** Adding an RMS reduction during the load phase (the FlashNorm/Mirage approach) is a natural fit for the GEMM's data access pattern.

4. **Attention has a different tiling structure.** Attention tiles over the sequence dimension (blocks of tokens), not the hidden dimension. The RMSNorm reduction is over the hidden dimension. These don't align well.

---

## What This Means for Our Worker

The question "should we fuse RMSNorm into the attention kernel?" has a clear industry answer: **no, fuse it into the QKV projection GEMM instead.**

The standard fusion point in the transformer data flow is:

```
OPTION A (industry standard):
  x -> [Fused: RMSNorm(x) + QKV_GEMM] -> Q, K, V -> Attention(Q, K, V)

NOT this:
  x -> RMSNorm(x) -> QKV_GEMM -> Q, K, V -> [Fused: RMSNorm + Attention](Q, K, V)
```

For our specific situation (standalone RMSNorm kernel at 4.1 us floor):
- **If the goal is eliminating the standalone kernel launch**: fuse RMSNorm into the QKV GEMM kernel (FlashNorm weight absorption or Mirage-style GEMM prologue)
- **If the goal is eliminating ALL inter-kernel overhead**: full megakernel (FlashFormer approach), but this is a massive engineering effort
- **The attention kernel itself should NOT be modified** to include RMSNorm -- this goes against the universal data flow pattern

### Exception: QK-Norm Models

Some models (Qwen, OpenELM) apply per-head RMSNorm to Q and K AFTER the QKV projection. For these models, the norm is between the GEMM and attention, so fusing into either the GEMM epilogue or the attention prologue is valid. vLLM's `fused_qk_norm_rope` handles this case. This is a different operation from the pre-attention layer norm.

---

## Caveats

- Our existing attention kernel operates on pre-projected Q, K, V. Fusing the pre-layer RMSNorm into it would change the kernel's interface to accept raw activations + weight matrices, effectively making it a fused QKV+attention kernel. This is the megakernel direction.
- The FlashNorm weight absorption trick (absorb gamma into W, defer scalar division) is the lowest-effort path to eliminate the standalone norm kernel, with ~20 lines of Python.
- All production frameworks (vLLM, TRT-LLM, TE) treat the fused residual_add + RMSNorm + quantization combination as the highest-value fusion target, not norm+attention.
