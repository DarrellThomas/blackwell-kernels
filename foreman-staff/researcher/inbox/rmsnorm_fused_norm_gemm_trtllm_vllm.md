# Fused RMSNorm + GEMM/Quantization: TensorRT-LLM and vLLM Production Patterns

**Sources:**
- [Pushing Latency Boundaries: DeepSeek-R1 on B200 (NVIDIA TensorRT-LLM Blog)](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog1_Pushing_Latency_Boundaries_Optimizing_DeepSeek-R1_Performance_on_NVIDIA_B200_GPUs.html)
- [vLLM Fusion torch.compile passes documentation](https://docs.vllm.ai/en/latest/design/fusions/)
- [vLLM + torch.compile blog post (August 2025)](https://blog.vllm.ai/2025/08/20/torch-compile.html)
- [SemiAnalysis InferenceMAX: vLLM + NVIDIA on Blackwell (October 2025)](https://blog.vllm.ai/2025/10/09/blackwell-inferencemax.html)
- [Unlocking DeepSeek with NVFP4 on Blackwell (Microsoft)](https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/unlocking-high-performance-inference-for-deepseek-with-nvfp4-on-nvidia-blackwell/4497936)

**Relevant to:** rmsnorm worker
**Worker's current problem:** 4.1 us pipelining floor for standalone RMSNorm. Fusion is the path forward. Existing briefs cover FlashNorm (weight absorption) and FlashFormer (megakernel). This brief covers production fusion patterns from TensorRT-LLM and vLLM.
**Supplements:** `rmsnorm_fusion_strategies.md`, `rmsnorm_fusion_new_findings_2026_03_14.md`

---

## What's New Here

The existing briefs cover FlashNorm (deferred normalization), FlashFormer (megakernel), and FlashInfer (JIT functors). This brief adds concrete patterns from two production inference engines that ship fused RMSNorm kernels on Blackwell:

1. **TensorRT-LLM's Module9 triple fusion**: AllReduce + RMSNorm + DynamicQuant in one kernel
2. **vLLM's torch.compile fusion passes**: Automatic graph-level fusion of RMSNorm with surrounding ops
3. **The fused RMSNorm + quantization pattern**: Normalize-then-quantize in a single memory pass

---

## TensorRT-LLM: Triple Fusion Kernel on Blackwell

For DeepSeek-R1 on 8xB200, TensorRT-LLM achieves 368 tokens/second (5.5x speedup over baseline) using several fused kernels. The most relevant:

### Module9: AllReduce + Add_RMSNorm + DynamicQuant (BF16->NVFP4)

This single kernel performs:
1. **AllReduce** -- sum tensor-parallel shards across GPUs
2. **Residual Add + RMSNorm** -- add residual connection, normalize
3. **Dynamic Quantization** -- convert BF16 output to NVFP4 for the next layer

All in one kernel launch, one pass through memory.

### Why this pattern matters (even without multi-GPU):

Strip out the AllReduce, and the remaining pattern is: **RMSNorm + Quantization** in a single kernel. This is directly applicable to our worker's fusion goals:

```cuda
// Fused RMSNorm + quantize kernel (single SM, single row)
// Read x once from DRAM
// Compute RMS(x) via reduction
// Normalize: x_hat = x * gamma / RMS
// Quantize: x_q = quantize(x_hat)
// Write x_q to DRAM (not x_hat -- skip the intermediate)
```

For our worker, the "quantize" step could be replaced with any point-wise epilogue (bias add, activation, etc.), but the principle is the same: fuse RMSNorm with whatever comes after it to avoid the intermediate DRAM write of the normalized tensor.

---

## vLLM: torch.compile Graph-Level Fusion

vLLM uses PyTorch's `torch.compile` with custom fusion passes to automatically detect and fuse operator patterns:

### Fused operator patterns:
- `Attention + Output Quantization`
- `AllReduce + RMSNorm + Quantization`
- `RMSNorm + Activation (SiLU/GELU)`

### How it works:
1. The model graph is traced by `torch.compile`
2. Custom passes (`FuseAllReduceRMSQuant`, `FuseRMSNormQuant`) scan for patterns
3. Matched patterns are replaced with a single fused CUDA kernel
4. The fused kernel is auto-generated or hand-written (via FlashInfer backend)

### Key design choice:
vLLM's fusion is pattern-based at the graph level, not kernel-level. The worker doesn't need to manually fuse ops -- the compilation infrastructure identifies fusible sequences. But for our standalone kernel project, the relevant learning is: **production systems prioritize fusing RMSNorm with quantization** because that's the most common and most impactful fusion.

---

## The Fused Norm + Quantize Pattern (Applicable to sm_120)

Even without NVFP4 or multi-GPU AllReduce, the core pattern transfers:

### Architecture:
```
Phase 1: Compute RMS (reduction across H dimension)
  - Each warp processes a chunk of the row
  - Warp reduction via __shfl_down_sync
  - Block reduction via shared memory
  - Result: rsigma = 1/sqrt(rms + eps) per row

Phase 2: Normalize + downstream op (element-wise, same threads)
  - Read x[i] again (or from shared memory if cached)
  - Compute y[i] = x[i] * gamma[i] * rsigma
  - Apply downstream op: y[i] = downstream(y[i])
  - Write y[i] to output
```

The key insight: Phase 2 can include ANY element-wise operation -- quantization, activation, bias add, etc. The cost of Phase 2 is dominated by the global memory write, which you're doing anyway. The normalize + downstream op adds minimal compute.

### For the rmsnorm worker specifically:

The backward pass is the immediate next step, not fusion. But understanding the fusion pattern informs the backward kernel design:
- **Cache rsigma** from the fused forward pass (as Transformer Engine does)
- **Don't cache x_hat** -- recompute it in backward from x + rsigma
- **Design the forward kernel API** to support an optional epilogue functor, so fusion can be added later without rewriting the kernel

---

## Caveats

- TensorRT-LLM's fused kernels target datacenter Blackwell (sm_100, B200) with features like AllReduce and NVFP4 that don't exist on sm_120. The RMSNorm + epilogue fusion pattern itself is architecture-agnostic.
- vLLM's torch.compile fusion works at the Python graph level. Our worker writes raw CUDA kernels -- the fusion must be manual. But the operator patterns identified by vLLM (what to fuse) are directly informative.
- The "fused norm + quant" pattern is most impactful for inference (where quantization is common). For training backward passes, the fusion target is different: fuse backward norm with the upstream gradient computation.
