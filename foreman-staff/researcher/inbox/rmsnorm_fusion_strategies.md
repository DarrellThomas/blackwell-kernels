# Research: RMSNorm Fusion Strategies to Break the 4.1us Pipelining Floor

**Source:** Multiple (see references below)
**Relevant to:** RMSNorm worker
**Worker's current problem:** 4.1us GPU pipelining floor is the hard limit for standalone RMSNorm. 16 experiments exhausted. The only path forward is eliminating the standalone kernel via fusion.

---

## Strategy 1: FlashNorm — Absorb RMSNorm into Weight Matrix (Zero-Cost Norm)

**Source:** https://arxiv.org/html/2407.09577v1

### What This Is

FlashNorm exploits the mathematical linearity of RMSNorm followed by a linear layer. Instead of computing `y = RMSNorm(x) @ W`, it rearranges to `y = (x @ W') / RMS(x)`, where `W' = diag(g) @ W` (the RMSNorm gain vector absorbed into the weight matrix).

### The Trick

RMSNorm computes: `y_i = (x_i * g_i) / sqrt(1/n * sum(x_j^2))`

Since matrix multiplication is linear, you can:
1. **Pre-absorb g into W:** Compute `W'[i,j] = g[i] * W[i,j]` once at init time. Now the norm is just `y = x / RMS(x)`, a scalar division.
2. **Defer the division:** Compute `output = (x @ W') / RMS(x)` — the GEMM and RMS computation run in parallel because they're independent.

This works because RMS(x) is a scalar that doesn't depend on W. The GEMM proceeds on unnormalized x while a separate small reduction computes the scalar RMS. Then one scalar division at the end.

### Why It Matters for Us

This approach eliminates RMSNorm as a standalone kernel entirely. For the attention path:
- Q = RMSNorm(x) @ W_Q becomes Q = (x @ W_Q') / RMS(x)
- K = RMSNorm(x) @ W_K becomes K = (x @ W_K') / RMS(x)
- V = RMSNorm(x) @ W_V becomes V = (x @ W_V') / RMS(x)

The RMS(x) computation (a simple reduction) can be fused into the QKV GEMM kernel as a prologue that runs in parallel with the GEMM main loop. The 4.1us standalone kernel disappears entirely.

### Caveats

- Requires bias-free linear layers (Llama, Mistral, GPT-2 all qualify).
- Weight matrix must be modified at init time (one-time cost).
- The scalar division at the end adds one multiply per output element.
- For attention specifically: if Q-norm and K-norm are used (some architectures), this extends cleanly since `RMS(Q)` and `RMS(K)` are also scalars.
- The paper reports only ~10% speedup when applied naively because the norm itself is not the bottleneck — the kernel launch is. But combined with fusion, the launch overhead is zero.

---

## Strategy 2: Mirage-Style Fused RMSNorm + MatMul (Single Kernel, One Load)

**Source:** https://mirage-project.readthedocs.io/en/latest/tutorials/rms-norm-linear.html

### What This Is

Mirage auto-discovers a kernel that loads input tensor X once from global memory and computes both (a) the RMS reduction for normalization and (b) the matrix multiplication, in a single kernel. The key: move the division after the MatMul.

### The Technique

Standard: `Y = (X * g / RMS(X)) @ W` — requires materializing normalized X in memory.
Mirage: `Y = (X @ W') / RMS(X)` — X loaded once, RMS computed as a side-channel reduction during the GEMM, division applied to GEMM output.

Performance: **1.5-1.7x faster** than separate RMSNorm + MatMul kernels.

### Why It Matters for Us

The 1.5-1.7x speedup comes from eliminating:
1. The standalone RMSNorm kernel launch (our 4.1us floor)
2. The intermediate memory write of normalized X
3. The re-read of normalized X by the GEMM

For our attention kernel specifically: the attention kernel already loads Q from global memory. If we fuse RMSNorm into that load path (compute RMS while loading Q tiles, divide after), we eliminate the standalone kernel.

### Caveats

- Mirage targets datacenter GPUs and uses auto-generation. We need to implement manually.
- The division-after-GEMM trick changes the GEMM output slightly (dividing accumulated FP32 values by a scalar vs dividing BF16 inputs). Numerics should be fine for BF16/FP8 inference.

---

## Strategy 3: Megakernel — Fuse Entire Transformer Block

**Source:** https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles

### What This Is

The Hazy Research "No Bubbles" megakernel fuses RMSNorm, QKV projection, RoPE, attention, output projection, and MLP into a single persistent kernel. On H100, it achieves 2.5x faster than vLLM; on B200, 3.5x faster.

A simpler implementation (MegaQwen, targeting RTX 3090 sm_86) fuses the same operations and achieves **3.9x faster than HuggingFace** (531 tok/s vs 136 tok/s).

### The Technique

- **Persistent kernel:** One kernel launch executes the entire transformer block. Thread blocks stay resident, processing operations sequentially with grid-wide synchronization (cooperative groups).
- **Paged shared memory:** Shared memory is divided into fixed-size pages (e.g., 16KB). Operations acquire/release pages, enabling cross-operation pipelining.
- **RMSNorm in the kernel:** Normalization is computed in shared memory between operations. The output feeds directly into the next operation's input path without a global memory round-trip.
- **Interpreter model:** An instruction interpreter dispatches operations to resident thread blocks. Memory loads for the next instruction begin before the current one finishes.

### Why It Matters for Us

This is the nuclear option for the pipelining floor. Instead of trying to make RMSNorm faster, you make it free by absorbing it into a larger kernel that was launching anyway. The 4.1us overhead becomes zero because there's no separate launch.

For our project: we already have a flash attention kernel. The natural fusion point is:
1. Attention kernel prologue loads Q tiles from global memory
2. Before feeding Q tiles to MMA, compute RMSNorm in shared memory or registers
3. The norm adds ~768 FLOPs (for D=768) of reduction + division, which is negligible vs the MMA compute

### Detailed Architecture (from 2026-03-14 research update)

**Hazy Research "No Bubbles" (H100/B200):**
- Shared memory divided into 13 x 16KB pages on H100 (213KB total)
- Instructions request/release pages from an on-GPU interpreter
- Released pages automatically pass to the next instruction, so weight loads begin as soon as shared memory frees up
- Synchronization uses global memory counter arrays (no cooperative groups): instructions increment counters on completion, next instructions poll until target values are reached
- MLP optimization: intermediate states divided into 4 chunks with independent counters, enabling earlier down-projection
- Result: sub-millisecond forward pass for Llama-1B on H100, 78% memory bandwidth utilization

**MegaQwen (RTX 3090, sm_86):**
- Uses cooperative groups `grid_sync()` instead of counter-based sync
- **Bottleneck: synchronization latency, NOT memory bandwidth** -- only 5% bandwidth utilization (47 GB/s of 936 GB/s peak)
- ~140 grid_sync calls per token at ~0.7us each = ~100us synchronization overhead
- This sets a ceiling of ~530 tok/s for single-batch bfloat16 cooperative megakernels
- Uses `__ldg()` for weights (texture cache path) while L1/L2 handles activations

**Implication for our approach:** The counter-based synchronization (Hazy Research) avoids cooperative group overhead. For fusing just RMSNorm into attention (not a full megakernel), no grid-wide sync is needed at all -- the norm is block-local.

### Caveats

- MegaQwen targets sm_86 (RTX 3090), not sm_120. But cooperative groups work on sm_120.
- Full megakernel (entire transformer block) is a huge engineering effort. But fusing just RMSNorm into the existing attention kernel is tractable.
- Cooperative groups `grid_sync()` requires launching with `cudaLaunchCooperativeKernel`. This limits the number of thread blocks to what can all be resident simultaneously.
- On sm_120 with 170 SMs: if attention uses 3 blocks/SM, max 510 blocks. This is fine for our configs.
- **MegaQwen's sync overhead (100us/token) is a warning:** cooperative group megakernels on consumer GPUs are sync-limited, not compute-limited. For our use case (fusing norm into attention, no grid sync), this is irrelevant.

---

## Strategy 4: CUTLASS Epilogue/Prologue Fusion Pattern

**Source:** https://github.com/NVIDIA/cutlass/tree/main/examples/37_gemm_layernorm_gemm_fusion

### What This Is

CUTLASS example 37 demonstrates fusing GEMM1 + LayerNorm + GEMM2 into a single kernel. The LayerNorm is split into two halves:
- First half (partial reduction) runs in the epilogue of GEMM1
- Second half (finalize + apply) runs in the prologue of GEMM2

The GEMM output accumulator stays in registers/shared memory; the norm is applied without a global memory round-trip.

### The Technique

For RMSNorm (simpler than LayerNorm — no mean subtraction):
1. GEMM1 epilogue: each thread block computes partial sum-of-squares over its output tile
2. Cross-block reduction: partial sums are reduced (atomics or cooperative groups)
3. GEMM2 prologue: each element is divided by the computed RMS before being fed to GEMM2's input

The key constraint: the output tile of GEMM1 must align with the input tile of GEMM2 so data stays on-chip.

### Why It Matters for Us

This is the CUTLASS-blessed pattern for exactly what the worker needs. For the attention path:
- The "GEMM1" is the previous layer's output projection or MLP
- The "norm" is RMSNorm
- The "GEMM2" is the QKV projection (or the attention kernel itself)

Even without the full GEMM1+Norm+GEMM2 fusion, the pattern of "compute norm reduction in registers during the load phase, apply division before MMA" is directly applicable to our hand-written attention kernel.

### Caveats

- CUTLASS example 37 targets Ampere (sm_80). The pattern is architecture-agnostic — it's about data flow, not instructions.
- The cross-block reduction for RMS is the tricky part. For RMSNorm on a vector of length D=768, if the attention kernel loads the full row, the reduction is block-local (no cross-block communication needed).

---

## Strategy 5: Transformer Engine's Fused LayerNormLinear (Quantized Norm + GEMM)

**Source:** https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/pytorch.html

### What This Is

NVIDIA Transformer Engine provides `LayerNormLinear` and `LayerNormMLP` modules that fuse RMSNorm + Linear into a single operation. When FP8 training is enabled, the norm output is quantized directly to FP8 without materializing the high-precision intermediate — the norm kernel writes FP8 data that the GEMM reads.

### The Technique

Two fusion levels:
1. **Memory fusion:** Norm output is written in FP8 format directly, GEMM reads it. Two kernels but no BF16 intermediate.
2. **Kernel fusion (with_quantized_norm=True):** The normalization kernel produces FP8 output in a single pass, eliminating the quantization kernel entirely.

The actual kernel-level implementation in TE launches norm and GEMM as separate but tightly coupled kernels — the norm output goes directly to the GEMM input through a staging buffer. True single-kernel fusion (norm in GEMM prologue) is done via the CUTLASS EVT (Epilogue Visitor Tree) system for sm_90+.

### Why It Matters for Us

For our FP8 attention path: if we compute RMSNorm and convert to FP8 in the same operation (as TE does), we eliminate both the standalone norm kernel AND the FP8 conversion kernel. The attention kernel receives pre-normalized FP8 Q/K/V.

### Caveats

- TE's implementation is tightly coupled to cuBLAS/CUTLASS. We'd need to adapt the pattern.
- The "quantized norm" approach (norm output directly in FP8) is particularly relevant for our FP8 attention kernel, which currently converts BF16 inputs to FP8 inside the kernel. If inputs arrive pre-normalized and pre-quantized, the conversion overhead (the 14:1 ratio from agent_state) drops significantly.

---

## Strategy 6: vLLM's Fused QK-Norm-RoPE Kernel

**Source:** https://docs.vllm.ai/en/latest/design/fusions/

### What This Is

vLLM implements `fused_qk_norm_rope`: a single CUDA kernel that performs split QKV, reshape, Q/K RMSNorm, reshape, and rotary embedding. It also implements `fuse_allreduce_rms` (AllReduce + RMSNorm) and `fuse_norm_quant` (RMSNorm + Quantization).

### The Technique

The fused kernel:
1. Reads QKV output from global memory in big aligned chunks
2. Caches the fp16 rows in shared memory
3. Computes Q-norm and K-norm reductions
4. Applies normalization + RoPE
5. Writes normalized, rotated Q/K/V back

Key optimization: shared memory reuse avoids a second global memory read. The reduction and normalization happen entirely on-chip.

### Why It Matters for Us

This is the closest existing implementation to what we need. The pattern of "load -> cache in smem -> compute reduction -> apply norm -> output" is exactly the RMSNorm fusion approach, just applied post-projection rather than pre-projection.

For pre-attention normalization: the same pattern applies. Load activation rows, compute RMS in shared memory, apply division, then feed to QKV projection or directly to attention.

### Caveats

- vLLM notes this fusion has "perf issues on H100" and is not enabled by default. The overhead of the shared memory staging may not always win vs. separate kernels.
- For sm_120 consumer GPU, the shared memory staging may work better than on H100 because we have higher SM counts relative to memory bandwidth.

---

## Register/Smem Budget Impact of Fusing Norm into Attention (2026-03-14 research update)

**Question:** Does fusing RMSNorm into the attention kernel's Q-load path blow the register/smem budget?

**Answer: No.** The cost is minimal:

1. **Register cost of RMSNorm reduction:**
   - 1 FP32 accumulator for sum-of-squares (1 register)
   - 1 FP32 for rsqrt result (1 register)
   - Warp shuffle reduction reuses existing registers
   - Total: ~2-4 extra registers. Current attention kernel uses 145 registers with 3 blocks/SM. Adding 4 registers to 149 does NOT change the block count.

2. **Shared memory cost:**
   - Zero additional smem if normalization is done during the Q load that already happens
   - Q tiles are already in shared memory; the reduction reads from the same smem buffer
   - The gain vector (D=768 = 1536 bytes for BF16) could be stored in smem or loaded from constant memory. 1.5KB is negligible vs the 32KB Q tile.

3. **Compute cost:**
   - For D=768: 768 FMAs for sum-of-squares + 1 rsqrt + 768 multiplies for normalization = ~1537 FLOPs
   - Attention MMA compute for one Q-tile (BQ=64, D=64): 64*64*64*2 = 524,288 FLOPs per KV block
   - Norm cost is 0.3% of one KV block's MMA compute. Negligible.

4. **CUDA 13 shared memory register spilling:**
   - CUDA 13 introduces `enable_smem_spilling` pragma that redirects register spills to shared memory instead of local memory
   - If the fused kernel does push register count slightly above the 3-block threshold, this pragma can help by spilling the least-used registers to smem at lower latency than local memory
   - Enabled via: `asm volatile(".pragma \"enable_smem_spilling\";" :::);`

**Bottom line:** Fusing RMSNorm into the attention kernel's Q-load path adds negligible register pressure (<4 registers), zero additional shared memory, and <0.3% compute overhead. The attention kernel's 3 blocks/SM occupancy is preserved.

---

## Recommended Path for the RMSNorm Worker

Given the 4.1us pipelining floor, the most actionable approaches in order of effort:

### Quick Win: FlashNorm Weight Absorption (Minimal Code Change)
- Absorb RMSNorm gains into QKV weight matrices at model init time
- RMSNorm becomes a scalar RMS(x) computation + scalar division
- The scalar division can be deferred to after the QKV GEMM (runs in parallel)
- **Eliminates the standalone kernel entirely at the model level**
- Effort: ~20 lines of Python to modify weight init

### Medium Effort: Fuse Norm into Attention Kernel Prologue
- In the attention kernel, before feeding Q tiles to MMA:
  1. Load a row of activations into shared memory (the kernel does this anyway for Q)
  2. Compute sum-of-squares reduction across the row (warp shuffle reduction, ~10 instructions)
  3. Compute rsqrt
  4. Multiply each element by rsqrt * gain before feeding to MMA
- For D=768: the reduction is 768 elements = 24 float4 loads per row. Trivial.
- **Saves 4.1us per attention call**
- Effort: ~50 lines of CUDA in the attention kernel's Q-loading path

### High Effort: Full Megakernel
- Fuse RMSNorm + QKV projection + attention + output projection
- Uses cooperative groups for grid-wide sync between phases
- **Saves all inter-kernel overhead in the attention block**
- Effort: Major rewrite, but MegaQwen proves it works on consumer GPUs (sm_86+)

---

## References

- FlashNorm paper: https://arxiv.org/html/2407.09577v1
- Mirage RMSNorm+Linear tutorial: https://mirage-project.readthedocs.io/en/latest/tutorials/rms-norm-linear.html
- Mirage Persistent Kernel paper: https://arxiv.org/html/2512.22219v1
- Hazy Research "No Bubbles" megakernel: https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles
- MegaQwen (RTX 3090 megakernel): https://github.com/Infatoshi/MegaQwen
- CUTLASS GEMM+LayerNorm+GEMM fusion: https://github.com/NVIDIA/cutlass/tree/main/examples/37_gemm_layernorm_gemm_fusion
- CUTLASS Epilogue Visitor Trees: https://research.colfax-intl.com/epilogue_visitor_tree/
- NVIDIA Transformer Engine fused modules: https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/pytorch.html
- vLLM fusion passes: https://docs.vllm.ai/en/latest/design/fusions/
- Nunchaku fused QKV+RMSNorm+RoPE: https://nunchaku.tech/docs/nunchaku/python_api/nunchaku.ops.fused.html
- Liger Kernel (fused Triton kernels): https://github.com/linkedin/Liger-Kernel
- CUDA Graphs constant-time launch: https://developer.nvidia.com/blog/constant-time-launch-for-straight-line-cuda-graphs-and-other-performance-enhancements/
