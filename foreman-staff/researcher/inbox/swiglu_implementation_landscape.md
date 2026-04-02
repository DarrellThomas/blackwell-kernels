# SwiGLU Fused Kernel Implementation Landscape (2024-2026)

**Source:** Multiple (see inline links)
**Relevant to:** fused-mlp worker (Phase 4: SwiGLU)
**Worker's current problem:** Need to implement SwiGLU (gated variant with two up-projections + element-wise multiply). Already has Bitdefender column-interleaving brief.

## Summary of All Known Approaches

There are four distinct strategies for fusing SwiGLU on GPU. They differ in
what they fuse (just the pointwise gating, GEMM1+gating, or the full GEMM chain)
and how they handle the two up-projections.

### Approach 1: Pointwise Fusion Only (Liger-Kernel)

**What it fuses:** Only the element-wise `SiLU(gate) * up` after two separate GEMMs.
GEMMs are still separate cuBLAS calls.

**How it works:**
- Compute `x1 = X @ W_gate` and `x2 = X @ W_up` via separate GEMMs (cuBLAS)
- Fuse `y = SiLU(x1) * x2` into one Triton kernel (forward + backward)
- Backward recomputes SiLU during backward pass (saves 1.6x peak memory)

**Performance:** No GEMM speedup. Saves one kernel launch + one intermediate tensor.
Memory savings ~1.6x at seq_len=16384.

**Relevance to us:** LOW. This is the minimal fusion. Our worker already has the
epilogue-fused GEMM approach which is strictly better -- it fuses the pointwise
ops into the GEMM epilogue, avoiding both a kernel launch AND an intermediate write.

**Source:** https://arxiv.org/html/2410.10989v2

### Approach 2: Column-Interleaved GEMM + Epilogue (Bitdefender / fal.ai)

**What it fuses:** Both up-projections + SiLU + element-wise multiply into a single GEMM.

**How it works:**
- Interleave W_gate and W_up columns: `W_combined[:, 2i] = W_up[:, i]`, `W_combined[:, 2i+1] = W_gate[:, i]`
- One GEMM of size `[M, D] x [D, 2*D_ff]` produces interleaved (up, gate) pairs
- In the epilogue, each thread naturally holds a (up, gate) pair due to mma.sync's
  column-pair thread mapping: `result = up * SiLU(gate)`
- Write only D_ff columns (half the GEMM width)

**Performance:** 95-98% of cuBLAS TFLOP/s for the GEMM portion. Memory halved
(no separate gate/up intermediates). ~1.28x faster than separate GEMM + pointwise.

**Key detail (from fal.ai):** The epilogue uses a coordinate transform `n >>= 1`
to map from the 2*D_ff GEMM columns to D_ff output columns. The SiLU computation
is just `gate / (1.0f + expf(-gate))` in registers -- negligible cost.

**Relevance to us:** HIGH. This is exactly what the worker should implement.
The Bitdefender brief already covers this. The fal.ai blog adds practical detail
about the CUTLASS epilogue visitor pattern (visit() per fragment, end_loop() for
the gated multiply). Our worker doesn't use CUTLASS but the register-level approach
is identical for hand-written mma.sync.

**Sources:**
- Bitdefender: https://bit-ml.github.io/blog/post/fused-swiglu-kernel/
- fal.ai: https://blog.fal.ai/crafting-efficient-kernels-with-epilogue-fusion/

### Approach 3: Full GEMM Chain Fusion (FlashFuser / DeepFusionKernel)

**What it fuses:** GEMM1(gate+up) + SiLU + element-wise multiply + GEMM2(down) into one kernel.

**FlashFuser (H100 only):**
- Uses Distributed Shared Memory (DSM) for inter-SM communication
- Producer SMs compute GEMM1 slices, consumers compute GEMM2
- dsm_shuffle for ring communication, dsm_reduce_scatter for output
- 3.3x vs cuBLAS, 58% fewer global memory accesses
- REQUIRES thread block clusters (sm_90+) and DSM -- NOT available on sm_120

**DeepFusionKernel (A100/H100):**
- Fuses all 4 SwiGLU ops into one kernel
- Profiler-driven scheduler selects row-major vs column-major tiling
- 9.7% speedup on A100, 13.2% on H100 (bandwidth-bound scenarios)
- Architecture-specific tuning, no sm_120 results

**Relevance to us:** LOW for direct use (requires hardware features we don't have).
BUT the DeepFusionKernel's modest 9.7-13.2% gains on datacenter GPUs confirm
that full GEMM chain fusion offers diminishing returns -- the worker's v2 post-mortem
already proved this with the O(D_out/BLOCK_N) redundancy analysis. These papers
validate that decision.

**Sources:**
- FlashFuser: https://arxiv.org/html/2512.12949
- DeepFusionKernel: https://arxiv.org/html/2602.11808

### Approach 4: NVIDIA cuDNN GEMM+SwiGLU (SM100 only)

**What it fuses:** Single GEMM + SwiGLU epilogue (same as Approach 2 conceptually).

**How it works:**
- Pairs consecutive 32-column blocks: [X0 | G0 | X1 | G1 | ...]
- N must be divisible by 64 (two consecutive 32-wide blocks)
- Produces both full GEMM output (AB12) and half-width SwiGLU result (C)
- Tile sizes: TILE_M in {128, 256}, TILE_N in {32..256}, default (128, 128)
- Thread block clustering with CLUSTER_M * CLUSTER_N <= 16

**Relevance to us:** NONE for direct use (SM100 only, explicitly excludes SM103/SM120).
But confirms NVIDIA's own approach is column-interleaving, validating Approach 2.

**Source:** https://docs.nvidia.com/deeplearning/cudnn/frontend/latest/fe-oss-apis/gemm_fusions/gemm_swiglu.html

### Approach 5: TensorRT-LLM GemmUniversalGated (SM90+)

**What it fuses:** GEMM + gated activation via CUTLASS GemmUniversalGated kernel class.

**How it works:**
- Uses `cutlass::gemm::kernel::GemmUniversalGated` with TMA warp-specialized schedule
- Tile shape 128x128x128, cluster shape 1x2x1
- FP8 support with fast accumulation
- ~560 us per layer at m=19731, n=9728, k=896

**Relevance to us:** NONE for direct use (requires TMA, warp specialization patterns
not available on sm_120). But the tile dimensions and FP8 support are informative.

**Source:** https://github.com/NVIDIA/TensorRT-LLM/issues/6361

## Recommendation for the Worker

**Implement Approach 2 (column-interleaved GEMM + epilogue fusion).**

This is the consensus approach across all implementations that target mma.sync-era
GPUs. It is:
1. The same algorithmic approach used by Bitdefender, fal.ai, NVIDIA cuDNN, and TensorRT-LLM
2. Directly compatible with the worker's existing epilogue-fused GEMM (v1)
3. The only approach that works on sm_120 (no DSM, no TMA, no thread block clusters)

The worker's existing v1 kernel already does GEMM + activation epilogue. SwiGLU
is an incremental change:
- Double the N dimension of the GEMM (2*D_ff columns)
- Interleave W_gate/W_up columns in the weight matrix (one-time Python setup)
- Replace the current activation epilogue with `result = up * SiLU(gate)`
- Write half the columns (D_ff instead of 2*D_ff)

Full GEMM chain fusion (Approaches 3-5) is not worth pursuing on sm_120, consistent
with the worker's v2 post-mortem analysis.
