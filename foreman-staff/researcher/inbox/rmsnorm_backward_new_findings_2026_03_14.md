# RMSNorm Backward Pass — New Findings (March 2026)

**Sources:** Chronicals (arxiv 2601.02609), MS-RMSNorm (arxiv 2406.16282), Mirage superoptimizer, Liger-Kernel NPU PR #1070, PyTorch AOTAutograd fused_rms_norm_backward discussion, CompAct (NAACL 2025), Dao-AILab/quack CuTe-DSL speed-of-light methodology
**Relevant to:** rmsnorm worker
**Worker's current problem:** Forward pass at 1.2-1.4x PyTorch (4.1 us pipelining floor). Next step is backward pass for training.
**Supplements:** `rmsnorm_backward_pass_kernel.md` and `rmsnorm_backward_implementation_details.md` (already delivered)

---

## 1. Chronicals Framework: 6.2x Backward Speedup Over PyTorch (January 2026)

**Source:** https://arxiv.org/html/2601.02609

Chronicals is a January 2026 LLM fine-tuning framework that achieves 3.51x overall
speedup over Unsloth. Its fused RMSNorm Triton kernel reports **6.2x speedup** for
the backward pass vs PyTorch on A100.

**Key backward pass details from their Algorithm 5:**

The gradient formula uses the form:
```
dx_i = (gamma_i / r) * (dL/dy_i - (x_i / r^2) * sum_j(dL/dy_j * x_j * gamma_j))
```

This is mathematically equivalent to what we already have, but note the implementation
choice: they compute `gamma_i / r` (weight divided by RMS) rather than first computing
`x_hat = x * rstd` and then `rstd * (wdy - x_hat * c1)`. Both are equivalent but
the Chronicals version avoids computing x_hat as a separate intermediate.

**What matters for us:** The 6.2x backward speedup suggests PyTorch's backward is
very weak. The existing brief already noted PyTorch issue #157345 (RMSNorm slower
than LayerNorm in some configs). A straightforward fused CUDA backward should easily
beat PyTorch by 2-4x, possibly more, because PyTorch likely launches multiple
separate kernels for the backward.

**Verification step:** Profile PyTorch's backward with nsys to count how many CUDA
kernels it launches. If it's 3-4 separate kernels, the kernel launch overhead alone
accounts for 10-20 us on top of compute.

---

## 2. MS-RMSNorm: Merge Affine Parameters into Following Linear Layer (2024)

**Source:** https://arxiv.org/html/2406.16282

This technique eliminates the need to store x_hat (normalized x) for the backward
pass by merging RMSNorm's affine weight gamma into the weight matrix of the
following linear layer:

```
W_merged = W * diag(gamma)
```

The backward pass then only needs to store:
- One vector in R^D (the raw input x)
- One scalar (rstd = 1/rms)

**Memory savings:** Eliminates the [rows, D] activation tensor that would normally
be saved between forward and backward. For LLaMA-13B, this reduces peak memory by
9.1 GiB.

**Why it matters for us:** This is a **training-level architectural optimization**,
not a kernel optimization. It changes the computation graph so that the standalone
RMSNorm backward pass can be simpler. However, for our kernel-level work, it is not
directly applicable -- we are writing standalone RMSNorm forward/backward kernels.
The technique is more relevant for framework-level integration.

**Takeaway:** Our backward kernel should NOT save x_hat -- only x and rstd. This
is already the recommendation in the existing brief. MS-RMSNorm validates that
approach from a different angle.

---

## 3. Mirage Superoptimizer: RMSNorm + Linear Fusion via Commutativity (2025)

**Source:** https://mirage-project.readthedocs.io/en/latest/tutorials/rms-norm-linear.html

Mirage discovered that the division in RMSNorm commutes with the multiplication
in the following MatMul:

```
Standard:  y = (x / rms) * gamma;  z = y * W
Mirage:    temp = x * W;  z = temp / rms  (after per-row scaling)
```

This allows loading x once to compute both `sum(x^2)` (for rms) and `x * W`
(for the matmul), fusing both into a single kernel.

**Performance:** 1.5-1.7x faster than running RMSNorm + Linear separately.

**Why it matters for us:** This is about fusing RMSNorm FORWARD with the downstream
linear layer. The backward analog would be fusing the RMSNorm backward with the
upstream gradient computation (the linear layer's backward). This is a phase-2/3
optimization after the standalone backward kernel works.

**Backward fusion implication:** In training, the gradient flows:
```
Linear backward: dL/dx_linear = dL/dy * W^T
RMSNorm backward: dL/dx_rmsnorm = f(dL/dx_linear, x, rstd, gamma)
```
If these could be fused, we would avoid materializing the intermediate
`dL/dx_linear` tensor entirely. The Mirage commutativity trick for the
forward pass suggests there may be a similar algebraic reordering for
the backward, but this has not been demonstrated.

---

## 4. PyTorch's fused_rms_norm_backward Under torch.compile (2025)

**Source:** https://discuss.pytorch.org/t/better-understanding-why-aotautograd-decomposes-fused-rms-norm-backward-for-cuda-but-not-for-meta-tensors/223096

PyTorch's AOTAutograd keeps `_fused_rms_norm_backward` as a single fused CUDA
operation when compiling for CUDA devices, but decomposes it into individual
operations (mul, sum, pow, add) for Meta tensors.

**What this tells us about the reference:**
- PyTorch DOES have a fused CUDA backward for RMSNorm (the `_fused_rms_norm_backward` op)
- But it may not always be used -- torch.compile's autograd tracing can decompose it
- The decomposed version (multiple small kernels) is what we're likely benchmarking against

**Implication:** When benchmarking our backward kernel vs PyTorch, we should test
both `nn.RMSNorm` (eager mode) and `torch.compile(nn.RMSNorm)` to see which
backward path PyTorch uses. The compiled version might be faster or slower
depending on whether AOTAutograd keeps the fused op.

---

## 5. Transformer Engine 2.12: nvte_rmsnorm_bwd_add (Reference Architecture)

**Source:** https://github.com/NVIDIA/TransformerEngine/releases

Transformer Engine added `nvte_rmsnorm_bwd_add` in release 2.8, providing a
fused RMSNorm backward + residual add kernel. The latest version is 2.12
(February 2025).

**Architecture details:**
- Separate workspace buffer for dg partial sums (dgamma_part)
- Multi-SM coordination via barriers
- Fused residual add: `d_residual += dx` happens inside the kernel
- Uses `cp.async.bulk` (TMA) on Hopper -- NOT applicable to sm_120

**What matters for us:** The fused residual add pattern is a good phase-2
optimization. After the basic backward works:
1. Add an optional `d_residual` output parameter
2. When provided, atomically add dx to d_residual instead of writing to a separate buffer
3. This eliminates one read + one write of the dx tensor by downstream consumers

TE's approach uses barriers and workspace -- we should use the simpler Liger-Kernel
approach (per-block partial buffers, sum in a second kernel) since we don't have TMA.

---

## 6. CompAct: Compressed Activations for Backward Pass (NAACL 2025)

**Source:** https://arxiv.org/abs/2410.15352

CompAct stores low-rank compressed activations for the backward pass using random
projection matrices. For LLM pretraining, this reduces peak GPU memory by 25-30%.

**RMSNorm handling:** CompAct does NOT compress RMSNorm activations. The reason:
RMSNorm activations are already small (just x and rstd) compared to linear layer
activations. The compression overhead wouldn't be worth it.

**Takeaway:** This confirms that our approach (save x + rstd, recompute x_hat on
the fly) is already near-optimal for memory. There's no need to compress these
saved tensors further.

---

## 7. Liger-Kernel NPU fused_add_rms_norm (February 2026)

**Source:** https://github.com/linkedin/Liger-Kernel/pulls (PR #1070)

Active development as of February 2026 on NPU-optimized fused_add_rms_norm forward
kernel. This indicates the fused residual + RMSNorm pattern continues to be an
active area across hardware platforms.

The Liger-Kernel backward approach (per-block partial buffers for dg, no locks,
no atomics) remains the simplest and most portable. Their row-mode vs block-mode
selection at runtime is a good pattern for handling different row counts.

---

## 8. Dao-AILab/quack: CuTe-DSL Speed-of-Light for Memory-Bound Kernels (July 2025)

**Source:** https://github.com/Dao-AILab/quack/blob/main/media/2025-07-10-membound-sol.md

The quack methodology for reaching speed-of-light on memory-bound kernels uses
CuTe-DSL (NVIDIA's Python DSL on CUTLASS). Key principles:

1. **Thread-Value (TV) layout:** Partition data so each thread handles 128-bit
   (8x BF16 or 4x FP32) per memory instruction for maximum coalescing
2. **Multi-level hierarchical reduction:**
   - Register: FMA accumulation
   - Warp: shuffle butterfly reduction
   - Block: shared memory reduction
   - Grid: partial buffers in global memory
3. **Cluster reduction via DSMEM** (Hopper only -- NOT applicable to sm_120)

**What matters for us:** The first two principles directly apply. Our forward
kernel already uses vectorized int4 (128-bit) loads and warp shuffle reduction.
The backward kernel should follow the same pattern:
- 128-bit loads for x and dy
- Per-thread FMA for wdy and x_hat computation
- Warp shuffle for c1 reduction
- Cross-warp shared memory for block-level c1
- Per-block partial buffers for dg grid-level accumulation

The DSMEM cluster reduction is Hopper-only (sm_90+) and does NOT apply to sm_120.
For our case, the grid-level dg reduction is the only multi-block step, and it's
small enough ([num_blocks, D]) to handle with a simple second kernel.

---

## Summary: New Information vs Existing Briefs

| Finding | New info? | Impact on backward kernel |
|---------|-----------|--------------------------|
| Chronicals 6.2x backward speedup | YES | Confirms PyTorch backward is very weak |
| MS-RMSNorm parameter merging | New paper, but confirms existing recommendation | Validates save x+rstd only |
| Mirage commutativity trick | YES (forward fusion) | Backward fusion analog is future work |
| AOTAutograd fused backward | YES | Test both eager and compiled PyTorch |
| TE nvte_rmsnorm_bwd_add | Known API, new context | Phase-2: fused residual add |
| CompAct no-compress for RMSNorm | YES | Confirms our memory approach is optimal |
| Liger NPU fused_add PR | New dev activity | Validates fused residual pattern |
| quack CuTe-DSL SOL | YES (methodology) | Directly applicable to our backward kernel |

**Bottom line:** No novel backward pass algorithm has emerged since the existing
briefs. The techniques remain: (1) fused dx+partial_dg kernel, (2) per-block
partial buffers for dg, (3) optional second pass to sum dg. The news is that
PyTorch's backward appears to be weaker than expected (6.2x beatable), and the
residual add fusion is a confirmed phase-2 win.
