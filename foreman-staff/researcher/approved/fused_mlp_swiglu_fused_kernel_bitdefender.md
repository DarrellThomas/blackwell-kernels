# Fused SwiGLU Kernel via Column Interleaving

**Source:** https://bit-ml.github.io/blog/post/fused-swiglu-kernel/ | https://github.com/fattorib/fusedswiglu
**Relevant to:** fused MLP worker (fused-mlp/)
**Worker's current problem:** Phase 4 is SwiGLU support. Phase 2 full fusion failed due to O(D_out/BLOCK_N) redundancy. Need a practical approach to fuse the gate + up-projection + activation.

## What This Is

Bitdefender Research published a CUDA C++ fused SwiGLU kernel using CuTe/CUTLASS. The key insight: instead of two separate GEMMs for gate and up projections, **interleave gate and up-projection columns in a single concatenated weight matrix**. One GEMM produces both projections, and the gating (SiLU + element-wise multiply) is applied in the epilogue.

## Why It Matters for Us

The fused MLP worker's Phase 4 is SwiGLU. SwiGLU requires:
1. `gate = X @ W_gate` (GEMM1a)
2. `up = X @ W_up` (GEMM1b)
3. `hidden = SiLU(gate) * up` (activation + element-wise multiply)
4. `Y = hidden @ W_down` (GEMM2)

Naively this is 3 GEMM kernel launches + 1 activation kernel. The Bitdefender approach fuses steps 1-3 into a single kernel by:
- Concatenating W_gate and W_up with interleaved columns: `W_combined[:, 2i] = W_up[:, i]`, `W_combined[:, 2i+1] = W_gate[:, i]`
- Running a single GEMM that produces interleaved gate/up outputs
- Applying `SiLU(gate) * up` in the epilogue (register-only, no extra memory traffic)

This avoids the v2 redundancy problem entirely because it's a standard GEMM with an epilogue — no second GEMM fused in.

## Key Technique

### Weight interleaving (one-time setup):
```python
# Interleave gate and up columns
W_combined = torch.empty(D, 2 * D_ff)
W_combined[:, 0::2] = W_up    # even columns = up-projection
W_combined[:, 1::2] = W_gate  # odd columns = gate
```

### Kernel epilogue:
```cpp
// After GEMM accumulation, each thread has pairs of (up, gate) values
// Apply SwiGLU in registers:
float up_val = acc[2*i];
float gate_val = acc[2*i + 1];
float silu_gate = gate_val / (1.0f + expf(-gate_val));  // SiLU
float result = up_val * silu_gate;  // element-wise multiply
// Write result (half the width of GEMM output)
```

### Performance:
- Achieves 95-98% of cuBLAS throughput for the GEMM portion
- Eliminates one kernel launch and one intermediate tensor write/read
- Memory savings: ~50% (no separate gate and up intermediate tensors)

## Caveats

- **Only fuses GEMM1 (gate+up), not GEMM2 (down-projection).** This is the same scope as the worker's v1 epilogue-fused approach, just extended to SwiGLU's dual projections. GEMM2 remains separate.
- **The Bitdefender kernel uses CuTe/CUTLASS** which we don't use (our kernels are hand-written mma.sync PTX). But the technique (column interleaving + epilogue fusion) transfers directly — it's an algorithmic trick, not a framework dependency.
- **Output tile width is halved.** The GEMM output has 2*D_ff columns, but the final result has D_ff columns (after gating). This means the effective compute is the same as doing two separate D_ff-wide GEMMs, but with better memory access (single contiguous weight matrix, single kernel launch).
- **Weight layout change required.** The interleaved weight matrix must be prepared once at model load time. This changes the model checkpoint format or requires a conversion step.
- **SiLU (sigmoid linear unit) uses `1/(1+exp(-x)) * x`** — maps to MUFU.EX2 + arithmetic in SASS. This is a few instructions in the epilogue, negligible cost.
- **Targets Ampere.** The Bitdefender code targets sm_80, but the mma.sync approach transfers to sm_120 directly (same ISA family).
