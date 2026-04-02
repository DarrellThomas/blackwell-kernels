# RMSNorm Backward Pass — Supplementary Implementation Details

**Sources:** Liger-Kernel source (rms_norm.py), Triton LayerNorm tutorial backward kernels, vLLM PR #22602, Dao-AILab/quack speed-of-light methodology, PyTorch issue #157345, Tri-RMSNorm benchmarks
**Relevant to:** rmsnorm worker
**Supplements:** `rmsnorm_backward_pass_kernel.md` (the math and architecture brief)

---

## 1. Production-Quality Backward Kernel Code (from Liger-Kernel)

The existing brief has the formulas; here is the exact production Triton kernel code
that implements them. Two key details emerge from reading the actual code:

### dx Computation (Per-Row)

```python
# The core dx computation per row (from Liger-Kernel _rms_norm_backward_kernel):
X_row = X_row.to(tl.float32)                              # upcast to FP32
m = (dY_row * W_row).to(tl.float32)                       # wdy = dy * g, in FP32
dX_row = rstd_row * m                                      # direct term: rstd * wdy
dX_row += rstd_row * (-(1/n_cols) * rstd_row * rstd_row   # indirect term:
          * tl.sum(m * X_row, axis=0) * X_row)             #   -rstd^3/N * dot(wdy,x) * x
```

**Key insight:** The formula uses `rstd^3` (rstd_row * rstd_row * rstd_row) rather than
the common presentation of `rstd * (... - x_hat * c1)`. Both are mathematically equivalent:
- Textbook: `dx = rstd * (wdy - x_hat * (1/N) * dot(wdy, x_hat))`
- Liger: `dx = rstd * wdy - rstd^3 * (1/N) * dot(wdy, x) * x`

The Liger version avoids computing x_hat = x * rstd separately, saving one multiply
per element. Instead it folds the extra rstd factors into the indirect term. This is
a minor optimization but shows attention to detail.

### dg Accumulation (The Simpler Strategy)

Liger-Kernel does NOT use the Triton tutorial's lock-based partial sums. Instead:

```python
# Each block accumulates partial_dg in registers across its rows:
for row_idx in range(row_start, row_end):
    ...
    dW_row += dY_row * (X_row * rstd_row)   # partial_dg in FP32 registers

# After loop, write partial to a per-block buffer (no locks, no atomics):
tl.store(dW_ptr + row_block_id * dW_row_stride + col_offsets, dW_row, mask=mask)
```

Then in Python: `dW = _dW.sum(dim=0).to(W.dtype)` -- a simple sum over partial buffers.

**Why this works better than locks for our case:**

- Grid = ceil(rows / rows_per_program) blocks. For 2048 rows and 32 rows/block = 64 blocks.
- Each block writes its own row in a [64, D] partial buffer. Zero contention.
- The final sum is a tiny [64, 768] reduction -- negligible vs the main kernel.
- No spin-loops, no atomic CAS, no lock overhead. Simpler code, same performance.

**Recommendation for CUDA implementation:** Allocate a `float[num_blocks][D]` buffer.
Each block writes its partial dg. After the kernel, sum across blocks with a small
second kernel or `cublasSgemv` with a ones vector. This is simpler than the Triton
tutorial's lock pattern and avoids atomic contention entirely.

---

## 2. Block-Mode vs Row-Mode: Two Kernel Strategies

Liger-Kernel implements TWO backward kernels and selects at runtime:

### Row-Mode (_rms_norm_backward_kernel)
- Grid: `(num_programs,)` where each program handles `rows_per_program` consecutive rows
- Each thread handles `BLOCK_SIZE` columns (one row width)
- Simple 1D tiling: threads iterate over rows in their assigned range
- Best for small-to-medium row counts where occupancy is adequate

### Block-Mode (_block_rms_norm_backward_kernel)
- Grid: `(NUM_SMS,)` -- one program per SM (persistent grid)
- Each program handles a BLOCK_ROW-sized chunk of rows per iteration
- Uses 2D indexing: `row_idx[:, None] * stride + col_offsets[None, :]`
- Processes multiple rows simultaneously per program invocation
- Best for large row counts where SM cycling improves cache utilization

**For our case (2048 rows, 768 D):** Row-mode with 64 blocks (32 rows/block) is likely
optimal. Block-mode with 170 programs (1 per SM) cycling through rows may have cache
locality benefits but adds indexing complexity.

---

## 3. PyTorch's RMSNorm Backward: The Benchmark Target is Weak

PyTorch issue #157345 reveals that `nn.RMSNorm` is currently SLOWER than `nn.LayerNorm`
in some configurations, which is the opposite of what the math predicts. The issue exists
because PyTorch's RMSNorm backward may not have a dedicated optimized CUDA kernel and
instead falls back through the autograd machinery.

**What this means for us:** The bar to beat may be even lower than expected. If PyTorch's
backward for RMSNorm is poorly optimized, achieving 1.5-2x speedup on the backward pass
is plausible, even with a straightforward implementation.

To confirm: profile PyTorch's backward with `torch.autograd.profiler` to see how many
CUDA kernels it launches and what they are. If it launches 3-4 small kernels (the
layer_norm_grad_input_kernel, a separate column reduction, etc.), the kernel launch
overhead alone is beatable.

---

## 4. Speed-of-Light Methodology for Memory-Bound Backward Kernel

From Dao-AILab/quack's approach to getting memory-bound kernels to speed-of-light:

### Two Key Ingredients
1. **Global memory coalesced load/store** -- ensure all reads/writes are 128-byte aligned
   and access consecutive addresses within a warp.
2. **Hardware-aware hierarchical reduction** -- reduce at each level of the memory hierarchy:
   - Thread-level: FMA accumulation in registers
   - Warp-level: `__shfl_xor_sync` butterfly reduction
   - Block-level: shared memory reduction across warps
   - Grid-level: partial buffers in global memory (for dg)

### Memory Hierarchy Exploitation
Allocate most local reduction at the highest (register) level. Only forward small
locally-reduced values to the next level. For the backward kernel:
- The dot product `sum(wdy * x)` should be accumulated per-thread first (register),
  then reduced across the warp (shuffle), then across warps (shared memory). Three
  levels, not one big shared memory reduction.
- This matches what the forward kernel already does for sum-of-squares.

### Bandwidth Target
For D=768, rows=2048:
- Read: dy (3.0 MB) + x (3.0 MB) + rstd (8 KB) + weight (1.5 KB) = ~6.0 MB
- Write: dx (3.0 MB) + partial_dg (64*768*4 = 192 KB) = ~3.2 MB
- Total: ~9.2 MB
- At 1792 GB/s: theoretical minimum = 5.1 us
- At 88% utilization (Triton RMSNorm forward achieves this on A100): ~5.8 us
- **Target: 5-7 us for the backward kernel**

---

## 5. Casting Modes: BF16 Precision Handling

Liger-Kernel implements three casting modes for the backward pass that affect where
FP32 upcasting occurs:

### LLAMA Mode (default for most models)
```
wdy = (dy * weight).to(float32)        # multiply in BF16, THEN upcast
dg += dy * (x * rstd).to(input_dtype)  # normalize in FP32, downcast, then multiply
```

### GEMMA Mode
```
dy = dy.to(float32)                     # upcast dy FIRST
wdy = dy * weight                       # multiply in FP32
dg += dy * (x * rstd)                   # everything in FP32
```

### Default Mode
```
wdy = dy * weight                       # type depends on input
dg += dy * (x * rstd)                   # type depends on input
```

**For our BF16 implementation:** Use LLAMA mode. The key insight is that `dy * weight`
can be done in BF16 before upcasting, because the weight vector is small and the product
is immediately accumulated in FP32. This saves one upcast per element for the dy*weight
multiply without meaningful precision loss.

---

## 6. Fused Backward with Residual Add (Training Optimization)

In transformer training, the common pattern is:
```
residual = residual + sublayer_output     # residual add
y = rmsnorm(residual)                     # normalize
```

The backward of this is:
```
d_residual += dx                          # gradient flows through residual connection
d_sublayer = dx                           # gradient also flows to sublayer
```

Where `dx` is the RMSNorm backward output. The residual add backward is trivial
(identity on both paths), but it means **dx is read twice** by downstream consumers.

**Fusion opportunity:** If the backward kernel also accumulates the residual gradient
(i.e., `d_residual += dx` happens inside the kernel), it eliminates one extra read
of the dx tensor. PaddlePaddle and Transformer Engine both offer this fused variant
(`nvte_rmsnorm_bwd_add`). For our implementation: consider an optional `d_residual`
parameter that, when provided, atomically adds dx to it instead of writing dx to
a separate buffer.

This is a phase-2 optimization after the basic backward works correctly.

---

## 7. Performance Benchmarks from Reference Implementations

| Implementation | Hardware | Forward+Backward Speedup | Notes |
|---|---|---|---|
| Tri-RMSNorm (Triton) | A100 | ~28.6% vs LayerNorm custom kernel | Both fwd+bwd combined |
| Tri-RMSNorm (Triton) | A100 | ~10.2% vs PyTorch standalone RMSNorm | Both fwd+bwd combined |
| Liger-Kernel RMSNorm | A100 | 2.98-3.82x vs standard (forward only) | Backward not separately benchmarked |
| vLLM vectorized RMSNorm | Various | -32% to -60% latency (forward only) | Backward not applicable (inference) |
| Custom Triton (subhadipmitra) | A100 | 8.1x vs PyTorch (forward only, 88% BW util) | Shows BW ceiling |

**Key takeaway:** No one publishes isolated backward pass benchmarks. The backward
is always tested as part of forward+backward combined timing. This makes sense because
in training, you always run both. Our benchmark should measure combined fwd+bwd time
vs PyTorch's combined fwd+bwd time.

---

## Sources

- [Liger-Kernel RMSNorm source](https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/rms_norm.py)
- [Liger-Kernel paper](https://arxiv.org/html/2410.10989v2)
- [Triton LayerNorm tutorial (backward kernels)](https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html)
- [vLLM vectorized RMSNorm PR](https://github.com/vllm-project/vllm/pull/22602)
- [Dao-AILab quack speed-of-light methodology](https://github.com/Dao-AILab/quack/blob/main/media/2025-07-10-membound-sol.md)
- [PyTorch RMSNorm performance issue](https://github.com/pytorch/pytorch/issues/157345)
- [Tri-RMSNorm](https://github.com/dtunai/Tri-RMSNorm)
- [Triton RMSNorm kernels](https://subhadipmitra.com/blog/2025/triton-kernels-llm-inference/)
- [LayerNorm gradient derivation](https://robotchinwag.com/posts/layer-normalization-deriving-the-gradient-for-the-backward-pass/)
- [NVIDIA Transformer Engine normalization API](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/c/normalization.html)
