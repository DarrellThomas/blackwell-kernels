# RMSNorm Backward Pass Kernel — Research Brief

**Sources:** PyTorch `aten/native/cuda/layer_norm_kernel.cu`, Triton LayerNorm tutorial, NVIDIA Transformer Engine `nvte_rmsnorm_bwd`, Liger-Kernel RMSNorm, Tri-RMSNorm, LayerNorm gradient derivations (robotchinwag, shreyansh26)
**Relevant to:** rmsnorm worker
**Worker's current problem:** Forward pass at 1.2-1.4x PyTorch (4.1 us pipelining floor). Next direction is backward pass for training.

## 1. Mathematical Derivation

### Forward Pass (recap)

```
rms = sqrt( (1/n) * sum_j(x_j^2) + eps )
rstd = 1 / rms                               -- saved for backward
x_hat_i = x_i * rstd                         -- normalized x (= x_i / rms)
y_i = x_hat_i * g_i                          -- final output
```

### Backward Pass: dL/dx (input gradient)

Starting from `y_i = (x_i / rms) * g_i`, apply the chain rule. The key
complexity: rms depends on ALL x_j (via the sum-of-squares), so changing
any x_j affects every y_i through the shared rms denominator.

**Step 1: Direct path (through numerator)**

```
dy_i/dx_i|_direct = g_i / rms = g_i * rstd
```

**Step 2: Indirect path (through rms denominator)**

```
d(rms)/dx_j = (1/rms) * (1/n) * x_j = x_j * rstd / n

dy_i/dx_j|_indirect = -x_i * g_i / rms^2 * d(rms)/dx_j
                    = -x_i * g_i * rstd^2 * (x_j * rstd / n)
                    = -(g_i * x_hat_i * x_hat_j) * rstd / n
```

**Step 3: Chain rule (sum over loss gradient dy)**

```
dL/dx_j = sum_i( dL/dy_i * dy_i/dx_j )

For i=j (direct + indirect):
  dL/dy_j * g_j * rstd - (1/n) * rstd * sum_i( dL/dy_i * g_i * x_hat_i * x_hat_j )

For i!=j (indirect only):
  -(1/n) * rstd * dL/dy_i * g_i * x_hat_i * x_hat_j

Combining all i:
  dL/dx_j = rstd * ( dL/dy_j * g_j  -  x_hat_j * (1/n) * sum_i(dL/dy_i * g_i * x_hat_i) )
```

**Final formula:**

```
Let wdy_i = dL/dy_i * g_i                  -- element-wise (no reduction)
Let c1 = (1/n) * sum_i(wdy_i * x_hat_i)   -- ONE scalar reduction per row
Then:
    dL/dx_j = rstd * (wdy_j - x_hat_j * c1)
```

**Comparison to LayerNorm:** LayerNorm backward has TWO reductions (c1 and c2),
where c2 = (1/n) * sum(wdy). RMSNorm eliminates c2 entirely because there is
no mean subtraction. This makes RMSNorm backward strictly simpler than LayerNorm
backward.

### Backward Pass: dL/dg (weight gradient)

The weight gradient is straightforward per-element, but must be reduced across
rows (the batch/sequence dimension):

```
dL/dg_i = sum_rows( dL/dy_i * x_hat_i )
        = sum_rows( dL/dy_i * x_i * rstd )
```

This is a reduction across the row (batch) dimension, NOT across D. Each element
of dg accumulates contributions from every row.

## 2. Kernel Architecture

### What to Save from Forward Pass

| Saved Tensor | Shape | Bytes (D=768, rows=2048) | Purpose |
|---|---|---|---|
| x (input) | [rows, D] | 3.0 MB (BF16) | Needed to recompute x_hat |
| rstd | [rows] | 8 KB (FP32) | Reciprocal RMS, one per row |

**Do NOT save x_hat.** It is [rows, D] sized and would double memory. Instead,
recompute `x_hat_i = x_i * rstd` on the fly during backward. The recomputation
cost is 1 multiply per element -- negligible vs. the memory cost of saving it.

**Do NOT save y (output).** It's not needed for backward.

All major implementations (PyTorch, Transformer Engine, Liger-Kernel, Triton
tutorial) save `x` and `rstd` only. Some also save `weight` (g), but that's
a parameter tensor already available.

### Two-Kernel vs Fused Approach

There are two standard approaches:

**Approach A: Two separate kernels (PyTorch native)**

Kernel 1: `compute_grad_input` -- computes dx, one block per row
  - Load dy, x, g, rstd for this row
  - Compute wdy = dy * g, x_hat = x * rstd
  - Reduce c1 = (1/n) * sum(wdy * x_hat) via warp shuffle
  - dx = rstd * (wdy - x_hat * c1)
  - Write dx

Kernel 2: `compute_grad_weight` -- computes dg, one block per feature
  - For each feature index i, sum dy[row, i] * x_hat[row, i] across all rows
  - Write dg[i]
  - This is a column-wise reduction: reads dy and x across rows

**Approach B: Fused single kernel (Triton tutorial, Liger-Kernel)**

One kernel computes BOTH dx and partial dg:
  - Each block handles GROUP_SIZE_M rows (e.g., 32 rows)
  - Computes dx for each row (same as Kernel 1 above)
  - Accumulates partial_dg across its GROUP_SIZE_M rows
  - Uses locks + atomic accumulation to merge partial_dg from different blocks
  - A second tiny kernel sums the partial_dg across groups

The fused approach is faster because it reads x and dy once instead of twice.

### Recommended Architecture for sm_120

Use the **fused approach**. Here's the concrete design:

```
Kernel: rmsnorm_backward_fused
  Grid: (ceil(rows / GROUP_SIZE_M),)    -- e.g., ceil(2048/32) = 64 blocks
  Block: 256 threads
  Shared memory: D*2 bytes for x cache (same as forward)

  For each row r in [blockIdx.x * GROUP_SIZE_M, ... +GROUP_SIZE_M):
    1. Load x[r, :] into shared memory (vectorized int4, same as forward)
    2. Load dy[r, :] from global memory (vectorized int4)
    3. Load rstd[r] (single float)
    4. Compute wdy = dy * g, x_hat = x * rstd  (element-wise in registers)
    5. Reduce c1 = (1/n) * sum(wdy * x_hat)    (warp shuffle + cross-warp smem)
    6. Compute dx = rstd * (wdy - x_hat * c1)   (element-wise)
    7. Write dx[r, :] to global memory (vectorized int4)
    8. Accumulate partial_dg += dy * x_hat       (in registers, across rows)

  After all rows in group:
    9. atomicAdd partial_dg to global dg buffer
       (or use lock-based accumulation a la Triton tutorial)
```

### Weight Gradient (dg) Accumulation Strategy

The weight gradient is the tricky part. Every row contributes to the SAME dg
vector of shape [D]. Options:

**Option 1: atomicAdd (simplest)**
Each block atomically adds its partial_dg to global dg. With BF16, this requires
FP32 atomics (atomic BF16 add is not supported). Simple but has contention with
64 blocks writing to the same D=768 locations.

**Option 2: Lock-based partial sums (Triton tutorial pattern)**
Allocate `NUM_GROUPS` partial buffers of shape [D]. Each block writes to its
group's buffer. Use per-group locks to serialize blocks sharing a group. A
second kernel sums across groups. No atomicAdd needed.

**Option 3: Two-pass (PyTorch native)**
Separate kernel for dg that reads x and dy again. Simple code but 2x global
memory reads.

**Recommendation: Option 2** for best performance. GROUP_SIZE_M=32 means
ceil(2048/32)=64 blocks, and with e.g., 8 partial buffers, each buffer gets
8 blocks. The locks serialize only within a group, giving good parallelism.

## 3. Performance Analysis

### Memory Traffic (D=768, rows=2048)

**Forward pass:**
| Operation | Bytes | Direction |
|---|---|---|
| Read x | 2048 * 768 * 2 = 3.0 MB | Global -> SM |
| Read weight | 768 * 2 = 1.5 KB | Global -> SM (cached) |
| Write output | 3.0 MB | SM -> Global |
| **Total** | **~6.0 MB** | |

**Backward pass:**
| Operation | Bytes | Direction |
|---|---|---|
| Read dy | 3.0 MB | Global -> SM |
| Read x | 3.0 MB | Global -> SM |
| Read weight | 1.5 KB | Cached |
| Read rstd | 2048 * 4 = 8 KB | Global -> SM |
| Write dx | 3.0 MB | SM -> Global |
| Write dg | 768 * 4 = 3 KB | Global (negligible) |
| **Total** | **~9.0 MB** | |

The backward pass has **1.5x the memory traffic** of the forward pass (9 MB
vs 6 MB). With the fused approach (one pass), all reads and writes happen once.

### Arithmetic Intensity

Per element: ~8 FLOPs (2 multiplies for wdy and x_hat, 1 FMA for c1 reduction,
1 multiply + 1 subtract + 1 multiply for dx, plus the reduction overhead).

Arithmetic intensity = ~8 FLOPs / 6 bytes = ~1.3 FLOP/byte.

This is **deeply memory-bandwidth-bound** (the RTX 5090 balance point is 125
FLOP/byte). Same regime as the forward pass.

### Theoretical Minimum Time

```
Backward traffic: ~9 MB
Bandwidth: 1792 GB/s
Minimum time: 9e6 / 1.792e12 = 5.0 us
```

For comparison, the forward theoretical minimum is 6.0 MB / 1792 GB/s = 3.3 us.

The backward pass should take about **1.5x the forward pass time**. If the
forward runs at 4.1 us (including pipelining), expect the backward to run at
roughly **5-7 us** with similar pipelining.

### Is the backward harder to optimize?

**No, it's structurally identical to the forward.** Both are:
1. Read a row of data (vectorized int4 loads)
2. One reduction across D (warp shuffle + cross-warp smem)
3. Element-wise output computation (vectorized int4 stores)

The only addition is the dg accumulation, which is amortized across rows and
happens once per block-group, not once per row. The c2 reduction from LayerNorm
is absent, making RMSNorm backward simpler.

**The backward will hit the same 4.1 us pipelining floor** for the dx kernel
portion if rows and D are the same. The extra memory traffic (reading dy in
addition to x) may push the floor slightly higher.

## 4. Implementation Checklist

### Minimum Viable Backward

1. **Save rstd from forward pass.** Modify `rmsnorm_forward` to also return
   the rstd vector (shape [rows], FP32). Use `ctx.save_for_backward(x, rstd)`
   in the autograd wrapper.

2. **Write backward kernel.** Structure per-row:
   - Load x, dy, rstd, weight (vectorized int4)
   - Compute wdy = dy * g (element-wise)
   - Compute x_hat = x * rstd (element-wise)
   - Reduce c1 = (1/n) * dot(wdy, x_hat) (warp shuffle)
   - Compute dx = rstd * (wdy - x_hat * c1) (element-wise)
   - Store dx (vectorized int4)

3. **Weight gradient kernel.** Either fused (accumulate dg in the same kernel)
   or separate (second pass reading x and dy again).

4. **PyTorch autograd integration.** Wrap with `torch.autograd.Function`:
   ```python
   class RMSNorm(torch.autograd.Function):
       @staticmethod
       def forward(ctx, x, weight, eps):
           out, rstd = rmsnorm_forward_with_rstd(x, weight, eps)
           ctx.save_for_backward(x, weight, rstd)
           ctx.eps = eps
           return out

       @staticmethod
       def backward(ctx, dy):
           x, weight, rstd = ctx.saved_tensors
           dx, dg = rmsnorm_backward(dy, x, weight, rstd)
           return dx, dg, None  # None for eps
   ```

### Optimization Path (after correctness)

1. **Vectorized loads/stores** -- same int4 pattern as forward. 128-bit aligned.

2. **Shared memory caching** -- cache x in shared memory (same as forward).
   dy does not need caching since it's read once.

3. **Compile-time D specialization** -- same template trick as forward for
   D=768, 4096, 5120, 8192.

4. **Fused dg accumulation** -- accumulate partial dg in registers across
   the GROUP_SIZE_M rows each block handles, then write once.

5. **Register pressure** -- the backward kernel holds: x (int4), dy (int4),
   wdy (float4), x_hat (float), c1 (float), rstd (float), partial_dg
   (float accumulators). Expect ~30-40 registers. Should be fine for 48
   warps/SM occupancy.

## 5. Reference Implementations

| Implementation | Language | Approach | What to Study |
|---|---|---|---|
| PyTorch native (`layer_norm_kernel.cu`) | CUDA | Two-kernel (separate dx and dg) | The `rms_norm` template specialization eliminates the c2 term |
| Triton LayerNorm tutorial | Triton | Fused dx+partial_dg with locks | The lock-based partial sum pattern for dg |
| Liger-Kernel | Triton | Fused, similar to Triton tutorial | Production-quality with FP32/BF16 casting modes |
| NVIDIA Transformer Engine | CUDA | `nvte_rmsnorm_bwd` with workspace+barrier | Multi-SM coordination pattern, dgamma_part |
| Tri-RMSNorm | Triton | Fused with GROUP_SIZE_M=32 | Simple reference, includes perf numbers |

## 6. Caveats

- **sm_120 has no TMA.** All the Transformer Engine patterns using `cp.async.bulk`
  or TMA-based loads don't apply. Use standard `cp.async.cg` or direct loads.

- **atomicAdd for BF16 is not natively supported on sm_120.** Must use FP32
  atomics or the lock-based pattern for dg accumulation.

- **The backward reads 3 tensors (x, dy, weight) vs forward's 2 (x, weight).**
  L1/L2 cache pressure is higher. Weight is small (1.5 KB) and will stay cached.
  dy and x are both large and will stream from DRAM.

- **The pipelining floor may be higher for backward.** The forward hits 4.1 us
  because the GPU pipelines consecutive kernel launches. The backward kernel
  has more memory traffic, so its pipelining floor may be ~5-6 us for D=768.
  But the PyTorch backward is also slower, so the speedup ratio may be similar
  (1.2-1.5x).

- **dg reduction across rows is the hard part.** The dx computation is
  embarrassingly parallel across rows (identical to forward). The dg reduction
  requires cross-row communication. This is where the lock-based or atomicAdd
  approach matters.

- **For training, the backward is called once per forward.** The combined
  forward+backward time matters. If forward=4.1us and backward=6us, total is
  ~10us. PyTorch's combined is ~12-14us. The relative speedup may actually be
  better for the combined operation than for forward alone.
