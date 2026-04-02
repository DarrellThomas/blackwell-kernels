# RMSNorm Backward: NVIDIA Transformer Engine API and Implementation Patterns

**Sources:**
- [Transformer Engine rmsnorm.h API (v2.12)](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/c/normalization.html)
- [Transformer Engine rmsnorm.h API (v0.11)](https://docs.nvidia.com/deeplearning/transformer-engine-releases/release-0.11.0/user-guide/api/c/rmsnorm.html)
- [Tri-RMSNorm (dtunai)](https://github.com/dtunai/Tri-RMSNorm)
- [Liger-Kernel RMSNorm source](https://github.com/linkedin/Liger-Kernel)
- [Chronicals: 6.2x backward speedup (arxiv 2601.02609)](https://arxiv.org/html/2601.02609)

**Relevant to:** rmsnorm worker
**Worker's current problem:** Forward at 4.1 us pipelining floor. Next step: backward pass kernel.
**Supplements:** `rmsnorm_backward_implementation_details.md` and `rmsnorm_backward_new_findings_2026_03_14.md`

---

## What's New Here

The existing briefs cover the math (dx formula, dg accumulation) and the Liger-Kernel / Chronicals implementations. This brief adds:

1. **NVIDIA Transformer Engine's C API for backward** -- the production-quality reference
2. **The `multiprocessorCount` parameter pattern** -- why TE passes SM count to the backward kernel
3. **Tri-RMSNorm's atomic-free gradient accumulation** -- an alternative to Liger's lock-free approach

---

## Transformer Engine's nvte_rmsnorm_bwd API

```c
void nvte_rmsnorm_bwd(
    const NVTETensor dz,          // incoming gradient [N, H]
    const NVTETensor x,           // forward input [N, H]
    const NVTETensor rsigma,      // 1/RMS(x) per row [N]
    const NVTETensor gamma,       // scale weights [H]
    NVTETensor dx,                // output: grad wrt input [N, H]
    NVTETensor dgamma,            // output: grad wrt gamma [H]
    NVTETensor workspace,         // scratch buffer
    const int multiprocessorCount,// number of SMs
    const bool zero_centered_gamma,
    cudaStream_t stream
);
```

### Key design choices in the TE API:

1. **Saves `rsigma` (1/RMS) from forward, not `rstd`:** The forward pass caches `rsigma = 1 / sqrt(mean(x^2) + eps)` per row. This is the only intermediate needed for backward. The actual normalized values `x_hat = x * rsigma` are NOT cached -- they're recomputed from `x` and `rsigma` during backward. This is the standard memory-efficient choice.

2. **`multiprocessorCount` parameter:** The backward kernel uses this to size its grid. For the `dgamma` accumulation (reducing across N rows to get a per-feature gradient), the kernel launches one CTA per SM and has each CTA process a horizontal strip of rows. With `multiprocessorCount` CTAs, each CTA handles `ceil(N / multiprocessorCount)` rows. This avoids atomic contention -- each CTA accumulates partial `dgamma` in registers, then a final reduction across CTAs writes the result.

3. **`workspace` tensor:** Used for the inter-CTA reduction of `dgamma`. Size = `multiprocessorCount * H * sizeof(float)`. Each CTA writes its partial `dgamma` to `workspace[cta_id * H ... (cta_id+1) * H]`, then a final kernel (or the last CTA) reduces across the `multiprocessorCount` partials.

4. **`zero_centered_gamma`:** If true, the normalization uses `(1 + gamma)` instead of `gamma`. The backward formula changes to `dgamma += dz * x_hat` (same as standard) but the dx computation uses `(1 + gamma)` in place of `gamma`. This is a detail from Llama-style models.

---

## Implementation Pattern: Two-Phase dgamma

The hardest part of the backward is `dgamma` -- it requires reducing across the batch dimension (N rows). Three approaches in production:

### Approach 1: Transformer Engine (CTA-partitioned reduction)
- Launch `SM_count` CTAs, each processes `N/SM_count` rows
- Each CTA accumulates partial dgamma in FP32 registers
- After processing all assigned rows, CTA writes partial to workspace
- Final reduction pass sums `SM_count` partials per feature
- **Pro:** No atomics, deterministic
- **Con:** Requires workspace buffer and a final reduction pass

### Approach 2: Liger-Kernel (row-blocked partial sums)
- Each CTA processes a fixed block of rows (e.g., 16 rows)
- Accumulates partial dgamma in registers
- Writes partial to output buffer at designated offset
- Separate kernel reduces all partials
- **Pro:** Simple, works with Triton's programming model
- **Con:** Second kernel launch for the final reduction

### Approach 3: Tri-RMSNorm (atomic accumulation)
- Each CTA computes dgamma for its rows and atomicAdd to global dgamma
- Uses lock mechanism to prevent race conditions
- **Pro:** Single kernel, no workspace
- **Con:** Atomic contention at high CTA counts, non-deterministic

### Recommended for sm_120:
Use the TE approach (Approach 1). With 170 SMs on RTX 5090, launching 170 CTAs with each processing `ceil(N/170)` rows gives good load balance. The final reduction of 170 partials per feature is trivial. The workspace is `170 * H * 4 bytes` = 2.7 MB for H=4096 -- fits easily.

---

## Tri-RMSNorm Backward Implementation

The dtunai/Tri-RMSNorm implementation in Triton provides a simpler reference than Liger-Kernel:

```python
@triton.jit
def _rms_norm_bwd_dx_fused(
    DX, DY, DW, DB,      # gradient tensors
    X, W, B,              # forward inputs
    Mean, Rstd,           # cached forward values
    Lock,                 # for safe dgamma accumulation
    stride_dx, stride_dy, stride_x,
    N, eps,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr
):
```

Key detail: `GROUP_SIZE_M` controls how many rows each CTA group processes for the dgamma partial sum. Larger `GROUP_SIZE_M` = fewer atomics but more register pressure. The sweet spot is architecture-dependent.

---

## Caveats

- Transformer Engine's actual CUDA kernel source is not publicly documented in detail. The API is public but the kernel internals are proprietary. The patterns described here are inferred from the API design and common CUDA backward kernel practices.
- The `multiprocessorCount` parameter hardcodes the grid size to SM count. This works well for large N (>=170 for RTX 5090) but wastes SMs for small N. Consider dynamic grid sizing: `min(SM_count, N)` CTAs.
- For our use case (standard transformer sizes: N=batch*seq_len, H=4096), N is typically >> 170, so the TE approach is well-suited.
