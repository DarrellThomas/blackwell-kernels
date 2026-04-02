# RMSNorm Backward — Production CUDA Source Code Survey

**Sources:** PyTorch `layer_norm_kernel.cu`, NVIDIA APEX `layer_norm_cuda_kernel.cu`, NVIDIA Transformer Engine `rmsnorm_bwd_kernels.cuh`, Dao-AILab Flash-Attention `ln_bwd_kernels.cuh`, Karpathy llm.c `layernorm_backward.cu`
**Relevant to:** rmsnorm worker
**Worker's current problem:** Forward pass at 1.2-1.4x PyTorch. Next step: backward pass kernel.
**Supplements:** Existing briefs cover the math and Triton implementations. This brief covers the actual CUDA kernel source code from five production codebases.

---

## What's New Here vs Existing Briefs

The existing briefs (`rmsnorm_backward_pass_kernel.md`, `rmsnorm_backward_implementation_details.md`, `rmsnorm_backward_new_findings_2026_03_14.md`) cover the math, the Liger-Kernel Triton code, and the Transformer Engine API. This brief adds:

1. **PyTorch's actual CUDA backward kernel** -- template specialization, vectorized variant, adaptive grid sizing
2. **APEX's three-kernel backward** -- the original NVIDIA implementation still used by many frameworks
3. **Transformer Engine's actual kernel internals** -- the `rmsnorm_bwd_tuned_kernel` CUDA code
4. **Flash-Attention's `ln_bwd_kernels.cuh`** -- the most optimized CUDA backward kernel, with Vec loads and Reducer patterns
5. **Karpathy's llm.c progression** -- kernels 1-10 showing optimization from naive to production-quality

---

## 1. PyTorch Native: `layer_norm_kernel.cu` (RMSNorm specialization)

**Source:** https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/layer_norm_kernel.cu

PyTorch implements RMSNorm backward as a template specialization of the LayerNorm backward via `template<..., bool rms_norm>`.

### dx Kernel: `layer_norm_grad_input_kernel`

Two variants exist -- scalar and vectorized:

```cuda
template<typename T, typename T_ACC, bool rms_norm>
__global__ void layer_norm_grad_input_kernel(
    const T* dY, const T* X,
    const T_ACC* mean, const T_ACC* rstd,
    const T* gamma, T* dX, const int N)
```

The vectorized variant uses `aligned_vector<T, vec_size>`:

```cuda
template<typename T, typename T_ACC, bool rms_norm>
__global__ void layer_norm_grad_input_kernel_vectorized(...)
{
    using vec_t = aligned_vector<T, vec_size>;
    const vec_t* X_i_vec_ptr = reinterpret_cast<const vec_t*>(X_i);
    const vec_t* dY_i_vec_ptr = reinterpret_cast<const vec_t*>(dY_i);
    // 128-bit aligned vector loads
    for (unsigned int l = threadIdx.x * vec_size; l + vec_size - 1 < N;
         l += blockDim.x * vec_size) {
        X_i_vec_reg = X_i_vec_ptr[vec_idx];
        dY_i_vec_reg = dY_i_vec_ptr[vec_idx];
    }
}
```

**Key alignment check:** The vectorized path is only used when `addr % (sizeof(T) * vec_size) == 0`. Otherwise it falls back to the scalar kernel.

### dx Computation (RMSNorm path)

The `compute_gI` function computes two statistics first, then applies them:

```cuda
// Two reduction statistics (per row):
// stats_x1 = sum(dY * gamma)              -- NOT USED for RMSNorm (only LayerNorm)
// stats_x2 = sum(dY * gamma * X * rstd)   -- dot product of wdy and x_hat

T_ACC term1 = (T_ACC(1) / fH) * rstd_val;   // 1/N * rstd

for (int l = threadIdx.x; l < N; l += blockDim.x) {
    T_ACC f_grad_input = fH * gamma_val * dy;     // N * gamma * dy
    // RMSNorm path (no mean subtraction):
    f_grad_input -= (x) * rstd_val * stats_x2;    // subtract x * rstd * c1_unnormalized
    f_grad_input *= term1;                          // multiply by 1/(N*rstd) ... wait, no
    dX_i[l] = f_grad_input;
}
```

**Important numerical detail:** PyTorch multiplies by `fH` (=N, the hidden dimension) in the main term and divides by it in `term1`. This avoids a division per element and instead uses one multiply per element with the pre-divided `term1`. The formula is:
```
dx[i] = (1/N) * rstd * (N * gamma[i] * dy[i] - x[i] * rstd * sum_j(gamma[j] * dy[j] * x[j] * rstd))
```
This is equivalent to our standard formula but factored to minimize divisions.

### dweight Kernel: Adaptive Grid Sizing

PyTorch selects different grid/thread configs based on the number of rows M:

| M (rows) | Threads | Rows per Block |
|-----------|---------|----------------|
| M < 64    | (32, 1) | 8 |
| M < 128   | (32, 8) | 64 |
| M < 256   | (32, 16) | 128 |
| M >= 256  | (32, 32) | 256 |

For M < 128, a simpler single-kernel reduction (`GammaBetaBackwardSimpleCUDAKernel`) is used.
For M >= 128, a two-kernel approach is used: partial sums then final reduction.

### Shared Memory

```cuda
int nshared = (num_threads() / warp_size) * sizeof(T_ACC);  // for dx kernel
// For dweight:
int nshared2_a = 2 * sizeof(T_ACC) * threads2.y * threads2.y * (threads2.x + 1);
```

**Block/thread for dx:** Grid=(M), Threads=(warp_size, num_threads/warp_size). One block per row. The second dimension (multiple warps) handles intra-row reduction.

---

## 2. NVIDIA APEX: `layer_norm_cuda_kernel.cu`

**Source:** https://github.com/NVIDIA/apex/blob/master/csrc/layer_norm_cuda_kernel.cu

APEX uses **three kernels** for the backward pass:

### Kernel 1: `cuComputeGradInput` (dx)

```cuda
template <typename T, typename U, typename V, bool MemoryEfficient>
__global__ void cuComputeGradInput(
    const V* dout, const T* input_or_output,
    const int n1, const int n2,
    const U* invvar, U epsilon,
    const V* gamma, T* grad_input,
    const double eps, bool rms_only)
```

Grid: `(1, min(n1, maxGridY))`, Threads: `(32, 4)` -- 4 warps per block, grid covers rows.

**MemoryEfficient mode:** When `MemoryEfficient=true`, the kernel receives the output `y` instead of the input `x`, and recomputes `x` from `y` and the weights: `x = y / gamma`. This trades compute for memory by not saving `x` in the forward pass. For RMSNorm, this means we can choose between saving `x` (standard) or saving `y` and recomputing (memory-efficient).

**dx formula in APEX:**
```cuda
U f_grad_input = fH * c_loss * k_gamma;                // N * dy * gamma
if (MemoryEfficient) {
    f_grad_input -= c_h / k_gamma * sum_loss2;          // recomputed x term
} else {
    f_grad_input -= c_h * c_invvar * sum_loss2;         // cached x term
}
f_grad_input *= term1;   // term1 = (1/n2) * invvar
```

**Half precision fast path:** APEX uses `__half22float2` to load paired FP16 values and convert to float2 in a single operation, enabling efficient bandwidth utilization for FP16 inputs.

### Kernel 2: `cuComputePartGradGammaBeta` (partial dweight)

Grid: `((n2 + 31)/32, 16)` -- 16 partial reduction blocks across the row dimension.
Threads: `(32, 4)`

Threads load strided inputs across rows and accumulate partial sums in two shared memory buffers (`warp_buf1` for dgamma, `warp_buf2` for dbeta). Within-warp reductions combine values, then inter-warp reductions via shared memory synchronize.

### Kernel 3: `cuComputeGradGammaBeta` (final dweight)

Grid: `((n2 + 31)/32, 1)`, Threads: `(32, 8)`.

Sequential reduction across the 16 partial results from Kernel 2. Each warp processes multiple partial segments, then inter-warp reductions finalize the sum.

**Shared memory:**
```cuda
int nshared = threads1.y > 1 ? threads1.y * threads1.x * sizeof(U) : 0;  // Kernel 1
int nshared2 = 2 * sizeof(U) * threads2.y * threads2.y * (threads2.x + 1);  // Kernel 2
```

---

## 3. NVIDIA Transformer Engine: `rmsnorm_bwd_kernels.cuh`

**Source:** https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/common/normalization/rmsnorm/rmsnorm_bwd_kernels.cuh

TE has three backward kernels:

### Main Kernel: `rmsnorm_bwd_tuned_kernel<Ktraits, FusedAdd>`

This is the performance-critical kernel. Key details:

**Vector load types:** Uses specialized types `Ivec`, `Ovec`, `Wvec`, `Cvec` with `.load_from()` and `.store_to()` methods. Each thread processes `NUM_ELTS` elements per vector load, repeated `LDGS` times per row.

**dx formula:**
```cuda
// rs = reciprocal RMS (rstd)
// y = normalized x (x * rs)
// dy = weighted gradient (dz * gamma)
dx = rs * (dy - mdyy * y)
```

Where `mdyy = (1/N) * sum(dy * y)` is the single reduction scalar.

**When `FusedAdd` is true:** The kernel also loads an additional tensor and adds it to dx:
```cuda
dx += add_tmp;   // fused residual gradient addition
```

**dweight accumulation:**
```cuda
dzy_sum += dz * y;    // partial dgamma in FP32 registers
```

Each CTA processes multiple rows and accumulates `dzy_sum` across all its rows. Partial dgamma is stored to shared memory (`smem_wgrad`) for multi-warp reduction when `WARPS_M > 1`.

**Constraint:** `WARPS_M == 1 || CTAS_PER_ROW == 1` -- you cannot have both multi-warp M-dimension tiling and multi-CTA row tiling simultaneously.

### Finalize Kernel: `rmsnorm_bwd_finalize_tuned_kernel`

Performs cross-CTA reduction of partial dgamma values. Uses shared memory transpose buffers (`SMEM_BYTES_TRANSPOSE`) for efficient inter-warp allreduce.

Grid: `Kernel_traits_f::CTAS`, Threads: `32 x 32` (one full warp block).

### General Kernel: `rmsnorm_bwd_general_kernel`

Fallback for non-optimized shapes. Same logic but without the tuned tile sizes.

---

## 4. Flash-Attention: `ln_bwd_kernels.cuh` (Most Optimized)

**Source:** https://github.com/Dao-AILab/flash-attention/tree/main/csrc/layer_norm

This is the most sophisticated CUDA backward implementation. Key patterns:

### Kernel Signature

```cuda
template<typename Ktraits, bool Is_dropout, bool Has_colscale,
         bool Has_subset, bool Is_even_cols>
__global__ __launch_bounds__(Ktraits::THREADS_PER_CTA)
void ln_bwd_kernel(layer_norm::BwdParams params)
```

### dx Computation (with RMSNorm support)

```cuda
compute_t y_tmp = rs_r * (x_tmp - (!params.is_rms_norm ? mu_r : 0.f));
compute_t dy_tmp = compute_t(gamma[it].data.elt[jt]) *
                   compute_t(dz.data.elt[jt]);
compute_t dx_tmp = rs_r * (dy_tmp - (mdyy_local * y_tmp +
                  (!params.is_rms_norm ? mdy_local : 0.f)));
```

For RMSNorm: `mu_r = 0` and `mdy_local = 0`, so this simplifies to:
```
y_tmp = rs_r * x_tmp                         // x_hat
dy_tmp = gamma * dz                           // wdy
dx_tmp = rs_r * (dy_tmp - mdyy_local * y_tmp) // the standard formula
```

### Reducer Pattern (Custom warp allreduce)

```cuda
reduce_t result = reducer.allreduce({mdy_local, mdyy_local}, sum);
mdy_local = Get<0>::of<reduce_t, compute_t>(result) * params.inverse_cols;
mdyy_local = Get<1>::of<reduce_t, compute_t>(result) * params.inverse_cols;
```

The `Reducer` class coordinates cross-warp summation via shared memory. When `CTAS_PER_ROW > 1`, it uses cooperative barriers for cross-CTA reduction within a row.

### Vectorized Loads

Flash-Attention uses `Wvec`, `Rvec`, `Ovec` vector types with `.load_from()` methods:

```cuda
Wvec gamma[LDGS];
gamma[it].load_from(params.gamma, idx);    // vector load of gamma
Rvec x;
x.load_from(params.x, idx_x);             // vector load of x
Ovec dz;
dz.load_from(params.dz, idx_z);           // vector load of dz (gradient)
```

Elements processed via `dz.data.elt[jt]` indexing within vectors of size `NUM_ELTS`.

### dweight Accumulation

**Phase 1 (in main kernel):** Per-row accumulation in registers:
```cuda
dzy_sum[it].data.elt[jt] += dz_tmp * y_tmp;    // dgamma contribution
dz_sum[it].data.elt[jt] += dz_tmp;             // dbeta contribution
```

Multi-warp reduction via shared memory:
```cuda
for (int it = 0; it < ROWS_PER_CTA; it++) {
    for (int jt = 0; jt < NUM_RES; jt++) {
        cta_dzy_sum[jt] +=
          smem_wgrad[it * COLS + tidx + jt * Ktraits::THREADS_PER_CTA];
    }
}
```

**Phase 2 (finalize kernel):** Cross-CTA reduction with shared memory transpose:
```cuda
g_i = reducer.allreduce(g_i, sum);   // dgamma final reduction
```

### Shared Memory Layout

```cuda
// Main kernel:
compute_t * smem_wgrad = reinterpret_cast<compute_t*>(smem_);
char *smem_dgrad = smem_ + Ktraits::SMEM_BYTES_WGRAD;

// Finalize kernel:
void * smem_gamma = smem_;
void * smem_beta = &smem_[Ktraits::SMEM_BYTES_TRANSPOSE];
```

### Thread/Block Config

```
Main kernel:
  WARPS_M x WARPS_N warp grid per CTA
  THREADS_PER_CTA total threads
  Grid = CTAS_PER_ROW * ctas_per_col blocks

Finalize kernel:
  32 x 32 threads (one full warp block)
  Grid = Kernel_traits_f::CTAS blocks
```

### Prenorm Support

When `prenorm=true`, the kernel adds the incoming gradient from a residual stream:
```cuda
dx_tmp_res = prenorm ? dx_tmp + compute_t(dx[it].data.elt[jt]) : dx_tmp;
```

This is the backward analog of the fused residual add in the forward pass.

---

## 5. Karpathy llm.c: `layernorm_backward.cu` (Educational Progression)

**Source:** https://github.com/karpathy/llm.c/blob/master/dev/cuda/layernorm_backward.cu

This file contains 10 kernel versions showing the optimization progression from naive to production-quality. Key lessons:

### dweight Accumulation Evolution

| Kernel | Strategy | Performance |
|--------|----------|-------------|
| 1-3 | Global atomicAdd per element | Baseline -- severe contention |
| 4-5 | Shared memory atomicAdd, then global write | Better but still contention |
| 6-7 | Per-block shared memory staging, last block sums | No atomics on global |
| 8-10 | Vectorized 128-bit loads + shared memory staging | Best |

### Vectorized Loads (Kernel 8-10)

```cuda
x128 dout128 = load128cs(dout_bt + global_index);
x128 inp128 = load128cs(inp_bt + global_index);
x128 weight128 = load128(weight + global_index);
```

The `cs` suffix = "cache streaming" to bypass L1 cache thrashing. This is important: using `.cs` for the large tensors (dout, input) avoids polluting L1 with data that won't be reused, leaving L1 for the weight vector.

### dx Formula in llm.c

```cuda
float dval = 0.0f;
dval += dnorm_i;              // term 1: weight * dout
dval -= dnorm_mean;           // term 2: mean of wdy (LayerNorm only, zero for RMSNorm)
dval -= norm_bti * dnorm_norm_mean;  // term 3: x_hat * mean(wdy * x_hat)
dval *= rstd_bt;              // scale by reciprocal std
```

### Shared Memory Organization (Kernel 9-10)

```cuda
extern __shared__ float shared[];
float* dbias_shared = shared;                    // [C]
float* dweight_shared = shared + C;              // [C]
float* dbias_tmp_shared = shared + 2*C;          // [BLOCK_SIZE]
float* dweight_tmp_shared = shared + 2*C + BLOCK_SIZE;  // [BLOCK_SIZE]
```

Total: `(2*C + 2*block_size + 1) * sizeof(float)`.

### Warp Reduction

Kernels 3-6 use cooperative groups:
```cuda
dnorm_mean = cg::reduce(warp, dnorm_mean, cg::plus<float>{});
```

Kernels 7-10 use manual warp shuffle for lower overhead:
```cuda
dnorm_mean = warpReduceSum(dnorm_mean) / C;
```

---

## Summary: Design Decisions Across All Implementations

| Decision | PyTorch | APEX | Transformer Engine | Flash-Attention | llm.c |
|----------|---------|------|--------------------|-----------------|-------|
| **Kernels for backward** | 2 (dx + dweight) or 3 (M>=128) | 3 (dx + partial_dg + final_dg) | 2 (main + finalize) | 2 (main + finalize) | 1 (fused) or 2 |
| **dx: one block per...** | row | row | multiple rows/CTA | configurable rows/CTA | batch element |
| **dweight strategy** | Partial sums + final reduce | 16 partial blocks + final reduce | SM-count partials + finalize | CTA partials + finalize | atomicAdd (early) / smem staging (late) |
| **Vectorized loads** | aligned_vector<T, vec_size> | __half22float2 | Ivec/Ovec/Wvec .load_from() | Wvec/Rvec/Ovec .load_from() | x128 load128cs() |
| **What's saved from fwd** | mean, rstd | invvar (+ optionally output) | rsigma (=rstd) | mu, rs (=rstd) | mean, rstd |
| **Cache hint for loads** | none | none | none documented | none documented | `.cs` (cache streaming) |
| **RMSNorm support** | `rms_norm` template bool | `rms_only` bool param | dedicated kernels | `is_rms_norm` flag | LayerNorm only (trivially adaptable) |
| **Fused residual add** | no | no | yes (`FusedAdd` template) | yes (`prenorm` flag) | no |
| **Shared memory (dx)** | warp reduction buffer | cross-warp reduction | dgrad reducer state | dgrad reducer + wgrad | C-sized staging buffers |

---

## Concrete Recommendations for sm_120 CUDA Backward

Based on all five codebases:

### Phase 1: Minimum Viable Backward

1. **Follow the Flash-Attention pattern:** Two kernels (main + finalize). The main kernel computes dx AND accumulates partial dgamma. The finalize kernel sums partials.

2. **Use 128-bit vectorized loads** with `.cs` cache hint (from llm.c) for x and dy. Weight stays in L1 (it's small). Load pattern:
```cuda
// BF16: 8 elements per 128-bit load
float4 x_vec = __ldcs(reinterpret_cast<const float4*>(x + offset));
float4 dy_vec = __ldcs(reinterpret_cast<const float4*>(dy + offset));
```

3. **dx formula** (use the PyTorch factoring -- it's cleanest):
```
stats_x2 = sum(gamma * dy * x * rstd)    // one warp reduction
dx[i] = rstd * (gamma[i] * dy[i] - x[i] * rstd * stats_x2 / N)
```

4. **dgamma accumulation:** Each CTA processes multiple rows (e.g., ceil(rows/170) rows per CTA on RTX 5090). Accumulate partial dgamma in FP32 registers. Write to workspace buffer. Finalize kernel sums 170 partials.

### Phase 2: Optimizations

5. **Fused residual add** (from TE/Flash-Attention): Optional `d_residual` output that adds dx in-place.

6. **Adaptive grid sizing** (from PyTorch): Different thread configs for different M values. For M < 128, use simpler single-kernel path for dweight.

7. **Memory-efficient mode** (from APEX): Option to save output y instead of input x, recomputing x in backward. Trades compute for memory.

---

## Source URLs

- [PyTorch layer_norm_kernel.cu](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/layer_norm_kernel.cu)
- [NVIDIA APEX layer_norm_cuda_kernel.cu](https://github.com/NVIDIA/apex/blob/master/csrc/layer_norm_cuda_kernel.cu)
- [NVIDIA Transformer Engine rmsnorm_bwd_kernels.cuh](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/common/normalization/rmsnorm/rmsnorm_bwd_kernels.cuh)
- [Dao-AILab Flash-Attention ln_bwd_kernels.cuh](https://github.com/Dao-AILab/flash-attention/tree/main/csrc/layer_norm)
- [Karpathy llm.c layernorm_backward.cu](https://github.com/karpathy/llm.c/blob/master/dev/cuda/layernorm_backward.cu)
- [Karpathy llm.c LayerNorm doc](https://github.com/karpathy/llm.c/blob/master/doc/layernorm/layernorm.md)
- [OneFlow LayerNorm optimization blog](https://oneflow2020.medium.com/how-to-implement-an-efficient-layernorm-cuda-kernel-oneflow-performance-optimization-731e91a285b8)
- [FlashNorm paper (inference only, no backward)](https://arxiv.org/abs/2407.09577)
- [Flash-Linear-Attention RMSNormLinear](https://github.com/fla-org/flash-linear-attention)
