# Flash-Attention SM120: Deep Code Analysis vs Our Kernel

**Sources:**
- [flash_fwd.py (SM80 base class)](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/flash_fwd.py)
- [flash_fwd_sm120.py](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/flash_fwd_sm120.py)
- [interface.py (dispatch)](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/interface.py)
- [softmax.py](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/softmax.py)
- [ampere_helpers.py (MMA/load utilities)](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/ampere_helpers.py)
- PRs: #2329 (fwd), #2330 (bwd), #2333 (varlen)
**Relevant to:** attention worker
**Worker's current problem:** BF16 at 1.76x SDPA (94% compiler ceiling), FP8 at 2.33x SDPA (latency-bound)
**Date:** 2026-03-15

---

## What This Is

A line-by-line analysis of the actual upstream flash-attention SM120 forward kernel
code (Cute-DSL Python compiled to PTX), compared against our hand-written CUDA C++
kernel. The goal: identify any technique or architectural decision we haven't tried.

---

## Architecture Comparison

### Tile Sizes

| Parameter | Our Kernel | Upstream SM120 |
|-----------|-----------|----------------|
| BQ (tile_m) | 64 (dynamic 128) | **128 always** |
| BKV (tile_n) | 64 | 128 (D<=64), 64 (D>64) |
| D pad | none | pad to multiple of 16 |
| Threads | 128 (4 warps) | 128 (4 warps) |
| MMA instruction | m16n8k16 BF16 | m16n8k16 BF16 (identical) |

**Key difference: BQ=128 x BKV=128 at D=64.** Upstream uses 128x128 tiles for D<=64,
which is dramatically larger than our 64x64. Their smem usage:

```
Q:  128 x 64 x 2 = 16 KB
K:  128 x 64 x 1 stage x 2 = 16 KB  (single-stage!)
V:  128 x 64 x 1 stage x 2 = 16 KB
Total: 48 KB
```

Vs our kernel:
```
Q:  64 x 64 x 2 = 8 KB
K:  64 x 64 x 2 stages x 2 = 16 KB  (double-buffer)
V:  64 x 64 x 2 stages x 2 = 16 KB  (double-buffer)
Total: 40 KB  (but needs 2x more KV iterations)
```

### Pipeline Depth

**Upstream uses 1 stage (single buffer) for SM120 forward.**

This is set explicitly in `interface.py` line 739: `num_stages=1`. This means NO
double-buffering of K/V loads. They rely on cp.async commit/wait patterns to overlap
load and compute within a single smem buffer.

Our kernel uses 2 stages (double-buffer). This was a 2.54x speedup historically
(v3 → v4), but that was for BQ=64, BKV=64. With 128x128 tiles, the compute-to-load
ratio is much higher (4x more MMA work per KV block), so the overlap from double-
buffering may be less critical.

**This is a potentially significant finding.** Single-stage saves half the K/V smem,
which is what enables 128x128 tiles within 48 KB. The tradeoff: less load/compute
overlap, but more MMA work per iteration (amortizing the softmax overhead).

### Q-in-Registers Mode

Upstream supports `Q_in_regs=True` (but sets it False for SM120 in interface.py).
When enabled, Q is loaded to smem, then read into registers via ldmatrix and held
there for the entire KV loop. The smem Q slot is then reused as V smem. This saves
smem by overlapping Q and V buffers.

Our kernel always keeps Q in smem (loaded once via cp.async, re-read via ldmatrix
per QK^T MMA iteration). This is correct for our BQ=64 case where Q smem is only
8 KB. For BQ=128, Q smem is 16 KB, and holding Q in registers would save that space.
However, at D=64 with 4 warps, each warp would need to hold 128x16 = 2048 elements
= 64 registers just for Q, which is too many.

### Smem Layout and Swizzle

Upstream uses `ampere_helpers.get_smem_layout_atom()`:
```python
swizzle_bits = 4  # for 128-byte rows (D=64, BF16: 64*2=128 bytes)
swizzle_base = 3  # for BF16 (2 bytes)
atom shape = (8, 64)  # 8 rows x 64 cols
```

This produces a composed swizzle layout: `Swizzle<4,3,3>` on an `(8, 64)` atom.
This is a standard XOR swizzle pattern, functionally equivalent to what our kernel
does with manual XOR on smem indices.

**No difference here.** Both use XOR swizzle for bank conflict elimination.

### MMA Configuration

```python
tiled_mma_qk = cute.make_tiled_mma(
    warp.MmaF16BF16Op(self.dtype, Float32, (16, 8, 16)),
    (self.num_threads // 32, 1, 1),          # 4 warps along M
    permutation_mnk=(4*16, 16, 16),          # = (64, 16, 16)
)
```

With 4 warps, each MMA is m16n8k16. The tiled MMA maps 4 warps along M:
- Warp 0: rows 0-15
- Warp 1: rows 16-31
- Warp 2: rows 32-47
- Warp 3: rows 48-63

For BQ=128: the tiled MMA covers 64 rows per MMA-tile, so 2 MMA-tiles in M
(128/64 = 2). For BKV=128: 128/16 = 8 MMA-tiles in N, 64/16 = 4 k-tiles in K.

Total MMA ops for QK^T with 128x128 tiles: 2 * 8 * 4 = 64 MMAs per block.
Our kernel with 64x64: 1 * 4 * 4 = 16 MMAs per QK^T.
But upstream does half as many KV iterations (128 per step vs 64).

**Net MMA count for N_kv=2048:** upstream = 64 * (2048/128) = 1024 MMAs for QK^T.
Ours = 16 * (2048/64) = 512 MMAs for QK^T. But PV is also proportionally larger.

### Smem Copy (ldmatrix) Configuration

```python
smem_copy_atom_QK = cute.make_copy_atom(
    warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),  # ldmatrix.x4
    self.dtype,
)
smem_copy_atom_V = cute.make_copy_atom(
    warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4),   # ldmatrix.x4.trans
    self.dtype,
)
```

**Identical to our kernel:** ldmatrix_x4 for Q and K (non-transposed), ldmatrix_x4_trans
for V (transposed). The upstream uses CUTLASS's tiled copy infrastructure to handle
fragment layout; our kernel uses the manual ldmatrix_x4_mma wrapper with baked
a1/a2 swap. Same underlying SASS instructions.

### GEMM Loop (ampere_helpers.gemm)

```python
def gemm(tiled_mma, acc, tCrA, tCrB, tCsA, tCsB, ...):
    # Load first k-tile of A and B from smem
    copy(smem_thr_copy_A, tCsA[..., 0], tCrA_copy_view[..., 0])
    copy(smem_thr_copy_B, tCsB[..., 0], tCrB_copy_view[..., 0])
    for k in range_constexpr(K_tiles):
        if k < K_tiles - 1:
            # Pre-load next k-tile of A and B
            copy(smem_thr_copy_A, tCsA[..., k+1], tCrA_copy_view[..., k+1])
            copy(smem_thr_copy_B, tCsB[..., k+1], tCrB_copy_view[..., k+1])
        gemm(tiled_mma, acc, tCrA[..., k], tCrB[..., k], acc)
```

This is a standard software-pipelined GEMM: pre-load the next K tile from smem while
computing the current MMA. This is functionally what our kernel does with `#pragma unroll`
on the k-tile loop.

**No novel scheduling trick here.** The CUTLASS framework handles the unroll; the
compiler produces the same interleaved LDSM+HMMA pattern.

### PV GEMM (gemm_rs — register-to-smem A operand)

```python
def gemm_rs(tiled_mma, acc, tCrA, tCrB, tCsB, smem_thr_copy_B, ...):
    # A (P matrix) is already in registers (from softmax)
    # Only B (V transposed) needs to come from smem
    copy(smem_thr_copy_B, tCsB[..., 0], tCrB_copy_view[..., 0])
    for k in range_constexpr(K_tiles):
        if k < K_tiles - 1:
            copy(smem_thr_copy_B, tCsB[..., k+1], tCrB_copy_view[..., k+1])
        gemm(tiled_mma, acc, tCrA[..., k], tCrB[..., k], acc)
```

This confirms that upstream also does **register-only P-to-A conversion** (no smem
round-trip for the P matrix), same as our kernel. The `layout_utils.reshape_acc_to_frgA`
handles the FP32 accumulator -> BF16 fragment conversion in registers.

---

## Softmax Deep Dive

The softmax implementation in `softmax.py` is the most interesting part for comparison.

### Online Softmax Algorithm

```python
def online_softmax(self, acc_S, is_first=False, check_inf=True):
    acc_S_mn = reshape_acc_to_mn(acc_S)  # shape: (num_rows, n_cols)

    for r in range(num_rows, unroll_full=True):
        acc_S_row = acc_S_mn[r, :].load()  # load row as SSA

        # 1. Compute row max (with previous max if not first block)
        row_max_cur = fmax_reduce(acc_S_row, init_val=row_max[r] if not is_first)
        row_max_cur = warp_reduction_max(row_max_cur, threads_in_group=4)  # <-- KEY

        # 2. Handle -inf case
        if check_inf:
            row_max_cur = 0.0 if row_max_cur == -inf else row_max_cur

        # 3. Scale and exp2
        row_max_cur_scaled = row_max_cur * scale_log2
        acc_S_row_exp = exp2(acc_S_row * scale_log2 - row_max_cur_scaled)

        # 4. Row sum (with rescaled previous sum if not first)
        if is_first:
            acc_S_row_sum = fadd_reduce(acc_S_row_exp)
            row_scale[r] = 1.0
        else:
            row_scale[r] = exp2((row_max_prev - row_max_cur) * scale_log2)
            acc_S_row_sum = fadd_reduce(acc_S_row_exp, init_val=row_sum[r] * row_scale[r])

        row_sum[r] = acc_S_row_sum
        acc_S_mn[r, :].store(acc_S_row_exp)

    return row_scale
```

**Critical observation: `warp_reduction_max` with `threads_in_group=4`.**

This is a 4-thread (quad) warp reduction for the row max, matching our kernel's
4-thread group shuffles with XOR masks 1 and 2. Upstream calls it
`cute.arch.warp_reduction_max(row_max_cur, threads_in_group=4)`.

**The row_sum reduction is DEFERRED to finalize().** In the per-block
`online_softmax`, they compute `fadd_reduce` within each thread's values but do NOT
do the cross-thread warp reduction. The quad reduction for row_sum happens only once
in `finalize()`:

```python
def finalize(self):
    row_sum.store(warp_reduce(row_sum.load(), operator.add, width=4))
    ...
    row_scale[r] = rcp_approx(row_sum[r]) * final_scale
    row_sum[r] = (row_max[r] * scale_log2 + log2(row_sum_cur)) * LN2  # = log(sum * exp(max))
```

**This is a key difference from our kernel.** Our kernel does the shuffle sum
immediately in each softmax iteration. Upstream defers the sum shuffle to the
very end (after all KV blocks are processed). This saves 2 shuffle instructions
per KV block per row.

**Impact estimate:** With N=2048 and BKV=64, our kernel does 32 shuffle-sum
operations (2 shuffles/row * 8 rows * 2 iterations — but actually more complex).
Deferring saves those shuffles during the inner loop. At ~4 cycles each, this
could save ~128 cycles per KV block, or ~4096 cycles total. On a ~68 us kernel
at 2.5 GHz, that's ~1.6 us, or about 2.4% — at the noise floor but possibly real.

### exp2f and Scale Folding

```python
softmax_scale_log2 = softmax_scale * LOG2_E
acc_S_row_exp = exp2(acc_S_row * scale_log2 - row_max_cur_scaled)
```

**Identical to our kernel.** Both fold LOG2_E into the Q scale and use exp2 (MUFU.EX2).

### rcp_approx in Finalize

```python
row_scale[r] = cute.arch.rcp_approx(row_sum[r]) * final_scale
```

Upstream uses `rcp_approx` (MUFU.RCP) for the 1/row_sum division. This is a fast
approximate reciprocal (1 cycle, ~23-bit accuracy). Our kernel likely uses a full
division or similar. Using MUFU.RCP could save a few cycles in the epilogue but is
only executed once per query row (not in the hot inner loop).

---

## Compute Loop Structure (compute_one_n_block)

The upstream inner loop structure:

```
1. acc_S = 0
2. sync (wait for K/V in smem)
3. Start async load of V for next iteration (load_V_next)
4. GEMM: acc_S = Q * K^T  (using smem K)
5. [optional: apply score_mod]
6. If single-stage: sync, then start async load of K for next-next iteration
7. Apply mask (if needed)
8. online_softmax(acc_S) -> row_scale
9. rescale_O(acc_O, row_scale)
10. rP = acc_S.to(BF16)  (register-only conversion)
11. If multi-stage: sync, then start async load of K for next-next iteration
12. GEMM: acc_O += rP * V^T  (P from registers, V from smem)
```

**Scheduling observation for single-stage (SM120 path):**

Steps 3 and 6 are relevant. For single-stage:
- V is loaded BEFORE the QK^T GEMM (step 3, via cp.async)
- After GEMM + softmax, K_next is loaded AFTER sync (step 6)
- Then PV GEMM executes with V already in smem

The sequence is: `load_V_next -> QK^T GEMM -> sync -> load_K_next -> mask -> softmax -> BF16 convert -> PV GEMM`

Our kernel (double-buffered): `QK^T GEMM with K[buf0] -> prefetch K[buf1]+V[buf1] -> softmax -> PV GEMM with V[buf0]`

The upstream single-stage approach loads V FIRST (before QK^T), which means V is
ready in smem by the time PV needs it. It loads K AFTER QK^T. This inverts the
typical double-buffer pattern. The advantage: only 1 smem buffer needed for K and V.

---

## Mainloop Structure (Masking Strategy)

```python
# First iteration: seqlen masking (handles seqlen not divisible by tile_n)
compute_one_n_block(n_block, ..., is_first_n_block=True, mask_fn=seqlen_mask)

# Causal iterations: causal + seqlen masking
for n_tile in range(n_block_max - 1 - n_block_min_causal_local_mask):
    compute_one_n_block(..., mask_fn=causal_mask)

# Remaining iterations: NO masking (pure GEMM + softmax)
for n_tile in range(n_block):
    compute_one_n_block(..., mask_fn=no_mask, is_first_n_block=False)
```

**This is the same 3-phase loop structure our kernel uses:**
1. Last KV block (largest n) with seqlen boundary masking
2. Causal blocks (decreasing n) with causal masking
3. Unmasked blocks (smallest n) with no masking

**The causal softmax bug (fixed in this PR):** The original SM80 code passed
`is_first_n_block=True` to ALL causal iterations, resetting the softmax running
max/sum each time. This produced wrong results when D>64 with tile_n=64 because
there were 2+ causal blocks. The fix: only pass `is_first_n_block=True` for the
very first block.

Our kernel iterates KV blocks in the same direction (last to first) and should
be verified to correctly handle the `is_first` flag.

---

## Techniques We Haven't Tried

### 1. Deferred Row-Sum Shuffle (HIGH POTENTIAL)

**What:** Skip the cross-thread row_sum shuffle during the inner KV loop. Only do
the quad reduction once after all KV blocks are processed, in the finalize step.

**Why it could help:** Removes 2 `__shfl_xor_sync` calls per row per KV block from
the inner loop. For N=2048, BKV=64, that's 32 KV blocks * 2 shuffles * 8 rows =
512 fewer shuffle instructions.

**Why our worker may have already tried this:** Experiment notes mention "deferred
sum shuffles" as producing a bench regression. However, the upstream implementation
proves this is correct mathematically. The regression may have been from an
implementation bug or interaction with other changes.

**Recommendation:** Re-examine this with a clean implementation. The key insight
from upstream: the row_sum only needs to be accurate at finalize time, not during
each softmax iteration. During rescaling, `row_scale[r] = exp2((max_prev - max_cur) * log2e)`
depends only on row_max (which IS shuffled every iteration), not on row_sum.

### 2. Single-Stage Pipeline with Larger Tiles (MEDIUM POTENTIAL)

**What:** Drop double-buffering, use 1-stage pipeline with 128x128 tiles at D=64.

**Smem math:**
```
1-stage 128x128:  Q=16KB + K=16KB + V=16KB = 48KB  -> 2 blocks/SM = 8 warps
2-stage 64x64:    Q=8KB  + K=16KB + V=16KB = 40KB  -> 3 blocks/SM = 12 warps (current)
```

**Tradeoff:** Fewer warps/SM (8 vs 12) but 4x more MMA work per KV block. Each KV
block processes 128 KV positions instead of 64. This means half as many softmax
passes (key for attention where softmax is the bottleneck).

**Why it matters:** Our worker identified that softmax is the irreducible bottleneck
(~48% math_throttle from softmax between MMA phases). Halving the number of softmax
passes by doubling BKV could be worth more than the occupancy loss.

**Counter-argument:** Our experiments 50-53 tried BKV=32 with more blocks/SM and it
was worse (softmax doubled). But upstream goes the OTHER direction: fewer blocks,
larger tiles, fewer softmax passes. This is the inverse of what we tried.

**Risk:** 2 blocks/SM may be too few for the warp scheduler. gau-nernst achieved
94.4% peak with BQ=128, so large Q tiles are proven viable on sm_120.

### 3. V-Before-QKT Load Ordering (LOW POTENTIAL)

**What:** Start the V cp.async BEFORE the QK^T GEMM (single-stage approach).

**In upstream's compute_one_n_block:**
```
load_V_next()         # V cp.async starts
QK^T GEMM             # 16-64 MMAs executing while V loads
sync()                # V is ready
load_K_next()         # K for next-next iteration
softmax + PV GEMM     # V data available immediately
```

**Why interesting:** V load latency is completely hidden behind the QK^T GEMM. K load
for the *next-next* iteration starts during softmax. This is a different ordering
from our "prefetch after QK^T" pattern.

**Why low potential:** Our double-buffer pattern already achieves load/compute overlap.
The single-stage V-first pattern only makes sense if we also adopt single-stage
pipeline (point 2 above). They are a package deal.

### 4. BQ=128 as Default (not just dynamic dispatch) (LOW-MEDIUM)

**What:** Use BQ=128 unconditionally for the primary config (B=2, H=8, N=2048, D=64).

**Grid calculation:** B*H * ceil(N/BQ) = 2*8 * ceil(2048/128) = 16 * 16 = 256 blocks.
With 170 SMs, that's 1.5 blocks/SM average — not great. Our dynamic dispatch
threshold already handles this: BQ=128 only when grid >= 340.

**But:** With 2 blocks/SM (single-stage 128x128), the grid only needs 340 blocks to
fill the GPU. At BQ=128, that's 340 * 128 = 43,520 query positions. For B=2, H=8
(16 heads), that's 43520/16 = 2720 seq positions. So N>=2720 would fill the GPU
with BQ=128.

**Our primary config (N=2048):** 16 * 16 = 256 blocks. With 2 blocks/SM target:
256/2 = 128 SMs used, out of 170. Only 75% GPU utilization. This is a real concern.

### 5. Head-Dim Padding to Multiple of 16 (NEGLIGIBLE)

Upstream pads `head_dim` to a multiple of 16 for smem layout alignment:
```python
self.tile_hdim = ceil(head_dim / 16) * 16
```

Our D=64 already is a multiple of 16. No impact for our primary config.

---

## Techniques They Use That We Already Have

| Technique | Upstream | Our Kernel | Match? |
|-----------|----------|-----------|--------|
| mma.sync m16n8k16 BF16 | Yes | Yes | Exact |
| cp.async with global cache | Yes | Yes | Exact |
| XOR swizzle for smem | Yes (Swizzle<4,3,3>) | Yes (manual XOR) | Equivalent |
| ldmatrix_x4 for Q, K | Yes | Yes (with baked a1/a2 swap) | Equivalent |
| ldmatrix_x4_trans for V | Yes | Yes | Exact |
| Register-only P->A conversion | Yes (reshape_acc_to_frgA) | Yes (pack_bf16x2) | Equivalent |
| Online softmax with exp2 | Yes | Yes | Exact |
| LOG2_E folded into Q scale | Yes | Yes | Exact |
| 4-thread quad warp reduction (softmax) | Yes (threads_in_group=4) | Yes (XOR masks 1, 2) | Exact |
| Causal 3-phase loop (masked/causal/unmasked) | Yes | Yes | Same structure |
| Non-volatile MMA | Not applicable (Cute-DSL) | Yes | N/A |

---

## What Upstream Does NOT Have (Our Advantages)

1. **FP8 attention (m16n8k32 e4m3)** -- upstream is BF16 only. Our FP8 at 2.33x SDPA
   is ahead.

2. **Dynamic BLOCK_Q dispatch** -- upstream uses fixed BQ=128. We adapt BQ=64/128
   based on grid size.

3. **V preload separation** -- our explicit V preload before PV MMA, which the compiler
   hoists into the softmax gap. Upstream relies on the Cute-DSL compiler for this.

4. **Skip mask optimization** -- our unconditional exp in softmax for unmasked blocks.
   Upstream applies masking uniformly (though the 3-phase loop skips it for unmasked
   blocks at the loop level).

5. **Hand-tuned register budget** -- 145 regs, 0 spills, 3 blocks/SM. Upstream relies
   on nvcc's register allocator via the Cute-DSL compilation chain.

---

## Summary: What to Try Next

**Priority 1 (re-examine): Deferred row-sum shuffle.** Our worker tried this and
it regressed, but upstream proves it's correct. Worth a clean re-implementation.
The savings are 2 shuffles per row per KV block, which adds up over 32 blocks.

**Priority 2 (new experiment): Single-stage 128x128 tiles at D=64.** This is the
opposite direction from our occupancy-first experiments. Instead of more-blocks-
smaller-tiles, try fewer-blocks-larger-tiles. The key hypothesis: halving softmax
passes (from 32 to 16 for N=2048) outweighs the occupancy loss (2 vs 3 blocks/SM).
However, GPU utilization may be too low for our primary config (N=2048 gives only
256 blocks for 170 SMs with BQ=128, BKV=128).

**Priority 3 (informational): No TMA, no FP8 upstream.** Confirms we're on the right
track with cp.async and ahead on FP8. No upstream techniques to port for FP8.

---

## Caveats

1. **Cute-DSL vs hand-written CUDA C++:** Upstream is Python compiled to PTX through
   CUTLASS's DSL. The compiler may produce different scheduling than nvcc on our
   hand-written C++. Performance may not be directly comparable.

2. **No upstream benchmarks.** PRs only report correctness (max_diff). We don't know
   if their 128x128 single-stage kernel is faster or slower than our 64x64 double-buffer.

3. **SM121a (DGX Spark) vs SM120 (RTX 5090).** Tested on different hardware. SM count
   and clocks differ, which affects occupancy tradeoffs.

4. **The deferred shuffle "regression" may have been real.** Our worker notes say
   "`__shfl_xor_sync` doesn't block non-volatile MMA pipeline." If the shuffles were
   being overlapped with MMA anyway, deferring them wouldn't save anything. Worth
   verifying with ncu stall analysis on the specific shuffle instructions.
