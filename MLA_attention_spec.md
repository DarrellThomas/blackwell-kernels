# Multi-Latent Attention (MLA) Kernel Specification — sm_120

## Overview

Custom CUDA kernel for Multi-Latent Attention (DeepSeek-V2 / OpenMythos) on RTX 5090 Blackwell (sm_120). MLA differs from standard multi-head attention in three ways:

1. **Split Q/K head dimensions**: Q and K are concatenations of a non-positional part (`nope`) and a RoPE-encoded part (`rope`), with different widths.
2. **Asymmetric V dimension**: `d_v != d_qk` in most configurations (e.g., `d_qk=64, d_v=48`).
3. **Latent KV compression**: K_nope and V are reconstructed from a low-rank latent `c_kv` via a linear projection. The kernel can fuse this reconstruction (Phase 2).

Target: **2x cuDNN SDPA** for training workloads, matching the BF16 v2 flash attention speedup profile.

---

## 1. TARGET CONFIGURATIONS

From OpenMythos `variants.py`, these are the concrete dimension sets:

| Variant | d_nope | d_rope | d_qk (nope+rope) | d_v | kv_lora_rank (d_c) | n_heads | Seq len |
|---------|--------|--------|-------------------|-----|---------------------|---------|---------|
| Tiny (validation) | 48 | 16 | 64 | 48 | 128 | 12 | 1024 |
| 1B | 64 | 32 | 96 | 64 | 256 | 16 | 4096 |
| 3B | 96 | 32 | 128 | 96 | 384 | 24 | 4096 |
| 10B | 128 | 64 | 192 | 128 | 512 | 32 | 8192 |

**Primary targets:** Tiny (validation) and 3B (production training).

**Template parameters:** `D_NOPE`, `D_ROPE`, `D_V` — compile-time constants. Instantiate for each variant.

---

## 2. COMPUTATION

### 2.1 Inputs

| Tensor | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| `Q_nope` | `[B, H, T, D_NOPE]` | BF16 | Query, no positional encoding |
| `Q_rope` | `[B, H, T, D_ROPE]` | BF16 | Query, RoPE already applied |
| `K_nope` | `[B, H, S, D_NOPE]` | BF16 | Key (reconstructed from latent) |
| `K_rope` | `[B, H, S, D_ROPE]` | BF16 | Key, RoPE already applied |
| `V` | `[B, H, S, D_V]` | BF16 | Value (reconstructed from latent) |
| `scale` | scalar | FP32 | `1.0 / sqrt(D_NOPE + D_ROPE)` |
| `causal` | bool | — | Enable causal masking |

### 2.2 Outputs

| Tensor | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| `O` | `[B, H, T, D_V]` | BF16 | Attention output |
| `L` | `[B, H, T]` | FP32 | Logsumexp (for backward pass) |

### 2.3 Forward Computation

```
# Split-head dot product (no concatenation needed)
score[b,h,t,s] = scale * (Q_nope[b,h,t,:] @ K_nope[b,h,s,:]^T
                         + Q_rope[b,h,t,:] @ K_rope[b,h,s,:]^T)

# Causal mask
if causal and s > t:
    score[b,h,t,s] = -inf

# Online softmax
P = softmax(score, dim=-1)

# Weighted sum over values
O[b,h,t,:] = P[b,h,t,:] @ V[b,h,s,:]
```

**Key insight:** The QK^T dot product decomposes into two independent sub-products over different dimension ranges. Both contribute additively to the same score matrix. This means the MMA schedule for QK^T has two phases with different tile dimensions, but the softmax and PV phases are identical to standard flash attention.

---

## 3. KERNEL ARCHITECTURE

### 3.1 Grid & Block Layout

```
grid:  (ceil(T / BLOCK_Q), B * H, 1)
block: (128, 1, 1)   — 4 warps, consistent with proven v2 flash attention
```

One threadblock owns a BLOCK_Q-row slice of Q and iterates over all KV blocks in the S dimension.

### 3.2 Tile Sizes

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `BLOCK_Q` | 64 (default), 128 (large grid) | Dynamic dispatch: 128 when total blocks >= 340 (proven in v2) |
| `BLOCK_KV` | 64 | Matches v2; 32 if register pressure from split-head logic is too high |
| `WARP_Q` | `BLOCK_Q / 4` | 16 or 32 rows per warp |
| `WARP_Q_TILES` | `WARP_Q / 16` | 1 or 2 MMA tiles per warp in M dimension |

### 3.3 MMA Tile Mapping

All MMA operations use `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`.

**QK^T phase — two sub-products accumulated into the same S_rmem:**

```
// Sub-product 1: Q_nope @ K_nope^T
D_NOPE_CHUNKS = D_NOPE / 16
for dc in 0..D_NOPE_CHUNKS:
    load K_nope fragments for this dc
    for t in WARP_Q_TILES:
        mma S[t][nc] += Q_nope[t][dc] * K_nope[nc][dc]

// Sub-product 2: Q_rope @ K_rope^T (accumulated into same S)
D_ROPE_CHUNKS = D_ROPE / 16
for dc in 0..D_ROPE_CHUNKS:
    load K_rope fragments for this dc
    for t in WARP_Q_TILES:
        mma S[t][nc] += Q_rope[t][dc] * K_rope[nc][dc]
```

**Important:** Both sub-products share the same `S_rmem` accumulators. The second sub-product does NOT zero the accumulators — it adds to the result of the first. This avoids materializing two separate score matrices.

**PV phase — standard, uses D_V:**

```
D_V_CHUNKS = D_V / 16
// P->A conversion in registers (pack_bf16x2, no smem round-trip)
for kc in P_K_CHUNKS:
    preload V fragments via ldmatrix_x4_trans
    for nc in D_V_CHUNKS:
        for t in WARP_Q_TILES:
            mma O[t][nc] += P_a[t][kc] * V[kc][nc]
```

### 3.4 Handling Non-Multiple-of-16 Dimensions

`D_ROPE` can be 16 or 32 (always multiple of 16 in current variants). `D_NOPE` can be 48, 64, 96, 128 (always multiple of 16). `D_V` matches `D_NOPE`.

If future configs introduce non-16-aligned dimensions: zero-pad the last chunk in shared memory during load (use `cp_async_128_zfill` with bounds checking, same pattern as v2 flash attention).

---

## 4. SHARED MEMORY LAYOUT

### 4.1 Buffer Allocation

Double-buffered K and V, with separate regions for nope/rope K components.

```
Buffer layout (per ping/pong slot):

Slot 0 (buf=0):
  K_nope_0: [BLOCK_KV, D_NOPE]  BF16   = BLOCK_KV * D_NOPE * 2 bytes
  K_rope_0: [BLOCK_KV, D_ROPE]  BF16   = BLOCK_KV * D_ROPE * 2 bytes
  V_0:      [BLOCK_KV, D_V]     BF16   = BLOCK_KV * D_V    * 2 bytes

Slot 1 (buf=1):
  K_nope_1: [BLOCK_KV, D_NOPE]  BF16   (same sizes)
  K_rope_1: [BLOCK_KV, D_ROPE]  BF16
  V_1:      [BLOCK_KV, D_V]     BF16
```

**Q stays in registers** (loaded once from global, never evicted — proven pattern from v2).

### 4.2 Memory Budget (BLOCK_KV=64)

| Variant | K_nope | K_rope | V | Per-slot | Total (2 slots) |
|---------|--------|--------|---|----------|------------------|
| Tiny (48+16+48) | 6 KB | 2 KB | 6 KB | 14 KB | 28 KB |
| 1B (64+32+64) | 8 KB | 4 KB | 8 KB | 20 KB | 40 KB |
| 3B (96+32+96) | 12 KB | 4 KB | 12 KB | 28 KB | 56 KB |
| 10B (128+64+128) | 16 KB | 8 KB | 16 KB | 40 KB | 80 KB |

All variants fit within the 96 KB sweet spot (preserves L1 cache). The 10B variant at 80 KB is tight — if register pressure forces BLOCK_KV=32, smem halves to 40 KB.

### 4.3 Swizzle

Apply XOR swizzle to all shared memory regions. Each region swizzles independently based on its column count:

```cpp
// K_nope: swizzle_idx<D_NOPE>(row, col)
// K_rope: swizzle_idx<D_ROPE>(row, col)
// V:      swizzle_idx<D_V>(row, col)
```

Use `swizzle.cuh` from `bwk/common/csrc/common/`. The template parameter `COLS` determines `SWIZZLE_BITS` automatically.

---

## 5. REGISTER ALLOCATION

### 5.1 Register Inventory (per thread, BLOCK_Q=64)

| Register set | Count | Purpose |
|---|---|---|
| `Q_nope_rmem[WARP_Q_TILES][D_NOPE/16][4]` | 4 * (D_NOPE/16) * WARP_Q_TILES | Q nope fragments |
| `Q_rope_rmem[WARP_Q_TILES][D_ROPE/16][4]` | 4 * (D_ROPE/16) * WARP_Q_TILES | Q rope fragments |
| `S_rmem[WARP_Q_TILES][BLOCK_KV/8][4]` | 4 * (BLOCK_KV/8) * WARP_Q_TILES | Score accumulators (FP32) |
| `O_rmem[WARP_Q_TILES][D_V/8][4]` | 4 * (D_V/8) * WARP_Q_TILES | Output accumulators (FP32) |
| `row_max[2*WARP_Q_TILES]` | 2 * WARP_Q_TILES | Online softmax state |
| `row_sum[2*WARP_Q_TILES]` | 2 * WARP_Q_TILES | Online softmax state |
| Temporaries (K/V fragments, P conversion) | ~16-24 | Reused across phases |

### 5.2 Estimated Register Pressure

| Variant | Q_nope | Q_rope | S | O | Softmax | Temps | Total (est.) |
|---------|--------|--------|---|---|---------|-------|------|
| Tiny (48/16/48) | 12 | 4 | 32 | 24 | 4 | 20 | ~96 |
| 1B (64/32/64) | 16 | 8 | 32 | 32 | 4 | 20 | ~112 |
| 3B (96/32/96) | 24 | 8 | 32 | 48 | 4 | 20 | ~136 |
| 10B (128/64/128) | 32 | 16 | 32 | 64 | 4 | 20 | ~168 |

**Target occupancy:**
- Tiny/1B: ~96-112 regs → 4-5 blocks/SM (excellent)
- 3B: ~136 regs → 3 blocks/SM (matches v2 flash attention sweet spot)
- 10B: ~168 regs → exceeds 145-reg sweet spot. **Mitigation:** reduce BLOCK_Q to 64 only (no 128 dispatch), or reduce BLOCK_KV to 32, or split Q_nope loading across D_NOPE chunks (evict/reload).

Use `__launch_bounds__(128, 3)` for the 3B target. Template-specialize launch bounds per variant if needed.

---

## 6. ALGORITHMIC FLOW

### Phase A: Load Q to Registers (once, reused across all KV blocks)

```
// Q_nope: global → shared (temporary) → registers
for each 128-bit chunk:
    cp_async_128_zfill(smem[swizzle_idx<D_NOPE>(row, col)], &Q_nope_global[...])
cp_async_commit + wait
__syncthreads()

for t in WARP_Q_TILES:
    for dc in D_NOPE/16:
        ldmatrix_x4_mma(Q_nope_rmem[t][dc], &smem_Q_nope[warp_row + t*16][dc*16])
        // Pre-multiply by (scale * LOG2E) for log2-space softmax
        Q_nope_rmem[t][dc] *= (scale * 1.4426950408889634f)

// Q_rope: same pattern, reuse same smem region (Q_nope no longer needed in smem)
__syncthreads()
for each 128-bit chunk:
    cp_async_128_zfill(smem[swizzle_idx<D_ROPE>(row, col)], &Q_rope_global[...])
// ... load to Q_rope_rmem, pre-scale
```

**Note:** Q_nope and Q_rope are loaded sequentially to reuse shared memory. After both are in registers, the smem region is free for K/V double buffering.

### Phase B: Initialize Accumulators

```
O_rmem[*][*][*] = 0.0f
row_max[*] = -FLT_MAX
row_sum[*] = 0.0f
```

### Phase C: Prologue — Prefetch First KV Block

```
// Load K_nope_0, K_rope_0, V_0 into buffer slot 0
for each 128-bit chunk:
    cp_async_128_zfill(K_nope_smem[0][swizzle(row, col)], &K_nope_global[kv_start=0][...])
    cp_async_128_zfill(K_rope_smem[0][swizzle(row, col)], &K_rope_global[kv_start=0][...])
    cp_async_128_zfill(V_smem[0][swizzle(row, col)],      &V_global[kv_start=0][...])
cp_async_commit_group()
```

### Phase D: Main KV Loop

```
for kv_block in 0..ceil(S / BLOCK_KV):
    buf = kv_block % 2

    // D.0: Wait for current KV block
    cp_async_wait_group<1>()   // wait for buf, allow next prefetch in flight
    __syncthreads()

    // D.1: Prefetch NEXT KV block (if not last)
    if kv_block + 1 < num_kv_blocks:
        next_buf = 1 - buf
        cp_async K_nope, K_rope, V for (kv_block+1) into next_buf
        cp_async_commit_group()

    // D.2: QK^T — Sub-product 1: Q_nope @ K_nope^T
    //   Zero S_rmem before first sub-product
    S_rmem[*][*][*] = 0.0f

    for dc in 0..D_NOPE_CHUNKS:
        for nc in 0..S_N_CHUNKS step 2:
            // Load 2 K_nope fragments from smem
            ldmatrix_x4(K_frag, &K_nope_smem[buf][nc*8 + ...][dc*16])
            for t in WARP_Q_TILES:
                mma_m16n8k16_bf16_nv(Q_nope_rmem[t][dc], K_frag[0], S_rmem[t][nc])
                mma_m16n8k16_bf16_nv(Q_nope_rmem[t][dc], K_frag[1], S_rmem[t][nc+1])

    // D.3: QK^T — Sub-product 2: Q_rope @ K_rope^T (ACCUMULATE into same S)
    for dc in 0..D_ROPE_CHUNKS:
        for nc in 0..S_N_CHUNKS step 2:
            ldmatrix_x4(K_frag, &K_rope_smem[buf][nc*8 + ...][dc*16])
            for t in WARP_Q_TILES:
                mma_m16n8k16_bf16_nv(Q_rope_rmem[t][dc], K_frag[0], S_rmem[t][nc])
                mma_m16n8k16_bf16_nv(Q_rope_rmem[t][dc], K_frag[1], S_rmem[t][nc+1])

    // D.4: Causal mask
    if causal:
        kv_start = kv_block * BLOCK_KV
        for each element in S_rmem:
            if kv_col > q_row: S_rmem[...] = -FLT_MAX
        // Skip entirely if kv_start > q_end (all masked)
        // Process fully if kv_start + BLOCK_KV <= q_start (no masking needed)

    // D.5: Online softmax (identical to v2 flash attention)
    for t in WARP_Q_TILES:
        local_max = warp_reduce_max(max of S_rmem[t][*][*])
        new_max = max(row_max[t], local_max)

        // Rescale running O and sum
        correction = exp2f(row_max[t] - new_max)
        O_rmem[t][*][*] *= correction
        row_sum[t] *= correction
        row_max[t] = new_max

        // exp2f scores and accumulate sum
        for nc in S_N_CHUNKS:
            S_rmem[t][nc][*] = exp2f(S_rmem[t][nc][*] - new_max)
        row_sum[t] += warp_reduce_sum(sum of S_rmem[t][*][*])

    // D.6: P conversion — register only (NO shared memory round-trip)
    //   Pack FP32 S_rmem pairs into BF16 uint32 for MMA A-fragment
    for t, nc:
        P_a[t][nc_pair] = pack_bf16x2(S_rmem[t][nc][0], S_rmem[t][nc][1])

    // D.7: P @ V — uses D_V dimension (may differ from D_QK)
    for kc in P_K_CHUNKS:
        // Preload all V fragments for this kc
        for nc in D_V_CHUNKS:
            ldmatrix_x4_trans(V_frag[nc], &V_smem[buf][kc*16][nc*16])
        // MMA
        for nc in D_V_CHUNKS:
            for t in WARP_Q_TILES:
                mma_m16n8k16_bf16_nv(P_a[t][kc], V_frag[nc], O_rmem[t][nc])

    __syncthreads()  // free smem for next prefetch
```

### Phase E: Finalize & Store

```
for t in WARP_Q_TILES:
    inv_sum = 1.0f / row_sum[t]
    for nc in D_V_CHUNKS:
        O_rmem[t][nc][*] *= inv_sum
        // Pack to BF16 and store to global
        pack_and_store_bf16(O_global[...], O_rmem[t][nc])

    // Store logsumexp for backward pass
    L[row] = row_max[t] / LOG2E + logf(row_sum[t])
```

---

## 7. PHASE 2: ABSORBED MLA (FUTURE — INFERENCE OPTIMIZATION)

For inference with KV cache, avoid materializing K_nope and V entirely. Work directly on compressed latent `c_kv`:

```
// Absorb kv_up into Q: Q_absorbed = Q_nope @ W_k_nope^T   [B,H,T,d_c]
// Score: score = Q_absorbed @ C_kv^T + Q_rope @ K_rope^T
// Output: O_latent = P @ C_kv, then O = O_latent @ W_v^T
```

This changes the QK^T inner dimension from `D_NOPE` to `d_c` (kv_lora_rank, much larger: 128-512). The PV phase inner dimension also becomes `d_c`. Trade-off: larger inner dimension but eliminates the kv_up GEMM and K/V materialization.

**Do not implement Phase 2 until Phase 1 is validated.** Phase 1 handles training (where K_nope and V are already computed for the backward pass).

---

## 8. BACKWARD PASS

### 8.1 Inputs (in addition to saved forward tensors)

| Tensor | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| `dO` | `[B, H, T, D_V]` | BF16 | Gradient of output |
| `O` | `[B, H, T, D_V]` | BF16 | Forward output (saved) |
| `L` | `[B, H, T]` | FP32 | Logsumexp (saved) |

### 8.2 Outputs

| Tensor | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| `dQ_nope` | `[B, H, T, D_NOPE]` | BF16 | Gradient w.r.t. Q nope |
| `dQ_rope` | `[B, H, T, D_ROPE]` | BF16 | Gradient w.r.t. Q rope |
| `dK_nope` | `[B, H, S, D_NOPE]` | BF16 | Gradient w.r.t. K nope |
| `dK_rope` | `[B, H, S, D_ROPE]` | BF16 | Gradient w.r.t. K rope |
| `dV` | `[B, H, S, D_V]` | BF16 | Gradient w.r.t. V |

### 8.3 Backward Computation

```
# Recompute attention weights (memory-efficient: don't save P)
score = scale * (Q_nope @ K_nope^T + Q_rope @ K_rope^T)
P = softmax(score)  # use saved L for numerically stable recomputation

# Value gradient
dV = P^T @ dO                         # [B,H,S,D_V]

# Score gradient
dP = dO @ V^T                         # [B,H,T,S]
D_i = rowsum(dO * O)                  # [B,H,T] — the "D" correction term
dS = P * (dP - D_i)                   # [B,H,T,S]
dS *= scale

# Split key gradients
dK_nope = dS^T @ Q_nope               # [B,H,S,D_NOPE]
dK_rope = dS^T @ Q_rope               # [B,H,S,D_ROPE]

# Split query gradients
dQ_nope = dS @ K_nope                 # [B,H,T,D_NOPE]
dQ_rope = dS @ K_rope                 # [B,H,T,D_ROPE]
```

### 8.4 Backward Kernel Strategy

Follow the FlashAttention-2 backward structure:

1. **Outer loop** over KV blocks (each threadblock owns a BLOCK_KV slice)
2. **Inner loop** over Q blocks
3. Recompute S from Q and K (using the split dot product), apply softmax via saved L
4. Accumulate dK_nope, dK_rope, dV in registers (owned by this threadblock)
5. Compute dQ contributions and atomically accumulate to global dQ_nope, dQ_rope

The split-head structure doubles the number of MMA operations in the dQ/dK phases (one pass for nope, one for rope) but the shared softmax recomputation is done only once per QK block pair.

---

## 9. PYTHON INTERFACE

```python
# File: bwk/attention/python/blackwell_kernels/mla_attention.py

def mla_attn_forward(
    Q_nope: torch.Tensor,   # [B, H, T, D_NOPE] BF16
    Q_rope: torch.Tensor,   # [B, H, T, D_ROPE] BF16
    K_nope: torch.Tensor,   # [B, H, S, D_NOPE] BF16
    K_rope: torch.Tensor,   # [B, H, S, D_ROPE] BF16
    V: torch.Tensor,        # [B, H, S, D_V]    BF16
    causal: bool = True,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        O: [B, H, T, D_V]  BF16
        L: [B, H, T]       FP32  (logsumexp for backward)
    """
    if scale is None:
        d_qk = Q_nope.shape[-1] + Q_rope.shape[-1]
        scale = d_qk ** -0.5
    return _C.mla_attn_forward(Q_nope, Q_rope, K_nope, K_rope, V, scale, causal)


def mla_attn_backward(
    dO: torch.Tensor,       # [B, H, T, D_V]    BF16
    Q_nope: torch.Tensor,   # [B, H, T, D_NOPE] BF16
    Q_rope: torch.Tensor,   # [B, H, T, D_ROPE] BF16
    K_nope: torch.Tensor,   # [B, H, S, D_NOPE] BF16
    K_rope: torch.Tensor,   # [B, H, S, D_ROPE] BF16
    V: torch.Tensor,        # [B, H, S, D_V]    BF16
    O: torch.Tensor,        # [B, H, T, D_V]    BF16
    L: torch.Tensor,        # [B, H, T]         FP32
    causal: bool = True,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
        dQ_nope: [B, H, T, D_NOPE] BF16
        dQ_rope: [B, H, T, D_ROPE] BF16
        dK_nope: [B, H, S, D_NOPE] BF16
        dK_rope: [B, H, S, D_ROPE] BF16
        dV:      [B, H, S, D_V]    BF16
    """


class MLAAttentionFunc(torch.autograd.Function):
    """torch.autograd wrapper for forward + backward."""

    @staticmethod
    def forward(ctx, Q_nope, Q_rope, K_nope, K_rope, V, causal, scale):
        O, L = mla_attn_forward(Q_nope, Q_rope, K_nope, K_rope, V, causal, scale)
        ctx.save_for_backward(Q_nope, Q_rope, K_nope, K_rope, V, O, L)
        ctx.causal = causal
        ctx.scale = scale
        return O

    @staticmethod
    def backward(ctx, dO):
        Q_nope, Q_rope, K_nope, K_rope, V, O, L = ctx.saved_tensors
        dQ_nope, dQ_rope, dK_nope, dK_rope, dV = mla_attn_backward(
            dO, Q_nope, Q_rope, K_nope, K_rope, V, O, L, ctx.causal, ctx.scale
        )
        return dQ_nope, dQ_rope, dK_nope, dK_rope, dV, None, None


def mla_attention(
    Q_nope: torch.Tensor,
    Q_rope: torch.Tensor,
    K_nope: torch.Tensor,
    K_rope: torch.Tensor,
    V: torch.Tensor,
    causal: bool = True,
    scale: float | None = None,
) -> torch.Tensor:
    """Drop-in replacement for the attention portion of MLAttention.forward()."""
    if scale is None:
        d_qk = Q_nope.shape[-1] + Q_rope.shape[-1]
        scale = d_qk ** -0.5
    return MLAAttentionFunc.apply(Q_nope, Q_rope, K_nope, K_rope, V, causal, scale)
```

---

## 10. INTEGRATION INTO OPENMYTHOS

Replace the attention computation in `open_mythos/main.py` `MLAttention.forward()` (lines 378-388):

```python
# BEFORE (PyTorch):
q = q.transpose(1, 2)
k = k.transpose(1, 2)
v = v.transpose(1, 2)
scale = self.q_head_dim ** -0.5
attn = torch.matmul(q, k.transpose(-2, -1)) * scale
if mask is not None:
    attn = attn + mask
attn = self.attn_drop(F.softmax(attn, dim=-1))
out = torch.matmul(attn, v)

# AFTER (custom kernel):
from blackwell_kernels.mla_attention import mla_attention

q_nope = q_nope.transpose(1, 2)  # [B, H, T, d_nope]
q_rope = q_rope.transpose(1, 2)  # [B, H, T, d_rope]
k_nope = k_nope.transpose(1, 2)  # [B, H, S, d_nope]
k_rope = k_rope.transpose(1, 2)  # [B, H, S, d_rope]
v = v.transpose(1, 2)            # [B, H, S, d_v]

out = mla_attention(q_nope, q_rope, k_nope, k_rope, v, causal=True)
```

This avoids the `torch.cat` calls that concatenate nope+rope before the dot product, and avoids materializing the full `[B, H, T, S]` attention matrix.

---

## 11. TEST PLAN

### 11.1 Correctness Tests

| Test | Input | Validation |
|------|-------|------------|
| Tiny smoke | B=1, H=1, T=32, S=32, d_nope=48, d_rope=16, d_v=48 | `torch.allclose(kernel_O, ref_O, atol=1e-2, rtol=1e-2)` |
| Multi-head | B=2, H=12, T=64, S=64 (Tiny config) | Same |
| Causal mask | T=128, verify upper triangle is zeroed | Compare with masked PyTorch reference |
| Asymmetric seq | T=64, S=256 (prefill-like) | Correctness |
| All variant dims | Each row from Section 1 table | Correctness across all template instantiations |
| Backward | Compare dQ_nope, dQ_rope, dK_nope, dK_rope, dV against `torch.autograd.gradcheck` | Numerical gradient check |

### 11.2 Performance Benchmarks

| Config | Metric | Baseline | Target |
|--------|--------|----------|--------|
| B=2, H=12, T=1024, Tiny dims | Wall time (us) | cuDNN SDPA (concat Q/K then call) | 1.5x SDPA |
| B=2, H=24, T=2048, 3B dims | Wall time (us) | cuDNN SDPA | 1.7x SDPA |
| B=2, H=32, T=4096, 10B dims | Wall time (us) | cuDNN SDPA | 1.5x SDPA |
| Backward (3B dims) | Wall time (us) | PyTorch autograd | 1.5x |

### 11.3 Numerical Accuracy

| Test | Metric | Threshold |
|------|--------|-----------|
| BF16 forward | Max absolute error vs FP32 reference | < 5e-3 |
| BF16 forward | Mean relative error | < 1e-3 |
| Backward gradients | Max absolute error vs FP64 reference | < 1e-2 |

---

## 12. FILE STRUCTURE

```
bwk/attention/
  csrc/attention/
    flash_attn_v2_sm120.cu       # existing — standard flash attention
    flash_attn_fp8_sm120.cu      # existing — FP8 flash attention
    mla_attn_fwd_sm120.cu        # NEW — MLA forward kernel
    mla_attn_bwd_sm120.cu        # NEW — MLA backward kernel
  python/blackwell_kernels/
    attention.py                  # existing
    mla_attention.py              # NEW — Python bindings + autograd
  tests/
    test_attention.py             # existing
    test_mla_attention.py         # NEW
  benchmarks/
    bench_attention.py            # existing
    bench_mla_attention.py        # NEW
  docs/
    attention_agent_state.md      # existing
    mla_attention_agent_state.md  # NEW — optimization log
```

---

## 13. BUILD

```bash
cd /data/src/bwk/attention
CUDA_HOME=/usr/local/cuda-13 python3 setup.py build_ext --inplace
```

Add `mla_attn_fwd_sm120.cu` and `mla_attn_bwd_sm120.cu` to the `CUDAExtension` sources list in `setup.py`.

Compile flags (must match existing kernels):
```
-gencode arch=compute_120a,code=sm_120a
-O3
--use_fast_math
-lineinfo
```

The `a` suffix on `compute_120a` enables FP8 features needed if we add an FP8 MLA variant later.

---

## 14. CONSTRAINTS (from 04_HARD_WON_LESSONS.md)

These are non-negotiable — all empirically validated across 60+ experiments:

1. **XOR swizzle on ALL shared memory regions** — no plain row-major layouts
2. **cp.async double-buffer** — K_nope, K_rope, V all double-buffered
3. **Register-only P conversion** — pack_bf16x2 in registers, never write P to smem
4. **Non-volatile MMA** — use `mma_m16n8k16_bf16_nv` (allows compiler scheduling)
5. **ldmatrix_x4_mma** with baked a1/a2 swap for all A-fragment loads
6. **Separate V preload** before PV MMA loop — compiler hoists into softmax gap
7. **exp2f in log2 space** — pre-scale Q by LOG2E, use exp2f not expf
8. **99 KB smem ceiling** — CUDA reserves 1 KB/block; stay under 96 KB for L1 preservation
9. **80-145 regs/thread** for 3-6 blocks/SM occupancy sweet spot
10. **No 3-stage pipelining** — drops occupancy, not worth extra prefetch distance

---

## 15. ACCEPTANCE CRITERIA

- [ ] Forward kernel produces correct output for all 4 variant dimensions
- [ ] Backward kernel passes `torch.autograd.gradcheck`
- [ ] Forward achieves >= 1.5x cuDNN SDPA on 3B config (B=2, H=24, T=2048)
- [ ] Backward achieves >= 1.3x PyTorch autograd baseline
- [ ] Peak shared memory per block <= 96 KB for all variants
- [ ] Register spills = 0 for Tiny, 1B, 3B variants
- [ ] Integrates into OpenMythos `MLAttention` as drop-in replacement
- [ ] All 6 existing flash attention tests continue to pass (no regression)
