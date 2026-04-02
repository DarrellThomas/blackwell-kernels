# Sawtooth Wavefront Reordering: Deep Dive

**Source:** https://arxiv.org/abs/2601.16032 (full paper read)
**Authors:** Yifan Zhu, Yekai Pan, Chen Ding (University of Rochester)
**Relevant to:** attention worker
**Worker's current problem:** BF16 at 1.76x SDPA (68 us), compute-bound (math_throttle 48%). FP8 at 2.33x SDPA (52 us), latency-bound (SM 43.8%, scoreboard 18%).
**Date:** 2026-03-15 (deep dive update, replaces earlier abstract-only brief)

---

## What This Is

A CTA-level optimization that alternates the KV tile scan direction on each
query tile iteration -- even iterations scan KV tiles forward (0 to N), odd
iterations scan backward (N to 0). This "sawtooth" pattern approximately
**halves the LRU reuse distance** for KV data in L2 cache, cutting L2 misses
by 50-67% and improving throughput up to 60%.

Tested on NVIDIA GB10 (Grace Blackwell SoC), which is **sm_121** -- the same
sm_120 family as our RTX 5090 (sm_120). The technique is ISA-agnostic: it
modifies only loop iteration order, not MMA instructions or memory operations.

---

## CORRECTION: Previous Brief Was Wrong

The earlier brief (attention_sawtooth_wavefront_reordering_l2_cache.md) described
the technique as *staggering KV start offsets across CTAs*:

```cuda
// WRONG - this is NOT what the paper does
int kv_start = (blockIdx.x * stride) % num_kv_tiles;
```

The actual technique is simpler: **alternating the inner loop direction per
iteration within each CTA.** All CTAs still start at KV tile 0 on their first
Q tile. The win comes from adjacent CTAs (which are roughly synchronized) seeing
recently-cached KV data from the CTA that just traversed backward.

---

## The Algorithm (from paper's Algorithm 4)

```
Input: Q_seq = sequence of query tiles assigned to this CTA
Input: N_KV = total number of KV tiles
i_local = 0

for each query tile q in Q_seq:
    if i_local % 2 == 0:
        // Forward scan
        for j = 0 to N_KV-1:
            load K[j], V[j]
            compute Attention(q, K[j], V[j])
    else:
        // Backward scan
        for j = N_KV-1 downto 0:
            load K[j], V[j]
            compute Attention(q, K[j], V[j])

    i_local += 1
```

That is the entire optimization. One integer counter, one branch on parity.

---

## Why It Works: Reuse Distance Analysis

**Cyclic ordering (standard):** Every Q tile scans KV from 0 to N. After CTA
finishes Q tile i, it starts Q tile i+1 at KV=0. Between the last access to
K[N-1] (end of tile i) and the next access to K[N-1] (end of tile i+1), the
CTA accesses ALL N KV tiles. Reuse distance = N for every tile.

**Sawtooth ordering:** Q tile i scans 0->N, Q tile i+1 scans N->0. After
accessing K[N-1] at end of tile i, the NEXT access is K[N-1] at START of
tile i+1. Reuse distance = 0 for the boundary tiles, and at most N/2 on
average for interior tiles.

The paper states: "unlike the cyclic order where all reuse distances equal the
data size, the sawtooth order reduces the reuse distance for most data accesses
to be less than the data size."

**L2 hit rate scaling:** Hit Rate = approximately 1 - 1/N_SM. With 48 SMs on
GB10, theoretical hit rate approaches 98%. The key insight is that synchronous
wavefronts of CTAs (all 48 SMs progress roughly in lockstep) mean one CTA's KV
loads populate L2 lines that adjacent CTAs will soon need.

---

## Performance Results (All Configs)

### CUDA Implementation
| Metric | Before | After | Change |
|--------|--------|-------|--------|
| L2 non-compulsory misses | baseline | ~50% reduction | -50% |
| Throughput | 1.3 TFLOPS | 2.4 TFLOPS | +85% |

### CuTile Implementation (B=8, S=128K, D=64, T=64x64)
| Variant | L2 Misses (sectors) | Throughput | Gain |
|---------|---------------------|------------|------|
| Non-causal (before) | 370M | 61 TFLOPS | - |
| Non-causal (after) | 120M | 69 TFLOPS | **+13%** |
| Causal (before) | — | 41 TFLOPS | - |
| Causal (after) | — | 66 TFLOPS | **+60%** |

### Why Causal Gets 60% vs Non-Causal 13%

Causal masking reduces total KV accesses from (S/T)^2 to S(S-1)/(2T). With
fewer total accesses, the L2 miss reduction becomes a larger fraction of total
work. In non-causal, the kernel is already more compute-bound and L2 is a
smaller fraction of total time.

---

## Applicability to Our Attention Kernel (RTX 5090, sm_120)

### Hardware Comparison
| Spec | GB10 (sm_121) | RTX 5090 (sm_120) |
|------|---------------|-------------------|
| SMs | 48 | 170 |
| L2 Cache | 24 MiB | 96 MiB |
| Memory BW | ~300 GB/s (LPDDR5X) | 1792 GB/s (GDDR7) |
| SM family | sm_120 | sm_120 |

### Analysis for Our Primary Config (B=2, H=8, N=2048, D=64, causal)

KV working set per head:
- N_KV tiles = 2048 / 64 = 32 tiles
- Per KV tile: 64 * 64 * 2 bytes * 2 (K+V) = 16 KB
- Total KV per head: 32 * 16 KB = 512 KB
- Total KV across all heads in batch: 2 * 8 * 512 KB = 8 MB

RTX 5090 L2 = 96 MB. Our 8 MB working set fits **12x over** in L2.

**Verdict: L2 is NOT a bottleneck at our primary config.** The entire KV set
fits comfortably in L2. Sawtooth would have near-zero impact here.

### When Sawtooth WOULD Help on RTX 5090

L2 pressure becomes real when KV working set approaches 96 MB:
- N=32768, D=128, B=4, H=32: 4*32*32768*128*2*2 = 2 GB -- well over L2
- N=8192, D=64, B=8, H=32: 8*32*8192*64*2*2 = 256 MB -- well over L2
- N=4096, D=64, B=16, H=16: 16*16*4096*64*2*2 = 128 MB -- over L2

For inference serving with large batch sizes (B>=16) or long contexts (N>=8K),
sawtooth could provide meaningful L2 miss reduction.

### Key Insight: We Already Tried "Block Index Remapping" and it FAILED

From the worker's agent state (experiment log, "What Didn't Work"):
> "Block index remapping -- destroyed L2 locality"

The previous attempt remapped blockIdx to change CTA-to-tile assignment. Sawtooth
is fundamentally different: it does NOT remap CTAs. Each CTA processes the same
tiles as before. Only the inner loop direction alternates. This preserves all
existing CTA-level L2 locality while adding wavefront-level reuse.

---

## Implementation in Our Kernel

The change is trivial. In the KV iteration loop:

```cuda
// Current: always scan forward
for (int kv = kv_start; kv < kv_end; kv++) {
    // load K[kv], V[kv], compute QK^T, softmax, PV
}

// Sawtooth: alternate direction
static __shared__ int iter_count;
if (threadIdx.x == 0) iter_count = 0;  // init once

// ... per Q tile:
int forward = (iter_count % 2 == 0);
for (int i = 0; i < num_kv_tiles; i++) {
    int kv = forward ? (kv_start + i) : (kv_end - 1 - i);
    // load K[kv], V[kv], compute QK^T, softmax, PV
}
if (threadIdx.x == 0) iter_count++;
```

No changes to: MMA instructions, shared memory layout, register usage,
occupancy, tile sizes, or any other kernel internal.

### Causal Masking Interaction

With causal masking, the KV range is already variable per Q tile (only KV
tiles where k <= q are accessed). Sawtooth still works: just reverse the
order within the valid KV range. The paper confirms this works and gives
the largest speedup (60%) in the causal case.

---

## Persistent CTA vs Standard Grid

The paper tested both:
- **Persistent CTA:** Grid-stride loop over Q tiles (our kernel uses standard grid)
- **Standard grid:** One Q tile per CTA, hardware scheduling

Both show "nearly identical L1/L2 behavior." Sawtooth works with either.
Our kernel uses a standard grid launch, so we can implement sawtooth by
adding the direction flag to our existing loop without any grid changes.

However, if our kernel processes only one Q tile per CTA (which it does for
the primary config where grid = B*H*ceil(N/BQ) = 2*8*32 = 512 blocks), then
`i_local` only ever equals 0 -- every CTA does exactly one forward pass.

**This means sawtooth has NO effect when each CTA processes exactly one Q tile.**

For sawtooth to work, CTAs must process MULTIPLE Q tiles (persistent CTA or
grid-stride loop). This happens when:
- Grid size > total SMs * blocks/SM (standard launch, but HW scheduler
  reassigns SMs to new Q tiles -- the SM retains L2 residency)
- Persistent CTA grid (explicit loop over Q tiles)

With our 512-block grid on 170 SMs * 3 blocks/SM = 510 slots, each CTA
gets ~1 Q tile. **Sawtooth would require either longer sequences (more Q tiles
per CTA) or a persistent CTA approach.**

---

## Limitations

1. **Tile size constraint:** T=128 tiles may be split by compiler, breaking
   the access pattern. Our T=64 is safe.

2. **Not applicable when working set fits in L2.** At our primary config
   (8 MB working set, 96 MB L2), there's nothing to gain.

3. **Requires multiple Q tiles per CTA** to get the alternating pattern.
   Single-Q-tile-per-CTA grids (our primary config) see no benefit.

4. **No source code released.** Only pseudocode in the paper.

---

## Recommendation

**Low priority for our primary config (B=2, H=8, N=2048, D=64).** Two reasons:
1. KV working set (8 MB) fits 12x in L2 -- no L2 pressure
2. Grid = 512 blocks on 510 CTA slots means ~1 Q tile per CTA -- no alternation

**Medium-high priority for long-context or large-batch configs.** If we
benchmark at N>=8K, B>=8, or D=128, sawtooth becomes relevant. Implementation
cost is near-zero (one counter + one branch), so worth adding as a configurable
option when persistent CTA support is added.

**The finding that our previous "block index remapping" failed is important
context.** Sawtooth is NOT the same thing -- it preserves CTA-to-tile mapping
and only changes inner loop direction. But both target L2, and at our primary
config, L2 is not the bottleneck.
