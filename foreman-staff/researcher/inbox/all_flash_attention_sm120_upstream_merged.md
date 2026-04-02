# Flash-Attention SM120 Forward + Backward + Varlen Merged Upstream

**Sources:**
- [SM120 forward pass PR #2329](https://github.com/Dao-AILab/flash-attention/pull/2329) (merged 2026-03-12)
- [SM120 backward pass PR #2330](https://github.com/Dao-AILab/flash-attention/pull/2330) (merged 2026-03-12)
- [SM120 varlen support PR #2333](https://github.com/Dao-AILab/flash-attention/pull/2333) (merged 2026-03-13)
**Relevant to:** attention worker, all workers (reference implementation)
**Date:** 2026-03-15

---

## What This Is

Three PRs merged into Dao-AILab/flash-attention in the last 72 hours (March 12-13, 2026)
adding full SM120 (Blackwell GeForce / DGX Spark) support for flash attention:

1. **Forward pass** -- `FlashAttentionForwardSm120` subclass of `FlashAttentionForwardSm80`
2. **Backward pass** -- `FlashAttentionBackwardSm120` with SM120-tuned tile sizes
3. **Variable-length sequences** -- `flash_attn_varlen_func` for SM120

All contributed by Second Nature Computing (blake-snc), tested on DGX Spark (SM121a).

---

## Why It Matters for Us

This is the first official upstream flash attention implementation targeting sm_120. It
confirms several architectural decisions we've already made and reveals some we should review.

### Key Technical Details

**Forward pass architecture (PR #2329):**
- Uses SM80-era MMA instructions (`mma.sync.aligned.m16n8k16`) -- same as our kernel
- 99 KB shared memory capacity check (matches our empirical finding)
- Tile sizes: D<=64 uses 128x128 (48 KB smem), D>64 uses 128x64 (64 KB smem)
- Does NOT use TMA (even though sm_120 supports cp.async.bulk)
- BF16 only (no FP8 path)

**Causal bug fix discovered during SM120 testing:**
The SM80 causal masking loop had a bug: it passed `is_first_n_block=True` for ALL causal
iterations, which reset the running softmax max/sum on each iteration. This was only exposed
when testing with D>64 and tile_n=64 (2+ causal N-blocks). The fix: omit `is_first_n_block`
(default False) after the first block. This is something our kernel should verify.

**Backward pass (PR #2330):**
- Tile config: m=n=64, 128 threads (4 warps, all in M-direction)
- Stages: 1 for D>64, 2 for D<=64 (double buffering only for small head dims)
- Uses SM80 code paths for pre/postprocess kernels

**Varlen support (PR #2333):**
- Padding `cu_seqlens` with one extra element to avoid OOB reads from wasted scheduler tiles
- Clamping `n_block = max(n_block_max - 1, 0)` for zero-seqlen batches
- 12 varlen configs tested: D=64/96/128, causal/non-causal, equal/unequal sequence lengths

### Tile Size Comparison

| Head dim | Our kernel | Upstream SM120 | Notes |
|----------|-----------|----------------|-------|
| D=64 | BQ=64, BKV=64 | 128x128 | Upstream uses larger Q tile |
| D=128 | BQ=64, BKV=64 | 128x64 | Upstream uses larger Q, smaller KV |

Our kernel uses BQ=64 with dynamic dispatch to BQ=128 for large grids. The upstream
uses BQ=128 as default. This is worth studying -- larger BQ amortizes the Q load cost
but requires more smem.

### What We Can Learn

1. **BQ=128 is viable on sm_120 at D=64** -- the 48 KB smem fits. Our kernel uses BQ=64
   by default. Testing BQ=128 as default (not just for large grids) could help.

2. **The causal softmax bug** -- verify our online softmax correctly handles the
   `is_first_block` flag across causal iterations. Our implementation uses a different
   structure but should be audited for the same class of bug.

3. **No TMA used** -- confirms that cp.async (which we use) is the right approach for
   sm_120. TMA would add complexity with no clear benefit for this workload.

4. **BF16 only** -- upstream does not implement FP8 for sm_120. Our FP8 implementation
   (2.33x SDPA) remains ahead of upstream in this dimension.

---

## Caveats

1. **Python-based (CuTile/Cute DSL)** -- this is NOT hand-written CUDA C++. It uses
   the flash-attention framework's Python-to-PTX compilation pipeline. Performance
   comparisons should account for framework overhead.

2. **No performance numbers in PRs** -- the PRs only report correctness (max_diff).
   We don't know how fast this implementation is on RTX 5090 vs our kernel.

3. **SM121a (DGX Spark), not SM120 (RTX 5090)** -- tested on GB10, which is SM121a.
   SM120 and SM121a share the same MMA ISA but may differ in SM count and clocks.

4. **No FP8, no SplitKV, no paged KV cache** -- these are listed as "future PRs."
   We're ahead on FP8. SplitKV and paged KV cache could be valuable for inference
   and worth monitoring.
