# CCCL 3.2 / CUB New Primitives (CUDA 13.2)

**Source:** [CUDA 13.2 Blog Post](https://developer.nvidia.com/blog/cuda-13-2-introduces-enhanced-cuda-tile-support-and-new-python-features/)
**Relevant to:** rmsnorm worker, dotproduct worker, linalg worker
**Date:** 2026-03-14

---

## What This Is

CUDA 13.2 ships CCCL 3.2 (CUDA C++ Core Libraries) with several new CUB device
primitives that could be useful as reference implementations or building blocks.

## New Primitives

### 1. Fixed-Size Segmented Reduction

`cub::DeviceSegmentedReduce` now has a variant optimized for uniform segment sizes.

| Segment Size | Speedup vs General |
|-------------|-------------------|
| Small (< 64 elements) | Up to 66x |
| Large (> 1024 elements) | Up to 14x |

**Relevance:** RMSNorm and LayerNorm compute per-row reductions over fixed-size
segments (hidden dimension). If hidden_dim is uniform across rows, this optimized
path could serve as a fast reference baseline. Our custom kernels should beat this.

### 2. Top-K Selection

`cub::DeviceTopK::MaxKeys` -- up to 5x faster than radix sort for small K values.
Less memory consumption than full sort.

**Relevance:** Not directly applicable to our current kernel work, but useful for
future sparse attention or top-K sampling kernels.

### 3. Segmented Scan

`cub::DeviceSegmentedScan` -- prefix sum within segments.

**Relevance:** Could be useful for cumulative operations in attention (cumulative
softmax normalization) as a reference implementation.

### 4. Binary Search and Find-If

`cub::DeviceFind::[Upper/LowerBound]` and `cub::DeviceFind::FindIf` with up to 7x
speedup via early-exit logic.

**Relevance:** Minimal for our current work. Could be useful for sparse operations.

## Deterministic Reductions (from CCCL 3.1)

Three determinism modes for `cub::DeviceReduce`:
- **Not-guaranteed:** Single-pass atomics (fastest)
- **GPU-to-GPU:** Bitwise-identical across runs (slowest, ~20% overhead)
- **Run-to-run:** Default two-pass (middle performance)

**Relevance:** The GPU-to-GPU deterministic mode could help debug numerical issues
in our reduction kernels (softmax, RMSNorm). If we see non-deterministic results,
we can use this mode to verify our kernel's correctness against a deterministic
reference.

## Summary

Most of these are reference-quality implementations, not direct building blocks for
our hand-tuned PTX kernels. The fixed-size segmented reduction is the most interesting
as a benchmark baseline for RMSNorm-type operations.
