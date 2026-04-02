# MAGMA 2.9.0 Blackwell Support and HPL GPUPDFACT Update

**Source:** https://github.com/icl-utk-edu/magma/blob/master/ReleaseNotes (MAGMA 2.9.0, Jan 2025)
**Source:** https://dl.acm.org/doi/10.1145/3712285.3759875 (SC24 HPL PDFACT paper)
**Relevant to:** LU worker
**Worker's current problem:** Building monolithic LU kernel for N=4096 on sm_120. Need current state of open-source GPU LU implementations and cooperative-group panel factorization techniques.

---

## 1. MAGMA 2.9.0 Release (January 23, 2025)

### Blackwell GPU Support

MAGMA 2.9.0 officially supports **sm_100 and sm_120** (NVIDIA Blackwell),
requiring CUDA 12.8 or higher. This means:

- `magma_sgetrf_native()` should compile and run on RTX 5090
- The fused panel kernel (`sgetf2_native_kernel`) works on sm_120
- Batched getrf variants work on sm_120

### New LU Features

**Variable-batch no-pivot LU:**
- `magma_<T>getrf_nopiv_vbatched()` -- non-pivoting LU on non-uniform batches
- Expert interface: `magma_<T>getrf_nopiv_expert_vbatched()` handles small
  diagonal elements below a user-defined threshold

The expert interface is notable: it provides threshold-based handling of small
pivots, similar to threshold pivoting but without the full row swap logic.

### Performance Improvements

- Batch Cholesky (`magma_<T>potrf_batched`) improved
- Batch TRSV (`magma_<T>trsv_batched`) improved
- Small-size LU, QR, and Cholesky factorizations tuned

### What's NOT New

- No changes to the native getrf panel kernel architecture
- No tensor core acceleration for LU trailing updates
- No monolithic kernel for large-N getrf
- No cooperative groups-based panel factorization

MAGMA's architecture for large-N LU remains: GPU-native panel kernel (spin-wait
inter-block sync) + cuBLAS trailing GEMM + host-side blocked loop.

---

## 2. Practical Value of MAGMA 2.9 for Our Worker

### As a Reference Implementation

MAGMA 2.9's `sgetf2_native_kernel` is the most relevant open-source GPU panel
kernel. Since it now officially supports sm_120, the worker can:

1. **Build MAGMA on our system** and benchmark `magma_sgetrf_native(N=4096)`
2. Compare against cuSOLVER's 9.4ms baseline
3. Study the source code for the panel kernel and spin-wait sync pattern

### As a Building Block

The MAGMA panel kernel could potentially be called from within a monolithic
kernel (if refactored). However, MAGMA uses a host-side blocked loop, so
significant modification would be needed.

### Expected Performance

Based on MAGMA's published results (2024 Exascale paper):
- MAGMA's native getrf is competitive with cuSOLVER for large N
- For N=4096 (relatively small), cuSOLVER's monolithic kernel likely wins
  because MAGMA still uses multiple kernel launches

---

## 3. HPL GPUPDFACT: Cooperative-Groups Panel Factorization (SC24 Update)

### What Changed Since Our Last Brief

The SC24 paper provided concrete performance numbers for GPU-based panel
factorization (GPUPDFACT) vs CPU-based panel factorization:

**Three PDFACT variants evaluated:**
| Variant | Description | HPL Performance (Frontier) |
|---------|-------------|---------------------------|
| Original PDFACT | CPU-based panel | 1.0 exaFLOPS |
| Dedicated-Thread (DT) | Multi-threaded CPU | 1.2 exaFLOPS |
| **GPUPDFACT** | GPU cooperative groups | **1.35 exaFLOPS** |

GPUPDFACT delivered **35% higher HPL performance** than the baseline by
eliminating CPU-GPU data transfers for the panel.

### Cooperative Groups Implementation Details

The paper reveals the exact synchronization pattern:

```
For each column k in the panel:
  1. Each threadblock performs rank-1 update on its portion of column k
  2. Each threadblock selects its block-local pivot candidate via
     intra-block reduction (__syncthreads)
  3. grid.sync() -- grid-level barrier
  4. Designated threadblock selects final pivot from block-local candidates
  5. grid.sync() -- broadcast pivot decision
  6. Non-owning threadblocks proceed with rank-1 updates on subsequent
     columns, OVERLAPPING with the pending pivot row swap
```

### Overlap Optimization

The key optimization: after determining the pivot, non-owning threadblocks
**do not wait for the row swap to complete**. They proceed with rank-1 updates
on subsequent columns immediately. Only the owning threadblock performs the
physical row swap. This overlaps swap latency with computation.

This is a fine-grained overlap that goes beyond the simple grid.sync() pattern
in our existing brief. It uses atomic flags (not grid.sync) for the overlap:
- grid.sync() for barrier points (end of each column)
- Atomic flags for non-blocking pivot broadcast

### Effective Bandwidth Improvement

GPUPDFACT achieved up to **3.5x improvement in effective panel bandwidth**
with optimal process layout at smaller scale.

---

## 4. Implications for Our Monolithic LU Kernel

### Panel Factorization Design

Our existing brief recommended a single-block panel (simplest, panel is only
~5-10% of compute). The GPUPDFACT results suggest multi-block panel is worth
considering IF:

1. Panel factorization becomes the bottleneck after BF16x9 accelerates the
   trailing GEMM (which will happen -- if trailing GEMM drops from ~8ms to
   ~2.7ms, the panel becomes a larger fraction)

2. The overlap pattern (rank-1 updates overlapping with row swaps) provides
   measurable benefit

### Recommended Architecture (Updated)

```
Iteration k:
  // Phase 1: Panel factorization
  IF trailing GEMM is dominant (early iterations, large trailing):
    Single-block panel (simple, panel is small fraction)
  IF panel becomes bottleneck (late iterations, or after BF16x9):
    Multi-block panel with GPUPDFACT-style overlap

  grid.sync();

  // Phase 2: LASWP + TRSM (distributed across all blocks)
  // Use threshold pivoting to skip unnecessary row swaps

  grid.sync();

  // Phase 3: Trailing GEMM (BF16x9 for FP32 accuracy at 3x speed)
  // All blocks participate, grid-stride loop over tiles

  grid.sync();
```

### When to Use Multi-Block Panel

For N=4096, NB=64:
- Panel factorization: ~0.5ms (64 columns * ~8us each)
- Trailing GEMM with BF16x9: ~2.7ms (first iteration, shrinks after)
- Ratio: panel is ~15-20% of total after BF16x9

This is borderline. Multi-block panel adds complexity for modest gain. Start
with single-block panel and measure. If panel time > 20% of total, consider
GPUPDFACT-style multi-block.

---

## Sources

- [MAGMA 2.9.0 Release Notes](https://github.com/icl-utk-edu/magma/blob/master/ReleaseNotes)
- [MAGMA Homepage](https://icl.utk.edu/magma/)
- [HPL PDFACT Optimization (SC24)](https://dl.acm.org/doi/10.1145/3712285.3759875)
- [MAGMA 2024 Exascale Paper](https://journals.sagepub.com/doi/10.1177/10943420241261960)
