# Panel/Update Overlap via CUDA Streams for Cholesky

**Source:** https://icl.utk.edu/files/publications/2017/icl-utk-987-2017.pdf | https://arxiv.org/html/2601.03754v1 | https://icl.utk.edu/projectsfiles/magma/doxygen/group__magma__potrf.html
**Relevant to:** cholesky worker (new kernel)
**Worker's current problem:** The panel factorization (potf2) is the serial bottleneck in blocked Cholesky — how to hide its latency.

## What This Is

GPU Cholesky performance is limited by the sequential panel factorization (potf2). The key optimization is **overlapping** the panel factorization of step k+1 with the trailing matrix update of step k using CUDA streams. This is the technique that MAGMA's GPU-only `dpotrf_native` uses, and it's what makes GPU-only Cholesky competitive with hybrid CPU+GPU approaches.

## Why It Matters for Us

Without overlap, the GPU sits idle during panel factorization (~5% of FLOPs but significant wall-clock time for small panels). With overlap, the panel runs on one SM while 169 other SMs work on the trailing GEMM/SYRK update. This is essentially free — the panel latency is hidden behind the much larger trailing update.

## Key Technique

### Basic Stream Overlap (2 streams)

```
Stream 0: [panel k] [trailing update k] [panel k+2] [trailing update k+2] ...
Stream 1:           [panel k+1]          [trailing update k+1]              ...
                    ↑ overlaps with trailing update k
```

**Dependency chain per step k:**
1. Panel k factorization depends on: trailing update k-1 completing for the panel columns
2. Trailing update k depends on: panel k completing
3. Panel k+1 can START as soon as trailing update k has updated the k+1 panel columns (NOT all of trailing update k)

### MAGMA GPU-Only Approach (Haidar et al., 2017)

The "GPU-only" Cholesky eliminates the CPU entirely:

1. **Panel factorization on GPU:** Use a custom potf2 GPU kernel (single SM) instead of sending the panel to CPU
2. **Trailing update on GPU:** Standard cuBLAS SYRK + GEMM across all SMs
3. **Overlap:** CUDA events signal when the panel columns needed by the next panel have been updated

**Key insight from Haidar:** The CPU was only used for the panel because GPU potf2 was slow. By optimizing the GPU potf2 kernel (batched column operations, warp-level reductions), the panel runs fast enough on GPU that the CPU data transfer overhead is eliminated.

### 3-Stream Overlap (Block Tridiagonal, 2026)

The recent block tridiagonal paper uses three concurrent streams for even better overlap:

```
Stream 0: [potrf(k)]                    [potrf(k+1)]
Stream 1:           [trsm(k)]                        [trsm(k+1)]
Stream 2:                     [syrk(k)]                          [syrk(k+1)]
```

With CUDA events for synchronization:
- `trsm(k)` waits for `potrf(k)` completion (event on stream 0)
- `syrk(k)` waits for `trsm(k)` completion (event on stream 1)
- `potrf(k+1)` waits for `syrk(k)` completion (event on stream 2)
- **But:** `gemm(k)` (off-diagonal update) can overlap with `syrk(k)` since they're independent

This reduces the critical path from 4 sequential operations to 3, improving throughput by ~12%.

### Atomic-Based Right-Looking Overlap

The block tridiagonal paper also introduces atomic operations for even more overlap:
- Multiple trailing update blocks can write to the same output tile using atomicAdd
- Low contention (≤2 writers per tile) makes atomic overhead negligible
- Enables fully right-looking factorization where ALL updates for a tile can proceed concurrently

### Implementation Pattern for Our Kernel

```cpp
cudaStream_t stream_panel, stream_update;
cudaEvent_t panel_done, update_done;

for (int k = 0; k < N; k += nb) {
    // Wait for previous trailing update to finish updating our panel columns
    cudaStreamWaitEvent(stream_panel, update_done);

    // Panel factorization (single SM, small kernel)
    potf2_kernel<<<1, 256, smem, stream_panel>>>(A + k*lda + k, nb, lda);
    cudaEventRecord(panel_done, stream_panel);

    // TRSM: solve for column tiles below diagonal
    cudaStreamWaitEvent(stream_update, panel_done);
    trsm_kernel<<<grid_trsm, 128, smem, stream_update>>>(
        A + k*lda + k, A + k*lda + k+nb, nb, N-k-nb, lda);

    // Trailing SYRK + GEMM update (all SMs, tensor cores)
    syrk_gemm_kernel<<<grid_update, 128, smem, stream_update>>>(
        A + k*lda + k+nb, A + (k+nb)*lda + k+nb, nb, N-k-nb, lda);
    cudaEventRecord(update_done, stream_update);
}
```

## Caveats

- **Stream overlap only helps when trailing update is large enough.** For small matrices (N < 1024 with nb=64), the trailing update finishes too quickly to overlap with the next panel. The technique pays off for larger matrices.
- **Panel factorization must be a separate kernel launch** (or in a separate stream) to overlap with trailing updates. A fully fused kernel that does panel + update in one launch loses the ability to overlap.
- **CUDA event overhead is negligible** (~1-2 us per event record/wait). Use events, not `cudaStreamSynchronize()`.
- **Graph capture:** For repeated factorizations (e.g., in an optimization loop), capture the stream pattern as a CUDA graph for reduced launch overhead.
- **We already use multiple streams for other kernels** (GEMM, attention). The stream infrastructure should be reusable.
