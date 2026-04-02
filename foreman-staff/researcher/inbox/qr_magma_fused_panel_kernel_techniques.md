# MAGMA Fused Panel Kernel: Register-Tiled QR on GPU

**Source:** Haidar, Tomov, Dongarra, Luszczek. "Batch QR Factorization on GPUs: Design, Optimization, and Tuning." ICCS 2022. (https://www.netlib.org/utk/people/JackDongarra/PAPERS/batchqr-gpu-2022.pdf)
**Also:** MAGMA source: `magmablas/sgeqr2_batched_fused_sm.cu` (https://github.com/CEED/MAGMA)
**Relevant to:** QR worker
**Worker's current problem:** Needs to implement the GEQR2 panel factorization kernel. This is the most complex panel kernel in the numerical projects -- more complex than LU's getf2 or Cholesky's potf2.

## What This Is

MAGMA's fused panel kernel for QR factorization caches the entire panel in the GPU register file (one row per thread) and performs unblocked Householder factorization entirely in fast memory. The key innovations are: (1) fusing GEQR2 + LARFT + LARFB into a single kernel, (2) register-tiled storage that eliminates global memory traffic during the panel, and (3) a multi-level fusion strategy that adapts to matrix size.

## Why It Matters for Us

The panel factorization (GEQR2) is the serial bottleneck in QR. It's BLAS-2 dominated -- each column involves a norm computation (reduction), a Householder reflector generation, and application of that reflector to remaining columns (GEMV + rank-1 update). On GPU, these are memory-bound unless the panel fits in fast memory.

MAGMA's approach eliminates global memory traffic during the panel by keeping everything in registers. After incorporating these fused panel kernels, **the panel is no longer the dominant cost** for any tested matrix size. This is the technique our worker should use if cuSOLVERDx's device-side geqrf proves limiting.

## Key Technique

### Thread Organization
- **One thread per row**: For a panel of size m x nb, launch m threads (one per row)
- Each thread "owns" one row of the panel in registers
- At least m threads required per thread block
- The panel width nb must be known at compile time (template parameter) to avoid register spilling

### Register Storage
- Each thread stores nb values in registers (its row of the panel)
- For m=256, nb=32: each thread holds 32 FP32 values = 128 bytes of registers
- Total register usage per thread block: m * nb * 4 bytes
- **Critical constraint**: nb must be small enough that registers don't spill. Practical limit on sm_120: nb <= 32-64 depending on m

### Householder Reflector Generation (per column j)
```
1. Compute column norm: ||A(j:m, j)||
   - Each thread contributes A(j, threadRow)^2 to shared memory
   - Tree reduction in shared memory across all threads
   - Two-layer reduction: warp-level __shfl_xor for threads within a warp,
     then shared memory reduction across warps

2. Generate Householder vector v and scalar tau:
   - Thread j computes: v(j) = 1 (implicit), tau = standard Householder formula
   - Threads j+1..m have v values in their registers (the column below diagonal)

3. Apply reflector to remaining columns j+1..nb:
   - Each thread computes v_local * A(threadRow, k) for columns k = j+1..nb
   - This is a dot product (v^T * A(:,k)) requiring reduction
   - **Multi-column reduction**: Threads reorganized into independent groups,
     each group collaboratively reduces one column
   - After reduction, each thread updates: A(threadRow, k) -= tau * v_local * dot_result
```

### Kernel Fusion Strategy (Three Levels)

MAGMA uses three strategies depending on matrix size:

1. **Fully fused (sizes <= 32)**: Single kernel does entire QR factorization. GEQR2 + LARFT + LARFB all in one kernel. Panel stays in registers, trailing update applied immediately. No global memory traffic except initial load and final store.

2. **Panel+Update fusion (medium sizes)**: Fused panel kernel handles GEQR2 within the panel, then applies reflectors directly to the trailing matrix without forming T. The elementary reflectors are applied one-by-one to the trailing columns (BLAS-2 style but in-register). Avoids LARFT overhead.

3. **LAPACK-style (large sizes)**: Standard blocked algorithm with separate GEQR2 kernel (register-tiled), LARFT kernel, and LARFB via batched GEMM. Uses the register-tiled panel kernel for the panel step, then launches GEMM for trailing update.

### Avoiding T Matrix Formation

For the fused panel+update kernel (strategy 2), MAGMA applies reflectors directly without forming T:
```
For each reflector j:
    v = column j of panel (in registers)
    // Apply to trailing columns in registers:
    for k = nb..n_trail:
        dot = reduce(v * A(:, k))  // shared memory reduction
        A(:, k) -= tau * v * dot   // register update
```

This eliminates LARFT entirely at the cost of O(nb * m * n_trail) work per panel instead of O(nb * m + nb^2 * n_trail). For small nb, the difference is small and the reduced kernel launches win.

### Shared Memory Usage
- Tree reduction workspace: nb floats per warp (for multi-column reduction)
- The panel V matrix is loaded into shared memory after GEQR2 for the LARFT computation (in the strategy-3 path)
- Upper triangular part of V is zeroed, diagonal set to 1 (unit lower triangular)

## Performance Impact

After incorporating the fused panel kernels:
- **Panel is no longer the dominant cost** for any tested size
- MAGMA batched QR is **1.6x-1.7x faster than KBLAS** and **5.8x-7.4x faster than cuBLAS** for single/double precision
- The fused kernels work on the "fastest memory levels" (registers + shared memory), which is why the simplification (apply reflectors directly, skip T) works well

## Application to sm_120

### Direct applicability:
- The thread-per-row design maps well to sm_120's warp structure
- For nb=32, each thread needs 32 registers for the panel -- leaves room for other state
- Warp shuffle reductions (__shfl_xor_sync) are identical on sm_120
- Shared memory tree reductions are standard

### Practical considerations for our worker:
1. **Start with cuSOLVERDx geqrf** for the panel (it likely uses a similar approach internally)
2. **If cuSOLVERDx panel becomes the bottleneck**, implement a custom register-tiled kernel using MAGMA's design
3. **Template the panel width**: nb must be a compile-time constant for register allocation
4. **Two-layer reduction**: Use warp shuffles within warps (32 threads), then shared memory across warps. This matches our attention kernel's softmax reduction pattern.

### Key sizing constraints on sm_120:
- Max 255 registers per thread (compiler limit)
- For nb=64 panel width: 64 FP32 values = 64 registers just for the panel. Plus working registers, leaves ~128 for computation. Tight but feasible.
- For nb=32: 32 registers for panel, plenty of room. Recommended starting point.
- Thread count m must be <= 1024 (max threads per block). For m > 1024, use multiple thread blocks or chunk rows.

## Caveats

1. **Batched context vs single large QR**: MAGMA's fused kernels were designed for batched QR (many small matrices). For a single large QR, only the panel step uses this kernel -- the trailing update uses standard GEMM. The register-tiled technique is for the panel only.

2. **nb compile-time constant**: The panel width must be known at compile time. If trying different block sizes, need multiple compiled kernel variants (template specialization).

3. **Thread count scales with m**: For m=4096 (a full 4096-tall panel), you'd need 4096 threads per block. sm_120 supports at most 1024 threads per block. Solution: for large panels, process multiple rows per thread or use the CAQR tile-based approach (256-row tiles in shared memory).
