# LU Panel: Register-Resident with Lazy Pivoting (MAGMA Technique)

**Sources:**
- https://icl.utk.edu/files/publications/2014/icl-utk-792-2014.pdf (LU Small Matrices, ACM 2014)
- https://www.netlib.org/utk/people/JackDongarra/PAPERS/icl-utk-1237-2018.pdf (Progressive Batched LU, 2018)
- https://icl.utk.edu/projectsfiles/magma/doxygen/group__magma__getf2__batched.html (MAGMA API)

**Relevant to:** lu worker
**Worker's current problem:** Building v1 blocked LU. Strategy calls for v3 monolithic kernel to beat cuSOLVER (9.4ms at N=4096). The Cholesky project showed that many small kernel launches cannot beat a monolithic kernel. The LU worker needs to understand how to build the panel factorization kernel.

---

## What This Is

MAGMA's register-resident panel factorization technique caches the **entire panel** in the GPU register file, performs LU factorization with pivoting entirely in registers, and writes back only at the end. This eliminates all intermediate shared memory and global memory traffic for the panel.

---

## Why It Matters for Us

The LU worker's v3 (monolithic kernel) needs an efficient panel factorization that runs entirely on the GPU device. cuSOLVER achieves this with a single monolithic kernel using 202 registers and 52KB smem. Our approach needs a device-side panel factorization that can be called from within the monolithic kernel.

The register-resident technique is the MAGMA answer to this problem, and it's the most efficient approach for panels that fit in registers.

---

## Key Technique: Register-Resident Panel

### Architecture
```
1 thread block, m threads (m = panel rows)
Each thread holds 1 complete row of the panel (nb values)
nb is a compile-time template parameter (enables unrolling)
```

### Algorithm
```
for j = 0 to nb-1:          // iterate over columns
    // 1. Find pivot: parallel reduction to find max |A[j:m, j]|
    //    Each thread contributes its own row's value
    //    Use shared memory for warp-level reduction
    pivot_row = argmax(|thread_data[j]| for threads j..m-1)

    // 2. Record pivot (in shared memory pivot array)
    ipiv[j] = pivot_row

    // 3. Swap rows: threads swap their register values
    //    Thread pivot_row and thread j exchange ALL nb values
    //    Uses __shfl_sync or shared memory relay

    // 4. Scale column: thread_data[j] /= pivot (for threads > j)
    //    Each thread scales its own copy of column j

    // 5. Rank-1 update: for threads > j, for k > j:
    //    thread_data[k] -= thread_data[j] * row_j_data[k]
    //    row_j_data[k] is broadcast from thread j via shuffle
```

### Why It's Fast
- **Zero intermediate memory traffic** — panel is read from global once, written back once
- **Perfect parallelism for pivoting** — all threads participate in argmax reduction
- **No synchronization between columns** — each thread works on its own row
- **Compiler unrolls the inner loop** — nb is a template parameter, so the k-loop unrolls completely

### Constraints
- **Panel width nb ≤ 32** (typical) — limited by register file capacity
  - m threads × nb registers per thread = m × nb registers total
  - sm_120: 64K registers/SM. For m=256, nb=32: 256 × 32 = 8192 registers (fits comfortably)
  - For m=1024 (needed for N=4096 with NB=64): 1024 × 64 = 65536 registers (exactly at limit!)
- **FP32 only** — panel factorization needs full precision for pivoting stability
- **m ≤ 1024** (limited by max threads per block on sm_120)

### Lazy Pivoting
Instead of performing row swaps on the trailing matrix after each panel step, **delay all row interchanges to the end of the panel**. This saves:
- Multiple global memory accesses for row swaps mid-factorization
- Synchronization overhead between panel and trailing update

After the panel kernel writes L and U factors back to global memory, a single batched row-swap kernel applies all pivots to the trailing matrix columns.

---

## Application to Our v3 Monolithic Kernel

### Panel Size Selection for N=4096
- **NB=32, IB=16**: Panel is 4096×32. Too many threads (4096 > max block size).
  - Solution: Process panel in chunks of 1024 rows × 32 columns
  - Or use IB=16 sub-blocking: factorize 1024×16 panels

- **NB=64**: Our current selection. Panel is 4096×64.
  - Thread-per-row: needs 4096 threads (too many for one block)
  - Solution: Multiple blocks or sub-panel approach

### Recommended Panel Strategy for N=4096
```
NB = 64 (outer blocking)
IB = 16 (inner blocking / sub-panel width)

For each NB panel:
  for ib = 0 to NB-1 step IB:
    // Register-resident factorization of sub-panel (m × IB)
    register_panel_factorize(panel[ib:, ib:ib+IB])

    // Trailing update within NB panel (small GEMM)
    device_trsm(L[ib:ib+IB, ib:ib+IB], panel[ib:ib+IB, ib+IB:NB])
    device_gemm(panel[ib+IB:, ib:ib+IB], panel[ib:ib+IB, ib+IB:NB], panel[ib+IB:, ib+IB:NB])

  // Apply pivots to trailing matrix columns (NB to N)
  apply_row_swaps(ipiv, A[:, NB:])

  // Large trailing update (cuBLAS or device-side GEMM)
  trsm(L_panel, A[:NB, NB:])
  gemm(A[NB:, :NB], A[:NB, NB:], A[NB:, NB:])
```

### Device-Side GEMM for Trailing Update
The trailing GEMM is the compute bottleneck (O(N^2 × NB)). cuSOLVER uses TF32 tensor cores internally for this. Our options:
1. **cuBLAS from device** — not possible (cuBLAS is host API)
2. **cuBLASDx device-side GEMM** — available on sm_120, limited size
3. **Our own BF16 MMA GEMM** — proven at 0.97x cuBLAS, available as a primitive
4. **CDP2 (dynamic parallelism)** — launch child kernels from device

Option 3 (our GEMM primitive) is the most promising since we already have it working and can inline it into the monolithic kernel.

---

## Caveats

1. **Register-resident approach for NB=64 is tight** — 4096×64 = 262K values, but the panel is processed in sub-blocks (IB=16), so only 4096×16 = 65K values need to be in registers at once.

2. **FP32 panel factorization** — the LU panel MUST use FP32 for numerical stability with pivoting. The trailing GEMM can use BF16 MMA with FP32 accumulators.

3. **Row swap performance** — pivoting requires row swaps across the full trailing matrix. This is a bandwidth-bound operation. Our linalg worker's `swap_rows` kernel (6.06x cuBLAS) can be reused.

4. **cuSOLVERDx alternative** — cuSOLVERDx v0.3.0 provides `getrf_partial_pivot` that runs on device. This could replace the register-resident panel kernel, but may have size limitations and the sm_120 bug (NVBUG 5288270) is a risk.
