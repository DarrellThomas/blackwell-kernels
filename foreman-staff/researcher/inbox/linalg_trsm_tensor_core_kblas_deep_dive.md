# TRSM: Tensor Core Opportunities, KBLAS Deep Dive, and cuBLAS Internals

**Sources:**
- [KBLAS-GPU library (KAUST BLAS)](https://github.com/ecrc/kblas-gpu)
- [Charara et al., "Redesigning Triangular Dense Matrix Computations on GPUs" (Euro-Par 2016)](https://link.springer.com/chapter/10.1007/978-3-319-43659-3_35)
- [Charara et al., "A framework for dense triangular matrix kernels" (CCPaE 2017)](https://onlinelibrary.wiley.com/doi/full/10.1002/cpe.4187)
- [Carrica & Onyango, "Toward Portable GPU Performance: Julia Recursive TRMM and TRSM" (2025)](https://arxiv.org/html/2504.13821v1)
- [Zhang, "Matrix Computations on TensorCore GPU" (UH thesis)](https://uh-ir.tdl.org/server/api/core/bitstreams/718e26c5-4ae9-4fe9-a37b-623ecdf1538b/content)
- [Haidar et al., "Harnessing GPU Tensor Cores for Fast FP16" (SC 2018)](https://www.netlib.org/utk/people/JackDongarra/PAPERS/haidar_fp16_sc18.pdf)
- [Hogg, "Optimizing triangular solve (TRSV) in CUDA" (STFC)](https://epubs.stfc.ac.uk/manifestation/7800/dtrsv.pdf)
- [Nath, Tomov, Dongarra, "A Fast Dense Triangular Solve in CUDA" (SIAM SISC 2010)](https://epubs.siam.org/doi/10.1137/12088358X)
- [MAGMA TRSM batched documentation](https://icl.utk.edu/projectsfiles/magma/doxygen/group__magma__trsm__batched.html)
- [cuSolverDx documentation](https://docs.nvidia.com/cuda/cusolverdx/)

**Relevant to:** linalg worker
**Worker's current problem:** TRSM at 0.82-1.00x cuBLAS (47-57 us at N=256, NRHS=64). Current approach: delegation to torch.linalg.solve_triangular (cuBLAS F32 under the hood). Worker wants native BF16 TRSM.

---

## 1. What cuBLAS TRSM Actually Does Internally

cuBLAS TRSM uses the MAGMA-derived diagonal-block-inversion approach:

**Step 1: Pre-invert all diagonal blocks** (trtri_diag kernel)
- Divides the NxN triangular matrix into ceil(N/NB) blocks of size NB x NB
- cuBLAS default NB = 128 (from MAGMA `get_nb.cpp`)
- Each NB x NB diagonal block is inverted independently (embarrassingly parallel)
- This is a single batched kernel launch

**Step 2: Blocked forward/back substitution**
For each block column k (sequential in k):
1. Panel solve: `X[k:k+NB] = dinvA[k] @ B[k:k+NB]` -- this is a GEMM
2. Trailing update: `B[k+NB:] -= A[k+NB:, k:k+NB] @ X[k:k+NB]` -- this is a GEMM

**Key observation for F32 TRSM on sm_120:** cuBLAS uses F32 GEMM (NOT tensor cores) for the off-diagonal GEMM updates. Tensor cores on sm_120 only accelerate BF16/FP16/FP8/TF32 GEMM, not F32. This means cuBLAS F32 TRSM is leaving tensor core performance on the table.

**This is the primary opportunity:** A BF16 TRSM that uses BF16 tensor-core GEMM for the off-diagonal updates would run the GEMM portion ~2x faster than cuBLAS's F32 GEMM, while the base case (diagonal block solve) runs at similar speed. Since GEMM is 75-87% of total FLOPs, the overall speedup is significant.

---

## 2. KBLAS: The Gold Standard for GPU TRSM

KBLAS (KAUST BLAS) is the highest-performing open-source GPU TRSM. Key implementation details from the codebase and papers:

### Performance Claims
- Up to 8x speedup over cuBLAS for TRMM
- Up to 2x speedup over cuBLAS for TRSM
- Recursive formulation that converts most work to GEMM calls

### Recursive Algorithm
KBLAS uses divide-and-conquer, splitting the matrix at N/2 each level:
```
RecTRSM(A, B, N):
  if N <= BASE_NB:
    base_trsm(A, B)  // shared-memory warp-shuffle kernel
    return
  RecTRSM(A11, B1, N/2)           // solve top half
  B2 -= GEMM(A21, B1)             // off-diagonal update (this is where time goes)
  RecTRSM(A22, B2, N/2)           // solve bottom half
```

### Base Case Kernel Details (32x32 tile)
This is the most critical piece for a custom implementation:
- **Thread block:** (32, WPB) where WPB=8 (8 warps, each handling one column of B)
- **A loaded into shared memory** (32x32 = 4 KB, fits easily)
- **B elements stored in registers** (one per thread per column)
- **Warp-shuffle broadcast** for the solved value at each step:
  ```cuda
  // Column-by-column forward substitution
  for (int j = 0; j < 32; j++) {
      if (lane == j) b_reg /= smem_A[j][j];   // divide by diagonal
      float solved = __shfl_sync(0xFFFFFFFF, b_reg, j);  // broadcast
      if (lane > j) b_reg -= smem_A[lane][j] * solved;   // update below
  }
  ```
- **Why 32:** Matches warp size exactly. Every thread in the warp holds one row element. `__shfl_sync` provides zero-cost intra-warp communication.
- **Why WPB=8:** 8 independent B columns processed in parallel provide enough ILP to hide the sequential dependency along the triangular solve.

### KBLAS vs cuBLAS: Why KBLAS Wins
1. **No diagonal inversion:** KBLAS solves diagonal blocks directly (forward substitution). cuBLAS inverts them first. Inversion is numerically unnecessary and adds overhead.
2. **Better GEMM sizing:** The recursive decomposition produces GEMMs at (N/2 x N/2) * (N/2 x NRHS) at the top level -- these are large enough for good tensor core utilization. cuBLAS's blocked approach produces (M x NB) * (NB x NRHS) GEMMs which may be thinner.
3. **Register-cached base case:** The warp-shuffle base kernel avoids shared memory bank conflicts and synchronization overhead that a shared-memory-only implementation would have.

---

## 3. Tensor Core TRSM: How to Use MMA for Triangular Solve

Tensor cores cannot directly compute triangular solves. They can only compute `D = A * B + C` (dense matrix multiply-accumulate). But the blocked/recursive TRSM decomposes the problem so that 75-87% of FLOPs are dense GEMM -- and those GEMM calls CAN use tensor cores.

**The strategy (from Zhang's thesis and Haidar et al.):**

### For the GEMM portion (75-87% of FLOPs):
Use your existing BF16 GEMM kernel (0.97x cuBLAS, or 2.02x for FP8 batched). These calls are:
- Panel-diagonal multiply: (NB x NB) @ (NB x NRHS) -- small but parallelizable
- Trailing update: (M x NB) @ (NB x NRHS) -- can be large, good tensor core utilization

### For the base case (13-25% of FLOPs):
Keep as FP32 scalar warp-shuffle forward substitution (the KBLAS approach). At 32x32 blocks, this is a tiny amount of work per block.

### Mixed-precision approach:
- Off-diagonal GEMM: BF16 tensor cores (accumulate in FP32)
- Diagonal solve: FP32 scalar
- This gives tensor-core throughput for the bulk of computation while maintaining precision where it matters (the diagonal division)

### Performance ceiling estimate for N=256, NRHS=64:
- Recursion levels: log2(256/32) = 3
- 8 diagonal base-case solves (32x32 each)
- 7 GEMM calls of varying sizes
- GEMM fraction: ~75% of FLOPs
- If GEMM runs 2x faster (tensor core vs F32 scalar): overall ~1.6x
- If using FP8 GEMM (2.02x cuBLAS): GEMM runs ~4x faster, overall ~2.3x

---

## 4. Multi-SM Blocked TRSM: Distributing Work

For large N, distributing the TRSM across all 170 SMs is critical. The recursive approach naturally does this through the GEMM calls (your GEMM kernel already uses all SMs). But the diagonal base cases are sequential bottlenecks.

**KBLAS's approach to parallelism:**
- At each recursion level, the GEMM update uses ALL SMs (it's a standard GEMM launch)
- The base case solves are serialized (can't solve block k+1 until block k is done)
- But each base case processes WPB=8 RHS columns in parallel across warps

**For N=256, NB=32:**
- Sequential chain: 8 base-case solves + 7 GEMM calls = 15 sequential operations
- Each operation is one kernel launch (or one cooperative-group sync point)
- At ~3-5 us per launch: 45-75 us of launch overhead alone
- Current cuBLAS time: ~47-57 us

**This is the kernel-launch-overhead trap.** For small N (256), launch overhead dominates. Solutions:

1. **Persistent kernel with grid-level sync:** Launch once, use cooperative groups for inter-block synchronization between steps. Eliminates all intermediate launches.

2. **CUDA graphs:** Capture the sequence of GEMM + base-case launches in a graph. Graph launch overhead is ~5 us total instead of ~5 us per step.

3. **Larger base case (NB=64 or 128):** Fewer recursion levels, fewer launches. At NB=128 for N=256: only 2 base cases + 1 GEMM = 3 launches. Each base case is 128x128, which needs 4 warps (128/32 = 4 warp-rows) x WPB columns.

4. **Single monolithic kernel for small N:** For N<=256, handle everything in one kernel launch. Load the entire 256x256 matrix (128 KB in BF16) into shared memory and solve in-place with warp-level synchronization. This avoids all launch overhead.

---

## 5. Practical Recommendation for the Worker

Given the current codebase and benchmark (N=256, NRHS=64):

### Quickest path to beating cuBLAS:
The worker is already at 1.00x with cuBLAS BF16 delegation (exp11). The 0.82x measurement was vs F32 cuBLAS, which is no longer the reference.

If the reference is now cuBLAS BF16 TRSM (exp11 shows 1.00x), beating it requires a genuinely faster kernel. The GEMM-decomposition approach with tensor cores is the path:

1. **Write a 32x32 base-case kernel** using the KBLAS warp-shuffle pattern
2. **Implement 2-level recursion** for N=256: split into two 128x128 halves, each split into four 64x64, etc.
3. **Use existing BF16 GEMM** for off-diagonal updates
4. **For N=256:** Consider a single-kernel approach (monolithic) to eliminate launch overhead

### Code to study:
KBLAS source is at `https://github.com/ecrc/kblas-gpu`. Key files:
- `src/batch_triangular/Xtrsm_batch.cu` -- batched TRSM kernel
- `src/batch_triangular/Xtrsm_batch_drivers.cuh` -- launch logic
- `src/Xblas_core.cu` -- base case kernels

---

## Caveats

1. **N=256 with NRHS=64 is small.** The GEMM calls from recursive decomposition will be small GEMMs (128x128, 64x64). Your GEMM kernel is tuned for larger sizes. Rectangular GEMMs with one small dimension (like 192x128 * 128x64) may not hit the same performance as square 4096x4096. Benchmark these shapes specifically.

2. **Precision.** The existing cuBLAS delegation runs in F32. A BF16 TRSM has lower precision. If the benchmark validates against F32 results, the BF16 kernel must still pass accuracy tests. Use FP32 accumulation in GEMM and FP32 for the diagonal solve to maintain accuracy.

3. **The reference shifted.** exp11 shows TRSM at 1.00x with "python delegation" (BF16-typed cuBLAS). To beat 1.00x, the custom kernel must be genuinely faster than cuBLAS BF16 TRSM, not just faster than F32 TRSM. The tensor-core advantage over cuBLAS BF16 TRSM is smaller because cuBLAS BF16 may already use tensor cores for its internal GEMMs.
