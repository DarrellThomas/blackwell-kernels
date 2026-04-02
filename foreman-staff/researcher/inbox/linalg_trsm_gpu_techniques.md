# GPU TRSM (Triangular Solve) Techniques — Comprehensive Research Brief

**Source:** Multiple (see individual sections)
**Relevant to:** linalg worker
**Worker's current problem:** TRSM at 0.82x cuBLAS F32 via `torch.linalg.solve_triangular` delegation. Worker has custom 0.97x cuBLAS BF16 GEMM and 1.34x FP8 GEMM available. Needs a native TRSM to beat cuBLAS.

---

## 1. The Two Fundamental Approaches

There are two proven ways to build a fast GPU TRSM. Both decompose the problem into GEMM calls (which you already have) plus a small base-case solver.

### Approach A: Diagonal Block Inversion + GEMM (MAGMA style)

**Source:** MAGMA library (https://icl.utk.edu/magma/)

**Algorithm (blocked, for lower-triangular left-side: L * X = B):**
```
For k = 0 to N-1 in steps of NB:
  1. Invert diagonal block: invL_kk = inv(L[k:k+NB, k:k+NB])
  2. Solve current panel: X[k:k+NB, :] = invL_kk * B[k:k+NB, :]    ← this is a GEMM
  3. Update trailing blocks: B[k+NB:, :] -= L[k+NB:, k:k+NB] * X[k:k+NB, :]  ← GEMM
```

**Key details:**
- MAGMA uses NB = 128 for the block size
- Diagonal block inversion is done by `trtri_diag` kernel (NB x NB blocks)
- Step 2 becomes a dense GEMM (triangular structure already captured in invL)
- Step 3 is a standard GEMM
- Requires workspace: `ceil(N/NB) * NB * NB` for the inverted diagonal blocks
- The `flag` parameter lets you pre-compute inversions and reuse across multiple solves

**Performance:** This is what cuBLAS itself uses internally. The diagonal block inversion step is a serial bottleneck on GPU — it's fast for each block but they must be done before the GEMM. For large matrices this amortizes well. For moderate sizes it limits speedup.

**Numerical stability note:** Inverting diagonal blocks is numerically stable when the triangular matrix comes from a stable factorization (LU with pivoting, Cholesky), which is the typical TRSM use case.

### Approach B: Recursive Decomposition (KBLAS / Charara et al. style)

**Source:** "Redesigning Triangular Dense Matrix Computations on GPUs" — Charara, Ltaief, Keyes (Euro-Par 2016)
https://link.springer.com/chapter/10.1007/978-3-319-43659-3_35

**Algorithm (recursive, for lower-triangular L * X = B):**
```
function recursive_trsm(L, B):
  if size <= BASE:
    native_trsm(L, B)     ← small base-case kernel
    return

  Partition L into [L11  0 ; L21  L22], B into [B1; B2]

  recursive_trsm(L11, B1)                    ← recurse on top half
  B2 = B2 - L21 * B1                         ← GEMM (this is where the time goes)
  recursive_trsm(L22, B2)                    ← recurse on bottom half
```

**Key details:**
- Recursion halves the problem each time (partition at N/2)
- At each level, one GEMM handles the off-diagonal update
- Most FLOPs end up in GEMM calls at the higher recursion levels
- The base case fires at small sizes where a specialized kernel handles TRSM directly
- Each recursion level is a separate kernel launch (manageable overhead)

**Performance:** Up to 2x speedup over cuBLAS TRSM on various GPU generations. Up to 8x for TRMM. The key insight: the recursive approach eliminates the serial diagonal-inversion bottleneck of Approach A by parallelizing inversions across separate kernel launches.

**Why it's faster:** The diagonal inversion in Approach A is on the critical path — all blocks must be inverted before GEMM can proceed. The recursive approach eliminates this by solving small triangular systems directly and expressing all large operations as GEMM.

---

## 2. Base Case Kernel Implementation

**Source:** Charara et al. (2017), "A Framework for Dense Triangular Matrix Kernels on Various Manycore Architectures"
https://onlinelibrary.wiley.com/doi/full/10.1002/cpe.4187

The base case is the critical piece you need to write. Here is how the experts do it:

### Tile Size: 32x32

The base case operates on 32x32 diagonal blocks of A, solving against WPB columns of B simultaneously.

### Thread Configuration
- Thread block: (32, WPB) threads, where WPB = 8 gives best performance
- One warp (32 threads) processes one column of B
- WPB warps process WPB columns in parallel

### Register Caching Strategy
- **B columns cached in registers** — each thread holds one element of B
- **Values shared across warp via `__shfl_sync`** — no shared memory needed for B
- A diagonal block is loaded into shared memory once

### Forward Substitution (lower triangular, column-by-column):
```
For each column j of the 32x32 block:
  // Thread i holds B[i] for its column
  // Thread j broadcasts A[j,j] via shuffle
  if (thread_id == j):
    B_reg /= A[j][j]          // divide by diagonal
  // Broadcast solved value via __shfl_sync
  solved_val = __shfl_sync(mask, B_reg, j)
  // All threads below j update their value
  if (thread_id > j):
    B_reg -= A[thread_id][j] * solved_val
```

### Why Warp Shuffle Instead of Shared Memory
- Shared memory access has bank conflict risk and latency
- `__shfl_sync` within a warp is essentially free (no memory access)
- Eliminates need for `__syncthreads()` barriers
- The 32-thread warp maps perfectly to 32 rows of the triangular block

### Two Algorithmic Variants for the Full TRSM

**Right-looking (eager update):** After solving one B-column block with the diagonal A-block, immediately do a partial GEMM update of lower blocks, then write to memory before the next diagonal block. Better memory locality.

**Left-looking (lazy update):** Accumulate all GEMM updates from upper blocks first, then solve the diagonal block. Better for reducing kernel launch overhead in the recursive approach.

The KBLAS implementation uses the right-looking variant for better data reuse.

---

## 3. Using Tensor Cores for the GEMM Parts

**Source:** Haidar, Tomov, Dongarra, Higham — "Harnessing GPU Tensor Cores for Fast FP16 Arithmetic" (SC 2018)
https://dl.acm.org/doi/10.5555/3291656.3291719

**Key insight for your worker:** In the blocked/recursive TRSM, the GEMM calls dominate the FLOPs for matrices above ~256. You can replace those GEMM calls with your existing tensor-core GEMM.

**The technique (from MAGMA's TC-TRSM):**
- Keep the base case (diagonal block solve) in FP32 or BF16 scalar math
- Replace all off-diagonal GEMM updates with tensor core GEMM (BF16 or FP8)
- The base case is a tiny fraction of total FLOPs — precision there costs little

**Why this works:** Tensor cores can only do GEMM. They cannot do triangular solve directly. But the recursive decomposition converts TRSM into a sequence of GEMMs + tiny triangular base cases. So tensor cores handle 90%+ of the work.

**Numerical precision strategy (from recent work):**
- Large off-diagonal blocks: BF16 tensor core GEMM (fast, less precision-sensitive)
- Diagonal blocks: FP32 scalar solve (small, precision-critical)
- For your FP8 GEMM at 1.34x cuBLAS: use it for the off-diagonal GEMM if precision is acceptable; otherwise use BF16 GEMM at 0.97x

---

## 4. Mixed-Precision Recursive TRSM (State of the Art, 2025)

**Source:** "Hierarchical Precision and Recursion for Accelerating Symmetric Linear Solves on MXUs"
https://arxiv.org/html/2601.08082v1

This January 2025 paper by Carrica et al. is the most advanced approach found:

**Algorithm:**
```
function mixed_precision_trsm(L, B, depth):
  if size <= base_threshold:
    vendor_trsm(L, B)    ← cuBLAS FP32 for leaf blocks
    return

  Partition L into [L11 0; L21 L22], B into [B1; B2]

  mixed_precision_trsm(L11, B1, depth+1)
  B2 = B2 - GEMM_low_precision(L21, B1)    ← FP16/BF16 tensor core GEMM
  mixed_precision_trsm(L22, B2, depth+1)
```

**Key insight:** Positive-definite matrices (common TRSM input from Cholesky) are diagonally dominant. Off-diagonal blocks can safely use lower precision. Diagonal blocks need higher precision.

**Performance results (H200):**
- 5.3x speedup over cuSOLVER FP64 TRSM
- 100x more accurate than pure FP16
- Mixed-precision SYRK: up to 27x speedup

**Applicability to sm_120:** The algorithm is architecture-agnostic. It just needs a fast GEMM (which you have) and a base-case solver. The Julia implementation runs on NVIDIA, AMD, and Apple Silicon.

---

## 5. NVIDIA cuSolverDx — Device-Side TRSM

**Source:** NVIDIA cuSolverDx documentation
https://docs.nvidia.com/cuda/cusolverdx/

**What it is:** Device-side (in-kernel) TRSM that can be called from your CUDA kernel, part of the MathDx library ecosystem.

**Key facts:**
- TRSM added in cuSolverDx 0.2.0
- sm_120 (RTX 5090) is supported since 0.2.0
- Performance improvements for TRSM on Hopper added in 0.3.0
- Used as the base case in cuSolverDx's own blocked Cholesky example (sequence: unblocked POTRF -> TRSM -> cuBLASDx GEMM)
- Part of libmathdx package (same as cuBLASDx)

**Limitation:** cuBLASDx itself (0.5.1) still only supports GEMM, NOT TRSM. TRSM is in cuSolverDx, not cuBLASDx. The data types and size constraints for TRSM are not publicly documented in the release notes — you'd need to check the API reference or header files after installing libmathdx.

**Practical use:** cuSolverDx TRSM could serve as your base case if writing a custom 32x32 kernel is too complex. Let cuSolverDx handle the small triangular solve, and your custom GEMM handles the off-diagonal updates.

---

## 6. KBLAS Library — Open Source Reference Implementation

**Source:** https://github.com/ecrc/kblas-gpu

**What it provides:**
- Complete recursive TRSM implementation in CUDA
- Batched TRSM for small matrices (up to 256)
- Base case kernels using register caching + warp shuffles
- Freely available, MIT-like license

**Performance claims:** Up to 2x speedup over cuBLAS for TRSM, with MAGMA POTRS getting up to 2x speedup when linked with KBLAS TRSM instead of cuBLAS TRSM.

**Code location:** `src/batch_triangular/Xtrsm_batch.cu` and related files in the `batch_triangular` directory. The `Xtrsm_batch_drivers.cuh` file contains the kernel launch logic.

**Recommendation:** This is the best reference implementation to study. It shows exactly how to build the base case kernel and wire it into the recursive framework.

---

## 7. Recommended Implementation Strategy for the Worker

Given the worker's existing assets (0.97x BF16 GEMM, 1.34x FP8 GEMM, proven MMA infrastructure):

### Phase 1: Recursive TRSM with cuBLAS Base Case
```
1. Implement recursive decomposition (trivial — ~50 lines)
2. Use existing BF16 GEMM for off-diagonal updates
3. Use cuBLAS (via torch.linalg.solve_triangular) for base case (N <= 128)
4. This alone should beat 0.82x because the GEMM portion uses your fast kernel
```

### Phase 2: Custom 32x32 Base Case Kernel
```
1. Write a single-warp TRSM kernel for 32x32 blocks
2. Use __shfl_sync for intra-warp communication
3. Thread block = (32, 8): one warp per B column, 8 columns at once
4. A block in shared memory, B elements in registers
5. Column-by-column forward substitution with warp shuffle broadcast
```

### Phase 3: Mixed Precision (Optional)
```
1. Use FP8 GEMM (1.34x cuBLAS) for large off-diagonal blocks
2. Keep FP32 for base case diagonal solves
3. Blockwise quantize/dequantize at GEMM boundaries
```

### Expected Performance
- Phase 1 alone should reach ~1.0-1.1x cuBLAS (your GEMM is faster, base case is same)
- Phase 2 eliminates kernel launch overhead for base case, could reach 1.2-1.5x
- Phase 3 with FP8 GEMM could push to 1.5-2.0x for large matrices

---

## 8. Caveats

1. **sm_120 specifics:** All techniques above use scalar math (not tensor cores) for the base case. Tensor cores only help for the GEMM portion. This is fine — the base case is a tiny fraction of FLOPs.

2. **Matrix sizes matter:** For small matrices (N < 256), the base case dominates and you won't beat cuBLAS much. For large matrices (N > 1024), GEMM dominates and your advantage compounds.

3. **cuBLASDx does NOT have TRSM** (as of 0.5.1). cuSolverDx does have TRSM (device-side). These are different libraries despite similar names.

4. **Numerical stability:** Block inversion is fine for matrices from stable factorizations. The recursive approach avoids inversion entirely and is unconditionally stable.

5. **The 0.82x baseline is cuBLAS F32.** A BF16 TRSM with BF16 GEMM is not an apples-to-apples comparison. If the spec requires F32-equivalent precision, the base case must stay in F32 and only GEMM can use lower precision (with accumulation in F32).

---

## Sources

- MAGMA TRSM: https://icl.utk.edu/projectsfiles/magma/doxygen/group__magma__trsm.html
- MAGMA trtri_diag: https://icl.utk.edu/projectsfiles/magma/doxygen/group__magma__trtri__diag__batched.html
- Charara et al. 2016 (recursive TRSM): https://link.springer.com/chapter/10.1007/978-3-319-43659-3_35
- Charara et al. 2017 (framework): https://onlinelibrary.wiley.com/doi/full/10.1002/cpe.4187
- KBLAS library: https://github.com/ecrc/kblas-gpu
- Carrica et al. 2025 (mixed precision recursive): https://arxiv.org/html/2601.08082v1
- Carrica et al. 2025 (Julia recursive TRSM): https://arxiv.org/html/2504.13821v1
- Haidar et al. 2018 (TC FP16 TRSM): https://dl.acm.org/doi/10.5555/3291656.3291719
- cuSolverDx TRSM: https://docs.nvidia.com/cuda/cusolverdx/
- cuSolverDx release notes: https://docs.nvidia.com/cuda/cusolverdx/release_notes.html
- Abdelfattah et al. 2019 (batched small TRSM): https://dl.acm.org/doi/10.1145/3267101
- cuBLASDx (GEMM only, no TRSM): https://docs.nvidia.com/cuda/cublasdx/
