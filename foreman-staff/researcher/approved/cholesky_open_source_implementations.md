# Open-Source GPU Cholesky Implementations to Study

**Source:** https://github.com/icl-utk-edu/magma | https://github.com/ecrc/kblas-gpu | https://docs.nvidia.com/cuda/cusolverdx/examples/advanced_example.html | https://github.com/facebookresearch/baspacho | https://arxiv.org/html/2601.03754v1
**Relevant to:** cholesky worker (new kernel)
**Worker's current problem:** Needs reference implementations to study before writing our own sm_120 kernel.

## What This Is

A survey of open-source GPU Cholesky implementations, ranked by relevance to our use case (custom mma.sync kernel on RTX 5090).

## Implementations

### 1. MAGMA (Most Relevant — Study First)

**Repo:** https://github.com/icl-utk-edu/magma (v2.10.0, March 2026)
**License:** BSD-3-Clause
**What it has:**
- `magma_dpotrf_gpu` — hybrid CPU+GPU blocked Cholesky
- `magma_dpotrf_native` — **GPU-only** blocked Cholesky (most relevant)
- `magma_dpotf2_gpu` — unblocked panel factorization on GPU (N ≤ 512)
- Batched variants for many small matrices
- Left-looking blocked algorithm
- Multiple CUDA streams for panel/update overlap

**Key source files to study:**
- `src/dpotrf.cpp` / `src/dpotrf_native.cpp` — blocked algorithm main loop
- `magmablas/dpotf2_kernels.cu` — GPU panel kernel (the critical inner kernel)
- `magmablas/dsyrk_batched.cu` — batched SYRK kernel

**Why study this:** MAGMA is the gold standard for dense GPU linear algebra. The GPU-only `_native` variant shows how to do panel factorization entirely on GPU without CPU fallback. The panel kernel (`potf2_kernels.cu`) is the critical code to understand.

**Caveats:** MAGMA uses cuBLAS for GEMM/SYRK trailing updates. We'd replace those with our own mma.sync GEMM/SYRK kernels.

### 2. cuSOLVERDx (NVIDIA's Newest API)

**Docs:** https://docs.nvidia.com/cuda/cusolverdx/
**What it has:**
- `blocked_potrf.cu` example — left-looking blocked algorithm using cuSOLVERDx
- Device-callable potrf for small tiles
- Designed to be composable with cuBLASDx GEMM
- Single thread block per matrix (batched workloads)

**Key example:** `examples/advanced_example.html` shows the blocked algorithm with cuSOLVERDx potrf + cuBLASDx GEMM.

**Why study this:** Shows NVIDIA's recommended approach for composing small potrf tiles with high-performance GEMM. The "device-callable" API means the potrf kernel runs within a larger kernel, avoiding launch overhead.

**Caveats:** cuSOLVERDx is a library, not source code. We can learn the algorithm structure but can't study the kernel internals. Also, it targets batched workloads (many small matrices), not single large matrix.

### 3. KBLAS (GPU-Resident potrf)

**Repo:** https://github.com/ecrc/kblas-gpu
**License:** BSD-3-Clause
**What it has:**
- GPU-resident potrf kernel (matrix stays on GPU)
- Batched Cholesky for small matrices
- Recursive and batch algorithms maximizing GPU bandwidth
- Real precisions (s/d)

**Why study this:** "GPU-resident" means no host-device transfers during factorization — everything stays on GPU memory. Good model for our approach.

**Caveats:** Older codebase, may not have tensor core support.

### 4. Block Tridiagonal Cholesky (2026 — Tested on RTX 5090)

**Paper:** https://arxiv.org/html/2601.03754v1
**Implementation:** Uses NVIDIA Warp library with cuBLASDx/cuSOLVERDx
**What it has:**
- Tested on **RTX 5090** (directly relevant hardware)
- CUDA stream overlap between potrf, trsm, syrk, gemm
- Fused kernel for small blocks (n ≤ 16)
- Blocked variant for larger blocks
- Performance: 100-500x vs sparse solvers, 25x vs CPU BLAS

**Why study this:** Only implementation we found that's been tested on RTX 5090. Shows achievable performance on our exact hardware. The stream overlap pattern is directly applicable.

**Caveats:** Block tridiagonal structure (not general dense matrix). Uses Warp/Python JIT, not hand-written CUDA. But the algorithm insights transfer.

### 5. BaSpaCho (Facebook, Sparse)

**Repo:** https://github.com/facebookresearch/baspacho
**License:** MIT
**What it has:** Supernodal sparse Cholesky with GPU CUDA support.

**Why study this:** Only relevant if we ever do sparse Cholesky. Skip for now.

### 6. Pedagogical Implementations

**Göttingen CUDA tutorial:** http://num.math.uni-goettingen.de/~stkramer/doc/autogen/CUDA_HPC_Praktikum/step_1.html
- Complete CUDA kernels for blocked Cholesky with 16×16 tiles
- Five kernels: factorize_diagonal_block, strip_update, diag_update, lo_update
- Clear, readable code — best for understanding the algorithm
- Not optimized (no tensor cores, small tiles)

**Simple CUDA Cholesky:** https://github.com/alexbenfica/CUDA-Cholesky-UFMG
- Basic implementation, good for understanding the serial algorithm

## Recommended Study Order

1. **Göttingen tutorial** — understand the 5-kernel blocked structure
2. **MAGMA dpotrf_native + dpotf2_kernels** — see production-quality GPU-only panel kernel
3. **cuSOLVERDx blocked_potrf example** — see NVIDIA's recommended composition pattern
4. **Block tridiagonal paper** — see RTX 5090 performance and stream overlap

## Caveats

- **None of these use mma.sync for the panel factorization.** The panel (potf2) is column-by-column with dot products and sqrt — not GEMM-shaped. Tensor cores only help in the trailing updates (SYRK, GEMM, TRSM).
- **All use cuBLAS for GEMM/SYRK.** We'd replace with our existing hand-written mma.sync GEMM kernel, which already beats cuBLAS at 1.34x (FP8) / 0.97x (BF16) on sm_120.
- **License compatibility:** MAGMA and KBLAS are BSD-3-Clause. Fine for studying and reimplementing. Do not copy-paste MAGMA code.
