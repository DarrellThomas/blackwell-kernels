# cuSOLVERDx v0.3.0 / MathDx 25.12.1 — Updated Capabilities for LU

**Source:** https://docs.nvidia.com/cuda/cusolverdx/release_notes.html
**Source:** https://docs.nvidia.com/cuda/cublasdx/release_notes.html
**Source:** https://docs.nvidia.com/cuda/mathdx/index.html
**Relevant to:** numerical/ worker (LU factorization device-side approach)
**Worker's current problem:** Need latest info on device-side factorization capabilities for sm_120 to decide between cuSOLVERDx-based and custom kernel approaches.

---

## MathDx Package: Current State (March 2026)

| Component | Version | Key Capability |
|-----------|---------|---------------|
| **MathDx** | 25.12.1 | Package container |
| **cuSOLVERDx** | 0.3.0 | Factorization, eigenvalue, SVD |
| **cuBLASDx** | 0.5.1 | GEMM (including FP8, BF16, mixed precision) |
| **cuFFTDx** | 1.6.1 | FFT |
| **cuRANDDx** | 0.2.2 | Random number generation |
| **nvCOMPDx** | 0.1.2 | Compression |

Requirements: C++17, CUDA 12.0.0+, GCC 7+/Clang 9+.

---

## cuSOLVERDx v0.3.0 — What's New Since Our Last Research

### New in v0.3.0 (latest)
- **SVD:** BDSVD (bidiagonal) and GESVD (general) -- batched and non-batched
- **Eigenvalue:** HTEV (tridiagonal) and HEEV (symmetric/Hermitian)
- **Tridiagonal solver:** GTSV_NO_PIVOT
- **Q generation:** UNGQR (from QR), UNGLQ (from LQ)
- **Performance improvements** on Hopper for TRSM, UNMQR, UNMLQ, GEQRF, GELQF, GELS
- **Breaking:** TRSM dimensions updated for LAPACK consistency; `cusolverdx_io.hpp` header now required

### Available for LU on sm_120

| Function | Description | Added | sm_120 Status |
|----------|-------------|-------|---------------|
| `getrf_no_pivot` | LU without pivoting | v0.1.0 | Supported (v0.2.0+) |
| `getrf_partial_pivot` | LU with partial pivoting | v0.2.0 | Supported, see caveat |
| `getrs` | Solve using LU factors | v0.2.0 | Supported |
| `gesv` | Combined LU + solve | v0.2.0 | Supported |
| `trsm` | Triangular solve | v0.2.0 | Supported |

### Known sm_120 Bug (STILL PRESENT in v0.3.0)

**CUDA 12.8, 12.9, and 13.0 may miscompile kernels using `gesv_no_pivot`** with high register pressure when sm_120 and `type::real` are combined.

Workarounds:
1. Define `CUSOLVERDX_IGNORE_NVBUG_5288270_ASSERT` macro
2. Use `-Xptxas -O0` compilation flag

**Note:** Only `gesv_no_pivot` is explicitly called out. `getrf_partial_pivot` is NOT mentioned as affected. But proceed with caution and verify correctness empirically.

---

## cuBLASDx v0.5.1 — What's New

### Key Features for LU Trailing GEMM

| Feature | Version | Relevance |
|---------|---------|-----------|
| sm_120 support | v0.4.0 | Basic Blackwell support |
| Suggested layouts + swizzle | v0.4.0 | Optimal shared memory access |
| ld.matrix/st.matrix generation | v0.4.0 | Hardware load/store instructions |
| FP8 matrices (e4m3, e5m2) | v0.2.0 | Not useful for LU (too low precision) |
| Mixed precision (TF32, BF16, FP16) | v0.2.0-0.3.0 | BF16 GEMM with FP32 accum for trailing |
| Register-based C storage | v0.3.0 | Keeps accumulator in registers |
| Experimental pipelining API | v0.5.0 | WGMMA/TMA/UTCMMA (sm_100 only, NOT sm_120) |
| Ozaki scheme (FP emulation) | v0.5.0 | "Significant perf upgrade for Blackwell B200" |

### What cuBLASDx Does NOT Have

- **No TRSM.** cuBLASDx is GEMM-only. Use cuSOLVERDx for TRSM.
- **No SYRK.** For Cholesky trailing update, use GEMM with A*A^T pattern.
- **sm_120 pipelining:** The v0.5.0 experimental pipelining (WGMMA, TMA) is for datacenter Blackwell (sm_100), not consumer (sm_120). Our sm_120 uses the mma.sync backend.

### Ozaki Scheme in cuBLASDx

cuBLASDx v0.5.0 added an Ozaki scheme example with "significant performance upgrade targeted for Blackwell B200 GPU." This is the device-side equivalent of cuBLAS's BF16x9 FP32 emulation -- it performs FP32-accurate GEMM using BF16 tensor cores within a kernel.

**This is potentially the most important feature for the monolithic kernel's trailing GEMM.** If the Ozaki/BF16x9 scheme can be used device-side via cuBLASDx, it would give 3-4x FP32 speedup within the monolithic kernel.

---

## libmathdx v0.2.1 Notable Addition

**TRSM support added.** libmathdx 0.2.1 added:
- Blackwell architecture support
- TRSM (triangular solve with matrix as right-hand-side)
- Integer GEMMs

This means device-side TRSM is now available for the monolithic kernel's triangular solve phase.

---

## Blocked LU Composition Pattern (Updated)

Based on the current MathDx capabilities:

```cpp
// Single kernel: cuSOLVERDx getrf + cuSOLVERDx trsm + cuBLASDx GEMM
__global__ void blocked_lu(float* A, int* ipiv, int N, int NB) {
    __shared__ char smem[max(SolverSmem, TrsmSmem, GemmSmem)];

    for (int k = 0; k < N/NB; k++) {
        // 1. Load panel to shared memory
        load_panel(A, smem, k, NB);

        // 2. Panel factorization with pivoting
        using LU = decltype(
            cusolverdx::Size<NB, NB>()
            + cusolverdx::Precision<float>()
            + cusolverdx::Function<cusolverdx::function::getrf_partial_pivot>()
            + cusolverdx::SM<1200>()
        );
        auto [As, ipivs] = cusolverdx::shared_memory::slice<float, int>(smem);
        LU().execute(As, ipivs, &info);
        __syncthreads();

        // 3. Apply pivots to rest of row (device-side LASWP)
        device_laswp(A, ipivs, k, NB, N);
        __syncthreads();

        // 4. TRSM: solve L * U_row = A_row
        using Trsm = decltype(
            cusolverdx::Size<NB, /* trailing cols */>()
            + cusolverdx::Precision<float>()
            + cusolverdx::Function<cusolverdx::function::trsm>()
            + cusolverdx::SM<1200>()
        );
        Trsm().execute(L_smem, U_smem);
        __syncthreads();

        // 5. Trailing GEMM update (tile-streamed through shared memory)
        // Use cuBLASDx GEMM or Ozaki scheme for BF16x9 accuracy
        for (each trailing tile) {
            load_tiles(A, smem, k);
            GEMM().execute(alpha, A_smem, B_smem, beta, C_smem);
            store_tile(A, smem, k);
            __syncthreads();
        }
    }
}
```

---

## Key Decision: cuSOLVERDx vs Custom

| Factor | cuSOLVERDx | Custom (MAGMA-style) |
|--------|-----------|---------------------|
| Development time | Days | Weeks |
| Panel factorization | Proven, tested | Must write argmax, swap, scale, rank-1 |
| TRSM | Provided | Must write or use invert-and-multiply |
| Trailing GEMM | cuBLASDx (good) or BF16x9 | Our proven BF16 MMA (0.97x cuBLAS) |
| Flexibility | Template params only | Full control |
| sm_120 bugs | Known issue with gesv_no_pivot | No dependency on NVIDIA code |
| Performance ceiling | Unknown (NVIDIA optimized) | Can match cuSOLVER if done right |

**Recommendation:** Start with cuSOLVERDx for v1 baseline (fastest development). If performance is insufficient or bugs are hit, switch to custom MAGMA-style panel + our BF16 MMA trailing GEMM for v2.

---

## Sources

- [cuSOLVERDx Release Notes](https://docs.nvidia.com/cuda/cusolverdx/release_notes.html)
- [cuBLASDx Release Notes](https://docs.nvidia.com/cuda/cublasdx/release_notes.html)
- [MathDx Package Documentation](https://docs.nvidia.com/cuda/mathdx/index.html)
- [cuSOLVERDx Blocked Cholesky Example](https://docs.nvidia.com/cuda/cusolverdx/examples/advanced_example.html)
- [cuSOLVERDx GETRF Documentation](https://docs.nvidia.com/cuda/cusolverdx/get_started/getrf.html)
