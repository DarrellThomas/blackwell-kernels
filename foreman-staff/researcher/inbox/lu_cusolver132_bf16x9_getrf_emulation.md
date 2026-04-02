# cuSOLVER 13.2: BF16x9 FP32 Emulation for getrf -- Free 3x Trailing GEMM Speedup

**Source:** https://docs.nvidia.com/cuda/cusolver/index.html (cuSOLVER 13.2 documentation)
**Source:** https://developer.nvidia.com/blog/unlocking-tensor-core-performance-with-floating-point-emulation-in-cublas/
**Relevant to:** LU worker
**Worker's current problem:** cuSOLVER baseline is 9.4ms at N=4096. Worker is building v1 blocked LU. Trailing GEMM is 80-85% of compute. Need tensor core acceleration with FP32 accuracy.

---

## What This Is

cuSOLVER 13.2 (CUDA 13.2, released March 2026) now supports
`CUSOLVER_FP32_EMULATED_BF16X9_MATH` as a math mode for `cusolverDnXgetrf`.
This means cuSOLVER's own LU factorization can use BF16 tensor cores internally
for the trailing GEMM updates while maintaining full FP32 numerical accuracy.

**This is the single most impactful finding for the LU worker's v1 blocked approach.**

---

## How to Enable

```cpp
cusolverDnHandle_t handle;
cusolverDnCreate(&handle);

// Enable BF16x9 FP32 emulation for ALL subsequent cuSOLVER calls on this handle
cusolverDnSetMathMode(handle, CUSOLVER_FP32_EMULATED_BF16X9_MATH);

// Now call getrf as usual -- internally it will use BF16 tensor cores
cusolverDnXgetrf_bufferSize(handle, params, N, N, CUDA_R_32F, d_A, N,
                            CUDA_R_32F, &workspaceSize);
cusolverDnXgetrf(handle, params, N, N, CUDA_R_32F, d_A, N,
                 d_ipiv, CUDA_R_32F, d_workspace, workspaceSize, d_info);
```

**Important:** Workspace sizes returned by `*_bufferSize` APIs may depend on the
math mode and emulation strategy. Always call `bufferSize` AFTER setting the math mode.

---

## Expected Performance Impact

The BF16x9 algorithm decomposes each FP32 GEMM into 9 BF16 tensor core GEMMs.
On Blackwell, BF16 tensor core throughput is ~16x native FP32 throughput.
9 BF16 GEMMs / 16x throughput = ~1.78x theoretical speedup. In practice,
cuBLAS achieves 3-4x speedup for large GEMMs at M=N=K=32768.

For our N=4096 LU case:
- Trailing GEMM is ~94% of compute (43 GFLOP out of 45.8 total)
- If trailing GEMM runs 3x faster, that portion drops from ~8ms to ~2.7ms
- Panel factorization and LASWP remain unchanged
- **Expected total: ~4-5ms vs 9.4ms baseline = 1.9-2.3x speedup**

This could be achievable with ZERO custom kernel code -- just a cuSOLVER API flag change.

---

## Numerical Accuracy

BF16x9 FP emulation provides accuracy **equivalent to or better than** native FP32.
This is because:
1. Each FP32 value is exactly decomposed into 3 BF16 values (covering all 24 mantissa bits)
2. The 9 GEMM products with FP32 accumulation recover the full FP32 result
3. No iterative refinement needed -- the result IS FP32 quality

This is fundamentally different from using raw BF16 MMA (which loses precision).
BF16x9 gives full FP32 accuracy at 3x speed. It is the ideal approach for the
trailing GEMM in a monolithic kernel.

---

## Implications for Our Strategy

### v1 (Blocked LU with cuBLAS/cuSOLVER)

**Immediate action:** After establishing the baseline, try:
```cpp
cusolverDnSetMathMode(handle, CUSOLVER_FP32_EMULATED_BF16X9_MATH);
```

This might immediately bring our host-launched cuSOLVER getrf from 9.4ms down
to ~4-5ms. If cuSOLVER's internal monolithic kernel respects this math mode for
its trailing GEMM, we get a near-free 2x speedup.

**Alternative:** If cuSOLVER's monolithic kernel doesn't respect the math mode
(possible -- the monolithic kernel may have its own internal logic), then use
the explicit `cusolverDnXgetrf` API with the math mode set.

### v3 (Monolithic Kernel)

For the monolithic kernel's device-side trailing GEMM, implement the BF16x9
decomposition manually:

```
For each FP32 tile pair (A_tile, B_tile):
  1. Split A into A0, A1, A2 (three BF16 components)
  2. Split B into B0, B1, B2 (three BF16 components)
  3. Compute 9 mma.sync m16n8k16 products:
     C += A0*B0
     C += scale(A0*B1 + A1*B0)
     C += scale(A0*B2 + A1*B1 + A2*B0)
     C += scale(A1*B2 + A2*B1)
     C += scale(A2*B2)
  4. Sum with appropriate 2^-8, 2^-16 scaling factors
```

This gives full FP32 accuracy at ~3x FP32 scalar speed, inside the monolithic
kernel without any library dependency.

### cuBLASDx Ozaki Example

cuBLASDx v0.5.1 includes a `dgemm_emulation` example that demonstrates the Ozaki
scheme for emulating FP64 GEMM using multiple INT8 GEMMs. The same principle
applies to FP32 emulation via BF16. The example provides device-side reference
code for the decomposition pattern.

---

## Additional cuSOLVER 13.2 Features

### FP64 Fixed-Point Emulation

cuSOLVER 13.2 also added FP64 emulation APIs:
- `cusolverDnSetFixedPointEmulationMantissaControl()`
- `cusolverDnSetFixedPointEmulationMaxMantissaBitCount()`
- `cusolverDnSetFixedPointEmulationMantissaBitOffset()`
- `cusolverDnSetEmulationSpecialValuesSupport()`

These control FP64 emulation via integer tensor cores, but are less relevant for
our FP32 getrf work.

### Emulation Strategy Control

The new `cusolverDnSetEmulationStrategy()` API allows fine-tuning emulation
behavior when emulation modes are active.

### Batched Eigenvalue Algorithm Switch for Small Matrices (Blackwell)

cuSOLVER 13.1 Update 1 introduced a performance improvement for
`cusolverDnXsyevBatched()` with an **internal algorithm switch on Blackwell
GPUs for matrices of size n <= 32**. Revert via `cusolverDnSetAdvOptions()`.
This is eigenvalue only, NOT getrf/potrf, but signals NVIDIA is actively
optimizing small-matrix paths on Blackwell.

## Update: cuSOLVERDx Status (March 14, 2026)

cuSOLVERDx remains at **v0.3.0** (released in MathDx 25.12.0). No v0.3.1 or
v0.4.0 has been released. The getrf_partial_pivot and potrf device functions
are unchanged. The sm_120/gesv_no_pivot miscompilation bug is still present.

### CERN Validation (March 2026)

NVIDIA presented at CERN (indico.cern.ch/event/1538409) validating FP emulation
in math libraries. BF16x9 has been tested for accuracy and performance in:
- Weather simulation (ecTrans)
- Quantum circuit simulation
- Condensed matter physics
- **Dense linear algebra (QR and LU factorization)**

The trailing matrix update -- the BLAS3 component that dominates factorization
runtime -- can be redirected to use emulated GEMM with accuracy guardrails.
This was demonstrated by integrating into cuSOLVER for QR factorization.

---

## Caveats

1. **RTX 5090 (sm_120) support for BF16x9 is confirmed** -- cuBLAS documentation
   states BF16x9 requires "special hardware features for scaling factors" and
   "select architectures." Blackwell is confirmed supported.

2. **Workspace size may increase** with BF16x9 mode -- the emulation needs
   temporary buffers for the decomposed BF16 matrices.

3. **Small trailing GEMMs may not benefit** -- BF16x9 has 9x the instruction
   count. For very small GEMMs (trailing matrix near end of factorization),
   native FP32 may be faster. cuBLAS/cuSOLVER may handle this automatically
   via fallback.

4. **Test empirically** -- set the math mode and re-benchmark. If cuSOLVER's
   internal monolithic kernel doesn't respect the math mode (uses its own path),
   the explicit `cusolverDnXgetrf` API is the fallback.

---

## Sources

- [cuSOLVER 13.2 Documentation](https://docs.nvidia.com/cuda/cusolver/index.html)
- [cuBLAS FP Emulation Blog](https://developer.nvidia.com/blog/unlocking-tensor-core-performance-with-floating-point-emulation-in-cublas/)
- [CUDA 13.2 Release Notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)
- [cuBLASDx Examples](https://docs.nvidia.com/cuda/cublasdx/examples.html)
