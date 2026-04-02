# TRSM: Julia Recursive Implementation — Portable GPU Performance (April 2025)

**Sources:**
- [Toward Portable GPU Performance: Julia Recursive Implementation of TRMM and TRSM (Carrica et al., April 2025)](https://arxiv.org/abs/2504.13821)
- [Quantumzeitgeist summary: High-Performance Recursive TRMM/TRSM](https://quantumzeitgeist.com/high-performance-recursive-trmm-trsm-implementation-in-julia-for-gpus-across-architectures/)
- [Charara et al.: A framework for dense triangular matrix kernels on GPUs (2017)](https://onlinelibrary.wiley.com/doi/full/10.1002/cpe.4187)

**Relevant to:** linalg worker
**Worker's current problem:** TRSM at 0.82x cuBLAS F32. Existing briefs cover the recursive algorithm and MAGMA-style diagonal inversion. This brief adds concrete implementation details from a 2025 paper that matches cuBLAS on NVIDIA GPUs.
**Supplements:** `linalg_recursive_trsm_to_gemm.md`, `linalg_trsm_gpu_techniques.md`, `linalg_trsm_gpu_optimization.md`

---

## What's New Here (vs. existing briefs)

The existing briefs describe the recursive algorithm and the MAGMA diagonal-inversion approach. This 2025 paper provides three concrete details missing from our coverage:

1. **Optimal base-case tile size** for the small TRSM at the recursion leaf
2. **Left-looking vs right-looking** variant choice and why it matters on GPU
3. **Warp shuffle for the base case** instead of shared memory

---

## Key Technique: Base-Case Implementation Details

### Base-case tile size should match warp geometry

The paper uses a base-case TRSM tile congruent with the warp size (32 threads). For their NVIDIA implementation:
- Base-case TRSM operates on tiles <= 32x32
- Each thread handles one column of the tile
- B columns are cached in registers (not shared memory)
- Threads share values via `__shfl_sync` (warp shuffle), which is lower latency than shared memory access

This is critical: the base-case TRSM is the serial bottleneck. Making it warp-native (shuffle-based, register-resident) minimizes its latency.

### Left-looking variant for GPU

The paper uses a **left-looking** variant for the base-case kernel:
- Left-looking trades fewer writes for an equivalent number of extra reads
- On GPU, reads are faster than writes (especially with texture caching / L2)
- Right-looking writes intermediate results that are immediately consumed -- this causes read-after-write hazards on GPU

For the base case, left-looking means:
```
for col j = 0 to NB-1:
    // Read column j of L (broadcast via shuffle)
    // Read column j of X (in registers)
    X[j] = X[j] / L[j,j]
    for col k = j+1 to NB-1:
        X[k] -= L[k,j] * X[j]  // update uses shuffle-broadcast of X[j]
```

Each iteration broadcasts L[k,j] and X[j] via `__shfl_sync`, avoiding shared memory entirely.

### Recursion structure

```
TRSM(L, B, n):
    if n <= NB_BASE (e.g., 32):
        return warp_trsm(L, B)  // shuffle-based base case

    n_half = n / 2
    TRSM(L[0:n_half, 0:n_half], B[0:n_half, :], n_half)     // top-left
    GEMM(B[n_half:, :] -= L[n_half:, 0:n_half] * B[0:n_half, :])  // update
    TRSM(L[n_half:, n_half:], B[n_half:, :], n_half)         // bottom-right
```

For N=1024 with NB_BASE=32:
- Recursion depth: log2(1024/32) = 5 levels
- 31 base-case TRSM calls + 31 GEMM calls
- The GEMM calls dominate for large N and can use your existing 0.97x BF16 GEMM

---

## Performance Results (from the paper)

On NVIDIA A100:
- **Square TRSM (N=N, NRHS=N):** matches cuBLAS for N >= 512, slightly faster for some rectangular cases
- **Rectangular TRSM (NRHS >> N):** surpasses cuBLAS because the GEMM fraction is higher
- The implementation is ~300 lines of Julia (kernel + recursion logic)

On sm_120 (our target):
- The paper doesn't test sm_120, but the approach is architecture-agnostic
- Our advantage: we have a 0.97x BF16 GEMM and 1.34x FP8 GEMM -- the paper uses vendor GEMM
- If the recursive GEMM calls use our custom GEMM, the GEMM-dominated regime should be faster than cuBLAS TRSM

---

## What This Means for the Worker

The worker's existing briefs already describe the recursive algorithm. This paper confirms:

1. **Base case = 32x32 warp-native kernel using shuffle** (not shared memory). This is a specific implementation decision the worker should adopt.
2. **Left-looking variant** for the base case (fewer writes, better for GPU memory model).
3. **The approach genuinely matches cuBLAS** on modern NVIDIA hardware -- it's not just theoretical.
4. **Our custom GEMM gives us an edge** that the paper authors didn't have (they use vendor GEMM).

---

## Caveats

- The paper's Julia implementation uses vendor GEMM via library calls. Our CUDA implementation would call our own GEMM kernel, which adds some integration complexity (kernel launch from kernel is possible via dynamic parallelism, or use host-side recursion with kernel launches).
- Host-side recursion with separate kernel launches adds launch overhead at each recursion level. For N=1024 with NB=32, that's ~62 kernel launches. Consider: (a) using larger NB (64 or 128) to reduce depth, or (b) persistent kernel that processes the recursion tree without returning to host.
- BF16 TRSM accuracy: triangular solve accumulates errors along the diagonal. BF16 has only 8 mantissa bits. The base-case 32x32 solve may need FP32 accumulation even if inputs are BF16. The recursive GEMM updates can stay BF16 since GEMM error is bounded per call.
