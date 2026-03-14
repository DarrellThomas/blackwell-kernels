# Reference: salykova — SGEMM Beating cuBLAS on RTX 3090

**Source:** https://salykova.github.io/sgemm-gpu | https://github.com/salykova/sgemm.cu
**Author:** salykova
**Attribution:** Techniques from this work should be cited in code and commit messages.

---

## Result: Beat cuBLAS by 3-4% on RTX 3090 (Ampere, GA102)

Pure CUDA/PTX, no SASS patching. Scalar FFMA (not tensor cores — this is FP32 SGEMM).

## Why This Matters for Us

Proves cuBLAS CAN be beaten from CUDA C++ without assembly-level tools. The memory
hierarchy techniques transfer directly to our BF16 tensor core GEMM.

## Architecture

### Two Kernel Configs (adaptive dispatch)

| Config | Block Tile | K-strip | Threads | launch_bounds | When |
|--------|-----------|---------|---------|---------------|------|
| 128×128×8 | 128×128 | 8 | 256 (8 warps) | (256,2) | M,N < 2500 |
| 128×256×8 | 128×256 | 8 | 256 (8 warps) | (256,1) | M,N ≥ 2500 |

### Tiling Hierarchy
- Block: 128×128 (or 128×256) tile of C
- Warp: 32×64 sub-tile (8 warps per block)
- Thread: 8×8 outer product via scalar FMA

## Key Techniques

### 1. Register-Level Double-Buffering
```cuda
float frag_a[2][8];  // ping-pong
float frag_b[2][8];
```
Inner K-loop: load next fragments into `frag[next]`, compute with `frag[curr]`, toggle.
Hides shared memory load latency behind compute. **This is the main reason it beats cuBLAS.**

### 2. A Transposed During Store to Shared
Matrix A is transposed on the fly (128×8 global → 8×128 shared). Converts row-major
accesses that cause bank conflicts into conflict-free column-major accesses.

### 3. Shared Memory Padding
A: leading dimension = 132 (128 + 4 padding) to shift rows into distinct banks.
B: leading dimension = 128 (naturally 32 distinct banks, no padding needed).

### 4. XOR Buffer Toggling
Double-buffer toggle via XOR on shared memory addresses:
- A: `sts_a_addr ^= 8192` (0x2000)
- B: `sts_b_addr ^= 4096` (0x1000)

### 5. PTX Inline Assembly
All critical memory ops use PTX directly:
- `cp.async.ca.shared.global` — async global→shared (128×256 kernel)
- `ld.shared.v4.f32` — 128-bit vectorized shared loads
- Predicated loads with bitmasks — no branch divergence in hot loop

### 6. Bitmask Boundary Guards
Pre-computed bitmasks encode which per-thread loads are in-bounds.
Avoids per-element branch checks entirely.

### 7. Power-Aware Design
128×128 kernel draws ~12% more power than cuBLAS, causing throttling at large sizes.
128×256 with cp.async is more power-efficient and maintains advantage at scale.

## Techniques Applicable to Our BF16 GEMM

| Technique | Status in Our Kernel | Action |
|-----------|---------------------|--------|
| Register double-buffering of fragments | Not done | **Implement** |
| XOR shared memory buffer toggle | Already using | Keep |
| cp.async for global→shared | Already using | Keep |
| PTX inline for critical loads | Partial | Expand |
| Predicated boundary loads (bitmasks) | Not done | **Implement** |
| Adaptive kernel dispatch by size | Not done | Consider |
| A transposition during store | Not applicable (BF16 MMA layout differs) | Skip |

## Key Insight for Beating cuBLAS

The gap between cuBLAS and "good CUDA" is mostly about **latency hiding in the inner loop**.
Register-level double-buffering of MMA fragments — loading the next iteration's data while
computing the current one — is the technique that closes the gap. cuBLAS apparently doesn't
fully optimize this at certain tile sizes, leaving room for a hand-tuned kernel.
