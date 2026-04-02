# CRITICAL: TF32 MMA B Fragment Defect on sm_120 — Cross-Pollination from Cholesky

**Source:** Empirically verified by Cholesky worker (experiments 23-24), documented in `/data/src/bwk/numerical/docs/cholesky_agent_state.md`
**Relevant to:** LU worker
**Worker's current problem:** Building v1 blocked LU, will eventually need monolithic kernel with device-side GEMM (v3). Choosing the right MMA instruction is critical.

## What This Is

The Cholesky worker spent 2 experiments discovering that TF32 MMA (`mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32`) has a **B operand broadcasting defect on sm_120** that makes it **unusable for general GEMM**. You MUST know this before attempting a monolithic LU kernel.

## The Defect

Each B register (b0, b1) is broadcast to TWO matrix positions:
```
b0 -> B[k_even, n_even] AND B[k_even+1, n_even+1]  (diagonal broadcast)
b0 == b1 (both registers map to identical positions)
```

This constrains `B[k, n] = B[k+1, n+1]` along diagonals. For arbitrary B matrices, this produces incorrect results. Verified with 25+ test cases on RTX 5090.

## Impact on LU

LU factorization needs general GEMM for the trailing update: `A[i:,j:] -= L[i:,k:k+nb] * U[k:k+nb,j:]`. Both L and U are general dense matrices (not banded, not diagonal). TF32 MMA will produce **silently wrong results** for this update.

### What cuSOLVER Does

The Cholesky worker profiled cuSOLVER's monolithic kernel and found:
- Uses `getrf_wo_pivot_params_<float, 0, 256, 1, 64, 64, 68>`
- Grid 1x1x1, Block 256, 202 registers, 52KB smem
- Achieves ~15 TFLOPS on 1 SM
- The name indicates no-pivot, but does use tensor cores for SYRK/GEMM internally
- It likely uses **BF16 MMA** (m16n8k16) with FP32->BF16 conversion, OR a proprietary MMA variant

## Viable Paths for Device-Side GEMM in Monolithic LU

### Path 1: BF16 MMA (m16n8k16) — RECOMMENDED
```
mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
```
- Proven working for general GEMM on sm_120 (our GEMM kernel at 0.97x cuBLAS)
- FP32 -> BF16 conversion + FP32 accumulators preserves ~1e-3 precision
- Well-understood fragment layout (ldmatrix_x4_mma with a1/a2 swap)
- Known tile sizes and occupancy sweet spots (64x64, 4 warps, 6 blocks/SM)

**Precision note for LU:** The trailing update is GEMM (not SYRK), so asymmetric errors don't cancel. But with FP32 accumulators and BF16 operands, the relative error is ~1e-3 per multiply — acceptable for an iterative refinement approach where you compute LU in mixed precision and refine with FP32 residuals.

### Path 2: FP8 MMA (m16n8k32) — AGGRESSIVE
- 2x throughput vs BF16 but ~3.7% relative error per multiply
- Only viable with iterative refinement or if FP8 accuracy is acceptable
- More complex: requires conversion + fragment layout management

### Path 3: FP32 scalar — BASELINE
- No tensor cores, pure FP32 scalar arithmetic
- Simple but leaves massive compute on the table
- Only useful for the panel factorization base case (column-by-column solve)

### Path 4: TF32 Decomposition — NOT RECOMMENDED
- Decompose each GEMM column into pairs that satisfy the diagonal constraint
- 2x MMA calls (halving throughput), plus complex setup
- Not worth it when BF16 MMA works correctly

## TF32 Output Layout (Also Different)

For reference, the TF32 MMA output layout also differs from BF16:
```
BF16: d0,d1 are adjacent columns in same row
TF32: d0→(gid, tid*2), d1→(gid+8, tid*2), d2→(gid, tid*2+1), d3→(gid+8, tid*2+1)
      (adjacent rows, not adjacent columns — 8-row stride)
```

This means even if you work around the B broadcasting, the output store logic must be rewritten. Another reason to use BF16 MMA.

## Recommendation

**Use BF16 MMA (m16n8k16) for all device-side GEMM in the monolithic LU kernel.** This is the proven, correct path. The ~1e-3 precision loss per operation is manageable with FP32 panel factorization and optional iterative refinement.

Do NOT attempt TF32 MMA for general GEMM on sm_120. The B fragment broadcasting makes it silently incorrect. This was verified with 25+ empirical tests.
