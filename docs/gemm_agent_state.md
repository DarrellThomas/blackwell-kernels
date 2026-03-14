# GEMM Kernel — Optimization State

**Last updated:** 2026-03-13
**Status:** FP8 GEMM beats cuBLAS on all configs. BF16 GEMM at 0.97x cuBLAS.

-----

## Hardware

- GPU: RTX 5090, sm_120 (consumer Blackwell, `mma.sync` ISA)
- Host: Threadripper PRO 7995WX, 512GB DDR5, Ubuntu 24.04
- CUDA 13 / PyTorch 2.10

-----

## Kernels

### BF16 GEMM (`bf16_gemm_sm120.cu`)

**Instruction:** `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`

**Tile config:** BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, 4 warps (128 threads), 6 blocks/SM

**Performance (vs cuBLAS):**
| Config | Duration | vs cuBLAS |
|--------|----------|-----------|
| 4096³ | 740 us | 0.97x |
| 4096×1024×4096 | 185 us | 1.19x |

**Status:** Mature. 51 experiments exhausted conventional optimization space. 0.97x is the ceiling for compiler-generated mma.sync code. Full inner-loop PTX is the only remaining path to 1.0x (major undertaking, ~500 lines).

### FP8 GEMM (`fp8_gemm_sm120.cu`)

**Instruction:** `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32` (2x throughput vs BF16)

**Architecture:** Loads BF16, converts to FP8 e4m3 in registers via `cvt.rn.satfinite.e4m3x2.f32`, executes FP8 MMA. FP32 accumulators, BF16 output.

**Dual-dispatch tile configs:**
- **Compute-bound (64×64):** BLOCK_M=64, BLOCK_N=64, BLOCK_K=64, 4 warps, `launch_bounds(128, 6)`
- **L2-bound (128×128):** BLOCK_M=128, BLOCK_N=128, BLOCK_K=32, 8 warps, `launch_bounds(256, 1)`

**Dispatch condition:** Use 128×128 when EITHER:
- Total input data (A+B) > 64MB — L2 capacity pressure
- Grid blocks with 64×64 tiles > 4096 — L2 contention from too many concurrent blocks

**Performance (vs cuBLAS):**
| Config | Path | Duration | vs cuBLAS |
|--------|------|----------|-----------|
| 1024³ | 64×64 | 16 us | 1.00x |
| 2048³ | 64×64 | 110 us | 1.04x |
| 4096³ | 64×64 | 657 us | **1.09x** |
| 4096×1024×4096 | 64×64 | 171 us | **1.29x** |
| 8192×2048×8192 | 128×128 | 1296 us | **1.08x** |
| 8192×4096×8192 | 128×128 | 2455 us | **1.02x** |

**Precision:** ~3.7% relative error vs torch.mm (FP8 e4m3). Acceptable for training — SGD absorbs the noise.

-----

## Key Architectural Decisions (load-bearing, do not remove)

Shared across both kernels:
- cp.async with double-buffer pipelining
- XOR swizzle for bank conflict elimination
- ldmatrix_x4 for A (with a1/a2 swap: r0, r2, r1, r3), ldmatrix_x2_trans for B
- Python-side padding to tile multiples — zero boundary checks in kernel
- Non-volatile MMA gives compiler full scheduling freedom
- ~32 KB smem sweet spot, leaving ~96 KB for L1 cache

FP8-specific:
- Vectorized `cvt.rn.satfinite.e4m3x2.f32` (empirically verified on sm_120; `bf16x2` variant does NOT work)
- CTA swizzle groups concurrent blocks to share B columns for L2 reuse
- Dual-dispatch based on data volume + grid pressure

-----

## Exhaustively Explored (BF16, all regressed or neutral)

51 experiments documented. Key dead ends:
- Tile sizes: 128×128 optimal for BF16, all others worse
- BLOCK_K=64: too much data per load
- 16 warps: race condition with 2-tile unroll
- 3-stage pipeline: occupancy/L1 loss exceeds latency benefit
- 2D warp tiling: bank conflicts from more A loads
- B fragment double-buffering: extra registers reduce occupancy
- Partial unroll: catastrophic (0.12x)
- wgmma: not available on sm_120

-----

## References

- [spatters.ca MMA matmul](../docs/reference_spatters_mma_matmul.md) — 93% peak on Ada, most applicable reference
- [math throttle guide](../docs/math_throttle_optimization.md) — diagnosis for compute-bound stalls
- [hard-won lessons](../.claude/04_HARD_WON_LESSONS.md) — empirical constraints, do not contradict
