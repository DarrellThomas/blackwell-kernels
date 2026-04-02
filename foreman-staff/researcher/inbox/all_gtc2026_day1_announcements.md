# GTC 2026 Day 1 (March 16) -- Keynote Status and Announcement Tracker

**Sources:**
- [GTC 2026 Keynote Page](https://www.nvidia.com/gtc/keynote/)
- [NVIDIA Newsroom](https://nvidianews.nvidia.com/news/latest)
- [NVIDIA Developer Blog](https://developer.nvidia.com/blog/)
- [NVIDIA GTC 2026 Blog](https://blogs.nvidia.com/blog/gtc-2026-news/)
- [CUTLASS Releases](https://github.com/NVIDIA/cutlass/releases)

**Relevant to:** all workers
**Date:** 2026-03-16

---

## KEYNOTE STATUS: NOT YET HAPPENED

Jensen Huang's GTC 2026 keynote is scheduled for **today, March 16, 2026** at
**11:00 AM - 1:00 PM Pacific Time** from the SAP Center in San Jose.

As of this research pass, the keynote has NOT yet occurred. No keynote-day press
releases have been published on nvidianews.nvidia.com (latest press releases are
from March 12). This brief documents what is known as of the morning of Day 1.

**Action required:** Re-run researcher after the keynote (after 1 PM PT / 4 PM ET
on March 16) to capture actual announcements.

---

## WHAT WE KNOW GOING INTO THE KEYNOTE

### Pre-Keynote Confirmed Announcements (recap)

These dropped before the keynote and are documented in detail in previous briefs
(`all_updates_march_14_16_2026.md`, `all_gtc2026_keynote_day_march16.md`):

1. **CUDA 13.2** (March 5) -- PTX ISA 9.2, CUDA Tile on Ampere/Ada, cuBLAS 13.2
   with MXFP8 Grouped GEMM, cuSOLVER FP64 emulation, Nsight Compute 2026.1
2. **CUTLASS 4.4.2** (March 13) -- sm_120f compilation target, NVFP4/MX profiler
3. **NVFP4 blog posts** (March 12-13) -- Blackwell Ultra at 15 PFLOPS, Rubin at 35-50 PFLOPS
4. **DGX Spark** -- available for purchase at GTC
5. **Nemotron 3 Super** (March 11) -- 120B parameter, 12B active, hybrid Mamba-Transformer MoE

### What the Keynote Is Expected to Cover

Based on pre-GTC reporting (unchanged from previous brief):

| Expected Topic | Relevance to sm_120 |
|---------------|---------------------|
| Vera Rubin platform details (H2 2026, 288GB HBM4, 22 TB/s) | None -- datacenter only |
| Feynman architecture preview (2028, TSMC A16) | None -- future datacenter |
| Blackwell Ultra (B300) refresh | None -- sm_100 variant |
| N1/N1X ARM laptop CPU | None |
| CUDA ecosystem/software stack updates | **POSSIBLY RELEVANT** |
| OpenClaw / agentic AI frameworks | None |
| DLSS 4.5 (confirmed on GeForce site) | None -- graphics pipeline |

**The only potentially actionable item for our kernel work is CUDA/software stack
updates.** Hardware announcements will be datacenter-focused.

---

## NEW FINDINGS SINCE LAST BRIEF (March 15)

### 1. CUTLASS 4.4.0 Details: GB300 Blockscaled GEMM + CuTe DSL Enhancements

From the GitHub releases page (more detail than previously captured):

**CuTe DSL in 4.4.0:**
- CUDA Toolkit 13.1 support
- `cute.experimental` layer for higher-level composable APIs
- Ahead-of-Time (AoT) compilation capability
- JAX framework integration
- `cutlass.version` and `cutlass.CUDA_VERSION` query APIs
- Device-side TMA descriptor management

**C++ Framework in 4.4.0:**
- **SM100 State Space Decomposition (SSD) kernel** (example 112) -- datacenter only
- **Ada FP8xFP8 blockwise dequantization GEMM** (example 94) -- sm_89, same ISA as sm_120
- Hopper e2m1 to FP32 optimized conversion with TF32 tensor core support
- Cluster swizzle improvements for Grouped GEMMs

**Relevance:** The Ada FP8xFP8 example (94) uses the same mma.sync ISA family as
sm_120. This is a reference implementation for blockwise dequantization patterns
that could inform our FP8 GEMM work.

### 2. PTX ISA 9.2 New Instructions (from CUDA 13.2 docs)

Additions beyond what was in previous briefs:

- **Packed integer arithmetic:** `add`, `sub`, `min`, `max`, `neg` for `.u8x4`
  and `.s8x4` types -- useful for 8-bit quantized integer workloads
- **Saturating addition:** `add.sat` for `u16x2`, `s16x2`, `u32`
- **128-bit async store:** `st.async` now supports `.b128` type
- **OOB-safe bulk copy:** `cp.async.bulk` with `.ignore_oob` qualifier
- **FP8-to-BF16x2 conversion:** `cvt` destination extended to `.bf16x2` from
  `.e4m3x2`, `.e5m2x2`, `.e3m2x2`, `.e2m3x2`, `.e2m1x2` sources

**Most relevant to workers:**
- The `.bf16x2` conversion from FP8 sources is new and could simplify FP8-to-BF16
  conversion paths in attention and GEMM kernels (currently using `cvt.rn.satfinite.e4m3x2.f32`)
- The `st.async .b128` extends async store capability
- `cp.async.bulk .ignore_oob` could simplify boundary handling in tiled kernels

### 3. CUDA Tile Flash Attention: sm_120 Autotuning Config Confirmed

The NVIDIA Flash Attention cuTile blog post confirms sm_120 support with specific
autotuning parameters:

```python
if gpu_capability in [(12, 0), (12, 1)]:
    # RTX 50 series (sm120, sm121)
    yield SimpleNamespace(TILE_M=64, TILE_N=64, num_ctas=1, occupancy=2)
```

This tells us NVIDIA's own autotuner for sm_120 attention selects:
- 64x64 tiles (matching our attention worker's tile choice)
- 1 CTA per cluster (no multi-CTA cooperation)
- Occupancy target of 2 blocks/SM (our worker uses 3 blocks/SM)

**Note:** No TFLOPS numbers on sm_120 were published -- all benchmarks are on B200.
The cuTile framework is Python-only and generates code through MLIR, so it's not
directly usable for our hand-written PTX kernels, but the autotuning configs are
informative about what NVIDIA considers optimal on sm_120.

### 4. CUDA Tile IR: MLIR-Based, sm_100 Primary Target

The CUDA Tile IR GitHub repo reveals:
- MLIR-based intermediate representation and compiler for tile-based GPU kernels
- Documentation examples target sm_100 primarily
- Python bindings available, C++ integration via FetchContent
- JIT or AOT compilation to cubin

**Not actionable for our work.** This is a high-level programming model that would
replace our hand-tuned PTX. Interesting as a future direction but not competitive
with hand-tuned code for peak performance today.

### 5. Vera Rubin Partnership Confirmation

Thinking Machines Lab announced a "multiyear partnership" for "gigawatt-scale
NVIDIA Vera Rubin systems" deployment. This confirms Vera Rubin is on track for
production deployment (previously expected H2 2026).

**Zero relevance to sm_120 kernel work.** Included for completeness.

### 6. DLSS 4.5 Confirmed on GeForce Site

The GeForce product page references "DLSS 4.5" with path tracing capabilities.
This is a graphics pipeline update, not relevant to compute kernels.

---

## WHAT TO WATCH FOR AFTER THE KEYNOTE

Priority-ordered items to check once the keynote concludes:

1. **Any CUDA 13.3 or CUDA 14 announcement** -- new toolkit releases are the
   highest-value finding for our kernel work
2. **Any sm_120-specific content** -- new instructions, features, or optimizations
   for consumer Blackwell
3. **cuBLAS/cuSOLVER performance updates** -- baseline shifts for our workers
4. **CUTLASS new releases or roadmap** -- new examples, sm_120 improvements
5. **Nsight Compute updates** -- profiling capabilities
6. **Session recordings from S81859 ("CUDA: New Features and Beyond")** -- this is
   the single most important session for us
7. **Session recordings from S81772 ("Don't Leave Tensors on the Table")** -- tensor
   core optimization techniques

---

## RESEARCH STATUS

| Search | Status | Finding |
|--------|--------|---------|
| GTC 2026 keynote | Keynote at 11am PT today, not yet happened | No announcements yet |
| NVIDIA press releases Mar 16 | None published yet | Check after 1pm PT |
| NVIDIA developer blog Mar 16 | No new posts yet | Check after keynote |
| CUDA 14 / CUDA 13.3 | No evidence found | May be announced in keynote |
| Feynman architecture | Pre-GTC speculation only | Expected keynote topic |
| Vera Rubin details | Partnership confirmed, specs from previous reporting | H2 2026 |
| Blackwell Ultra | Previous reporting unchanged | sm_100 datacenter only |
| cuBLAS / cuSOLVER updates | CUDA 13.2 details captured | Baseline shift if we upgrade |
| CUTLASS updates | 4.4.2 (Mar 13) is latest | sm_120f target enabled |
| New MMA instructions | None found in PTX ISA 9.2 | mma.sync unchanged for sm_120 |
| DLSS 4.5 | Confirmed on GeForce page | Graphics only, not relevant |
| OpenClaw | Featured prominently in GTC preview | AI agents, not relevant |

---

## BOTTOM LINE

**Nothing has changed for our kernel work since the March 15 brief.** The keynote
hasn't happened yet. The most impactful pre-GTC finding remains the **CUDA 13.2
upgrade path** (Nsight Compute register dependency visualization, 256b load/store
in PTX ISA 9.2, cuBLAS/cuSOLVER baseline shifts, bf16x2 conversion from FP8
sources). These are documented in `all_updates_march_14_16_2026.md`.

**This brief should be updated after the keynote (after 1 PM PT today).** The
keynote is the main event and could reveal new CUDA features, toolkit releases,
or sm_120-specific content that would be immediately actionable.
