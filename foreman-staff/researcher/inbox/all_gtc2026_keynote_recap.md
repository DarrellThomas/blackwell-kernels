# GTC 2026 Keynote -- Comprehensive Pre-Keynote Status Report

**Sources:**
- [NVIDIA GTC 2026 Live Updates Blog](https://blogs.nvidia.com/blog/gtc-2026-news/)
- [GTC 2026 Keynote Page](https://www.nvidia.com/gtc/keynote/)
- [NVIDIA Newsroom Latest](https://nvidianews.nvidia.com/news/latest)
- [Blackwell Ultra Technical Blog](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/)
- [Blackwell Ultra Newsroom](https://nvidianews.nvidia.com/news/nvidia-blackwell-ultra-ai-factory-platform-paves-way-for-age-of-ai-reasoning)
- [NVIDIA Dynamo Newsroom](https://nvidianews.nvidia.com/news/nvidia-dynamo-open-source-library-accelerates-and-scales-ai-reasoning-models)
- [CUDA 13.2 Release Notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)
- [CUDA 13.1 Blog Post](https://developer.nvidia.com/blog/nvidia-cuda-13-1-powers-next-gen-gpu-programming-with-nvidia-cuda-tile-and-performance-gains/)
- [PTX ISA 9.2 Documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)
- [cuBLAS Grouped GEMM Blog](https://developer.nvidia.com/blog/introducing-grouped-gemm-apis-in-cublas-and-more-performance-updates/)
- [Feynman Architecture Preview](https://pbxscience.com/nvidias-feynman-architecture-what-we-actually-know-ahead-of-gtc-2026/)
- [Feynman on TSMC A16](https://www.trendforce.com/news/2026/03/13/news-nvidia-may-offer-first-look-at-feynman-at-gtc-2026-tsmc-a16-and-taiwan-supply-chain-in-focus/)
- [Vera Rubin at CES 2026](https://www.tomshardware.com/tech-industry/artificial-intelligence/nvidia-ceo-confirms-vera-rubin-nvl72-is-now-in-production-jensen-huang-uses-ces-keynote-to-announce-the-milestone)
- [RTX 6090 / Rubin Consumer Timeline](https://www.tomshardware.com/pc-components/gpus/nvidias-next-gen-rtx-60-series-might-not-debut-until-the-second-half-of-2027-says-leaker-rumor-claims-rubin-architecture-will-power-future-consumer-gpus)
- [Analytics Insight GTC 2026](https://www.analyticsinsight.net/news/nvidia-gtc-2026-keynote-major-announcements-on-ai-gaming-cpus-and-computing)
- [GTC 2026 CUDA Sessions](https://www.nvidia.com/gtc/sessions/cuda-libraries-and-dev-tools/)
- [CUTLASS Releases](https://github.com/NVIDIA/cutlass/releases)

**Relevant to:** all workers
**Date:** 2026-03-15

---

## KEYNOTE STATUS: HAS NOT HAPPENED YET

Jensen Huang's GTC 2026 keynote is scheduled for **tomorrow, March 16, 2026** at
**11:00 AM - 1:00 PM Pacific Time** from the SAP Center in San Jose. No keynote
press releases have been published -- the latest NVIDIA newsroom entry is from
March 12.

**This brief consolidates everything known as of March 15 evening.** It should be
refreshed after the keynote (after 1 PM PT on March 16).

---

## SECTION 1: CONFIRMED PRE-GTC ANNOUNCEMENTS (Already Shipped)

These are real, shipped products/updates that dropped before the keynote.

### 1.1 CUDA 13.2 (Released March 5, 2026)

The most important pre-GTC release for our kernel work.

**PTX ISA 9.2 -- New Instructions:**
| Instruction | Details | sm_120 Relevant? |
|------------|---------|-----------------|
| `cvt .bf16x2` from FP8 sources | Convert `.e4m3x2`, `.e5m2x2`, `.e3m2x2`, `.e2m3x2`, `.e2m1x2` to `.bf16x2` | **YES** -- new FP8-to-BF16 conversion path |
| `add/sub/min/max/neg .u8x4/.s8x4` | Packed 8-bit integer SIMD | Marginal -- INT8 quantization |
| `add.sat .u16x2/.s16x2/.u32` | Saturating addition | Marginal |
| `st.async .b128` | 128-bit async store | Possibly useful |
| `cp.async.bulk .ignore_oob` | Out-of-bounds safe bulk copy | **YES** -- simpler boundary handling |
| `sm_120f` family-specific target | Compile for sm_120 family | **YES** -- forward compatibility |

**cuBLAS 13.3 (bundled with CUDA 13.2):**
- Extended Grouped GEMM API now supports MXFP8 inputs on Compute Capability 10.x and 11.0
- FP64 fixed-point emulation added to SYRK/HERK routines
- Up to 20% speedup on RTX PRO 6000 across multiple precisions
- Up to 3x improvement for selected MXFP8/NVFP4 shapes on DGX Spark
- Improved FP32 GEMM heuristics for M,N >> K shapes on Blackwell
- **NOTE:** Grouped GEMM MXFP8 specifies "Compute Capability 10.x and 11.0" -- unclear
  if sm_120 (12.0) is included. Need to test.

**cuSOLVER 12.1 (bundled with CUDA 13.2):**
- FP64 fixed-point emulation (BF16x9 Ozaki scheme) for QR/LU/Cholesky
- New API: `cusolverDnSetFixedPointEmulationMantissaControl()`
- New `cusolverDnXsygvd` API for larger problem sizes
- Performance improvements on Blackwell for matrices with n <= 32

**cuFFT:** Improved power-of-2/3/5/7 transforms on Blackwell
**cuSPARSE:** SpMVOp performance improved on B200
**libdevice:** `expm1f()` up to 20% faster, `erff()` 5-10% faster
**Compiler:** Unified CUDA Toolkit for Tegra and desktop GPUs

**Deprecations (watch for CUDA 14):**
- Legacy vector types (`double4`, `long4`, etc.) deprecated, removal planned in CUDA 14.0
- Multi-device cooperative launch APIs removed
- Nsight Eclipse Edition support deprecated

### 1.2 CUTLASS 4.4.2 (Released March 13, 2026)

- `sm_120f` compilation target added (family-specific)
- NVFP4/MX profiler support
- Ada FP8xFP8 blockwise dequantization GEMM (example 94) -- uses same mma.sync ISA as sm_120
- CuTe DSL: `cute.experimental` layer, AOT compilation, JAX integration
- SM100 SSD kernel (example 112) -- datacenter only

### 1.3 CUDA 13.1 -- CUDA Tile (Released January 2026)

- **CUDA Tile** is a new tile-based programming model with Virtual ISA
- cuTile Python DSL abstracts tensor core usage
- **NOT relevant for hand-written PTX kernels** -- this is a higher-level abstraction
- Supports sm_120 (compute_12x) but generates code via MLIR compiler
- C++ support planned for future releases (Python-only for now)
- NVIDIA's own Flash Attention cuTile autotuner for sm_120 selects: 64x64 tiles, 1 CTA, occupancy=2

### 1.4 NVFP4 Blog Posts (March 12-13, 2026)

- Blackwell Ultra: 15 PFLOPS dense NVFP4 (3x FP8 on same GPU)
- Rubin: 35-50 PFLOPS dense NVFP4
- NVFP4 training achieves FP16-equivalent accuracy at 1.9x speed over FP8
- **NOT available on sm_120** -- NVFP4 MMA requires sm_100/sm_101 (tcgen05)

### 1.5 NVIDIA Dynamo (Pre-GTC Announcement)

- Open-source inference framework, successor to Triton Inference Server
- Disaggregated serving: separates prefill/decode on different GPUs
- 2x performance on Hopper for Llama; 30x per-GPU for DeepSeek-R1 on GB200 NVL72
- Compatible with PyTorch, SGLang, TensorRT-LLM, vLLM
- GitHub: ai-dynamo/dynamo
- **Not relevant to kernel optimization directly** but sets context for inference workloads

### 1.6 NemoClaw (Pre-GTC Announcement, March 10)

- Open-source enterprise AI agent platform (Apache 2.0)
- Built on OpenClaw framework
- Enterprise security, orchestration, tool-use framework
- Hardware-agnostic (runs on non-NVIDIA too)
- **Not relevant to kernel work**

---

## SECTION 2: EXPECTED KEYNOTE ANNOUNCEMENTS (Tomorrow)

Based on pre-GTC reporting and leaks, the keynote will likely cover:

### 2.1 Vera Rubin Platform -- Full Launch Details

**Confirmed specs (from CES 2026 + pre-GTC reporting):**
- Successor to Blackwell for datacenter
- 288 GB HBM4, 22 TB/s bandwidth
- 35-50 PFLOPS dense NVFP4 (5x inference / 3.5x training vs Blackwell)
- Vera CPU: 88 Olympus-core ARM (Neoverse successor)
- TSMC 3nm process
- 208 billion transistors (2.6x Hopper)
- In full production (confirmed CES 2026)
- H2 2026 customer shipments

**sm_120 relevance: NONE.** Datacenter-only. But Vera Rubin's cuBLAS/cuSOLVER
will set new reference performance targets.

### 2.2 Blackwell Ultra (GB300) -- Deeper Technical Details

**Known specs:**
- 160 SMs (vs 170 SMs on RTX 5090 sm_120)
- 5th-gen Tensor Cores + 2nd-gen Transformer Engine
- 256 KB TMEM per SM (Tensor Memory -- datacenter only, NOT on sm_120)
- Doubled SFU exponential throughput (2x faster attention vs Blackwell)
- 15 PFLOPS dense NVFP4
- 288 GB HBM3E, 8 TB/s bandwidth per GPU
- NVLink 5: 1.8 TB/s per GPU
- PCIe Gen 6: 256 GB/s bidirectional
- Full backward CUDA compatibility

**sm_120 relevance: MINIMAL.** The doubled SFU throughput is interesting as a
direction (softmax/exponential acceleration) but this is sm_100 datacenter, not
sm_120. The specific techniques for exploiting this won't transfer.

### 2.3 Feynman Architecture Preview

**Expected announcement (not a product launch):**
- Next-next-gen after Rubin (2028 datacenter)
- TSMC A16 (1.6nm) process
- Silicon photonics for inter-chip communication (first NVIDIA GPU with optical I/O)
- HBM4 or HBM5 memory
- Groq LPU technology integration (deterministic logic for inference)
- Likely early samples on static display, architecture overview, not shipping
- TDP possibly >1000W for datacenter parts

**Consumer Feynman (RTX 70-series?):** 2029 at earliest. No details.

**sm_120 relevance: NONE.** This is 2028+ datacenter.

### 2.4 Rubin Consumer GPU (RTX 60-series) Timeline

**From leaks/reporting (NOT expected as keynote topic):**
- Rubin consumer GPUs (GR20x family: GR202, GR203, GR205, GR206, GR207)
- RTX 6090 expected H2 2027 or Q1 2027 announcement
- Will be the successor architecture to our RTX 5090
- NVIDIA confirmed no new consumer GPUs in 2026

**sm_120 relevance: FORWARD PLANNING ONLY.** Our RTX 5090 (sm_120) will be
current hardware for another ~18 months. No urgency.

### 2.5 Software/CUDA Stack Updates

**This is the wildcard -- highest potential for actionable findings.**

Jensen teased "several new chips the world has never seen before." Beyond hardware,
watch for:
- CUDA 13.3 or CUDA 14.0 preview announcement
- New PTX instructions or ISA features
- cuBLAS/cuSOLVER performance leaps
- CUTLASS major version bump
- New Nsight Compute capabilities
- sm_120-specific optimization guidance from NVIDIA engineers

### 2.6 N1/N1X ARM Laptop CPU

- NVIDIA's first Windows laptop CPU (ARM-based)
- Competition with Qualcomm Snapdragon X
- **sm_120 relevance: NONE**

---

## SECTION 3: WHAT MATTERS FOR OUR KERNEL WORK

### Priority 1: CUDA 13.2 Upgrade Path (Already Available)

The most impactful finding is the CUDA 13.2 release that is already shipped.
Workers should consider upgrading from CUDA 13.0 to CUDA 13.2 for:

1. **PTX ISA 9.2:** New `cvt .bf16x2` from FP8 sources -- could simplify FP8
   conversion paths in attention and GEMM kernels
2. **`cp.async.bulk .ignore_oob`** -- cleaner boundary handling in tiled kernels
3. **`sm_120f` target** -- forward-compatible compilation
4. **cuBLAS baselines** -- Grouped GEMM MXFP8, FP32 heuristics improvements
5. **cuSOLVER baselines** -- FP64 emulation for numerical workers
6. **libdevice** -- faster expm1f/erff

### Priority 2: Post-Keynote Monitoring (Tomorrow)

After the keynote (1 PM PT, March 16), check:
1. [NVIDIA Newsroom](https://nvidianews.nvidia.com/news/latest) for press releases
2. [NVIDIA Blog GTC 2026](https://blogs.nvidia.com/blog/gtc-2026-news/) for summaries
3. [NVIDIA Developer Blog](https://developer.nvidia.com/blog/) for technical posts

### Priority 3: Key GTC Sessions (March 17-19)

These sessions are the most likely to contain actionable kernel optimization info:

| Session ID | Title | Why It Matters |
|-----------|-------|---------------|
| S81859 | "CUDA: New Features and Beyond" | CUDA architect on what's new and roadmap |
| S81772 | "Don't Leave Tensors on the Table" | Tensor core optimization techniques for Blackwell |
| Connect With Experts | CUDA, Nsight sessions | Live Q&A -- ask about sm_120 specifics |

### What Has NOT Changed for sm_120

- **mma.sync instructions unchanged** in PTX ISA 9.2 (no new MMA variants for sm_120)
- **No TMEM on sm_120** (that's datacenter Blackwell only)
- **No tcgen05 on sm_120** (datacenter only)
- **No NVFP4 MMA on sm_120** (requires tcgen05)
- **No wgmma on sm_120** (datacenter only)
- **Shared memory limit still 99 KB/block on sm_120**
- **Register file still 64K/SM on sm_120**

The RTX 5090 / sm_120 ISA is fundamentally unchanged from CUDA 13.0. The upgrades
are in libraries, tooling, and minor PTX additions -- not architectural.

---

## SECTION 4: ARCHITECTURE ROADMAP SUMMARY

```
NOW (2025-2026)          2026 H2              2028              2029+
+-----------+        +-----------+       +-----------+     +-----------+
| Blackwell |        | Vera Rubin|       | Feynman   |     | ???       |
| sm_100/120|  --->  | Datacenter|  ---> | TSMC A16  |---> |           |
| RTX 5090  |        | HBM4, 3nm |       | Photonics |     |           |
+-----------+        +-----------+       +-----------+     +-----------+
                          |
                     +----------+
                     | Rubin    |
                     | Consumer |
                     | RTX 60xx |
                     | H2 2027  |
                     +----------+
```

- **Our RTX 5090 (sm_120)** remains current consumer hardware through at least H1 2027
- **Blackwell Ultra (GB300)** is datacenter sm_100 variant -- not our target
- **Vera Rubin** is datacenter successor -- different ISA, different target
- **RTX 60-series** (Rubin consumer) is ~18 months away

---

## BOTTOM LINE

**The keynote has NOT happened yet.** It is tomorrow, March 16, at 11am PT.

**What we already have that's actionable:** CUDA 13.2 (shipped March 5) with PTX ISA
9.2 additions (bf16x2 conversion from FP8, cp.async.bulk .ignore_oob) and library
updates (cuBLAS Grouped GEMM MXFP8, cuSOLVER FP64 emulation). CUTLASS 4.4.2 with
sm_120f target. These are documented in detail in previous briefs.

**What we're watching for tomorrow:** Any CUDA 13.3/14.0 announcement, new sm_120
PTX instructions, library performance improvements, or kernel optimization guidance
from NVIDIA engineers. The keynote will be hardware-heavy (Vera Rubin, Feynman,
Blackwell Ultra) -- software/toolkit announcements may be buried or saved for
breakout sessions (March 17-19).

**This brief should be refreshed after the keynote concludes (after 1 PM PT March 16).**
