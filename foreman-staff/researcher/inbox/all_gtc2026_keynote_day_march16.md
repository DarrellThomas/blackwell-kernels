# GTC 2026 Keynote Day -- What We Know as Conference Opens

**Sources:**
- [NVIDIA GTC 2026 Live Blog](https://blogs.nvidia.com/blog/gtc-2026-news/)
- [GTC 2026 Keynote Page](https://www.nvidia.com/gtc/keynote/)
- [GTC 2026 Session Catalog](https://www.nvidia.com/gtc/session-catalog/)
- [Feynman Architecture Pre-GTC](https://pbxscience.com/nvidias-feynman-architecture-what-we-actually-know-ahead-of-gtc-2026/)
- [Analytics Insight GTC 2026 Summary](https://www.analyticsinsight.net/news/nvidia-gtc-2026-keynote-major-announcements-on-ai-gaming-cpus-and-computing)
- [NVIDIA Developer Blog: CUDA 13.2](https://developer.nvidia.com/blog/cuda-13-2-introduces-enhanced-cuda-tile-support-and-new-python-features/)
- [NVIDIA Developer Blog: 3 Ways NVFP4](https://developer.nvidia.com/blog/3-ways-nvfp4-accelerates-ai-training-and-inference/)
- [NVIDIA Developer Blog: NVFP4 Training](https://developer.nvidia.com/blog/nvfp4-trains-with-precision-of-16-bit-and-speed-and-efficiency-of-4-bit/)
- [NVIDIA Community Guide to GTC 2026](https://forums.developer.nvidia.com/t/a-community-guide-to-gtc-2026/360161)

**Relevant to:** all workers
**Date:** 2026-03-15 (keynote is tomorrow, March 16 at 11am PT)

---

## Conference Status

GTC 2026 opens **tomorrow, March 16** in San Jose. Jensen Huang keynote at **11am PT**
from the SAP Center. Livestream free at nvidia.com/gtc (no registration required).

---

## Pre-Keynote Announcements Already Made

Several announcements dropped in the days leading up to GTC. These are CONFIRMED
(not speculation):

### 1. CUDA 13.2 Released (March 5, 2026)

CUDA Toolkit 13.2 shipped before GTC with:
- **CUDA Tile on Ampere/Ada** -- CUDA Tile IR now supports compute_80+ (previously
  Blackwell-only). This broadens the developer base but doesn't change our sm_120
  workflow.
- **CUDA 13.2 PTX ISA 9.2** -- Incremental instruction additions (covered in separate
  brief `all_cuda132_ptx_isa92_updates.md`).
- **cuBLAS 13.2**: GEMM_AUTOTUNE algo parameter, FP64 emulated matmuls (Ozaki-1),
  MXFP8 Grouped GEMM on datacenter Blackwell, DGX Spark FP8 improvements.
- **cuSOLVER 13.2**: FP64-emulated QR/LU/Cholesky using BF16x9 scheme on Blackwell.
  Uses new `cusolverDnSetMathMode` and `cusolverDnSetEmulationStrategy` APIs.

### 2. NVFP4 Training and Inference Blog Posts (March 12-13, 2026)

Two NVIDIA blog posts published just before GTC:
- **"3 Ways NVFP4 Accelerates AI Training and Inference"**: Blackwell Ultra at 15
  PFLOPS NVFP4 (3x FP8 on same GPU). Rubin at 35-50 PFLOPS NVFP4.
- **"NVFP4 Trains with Precision of 16-Bit"**: Demonstrated NVFP4 training with
  FP16-equivalent accuracy at 1.9x speed improvement over FP8.
- NOTE: NVFP4 MMA is sm_100/sm_101 only -- NOT available on sm_120. Relevant only
  as context for where NVIDIA is pushing precision boundaries.

### 3. Blackwell NVFP4 Kernel Hackathon Winners (Announced Feb 16)

GPU MODE / NVIDIA hackathon for FP4 kernels on Blackwell B200:
- 4 problems: Batched GEMV, GEMM, Gated Dual GEMM, Grouped GEMM
- Prizes: DGX Spark, RTX 5090, RTX 5080
- Top winners invited to GTC awards ceremony
- Blog posts from participants detail optimization approaches:
  - CuTe DSL implementations achieving ~22us from ~2000us starting point
  - Warp specialization + TMA integration for FP4 kernels
  - Micro-block scaling with FP8 scale factors

---

## Expected Keynote Announcements

Based on pre-GTC reporting, the keynote will likely cover:

### A. Feynman Architecture Preview

- Next-next-gen after Rubin, targeting 2028
- TSMC A16 (1.6nm) process
- "Inference-First" design philosophy
- Possibly >5000W TDP for datacenter parts
- Early architectural overview, not shipping product
- **Relevance to us:** Zero. This is 2028 datacenter. But good for understanding
  where NVIDIA is heading (inference-optimized silicon).

### B. Vera Rubin Platform Details

- 288GB HBM4, 22 TB/s bandwidth
- 35-50 PFLOPS dense NVFP4
- 88 Olympus-core ARM CPU (Vera)
- 5x inference perf, 3.5x training perf vs Blackwell
- H2 2026 production
- **Relevance to us:** Zero for sm_120. But sets the competitive bar for where
  cuBLAS/cuSOLVER/cuSPARSE performance targets will go.

### C. Software Stack Updates

Most likely to contain actionable items for our workers:
- CUDA ecosystem updates
- Possible CUDA 13.3 or 14.0 preview announcement
- CUTLASS updates
- New library APIs or performance improvements
- Agentic AI frameworks (OpenClaw, NemoClaw)
- **Watch for:** Any sm_120-specific kernel optimization guidance from NVIDIA engineers

### D. N1/N1X Laptop CPU

- NVIDIA's first Windows laptop CPU (ARM-based)
- Competition with Qualcomm Snapdragon X
- **Relevance to us:** None.

---

## Key Sessions to Watch Post-Keynote

Priority-ordered for our kernel work:

1. **S81859 -- "CUDA: New Features and Beyond"** (highest priority)
   - CUDA architect presenting what's new and what's coming next
   - Could reveal new PTX instructions, sm_120 optimizations, CUDA 14 roadmap

2. **S81772 -- "Don't Leave Tensors on the Table"**
   - Practical tensor core optimization techniques
   - CUTLASS-based coding patterns for Blackwell

3. **Connect With Experts sessions** (CUDA, Nsight)
   - Live Q&A with NVIDIA developers
   - Opportunity to ask about sm_120 specifics

---

## Action Plan

1. **March 16 (keynote day):** Check NVIDIA blog for keynote summary. Look for any
   CUDA/software announcements buried in the hardware-heavy keynote.

2. **March 17-19 (sessions):** Watch for session recordings/slides from S81859 and
   S81772. Check GTC on-demand (nvidia.com/en-us/on-demand/) for new uploads.

3. **March 20+ (post-GTC):** Deep-dive any new announcements. Write follow-up briefs
   for actionable findings. Check NVIDIA developer blog for technical follow-ups
   (NVIDIA typically publishes detailed blog posts within 1-2 weeks of GTC talks).
