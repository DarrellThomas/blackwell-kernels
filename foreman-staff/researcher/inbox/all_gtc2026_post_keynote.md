# GTC 2026 Post-Keynote Research Brief

**Sources:**
- [NVIDIA GTC 2026 Live Updates Blog](https://blogs.nvidia.com/blog/gtc-2026-news/)
- [NVIDIA Newsroom: Rubin Platform](https://nvidianews.nvidia.com/news/rubin-platform-ai-supercomputer)
- [GTC 2026 Keynote Page](https://www.nvidia.com/gtc/keynote/)
- [Analytics Insight: GTC 2026 Keynote](https://www.analyticsinsight.net/news/nvidia-gtc-2026-keynote-major-announcements-on-ai-gaming-cpus-and-computing)
- [IndexBox: GTC 2026 Keynote](https://www.indexbox.io/blog/nvidia-gtc-2026-kicks-off-with-ceo-keynote-on-ai-and-gaming-roadmap/)
- [NVIDIA-Groq Inference Chip](https://awesomeagents.ai/news/nvidia-groq-inference-chip-openai/)
- [Feynman Architecture Preview](https://pbxscience.com/nvidias-feynman-architecture-what-we-actually-know-ahead-of-gtc-2026/)
- [TrendForce: Feynman on TSMC A16](https://www.trendforce.com/news/2026/03/13/news-nvidia-may-offer-first-look-at-feynman-at-gtc-2026-tsmc-a16-and-taiwan-supply-chain-in-focus/)
- [CUDA 13.2 Release Notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)
- [CUDA 13.2 Technical Blog](https://developer.nvidia.com/blog/cuda-13-2-introduces-enhanced-cuda-tile-support-and-new-python-features/)
- [NVIDIA N1X Laptop Chip](https://www.tomsguide.com/computing/cpus/watch-out-intel-nvidia-finally-confirms-next-gen-n1x-and-n1-chips-for-ai-pcs-are-coming)
- [CNBC: NVIDIA CPU Push](https://www.cnbc.com/2026/03/13/nvidia-gtc-ai-jensen-huang-cpu-gpu.html)
- [NVIDIA NemoClaw](https://ryxel.ai/news/technology/2026/3/10/nvidia-launches-nemoclaw-open-source-ai-platform-gtc-2026)
- [NVIDIA Dynamo Newsroom](https://nvidianews.nvidia.com/news/nvidia-dynamo-open-source-library-accelerates-and-scales-ai-reasoning-models)
- [DLSS 4.5 at GDC 2026](https://www.nvidia.com/en-eu/geforce/news/gdc-2026-nvidia-geforce-rtx-announcements/)
- [GTC 2026 Preview: Groq SRAM-Decode](https://www.viksnewsletter.com/p/gtc-2026-preview-implications-of-sram-decode)
- [CUTLASS GitHub Releases](https://github.com/NVIDIA/cutlass/releases)

**Relevant to:** all workers
**Date:** 2026-03-16 (keynote day)

---

## KEYNOTE STATUS: IN PROGRESS / JUST CONCLUDED

Jensen Huang delivered the GTC 2026 keynote on **March 16, 2026 at 11:00 AM PT** from
the SAP Center in San Jose to 30,000+ attendees from 190 countries. The keynote was
a ~2 hour address covering chips, software, models, and applications.

**Note:** NVIDIA's newsroom has NOT yet published the keynote-day press releases as of
this research pass. The findings below are compiled from early post-keynote reporting,
confirmed pre-GTC announcements, and pre-keynote intelligence. This brief will be
supplemented when the official press releases and technical blog posts drop (typically
within hours of the keynote).

---

## SECTION 1: CONFIRMED KEYNOTE ANNOUNCEMENTS

### 1.1 Vera Rubin Platform -- Full Production Launch

The headline datacenter announcement. Six new chips, one AI supercomputer platform.

**Rubin GPU:**
- 50 PFLOPS dense NVFP4 inference compute
- 3rd-gen Transformer Engine with hardware-accelerated adaptive compression
- 3.6 TB/s NVLink bandwidth per GPU
- Up to 288 GB HBM4, 22 TB/s memory bandwidth
- 10x reduction in inference token cost vs Blackwell
- 4x reduction in GPUs needed to train MoE models vs Blackwell

**Vera CPU:**
- 88 custom "Olympus" ARM cores (Armv9.2)
- NVLink-C2C connectivity
- Designed for agentic reasoning workloads

**System Configurations:**
- Vera Rubin NVL72: 72 Rubin GPUs + 36 Vera CPUs, 260 TB/s total bandwidth
- HGX Rubin NVL8: 8 Rubin GPUs with NVLink interconnect (for x86 systems)
- Cable-free tray design: 18x faster assembly/servicing than Blackwell

**Supporting Chips:**
- NVLink 6 Switch (in-network compute)
- ConnectX-9 SuperNIC
- BlueField-4 DPU
- Spectrum-6 Ethernet Switch

**Availability:** H2 2026, AWS/Google Cloud/Azure/OCI first.

**sm_120 relevance: NONE.** Datacenter-only architecture. However, Vera Rubin's
cuBLAS/cuSOLVER performance will set new reference baselines.

### 1.2 NVIDIA-Groq Inference Chip (LPU-Based)

A new inference processor integrating Groq's LPU (Language Processing Unit) technology.

**Architecture:**
- Deterministic execution model (compiler schedules every operation at compile time)
- SRAM-based on-chip memory (~230 MB per chip), NOT HBM
- ~80 TB/s memory bandwidth (24x H100)
- Near 100% compute utilization (vs 30-40% GPU inference utilization)
- No caches, no branch predictor -- fully deterministic pipeline
- TSMC A16 process with 3D SRAM stacking

**Trade-offs:**
- 230 MB per chip capacity -- needs 600+ chips for a 70B model at FP16
- Excels at latency-sensitive workloads; struggles with large models

**Deployment:**
- LPX rack: 256 LPUs per rack (4x increase over gen 1)
- 52-layer M9 Q-glass PCBs
- Positioned as complement to Rubin GPUs for inference

**Lead Customer:** OpenAI -- 3 GW dedicated inference capacity for Codex.

**sm_120 relevance: NONE.** Completely different architecture (deterministic SRAM vs
GPU). Does not affect our mma.sync kernel work.

### 1.3 Feynman Architecture Preview (2028)

A teaser for the next-next-gen architecture after Rubin.

**Key details:**
- Slated for 2028 (production shipping 2028-2029)
- TSMC A16 process (1.6nm class)
- Super Power Rail (SPR) -- backside power delivery
- **Silicon photonics** -- optical inter-chip communication (first for NVIDIA)
- "Inference-First" architecture design philosophy
- May integrate Groq LPU technology
- Possibly >1000W TDP for datacenter parts

**sm_120 relevance: NONE.** 2028 datacenter. Consumer Feynman (RTX 70-series?) would
be 2029-2030 at earliest.

### 1.4 N1/N1X ARM Laptop CPUs

NVIDIA's re-entry into the consumer CPU market.

**N1X Specs (flagship):**
- 20 custom ARM cores
- Integrated GPU matching standalone RTX 5070 performance
- Joint venture with MediaTek
- TSMC fabrication

**N1 (mainstream):**
- Lower-power variant for thin-and-light laptops

**Launch:** H1 2026. Dell and Lenovo confirmed as first OEMs.

**sm_120 relevance: NONE.** These are laptop SoCs, not relevant to kernel optimization.

### 1.5 Groq Partnership Details

- $20B non-exclusive licensing deal (December 2025 acqui-hire)
- Groq founder Jonathan Ross and engineering team now at NVIDIA
- Groq's LPU IP integrated into new inference chip
- NVIDIA builds out dedicated LPU chip team

### 1.6 Strategic Investments and Partnerships

- $2B investment in Nebius (AI cloud)
- Gigawatt-scale partnership with Thinking Machines Lab for Vera Rubin systems
- ~$26B committed to open-source AI models
- OpenClaw described as "fastest-growing open source project in history"

---

## SECTION 2: SOFTWARE AND DEVELOPER ECOSYSTEM

### 2.1 NemoClaw (Pre-GTC, March 10)

- Open-source enterprise AI agent platform (Apache 2.0)
- Hardware-agnostic (runs on non-NVIDIA too)
- Not relevant to kernel work

### 2.2 NVIDIA Dynamo (Pre-GTC)

- Open-source inference framework, successor to Triton Inference Server
- Disaggregated serving: separate prefill/decode on different GPUs
- 2x on Hopper, 30x per-GPU for DeepSeek-R1 on GB200 NVL72
- Compatible with PyTorch, SGLang, TensorRT-LLM, vLLM
- Not relevant to kernel optimization directly

### 2.3 Nemotron 3 Super (March 11)

- 120B parameter, 12B active, hybrid Mamba-Transformer MoE
- 5x higher throughput for agentic AI
- Not relevant to kernel work

### 2.4 OpenClaw Playbook

- Step-by-step guide to run OpenClaw on DGX Spark
- Local-first AI agents working with files/apps/workflows
- Not relevant to kernel work

---

## SECTION 3: WHAT ACTUALLY MATTERS FOR SM_120 KERNEL WORK

### The honest assessment: NOTHING new from the keynote changes our work.

The GTC 2026 keynote was (as expected) a datacenter hardware and AI platform event.
The announcements that matter for our RTX 5090 (sm_120) kernel optimization work
were all **pre-GTC releases** that are already documented in previous briefs:

### 3.1 CUDA 13.2 (Released March 5 -- ALREADY SHIPPED)

This remains the most actionable finding. Key items for workers:

**PTX ISA 9.2 additions relevant to sm_120:**
| Instruction | What It Does | Worker Impact |
|------------|-------------|---------------|
| `cvt .bf16x2` from FP8 sources | Direct FP8-to-BF16x2 conversion | Potential simplification of FP8 conversion in attention/GEMM |
| `cp.async.bulk .ignore_oob` | Out-of-bounds safe bulk copy | Simpler boundary handling in tiled kernels |
| `st.async .b128` | 128-bit async store | Extended async store capability |
| `sm_120f` target | Family-specific compilation | Forward compatibility |

**cuBLAS 13.3 (bundled with CUDA 13.2):**
- Experimental Grouped GEMM with MXFP8 on CC 10.x and 11.0
- Up to 20% speedup on RTX PRO 6000
- Up to 3x for MXFP8/NVFP4 on DGX Spark (sm_121)
- Improved FP32 GEMM heuristics for M,N >> K on Blackwell
- **Open question:** Does Grouped GEMM MXFP8 work on sm_120 (CC 12.0)? Docs say "10.x and 11.0" -- needs testing.

**cuSOLVER 12.1:**
- FP64 fixed-point emulation (BF16x9 Ozaki scheme) for QR/LU/Cholesky
- Performance improvements for n <= 32 on Blackwell

**Developer Tools:**
- Nsight Compute 2026.1: register dependency correlation, report clustering
- Nsight Systems 2026.1: PyTorch profiling improvements
- Nsight Python: decorator-based kernel profiling
- Numba-CUDA: first GPU hardware debugging support

**New CCCL 3.2 Algorithms:**
- `cub::DeviceTopK`: up to 5x speedup over radix sort for small K
- Fixed-size segmented reduction: up to 66x speedup for small segments
- Segmented scan, binary search, conditional search primitives

**Math Library:**
- `expm1f()` up to 20% faster
- `erff()` 5-10% faster

### 3.2 CUTLASS 4.4.2 (Released March 13 -- ALREADY SHIPPED)

- `sm_120f` compilation target added
- NVFP4/MX profiler support
- Ada FP8xFP8 blockwise dequantization GEMM (example 94) -- uses same mma.sync ISA
  as sm_120, useful reference for our FP8 GEMM work
- CuTe DSL: `cute.experimental` layer, AOT compilation, JAX integration
- CuTe DSL Python achieves within 2% of handwritten C++ on Blackwell
- GEMM on Blackwell: ~100x compilation speedup over C++
- Flash attention on Blackwell: 30-50x compilation speedup

### 3.3 DLSS 4.5 Dynamic Multi Frame Generation (GDC 2026, March 10)

- Launches March 31, 2026
- Up to 35% higher 4K frame rates in path-traced titles
- 6X Multi Frame Generation
- Second-generation transformer AI model
- **Not relevant to compute kernels** -- graphics pipeline only

---

## SECTION 4: WHAT HAS NOT CHANGED FOR SM_120

Confirming from all sources -- these constraints remain:

- **mma.sync instructions unchanged** in PTX ISA 9.2 (no new MMA variants for sm_120)
- **No TMEM on sm_120** (datacenter Blackwell only)
- **No tcgen05 on sm_120** (datacenter only)
- **No NVFP4 MMA on sm_120** (requires tcgen05)
- **No wgmma on sm_120** (datacenter only)
- **No TMA on sm_120** (datacenter only)
- **Shared memory limit still 99 KB/block on sm_120**
- **Register file still 64K/SM on sm_120**
- **No CUDA 14 announced** -- latest is CUDA 13.2 (March 5, 2026)
- **No new consumer Blackwell GPU announced** -- RTX 5090 remains top consumer GPU through at least H1 2027

The RTX 5090 / sm_120 ISA is architecturally frozen. Improvements come only through
library updates, compiler optimizations, and PTX instruction additions -- not from
new hardware features.

---

## SECTION 5: WHAT TO WATCH THIS WEEK (March 17-19)

The breakout sessions are where developer-relevant content lives:

| Session | Title | Why It Matters |
|---------|-------|---------------|
| S81859 | "CUDA: New Features and Beyond" | CUDA architect on roadmap -- could reveal CUDA 14 timeline, new sm_120 features |
| S81772 | "Don't Leave Tensors on the Table" | Practical tensor core optimization for Blackwell |
| Connect With Experts | CUDA, Nsight sessions | Ask about sm_120-specific optimizations |
| Investor Q&A (Mar 17) | NVIDIA earnings/product timeline | Consumer GPU timeline clarity |

**Action:** Re-run researcher after March 19 to capture any session-derived technical
content that would be actionable for workers.

---

## SECTION 6: ARCHITECTURE ROADMAP (Updated)

```
NOW (2025-2026)          2026 H2              2028              2029+
+-----------+        +-----------+       +-----------+     +-----------+
| Blackwell |        | Vera Rubin|       | Feynman   |     | ???       |
| sm_100/120|  --->  | Datacenter|  ---> | TSMC A16  |---> |           |
| RTX 5090  |        | HBM4, 3nm |       | Photonics |     |           |
+-----------+        | +LPU infer|       | LPU integ |     |           |
                     +-----------+       +-----------+     +-----------+
                          |
                     +----------+
                     | Rubin    |
                     | Consumer |
                     | RTX 60xx |
                     | H2 2027  |
                     +----------+
```

- **Our RTX 5090 (sm_120):** Current consumer king through at least H1 2027
- **Vera Rubin (datacenter):** H2 2026 -- new cuBLAS baselines incoming
- **Feynman (datacenter):** 2028 -- inference-first, silicon photonics, possibly LPU
- **Rubin consumer (RTX 60-series):** H2 2027 at earliest -- different ISA
- **N1/N1X (laptop CPU):** H1 2026 -- not a GPU, not relevant

---

## BOTTOM LINE

**The GTC 2026 keynote does not change anything for our sm_120 kernel optimization
work.** The keynote was dominated by datacenter hardware (Vera Rubin, Groq LPU
inference chip, Feynman teaser) and AI platform announcements (NemoClaw, OpenClaw,
Dynamo). None of these affect the RTX 5090 / sm_120 ISA or available instructions.

**The most actionable items remain the pre-GTC releases:**
1. CUDA 13.2 (March 5) -- PTX ISA 9.2, cuBLAS/cuSOLVER updates, Nsight improvements
2. CUTLASS 4.4.2 (March 13) -- sm_120f target, Ada FP8 example 94

**Next research action:** Check back after March 19 for breakout session content,
especially S81859 ("CUDA: New Features and Beyond") which could reveal CUDA 14
roadmap or sm_120-specific optimization guidance from NVIDIA engineers.
