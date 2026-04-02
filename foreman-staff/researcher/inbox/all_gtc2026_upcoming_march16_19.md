# GTC 2026 Session Guide -- Relevant Talks for Kernel Workers

**Sources:**
- [GTC 2026 Session Catalog](https://www.nvidia.com/gtc/session-catalog/)
- [CUDA: New Features and Beyond (S81859)](https://www.nvidia.com/gtc/session-catalog/sessions/gtc26-s81859/)
- [Don't Leave Tensors on the Table (S81772)](https://www.nvidia.com/gtc/session-catalog/sessions/gtc26-s81772/)
- [GTC 2026 CUDA, Libraries and Dev Tools Track](https://www.nvidia.com/gtc/sessions/cuda-libraries-and-dev-tools/)
- [GTC 2026 Connect With Experts](https://www.nvidia.com/gtc/connect-with-experts/)
- [GTC 2026 News Blog](https://blogs.nvidia.com/blog/gtc-2026-news/)
- [Tuning Flash Attention in CUDA Tile](https://developer.nvidia.com/blog/tuning-flash-attention-for-peak-performance-in-nvidia-cuda-tile/)
- [CUDA Tile IR Backend for Triton](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton)
- [CUTLASS 4.2 GEMM Auto-Tuning](https://developer.nvidia.com/blog/improving-gemm-kernel-auto-tuning-efficiency-on-nvidia-gpus-with-heuristics-and-cutlass-4-2/)
- [CUDA 13.1 Release Blog](https://developer.nvidia.com/blog/nvidia-cuda-13-1-powers-next-gen-gpu-programming-with-nvidia-cuda-tile-and-performance-gains/)
- [FlashAttention-4 on Blackwell](https://bardai.ai/2026/01/22/overcoming-compute-and-memory-bottlenecks-with-flashattention-4-on-nvidia-blackwell/)
- [Blackwell Ultra Architecture](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/)
- [GTC 2026 Keynote Expectations](https://www.analyticsinsight.net/news/nvidia-gtc-2026-keynote-major-announcements-on-ai-gaming-cpus-and-computing)

**Relevant to:** all workers
**Date:** 2026-03-14 (updated -- conference starts in 2 days)

---

## Conference Overview

GTC 2026 runs **March 16-19, 2026** in San Jose. 700+ sessions, 70+ hands-on labs.
Jensen Huang keynote: **Monday March 16, 11:00 AM PT** (free livestream at nvidia.com/gtc).

---

## CONFIRMED SESSIONS RELEVANT TO OUR WORK

### 1. CUDA: New Features and Beyond [S81859]
- **Link:** https://www.nvidia.com/gtc/session-catalog/sessions/gtc26-s81859/
- **Speaker:** Presented by "one of the architects of NVIDIA CUDA" (likely Stephen Jones)
- **Description:** Engineering-focused talk covering what's new and what's coming next
  for CUDA and GPU computing as a whole. This is where new CUDA features for 2026 and
  support for new architectures will be announced.
- **Why it matters:** Could reveal new PTX instructions, sm_120 features, CUDA 13.2/14
  changes, or new programming model capabilities. This is the #1 session to watch.

### 2. Don't Leave Tensors on the Table: Programming and Optimizing Tensor Cores [S81772]
- **Link:** https://www.nvidia.com/gtc/session-catalog/sessions/gtc26-s81772/
- **Description:** Practical techniques for maximizing Tensor Core performance in CUDA
  kernels. Discusses CUTLASS-based coding patterns.
- **Why it matters:** Directly applicable to ALL workers. Could contain new optimization
  techniques for mma.sync, register layout tricks, shared memory strategies, or
  occupancy tuning that we haven't tried.

### 3. Connect With Experts: CUDA Developer Best Practices
- **Format:** Live Q&A with NVIDIA developers
- **Topics:** CUDA-accelerated libraries, profilers, debugging tools, CI pipelines,
  packaging, and builds across the entire development lifecycle.
- **Why it matters:** Opportunity to ask about sm_120-specific optimization strategies.

### 4. Connect With Experts: Developer Toolbox (Nsight Family)
- **Description:** Overview of NVIDIA Nsight family -- design, profiling, and debugging
  tools. Includes latest features and copilot integrations for CUDA compute, AI, and
  graphics applications.
- **Why it matters:** New Nsight Compute features could improve our profiling workflow.

### 5. GTC 2026 Keynote [S81595]
- **When:** Monday March 16, 11:00 AM - 1:00 PM PT
- **Speaker:** Jensen Huang
- **Expected:** Rubin architecture details, Blackwell Ultra (B300), CUDA ecosystem
  updates, Groq integration into CUDA platform, and "fifth layer" AI applications.
- **Why it matters:** CUDA roadmap updates, possible new toolkit announcements.

---

## EXPECTED ANNOUNCEMENTS TO WATCH FOR

### Hardware
- **Vera Rubin platform:** Next-gen after Blackwell. Up to 288GB HBM4, 22 TB/s
  bandwidth, 35-50 PFLOPS dense NVFP4. 5x Blackwell dense FP throughput. H2 2026
  availability. NOT relevant to our current sm_120 work but informs future direction.
- **Blackwell Ultra (B300):** Mid-cycle refresh. 288GB HBM3E, 8 TB/s. 1.5x tensor core
  perf vs standard Blackwell. 2x SFU throughput for attention. Doubled attention-layer
  compute. Datacenter only (sm_100 variant) -- not sm_120.
- **Feynman architecture:** Next-next-gen teased on roadmap. No details expected.

### Software (most relevant to us)
- **CUDA Toolkit updates:** Any CUDA 13.2+ or 14.0 preview. Watch for:
  - New PTX instructions for sm_120
  - CUDA Tile support expansion beyond Blackwell datacenter
  - cuBLAS performance improvements
  - New library APIs
- **CUTLASS updates:** CUTLASS 4.x releases. Watch for sm_120 examples, new tile
  strategies, or Python DSL improvements.
- **Nsight Compute 2026.1:** Already released -- adds Tile kernel profiling support.
  Watch for any GTC demonstrations of new profiling capabilities.

---

## KEY TECHNICAL CONTEXT FROM RECENT NVIDIA PUBLICATIONS

These were published in the lead-up to GTC and may be expanded upon in sessions:

### CUDA Tile Programming Model (CUDA 13.1)
CUDA Tile is NVIDIA's new tile-based GPU programming abstraction. Key facts:
- Introduced in CUDA 13.1 as "CUDA Tile IR" + "cuTile Python"
- Abstracts tensor cores, shared memory, and memory hierarchy
- Currently supports **compute capability 8.x, 10.x, 11.x, 12.x** (includes sm_120!)
- C++ support planned for future releases
- Triton backend (Triton-to-TileIR) available -- compiles Triton to CUDA Tile IR
  instead of PTX. Requires CUDA 13.1+ and Blackwell GPU.
- **sm_120 confirmed supported** -- Flash Attention blog shows separate autotuning
  configs for "sm120, sm121 (RTX 50 series)"

**Relevance to workers:** cuTile could eventually be a higher-productivity path for
writing kernels, but currently Python-only and less control than raw PTX/CUDA. Worth
monitoring but not actionable for our hand-tuned kernels yet.

### Flash Attention via cuTile (NVIDIA Blog, March 5 2026)
NVIDIA published a Flash Attention implementation using cuTile Python on B200:
- Achieved 548 TFLOPS at seq_len=1024, 918 TFLOPS at seq_len=16384
- Key optimizations: K-loop splitting (31% speedup), fast math flags, block ID
  remapping, autotuning tile sizes
- Uses `ct.mma()` which automatically maps to tensor cores
- FP16 inputs, FP32 accumulators
- **Separate autotuning configs exist for sm_120** -- confirms the framework works on
  consumer Blackwell
- No FP8 support shown

### FlashAttention-4 (FA4) -- Datacenter Blackwell Only
FA4 is hardware-software co-designed for datacenter Blackwell (sm_100):
- Uses Tensor Memory (TMEM, 256 KB/SM) -- **NOT available on sm_120**
- Uses tcgen05.mma -- **NOT available on sm_120**
- Peak 1,605 TFLOPS, 71% of theoretical peak
- 3.6x over FA2 at 32K sequence length
- Conditional softmax rescaling (updates only when running max crosses threshold)
- Software-emulated exponentials using FMA polynomial approximation alongside MUFU
- CuTe DSL in Python with 20-30x faster compile times than FA3

**What's portable to sm_120:**
- Conditional softmax rescaling technique (reduces redundant work)
- FMA-based polynomial exp approximation (alternative to MUFU bottleneck)
- K-loop splitting for causal attention
- Block ID remapping for load balancing
These algorithmic ideas are architecture-independent.

### CUTLASS 4.2 + nvMatmulHeuristics
New auto-tuning system for GEMM kernel selection:
- Supports Ampere, Ada, Hopper, and **(preliminary) Blackwell**
- All tensor core precisions: FP4, FP8, FP16/BF16, TF32, INT8
- On B200: achieves 99% of exhaustive search with 5x speedup
- Static cluster size optimization beats dynamic (104% of baseline)
- Available in early access with Python and C++ APIs

### Blackwell Ultra Architecture Details
- 160 SMs, 640 Tensor Cores, dual-reticle 208B transistors
- 15 PFLOPS dense NVFP4 (1.5x standard Blackwell)
- 2x SFU throughput for attention operations
- NVFP4: "nearly FP8-equivalent accuracy with ~1.8x less memory"
- Full backward compatibility with CUDA ecosystem
- **This is sm_100 datacenter only -- does NOT apply to sm_120**

### NVFP4 Precision Format
New 4-bit format for Blackwell:
- Microscaled FP4 with dynamic scaling
- 50% less memory than FP8 for KV cache
- Hardware support in 5th-gen tensor cores
- **sm_100/sm_101 only currently** -- unclear if sm_120 will get NVFP4 MMA support

---

## POST-GTC ACTION ITEMS

After the conference ends (March 19+), researcher should:

1. **Check for recordings/slides** of S81859 (CUDA New Features) and S81772 (Tensor
   Cores) -- these are the two highest-priority sessions
2. **Search for any new CUDA toolkit announcements** (CUDA 13.2, 14.0 preview, new
   PTX instructions)
3. **Search for any sm_120-specific content** that may have been presented
4. **Check for CUTLASS releases** or new examples shown at GTC
5. **Check for Nsight Compute updates** demonstrated at GTC
6. **Look for cuTile/CUDA Tile expansion** -- particularly C++ support timeline and
   sm_120 optimization guidance
7. **Search for any attention/GEMM kernel talks** not found in pre-conference catalog
8. **Check GTC 2026 on-demand** (nvidia.com/en-us/on-demand/) for session recordings
   as they become available (usually within days/weeks of the event)

---

## SESSIONS FROM GTC 2025 NOW AVAILABLE ON-DEMAND

These related talks from last year are available now and may contain useful techniques:

- **Programming Blackwell Tensor Cores with CUTLASS [S72720]** -- GTC 2025
  https://www.nvidia.com/en-us/on-demand/session/gtc25-s72720/
- **FlashAttention-3: Fast and Accurate Attention With Asynchrony and Low Precision [S71368]** -- GTC 2025
  https://www.nvidia.com/en-us/on-demand/session/gtc25-S71368/
- **Blackwell Programming for the Masses With OpenAI Triton [S72876]** -- GTC 2025
  https://www.nvidia.com/en-us/on-demand/session/gtc25-s72876/
- **CUDA Techniques to Maximize Memory Bandwidth and Hide Latency [S72683]** -- GTC 2025
  https://www.nvidia.com/en-us/on-demand/session/gtc25-s72683/

---

## COMMUNITY EVENTS

- **Post-GTC Meetup (March 23):** "GTC 2026 Conf Recap + Evolution of Flash Attention
  v1-v4 Optimizations" by Seth Weidman. Available on Meetup in DC and Dubai chapters.
  Could contain a useful summary of FA evolution and GTC highlights.
