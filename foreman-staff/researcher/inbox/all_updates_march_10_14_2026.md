# CUDA / PTX / Library / Driver Updates — March 10-14, 2026

**Sources:** NVIDIA CUDA 13.2 release notes, CUTLASS 4.4.2 changelog, NVIDIA driver 595.79 WHQL, GTC 2026 pre-event coverage, cuBLAS independent patch program, MLSys 2026 FlashInfer contest, gau-nernst FA for 5090
**Relevant to:** all workers
**Date:** 2026-03-14
**Supplements:** `all_cuda_updates_march2026.md` and `all_cuda132_ptx_isa92_updates.md` (already delivered)

---

## 1. cuBLAS Independent Patch Release Program (NEW — March 9, 2026)

**Source:** https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html

Starting **March 9, 2026**, NVIDIA is releasing cuBLAS patches independently of
CUDA Toolkit releases. This is a significant change in release strategy.

**What it means:**
- Critical cuBLAS bug fixes (like the cublasLtMatmul concurrent correctness issue
  in CUDA 13.0) can be delivered faster
- Workers can update cuBLAS without upgrading the entire toolkit
- Our cuBLAS reference baselines could shift at any time with a patch

**Action for foreman:** Check the cuBLAS patch downloads page periodically.
If patches fix correctness issues affecting our benchmarks, install them
without waiting for a full CUDA toolkit upgrade.

---

## 2. NVIDIA Driver 595.79 WHQL (March 10, 2026)

**Source:** https://www.nvidia.com/download/driverResults.aspx/265442/en-us/

The latest stable driver for RTX 5090 is **595.79 WHQL** (released March 10, 2026).
This is the current recommended driver.

**History of recent driver issues:**
- 595.59: Caused fan failures
- 595.71: Fixed fan issues but **capped voltages near 1V, causing up to 16% perf drops**
  in benchmarks. NVIDIA implemented this to prevent melted 16-pin connectors.
- 595.76: Hotfix removing the voltage cap
- **595.79: Current stable. Supports CUDA 13.2. No known CUDA compute regressions.**

**Impact on our benchmarks:** If we're on 595.71, our benchmark numbers could be
artificially LOW due to the voltage throttling. Verify our driver version:
```bash
nvidia-smi | head -3
```

If on 595.71, upgrade to 595.79 immediately and re-run baselines.

---

## 3. CUTLASS 4.4.2 (March 13, 2026)

**Source:** https://docs.nvidia.com/cutlass/latest/CHANGELOG.html

Released just yesterday. Key changes:

**sm_120f compilation enabled for examples:**
- The `f` suffix denotes "family-specific architecture features"
- sm_120f allows running the same binary on different chips in the sm_120 family
  (RTX 5090, RTX 5080, etc.) without recompiling
- This is primarily a compatibility convenience, NOT new ISA features

**Other changes:**
- NVFP4/MX Grouped GEMM exposed in CUTLASS Profiler
- Fixed Hopper FMHA causal attention performance regression (not us)
- Python 3.14 support for CuTe DSL
- Fixed memory fence for clc scheduler in Blackwell SM120 pingpong kernel
- Fixed missing SMEM alignment in Blackwell SM120 scale factors
- Blackwell SM100 and SM120 blockscaled sparse kernels added

**The sm_120f vs sm_120a question:** We should check whether compiling with
`-arch=compute_120f -code=sm_120f` produces different code than our current
`sm_120a`. Likely no functional difference for our mma.sync-based kernels,
but worth a quick test.

---

## 4. GTC 2026 (March 16-19) — Keynote is March 16

**Source:** https://blogs.nvidia.com/blog/gtc-2026-news/

The GTC 2026 keynote is **March 16 at 11:00 AM PT** (two days from now). As of
March 14, no technical announcements have been made yet. The keynote will be
streamed free at nvidia.com.

**Expected announcements:**
- **Vera Rubin GPU architecture:** Next-gen datacenter GPU with up to 288GB HBM4.
  This is sm_150+ territory -- NOT relevant to our sm_120 work.
- **Agentic-optimized CPUs:** NVIDIA N1/N1X laptop CPUs
- **NemoClaw:** Open-source platform for enterprise AI agents
- **CUDA/CUDA-X updates:** Possible, but no specifics leaked

**What to watch for (relevant to us):**
- Any CUDA 13.3 or 13.2.1 announcement
- New PTX ISA features for sm_120
- cuBLAS/cuSOLVER performance improvements for consumer Blackwell
- torch.compile improvements for custom CUDA extensions
- Any mention of consumer GPU compute improvements

**Action:** Re-run this research task after the keynote (March 16 evening) to
capture any announcements relevant to our kernel work.

---

## 5. CUDA 13.2 Confirmation — No Post-Release Patches Yet

**Source:** https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html

CUDA 13.2 was documented as of March 5, 2026. No point releases (13.2.1) or
patches have been released since, other than the independent cuBLAS patch program
announced March 9.

**Key items already covered in previous brief but worth reiterating:**
- PTX ISA 9.2 (incremental: FP8→BF16 cvt, st.async.b128, u8x4 ops)
- cuBLAS 13.3.0.5 (concurrent correctness fix, L3 improvements)
- cuSOLVER 12.1.0.51 (FP64 emulation APIs)
- Minimum driver: 595.45.04

We are still on CUDA 13.0. The upgrade to 13.2 is recommended for bug fixes.

---

## 6. PyTorch sm_120 Support Status

**Sources:** https://github.com/pytorch/pytorch/issues/159207, https://github.com/pytorch/pytorch/issues/164342

As of March 2026, PyTorch stable builds still do NOT ship pre-compiled sm_120
binaries. Users must:
- Build from source with CUDA 12.8+ or 13.x
- Use nightly builds (2.10.0.dev + cu12.9 or cu13.x)

We're already building from source with CUDA 13.0, so this doesn't affect us.
However, any future PyTorch upgrade should verify sm_120 compatibility.

**CXX_ABI change:** PyTorch now requires `CXX_ABI=1` for custom extensions.
If we upgrade PyTorch, our extension build scripts may need updating.

---

## 7. MLSys 2026 FlashInfer AI Kernel Generation Contest

**Source:** https://github.com/flashinfer-ai/flashinfer-bench-starter-kit

A new competition for high-performance GPU kernel development on NVIDIA Blackwell.

**Contest tracks:**
1. Fused Mixture-of-Experts kernel
2. Sparse attention for long-context inference
3. Gated delta network operations

**Hardware:** NVIDIA B200 (datacenter Blackwell, sm_100) — NOT sm_120.

**Relevance to us:** Low direct relevance (wrong architecture), but the
FlashInfer-Bench framework and published solutions may contain optimization
techniques transferable to sm_120. Worth monitoring after the competition
concludes.

---

## 8. gau-nernst Flash Attention for RTX 5090 — Update

**Source:** https://gau-nernst.github.io/fa-5090/

The gau-nernst blog post (August 2025) remains the best public reference for
sm_120 flash attention. No 2026 updates have been published.

**Confirmed details (from deep reading):**
- v5 kernel: 197.74 TFLOPS (94.39% of 209.5 TFLOPS SOL)
- Uses only Ampere-era features (cp.async.cg, mma.m16n8k16, ldmatrix)
- Does NOT use TMA/cp.async.bulk despite sm_120 supporting it
- MXFP8/NVFP4 MMA for sm_120 was the stated motivation but NOT implemented
- 4 warps, BLOCK_Q=128, BLOCK_KV=64, DIM=128
- XOR swizzle: XOR bits 4-6 of row address with bits 0-2 of row index

**Key insight for all kernels:** The author notes it's "rather easy to get good
performance out of 5090 compared to previous generations." This aligns with our
experience -- the hardware is forgiving. The challenge is the last 5-6% to SOL.

---

## 9. Transformer Engine Release Status

**Source:** https://github.com/NVIDIA/TransformerEngine/releases

Latest release: **v2.12** (February 24, 2025). No 2026 releases.

The v2.12 release fixed SM120 compilation with CUDA 12, added fused permute+pad
for FP8, and improved NVFP4 quantization. The `nvte_rmsnorm_bwd_add` fused
backward API was added in v2.8 (earlier release) and remains unchanged.

---

## Summary: What Changed Since March 10

| Item | Date | Impact |
|------|------|--------|
| cuBLAS independent patch program | Mar 9 | Medium: cuBLAS can update without full toolkit |
| Driver 595.79 WHQL | Mar 10 | Check: verify we're not on voltage-throttled 595.71 |
| CUTLASS 4.4.2 | Mar 13 | Low: sm_120f compilation, minor fixes |
| GTC 2026 keynote | Mar 16 | PENDING: watch for CUDA/PTX announcements |
| CUDA 13.2 patches | None yet | No new patches since release |
| PyTorch sm_120 | No change | Still requires source build |

**Highest priority action:** Verify our NVIDIA driver version. If on 595.71,
upgrade to 595.79 to avoid voltage throttling affecting benchmarks.

**Second priority:** After GTC 2026 keynote (March 16), re-search for any
CUDA toolkit, PTX ISA, or library announcements that affect our work.
