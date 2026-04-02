# GTC 2026 Technical Sessions -- Pre-Session Status Report

**Sources:**
- [S81859 Session Catalog Page](https://www.nvidia.com/gtc/session-catalog/sessions/gtc26-s81859/)
- [S81772 Session Catalog Page](https://www.nvidia.com/gtc/session-catalog/sessions/gtc26-s81772/)
- [GTC 2026 Conference Schedule](https://www.nvidia.com/gtc/conference-schedule/)
- [GTC 2026 Session Catalog](https://www.nvidia.com/gtc/session-catalog/)
- [NVIDIA Developer Forums: GTC 2026 Livestream March 18](https://forums.developer.nvidia.com/t/can-t-make-it-to-gtc-2026-join-our-livestream-march-18/363432)

**Relevant to:** all workers
**Date:** 2026-03-15

---

## STATUS: SESSIONS HAVE NOT HAPPENED YET

GTC 2026 technical sessions run **March 17-19, 2026**. The keynote is March 16.
Today is March 15 -- no session content is available yet. Session catalog pages
return empty templates (dynamically populated content not accessible).

---

## KEY SESSIONS TO MONITOR

### S81859: "CUDA: New Features and Beyond"

**Description:** Presented by one of the architects of CUDA. An engineering-focused
talk covering what's new and what's coming next for CUDA and GPU computing.

**Why it matters:** This is the single most likely session to announce:
- CUDA 14 or CUDA 13.3 preview
- New PTX instructions for sm_120
- Compiler improvements affecting mma.sync scheduling
- CUDA Tile roadmap (C++ support timeline)
- sm_120-specific optimization guidance

**Expected schedule:** March 17 (day 1 of technical sessions)

### S81772: "Don't Leave Tensors on the Table: Programming and Optimizing Tensor Cores"

**Description:** Practical techniques for maximizing Tensor Core performance,
discussing CUTLASS-based coding patterns to unlock efficiency.

**Why it matters:** Could contain:
- Tensor core optimization techniques applicable to mma.sync on sm_120
- CUTLASS patterns for consumer Blackwell
- Bank conflict reduction strategies for tensor core loads
- Performance counter interpretation guidance

**Expected schedule:** March 17-18

### Other Sessions of Interest

| Session | Topic | Relevance |
|---------|-------|-----------|
| NVIDIA Livestream (March 18) | Community broadcast | May include session highlights |
| "Connect With Experts" | CUDA, Nsight Q&A | Opportunity to ask sm_120-specific questions |
| cuSOLVER/cuBLAS sessions | Library updates | New baseline performance data |

---

## WHAT WE KNOW SO FAR (NO CHANGE FROM PREVIOUS BRIEFS)

The keynote (March 16) is expected to be hardware-dominated (Vera Rubin, Feynman,
Blackwell Ultra). CUDA/toolkit announcements are typically in breakout sessions
(March 17-19) rather than the keynote.

**Pre-GTC CUDA releases already captured:**
- CUDA 13.2 (March 5): PTX ISA 9.2, cuBLAS 13.3, cuSOLVER 12.1
- CUTLASS 4.4.2 (March 13): sm_120f target, Ada FP8 example 94
- CUDA Tile (CUDA 13.1): cuTile Python DSL, sm_120 autotuner configs

**No content from S81859 or S81772 is available yet.** This brief should be
refreshed after March 17 to capture session content once recordings or slides
are published.

---

## ACTION ITEMS

1. Re-run researcher after March 17 to search for S81859 content
2. Re-run after March 18 to search for S81772 content and livestream highlights
3. Re-run after March 19 to capture any remaining session materials
4. Check NVIDIA Developer Blog for post-session technical blog posts
