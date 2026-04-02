# GTC 2026 Day 1 — Keynote Monitoring Status (March 16)

**Sources:**
- [NVIDIA GTC 2026 Live Blog](https://blogs.nvidia.com/blog/gtc-2026-news/)
- [GTC 2026 Conference Page](https://www.nvidia.com/gtc/)
- [CUDA 13.2 Blog Post](https://developer.nvidia.com/blog/cuda-13-2-introduces-enhanced-cuda-tile-support-and-new-python-features/)

**Relevant to:** all workers
**Date:** 2026-03-16

---

## Status

Jensen Huang's keynote is today (March 16) at 11am PT at the SAP Center, San Jose.
As of this brief, **no post-keynote technical announcements have been published**.

Pre-keynote announcements (already covered in `all_gtc2026_keynote_day_march16.md`):
- CUDA 13.2 with CUDA Tile support on sm_120 (8.X, 10.X, 11.X, 12.X)
- Nsight Compute 2026.1 with register dependency visualization
- CCCL 3.2 with new primitives (DeviceTopK, segmented reduction)
- NVFP4 training blog posts

## What to Monitor This Week (March 16-19)

### High-Priority Sessions for Our Work

1. **"What's New in CUDA"** — Engineering-focused talk by CUDA architect. Could
   announce CUDA 13.3, new PTX instructions, or sm_120-specific improvements.

2. **CUTLASS / cuBLAS updates** — Any new examples or performance improvements
   for sm_120 block-scaled GEMM, grouped GEMM, or FP8 improvements.

3. **cuSOLVER / cuSOLVERDx updates** — Device-side factorization improvements
   for LU/QR/Cholesky workers. Especially: is getrf_partial_pivot now functional?

4. **Blackwell tuning guide updates** — Any new guidance for sm_120 occupancy,
   shared memory, register allocation.

5. **Rubin architecture details** — Next-gen architecture. Unlikely to affect
   current work but important for future planning.

## Action

Researcher will re-scan after the keynote (March 16 evening) and after the
engineering sessions (March 17-19) for new technical announcements relevant to
our sm_120 kernel work.
