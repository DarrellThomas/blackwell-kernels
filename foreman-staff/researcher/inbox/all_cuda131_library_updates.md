# CUDA 13.1 Math Library Updates Relevant to Workers

**Source:** https://developer.nvidia.com/blog/nvidia-cuda-13-1-powers-next-gen-gpu-programming-with-nvidia-cuda-tile-and-performance-gains/
**Relevant to:** all workers
**Date:** March 2026

## cuBLAS Updates

- **Grouped GEMM** (experimental): FP8/BF16 with CUDA Graph support. 4x speedup
  over multi-stream GEMM for MoE-style workloads. Could benefit fused-mlp SwiGLU
  (two up-projections can be batched as grouped GEMM).
- **Block-scaled FP4/FP8 matmuls** at production-level performance on Blackwell.
- **FP64 emulation** via tensor cores — double-precision GEMM using lower-precision
  tensor ops. Same principle as BF16x9 but for FP64.

## cuSOLVER Updates

- **Batched SYEV** (symmetric eigenvalue decomposition): ~2x speedup on Blackwell.
  Relevant for future eigenvalue project.
- **GEEV** (general eigenvalue): hybrid CPU/GPU algorithm, 1.7x speedup for N=5000-30000.

## cuSPARSE Updates

- **SpMVOp API**: New SpMV with improved CSR performance + user-defined epilogues.
  See separate brief for spmv worker.

## CCCL 3.1

- **Deterministic floating-point reductions**: three modes (not-guaranteed, run-to-run,
  GPU-to-GPU). Could be useful for debugging numerical issues in reduction kernels.
- **Single-phase CUB APIs** using memory resources — simpler scan/reduce API.

## Green Contexts

- SM resource partitioning via Runtime API. Allows dedicating specific SMs to
  priority workloads. Potentially useful for separating ComfyUI (GPU 0) from
  kernel work more cleanly, or for running multiple small kernels concurrently.

## CUDA Tile

- New tile-based programming model abstracting tensor cores. Python DSL (`cuTile`)
  available now; C++ support planned. Interesting for prototyping but probably not
  relevant for hand-tuned PTX kernels.
