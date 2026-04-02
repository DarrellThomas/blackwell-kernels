# Mustard: Device-Side Task Graph Execution — Eliminating Host Launch Overhead

**Source:** https://dl.acm.org/doi/10.1145/3721145.3730426 (ICS'25 — Turimbetov et al.)
**Relevant to:** numerical/ worker (LU factorization, kernel launch overhead)
**Worker's current problem:** Multi-kernel LU approach suffers from ~0.38ms CUDA Graph overhead. cuSOLVER avoids this with a monolithic kernel. Need intermediate approaches between "many kernel launches" and "write everything in one kernel."

---

## What This Is

Mustard (ICS 2025, June 2025) is a device-side execution model where the CPU initializes a task graph but the **GPU schedules and executes all tasks without returning to the CPU**. The CPU is removed from the critical path of task graph execution.

This is relevant because LU factorization has a natural task graph structure (panel -> LASWP -> TRSM -> GEMM, repeated), and removing the CPU from the loop eliminates kernel launch overhead.

---

## How It Works

### Architecture

1. **CPU side:** Constructs the task graph (DAG of kernel operations), partitions it across GPUs, uploads to device memory
2. **GPU side:** A persistent "scheduler kernel" reads the task graph from device memory, checks dependencies, and launches child tasks via CUDA Dynamic Parallelism (CDP) or direct execution within the persistent kernel
3. **Inter-GPU:** Dependencies between devices are tracked via circular buffers and atomic flags

### Key Technical Details

- **Graph partitioning:** Static partitioning between devices at setup time
- **Dependency tracking:** Each task has a dependency counter in device memory. When all predecessors complete, the counter reaches zero and the task becomes eligible
- **Data race avoidance:** cudaGraph dependencies encoded in the task graph structure
- **Communication buffers:** Circular buffers for inter-GPU data exchange

### Performance for Matrix Factorization

Mustard demonstrated good strong scaling for **tiled matrix factorization algorithms** (their primary benchmark). The approach eliminates the CPU-GPU round-trip that occurs at each iteration of the blocked algorithm.

However, the paper notes: "communication between devices and diminishing inter-GPU parallelism in tiled matrix factorization algorithms restrict performance improvements with higher GPU counts." This is the standard Amdahl's Law limitation for factorization.

---

## Relevance to Our Single-GPU LU

### What Transfers

The key insight is: **you don't need to write a monolithic kernel to eliminate launch overhead**. You can:

1. Write separate kernels for panel, LASWP, TRSM, GEMM (easier to develop and debug)
2. Compose them into a device-side task graph
3. Execute the entire graph without CPU involvement

This gives the **development simplicity** of multi-kernel approach with the **launch overhead elimination** of a monolithic kernel.

### Implementation Options on sm_120

**Option A: CUDA Graphs (Simplest)**
```cpp
// Capture the blocked LU iteration as a CUDA graph
cudaGraph_t graph;
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

for (int k = 0; k < N/NB; k++) {
    panel_kernel<<<1, 256, 0, stream>>>(A, k, NB, ipiv);
    laswp_kernel<<<gridL, blockL, 0, stream>>>(A, ipiv, k, NB);
    trsm_kernel<<<gridT, blockT, 0, stream>>>(A, k, NB);
    gemm_kernel<<<gridG, blockG, 0, stream>>>(A, k, NB);
}

cudaStreamEndCapture(stream, &graph);
cudaGraphInstantiate(&instance, graph, 0);
cudaGraphLaunch(instance, stream);  // ONE launch, entire factorization
```

**Problem:** CUDA Graph is static -- it captures fixed grid sizes, pointers, etc. But LU factorization has **shrinking** trailing matrix each iteration. Each GEMM kernel needs different dimensions.

**Solution 1:** Graph with `cudaGraphExecKernelNodeSetParams` to update kernel parameters between iterations. But this requires CPU involvement at each iteration.

**Solution 2:** Use `cudaGraphUpload` + `cudaGraphLaunch` with pre-captured per-iteration graphs.

**Solution 3:** Single persistent kernel with cooperative groups (the approach described in the companion brief).

**Option B: Persistent Kernel (What cuSOLVER Does)**
```cpp
__global__ void persistent_lu(float* A, int N, int NB) {
    // This kernel runs for the ENTIRE factorization
    for (int k = 0; k < N/NB; k++) {
        if (blockIdx.x == 0) panel_factorize(A, k, NB);
        __grid_sync();
        distributed_laswp_trsm(A, k, NB);
        __grid_sync();
        trailing_gemm(A, k, NB);
        __grid_sync();
    }
}
```

This is essentially what cooperative groups gives us. No task graph framework needed.

**Option C: Mustard-Style Device-Side Scheduling**

If we needed multiple separate kernels (e.g., using cuBLAS for trailing GEMM), we could implement a lightweight device-side scheduler:

```cpp
// Persistent scheduler kernel
__global__ void scheduler(TaskGraph* graph, float* A) {
    while (!graph->all_done()) {
        Task* task = graph->get_next_ready_task(blockIdx.x);
        if (task) {
            switch (task->type) {
                case PANEL: panel_factorize(A, task->k, task->NB); break;
                case LASWP: laswp(A, task->k, task->NB); break;
                case TRSM:  trsm(A, task->k, task->NB); break;
                case GEMM:  gemm_tile(A, task->k, task->tile_i, task->tile_j); break;
            }
            graph->mark_complete(task);
        }
    }
}
```

This is complex but allows fine-grained task scheduling without CPU involvement.

---

## Persistent Kernel Support Status

From NVIDIA forum (2024): Persistent kernels are **not officially guaranteed** in the CUDA programming model, but they work in practice:

- No explicit time limit on Linux with dedicated compute GPU
- Compute preemption (Pascal+) can interrupt persistent kernels on shared-GUI systems
- TCC driver (pro GPUs) eliminates kernel timeouts
- **Our setup:** GPU 1 is dedicated compute (no GUI), Linux -- persistent kernels should work fine

Cooperative groups launch (`cudaLaunchCooperativeKernel`) is the **officially supported** way to do persistent-style kernels with grid-level synchronization.

---

## Recommendation for Worker

1. **v1 (current):** Blocked LU with cuBLAS calls (already planned). Accept launch overhead.
2. **v2:** Cooperative groups persistent kernel (the companion brief has the architecture). This eliminates ALL launch overhead.
3. **Skip Mustard/task-graph complexity.** For single-GPU LU, cooperative groups is simpler and sufficient. Mustard's value is for multi-GPU.

---

## Sources

- [Mustard: Device-Side Task Graphs (ICS'25)](https://dl.acm.org/doi/10.1145/3721145.3730426)
- [NVIDIA Persistent Kernels Forum Discussion](https://forums.developer.nvidia.com/t/are-persistent-kernels-supported-now-and-in-the-future/288444)
- [CUDA Graphs Kernel Batching (arXiv, 2025)](https://arxiv.org/abs/2501.09398)
- [CUDA Cooperative Groups](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html)
