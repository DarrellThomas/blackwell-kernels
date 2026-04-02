# Nsight Compute Profiling Metrics Reference

Source: https://docs.nvidia.com/nsight-compute/ProfilingGuide/
Fetched: 2026-03-27

## Metric Types

### Counters
Raw GPU measurements with aggregate roll-ups:
- `.sum` — aggregated across all unit instances
- `.avg` — mean value per instance
- `.min` / `.max` — extremes
- Calculated sub-metrics: `.pct_of_peak_sustained_active`, `.per_cycle_elapsed`

### Ratios
- `.pct` — percentage
- `.ratio` — raw ratio
- `.max_rate` — theoretical maximum

### Throughputs
- `.pct_of_peak_sustained_active` — % during unit active cycles
- `.pct_of_peak_sustained_elapsed` — % across entire measurement window

## Warp Stall Reasons

Warps stall when waiting for instruction fetch, memory dependency, execution
dependency, or synchronization barrier. The profiler samples stall causes
periodically via the "Source Counters" section.

### Stall Categories

| Stall | Meaning |
|-------|---------|
| **math_throttle** | Arithmetic pipeline saturated (FMA, ALU, special function units) |
| **long_scoreboard** | Pending long-latency instruction (memory, FP64, transcendental) |
| **short_scoreboard** | Awaiting result from short-latency instruction |
| **barrier** | Block-level `__syncthreads()` blocking warp progress |
| **not_selected** | Scheduler chose a different warp (low occupancy indicator) |
| **wait** | Generic dependency stall |
| **imc_miss** | Immediate constant cache miss |
| **lg_throttle** | Local/global memory throttle |
| **tex_throttle** | Texture unit throttle |
| **mio_throttle** | Memory I/O throttle |
| **drain** | Warp draining at exit |

## SM Throughput Metrics

**sm__throughput** — Overall SM compute resource utilization

Breakdown by pipeline:
- FP32 arithmetic (FMA pipelines)
- Integer operations (ALU)
- Memory operations (LSU)
- Special functions (transcendental, conversions)
- Tensor core operations

## Memory Throughput Metrics

| Metric | What It Measures |
|--------|------------------|
| `l1tex__throughput` | L1 cache utilization vs peak bandwidth |
| `ltc__throughput` | L2 cache utilization |
| `dram__throughput` | Main memory (GDDR/HBM) bandwidth utilization |

Supporting metrics:
- Cache hit/miss rates (queries, sector hits, tag misses)
- Bank conflicts in shared memory (`l1tex__data_bank_conflicts_pipe_lsu`)
- Memory request serialization (wavefront count)

## Occupancy Metrics

| Metric | Meaning |
|--------|---------|
| `sm__maximum_warps_per_active_cycle_pct` | Theoretical Occupancy (%) |
| `smsp__maximum_warps_avg_per_active_cycle` | Theoretical Active Warps Per SM |
| `launch__occupancy_limit_registers` | Occupancy limit from register consumption |
| `launch__occupancy_limit_shared_mem` | Occupancy limit from shared memory |
| `launch__occupancy_limit_blocks` | Occupancy limit from max blocks per SM |

**Higher occupancy does NOT always mean higher performance.** But low occupancy
always reduces latency hiding ability.

## Warp Scheduler Metrics

| Metric | Meaning |
|--------|---------|
| `smsp__warps_active` | Warps allocated to subpartition |
| `smsp__warps_eligible` | Warps ready to issue (decoded, deps resolved, FU available) |
| `smsp__issue_active` | Cycles where instructions were issued |
| `smsp__issue_inst_issued` | Count of instructions issued |

High skipped issue slots = poor latency hiding.

## Tensor Core Metrics

`sm__sass_inst_executed_tensor` — Number of tensor (MMA) instructions executed

## Identifying Bottlenecks

| Pattern | Diagnosis |
|---------|-----------|
| SM throughput near 100%, memory low | **Compute-bound** |
| Memory throughput high, SM low | **Memory-bound** — check cache hit rates |
| Low SM throughput, high not_selected | **Occupancy-bound** — increase blocks or reduce regs |
| Low SM throughput, high barrier | **Sync-bound** — reduce __syncthreads frequency |
| High long_scoreboard | **Latency-bound** — increase occupancy or prefetch |
| High math_throttle | **Pipe saturated** — interleave MMA with memory ops |

## Metric Naming Convention

Pattern: `unit__[subunit_][pipestage_]quantity_[qualifiers]`

Examples:
- `sm__inst_executed` — instructions executed in SM
- `l1tex__data_bank_conflicts_pipe_lsu` — shared memory bank conflicts
- `smsp__average_warp_latency` — cycles per instruction between warp issues

## Precision Considerations

Multi-pass collection can introduce discrepancies when:
- Kernel < 20 us duration (doesn't reach steady state)
- Work distribution varies across replay passes
- Hit rates collected separately from miss rates

Increase kernel duration or reduce collected metrics to improve precision.
