# Phase: Development

You are in the **development phase** — building and optimizing your kernel.
Your job is the optimization loop: profile, analyze, fix, evaluate, record.

## The Loop

```
LOOP:
  1. HEARTBEAT — fb heartbeat <kernel> --task "exp<N>: <desc>" --job <id>
  2. PROFILE   — ./eval.sh --kernel <kernel> > eval.log 2>&1
  3. ANALYZE   — identify #1 bottleneck from ncu stall data
  4. DESIGN    — pick ONE technique from playbook or msearch
  5. IMPLEMENT — one optimization per iteration
  6. EVALUATE  — ./eval.sh again
  7. DECIDE    — >2% improvement = keep; ±2% = discard (noise)
  8. RECORD    — TSV row with all metrics
  9. UPDATE    — docs/<kernel>_agent_state.md (your persistent memory)
  10. CHECK     — stop criteria below
  11. GOTO 1 (or HALT)
```

## Bottleneck Quick Reference

| Stall | Meaning | First Try |
|-------|---------|-----------|
| long_scoreboard | Global memory latency | cp.async prefetch, double-buffer |
| math_throttle | Tensor core pipe saturated | Increase occupancy (smaller tiles, fewer regs) |
| barrier | __syncthreads overhead | Reduce sync points, async alternatives |
| wait | Waiting on cp.async/MMA | Better overlap, pipeline stages |
| not_selected | Low occupancy | __launch_bounds__, reduce registers |

When the playbook doesn't cover your stall, query the research DB:
```bash
msearch "your bottleneck" --kernel <kernel> --stall <top_stall> -k 3
```

## Key Hardware Rules (sm_120a)

- **99 KB max shared memory** per block (CUDA reserves 1 KB). Target 96 KB.
- **48 warps/SM** (not 64). Occupancy is the dominant axis for compute-bound kernels.
- **a1/a2 register swap required** for ldmatrix_x4 → mma.sync. Use `ldmatrix_x4_mma()`.
- **<2% is noise.** Don't keep or discard on sub-2% changes. Run 3 trials for 2-5%.
- **Double-buffer is the sweet spot.** Triple-buffer kills L1 on sm_120.
- **Non-volatile MMA always.** `asm` not `asm volatile` for mma.sync.

## Stop Criteria — HALT if ANY are true

1. **Plateau:** 5 consecutive discards
2. **Noise floor:** Last 3 keeps within 2% of each other
3. **Exhausted playbook:** Every relevant technique tried
4. **Diminishing returns:** Last 3 keeps each <1% improvement
5. **Target met:** Hit acceptance criteria in project spec
6. **Build stuck:** 3 consecutive iterations fail to compile/test

**When you halt:**
1. Set heartbeat complete: `fb heartbeat <kernel> --task "reason" --state complete --job <id>`
2. Post halt message: `fb message-create --from <kernel> --subject "Halt: <reason>" --type halt`
3. Update `docs/<kernel>_agent_state.md` with final state

**If you're stuck (not halted — just blocked):**
1. Set job state: `fb job-update <id> --state stuck_needs_research --by <kernel> --reason "what you need"`
2. Post question: `fb message-create --from <kernel> --subject "Need: <what>" --type question`
3. The researcher will be kicked automatically. Resume when you see `research_available`.

## Dead Ends — Do NOT Retry

- Full fusion when intermediates exceed tile size (7-51x slowdown)
- 3-stage pipeline on sm_120 (kills L1 cache)
- Manual PTX scheduling (compiler already optimal)
- Monolithic asm blocks with >50 output operands (silent misassignment)
- TF32 MMA for general GEMM (diagonal broadcast defect on sm_120)

## Pre-Flight Checklist (BEFORE every experiment)

1. Will the new registers fit within `__launch_bounds__`?
2. Will total smem (with double-buffer) stay ≤96 KB?
3. Does the kernel have irreducible sequential phases? If yes, occupancy-first won't work.
4. Is the expected gain >2%? If not, don't bother.
