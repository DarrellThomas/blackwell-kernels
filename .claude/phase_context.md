# Phase: Development

You are in the **development phase** — building and optimizing your kernel.

## First: Read Your Job Spec

Your job has a spec in the factory database. Read it before writing any code:
```bash
fb job-show <your-job-id>    # shows title, state, spec, and all job details
```
The spec tells you exactly what to implement, where, and how to test it.

If the spec conflicts with local markdown, the spec wins.

## Choose The Right Workflow

Not every development-phase job is an autokernel optimization loop.

### A. Kernel optimization jobs
Use the loop below. These jobs usually edit `csrc/...`, record experiment rows in
`factory_brain`, and run
`./eval.sh`.

### B. Bounded implementation / wiring jobs
Examples: BLAS/LAPACK exports in `libbwk_blas.so`, Octave plugin functions,
adding one missing operation, extending Python bindings, fixing stride/flag
handling, or writing targeted correctness tests.

For bounded jobs, do this instead:
1. Heartbeat immediately before substantial exploration: `fb heartbeat <worker> --job <id> --state working --task "what you are inspecting or implementing"`.
   Assume zero prior model context. Reconstruct the job only from `factory_brain`, the job spec, open messages, recent experiment history, and the local files named by the spec.
2. Read `fb job-show <id>` and identify the exact files to touch.
3. Find the closest working analogue in this repo and copy its structure.
4. Make the minimum change set that completes this one function/job.
5. Build with the project’s real command (`setup.py build_ext --inplace`, `make -C lib/`, etc.).
6. Run targeted tests first, then the project’s broader test entry point if it exists.
7. If the gate depends on `.claude/compliance_args.txt`, update it as part of the job.
8. Do not invent a benchmarking loop unless the spec explicitly requires one.
9. Refresh heartbeat during long work and before/after major test runs.
10. When the acceptance criteria are met, move the job to `testing_pass`.

## The Loop

```
LOOP:
  1. HEARTBEAT — fb heartbeat <kernel> --task "exp<N>: <desc>" --job <id>
  2. PROFILE   — ./eval.sh --kernel <kernel> > eval.log 2>&1
  3. ANALYZE   — inspect `experiment-summary`, then identify #1 bottleneck
  4. DESIGN    — pick ONE technique from playbook or /data/src/bwk/common/memory/msearch
  5. IMPLEMENT — one optimization per iteration
  6. EVALUATE  — ./eval.sh again
  7. DECIDE    — >2% improvement = keep; ±2% = discard (noise)
  8. RECORD    — DB experiment row with all metrics
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
/data/src/bwk/common/memory/msearch "your bottleneck" --kernel <kernel> --stall <top_stall> -k 3
```

Before choosing the next iteration, always inspect your recent structured
history:
```bash
python3 /data/src/bwk/common/memory/factory_brain.py experiment-summary --kernel <kernel> --recent 8
```
That is where you recover:
- what was kept vs discarded
- whether you are in a discard streak / plateau
- which stalls keep recurring
- what descriptions explain prior failures and successes

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
2. Post bounded research request: `fb message-create --from <kernel> --to researcher --job <id> --subject "Research needed: <what>" --body "Need: <exact question>. Constraints: <scope>. Why blocked: <reason>. Deliverable: <what you need back>." --type research_request`
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

## Gate Reality

- The watchdog only validates what the project exposes through tests and, when
  present, `.claude/compliance_args.txt`.
- Some backlog LAPACK and Octave integration jobs are not covered by
  `blas_compliance.py`. In those cases, add or extend focused project tests so
  the gate has something real to run.
- If your job spans two trees (for example an owning kernel project plus
  `octave-gpu/` integration), make sure the owning project still has the tests
  and compliance surface needed for the gate.
