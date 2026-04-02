# The Optimization Loop

This is your main operating program. Run this loop until you hit a stop
criterion, then set heartbeat to `complete` and post a halt message to the DB.

## The Loop

```
LOOP:
  1. HEARTBEAT
  2. PROFILE
  3. ANALYZE BOTTLENECK
  4. DESIGN FIX (consult playbook)
  5. IMPLEMENT
  6. EVALUATE
  7. DECIDE (keep or discard)
  8. RECORD (with timestamp)
  9. UPDATE STATE
  10. CHECK STOP CRITERIA
  11. GOTO 1 (or HALT)
```

### Step 1: HEARTBEAT

Report to the factory database that you're alive and what you're working on:
```bash
fb heartbeat <YOUR_KERNEL> --task "exp<N>: <short description>" --job <YOUR_JOB_ID>
```
This is one line. Do it first, every iteration. If your heartbeat goes stale
(>30 min), the factory assumes you crashed.

### Step 2: PROFILE

Use the project's declared evaluation path.

### For kernel-performance projects

Use `eval.sh` and profile with `ncu`:
```bash
./eval.sh --kernel <YOUR_KERNEL> > eval.log 2>&1
```

For fixed-shape kernels and other throughput-driven kernel projects, do NOT
silently skip profiling. Stall data is part of the objective.

### For general-shape library or numerical-method projects

The evaluation path may instead be:
- build
- correctness suite
- coverage suite
- numerical-quality checks
- then benchmark only after hard gates pass

In these modes, `ncu` is useful when kernel performance is the bottleneck, but
it is not mandatory on every iteration.

**Multi-function projects:** eval.sh automatically profiles all functions. To
focus on one op during optimization, use:
```bash
./eval.sh --kernel <YOUR_KERNEL> --func <OP_NAME> > eval.log 2>&1
```

### Step 3: ANALYZE BOTTLENECK

Analyze the current limiting factor.

Before inventing the next experiment, mine your own structured history:
```bash
python3 /data/src/bwk/common/memory/factory_brain.py experiment-summary --kernel <YOUR_KERNEL> --recent 8
```
Use that summary to answer:
- what was the last real keep
- what is the current discard streak
- which stall keeps recurring
- which ideas already failed and should not be retried

- For kernel-performance projects: read the profiler output.
- For library/numerical projects: read the failing tests, coverage gaps,
  residuals, convergence traces, or benchmark cliffs first.

If the problem is a CUDA performance bottleneck, **consult
`docs/sm120_optimization_playbook.md` first.** The playbook is built from 470+
experiments on this exact chip — it tells you what works and what doesn't for
each stall type. Do NOT invent approaches.

**If the playbook doesn't cover your stall or you've exhausted its techniques,**
query the research memory database:
```bash
/data/src/bwk/common/memory/msearch "your bottleneck description" --kernel <your_kernel> --stall <top_stall> -k 3
```
This searches 389 research briefs, reference docs, and source code semantically.
Only use when the playbook falls short — don't query every iteration.

Identify the #1 bottleneck and look it up in the playbook's quick reference:

| Bottleneck | Meaning | Playbook Section |
|------------|---------|-----------------|
| long_scoreboard | Global memory latency | LONG_SCOREBOARD |
| math_throttle | Tensor core pipe saturated | MATH_THROTTLE |
| barrier | __syncthreads overhead | BARRIER |
| wait | Waiting on cp.async or MMA result | WAIT |
| not_selected | Low occupancy | NOT_SELECTED |

**Multi-function projects:** each function gets its own stall profile.

### Step 4: DESIGN FIX

Pick ONE technique from the playbook's "Proven Techniques" table for your
top stall type. Check the "What FAILED" list and "Universal Dead Ends" to
avoid wasting an iteration.

Also consult:
- Your `docs/*_agent_state.md` for what you've already tried
- `factory_brain experiment-summary` for recent keeps/discards and recurring stalls
- `.claude/04_HARD_WON_LESSONS.md` for cross-project constraints
- `/data/src/bwk/common/memory/msearch` if you need a specific technique not in the playbook (e.g.,
  `/data/src/bwk/common/memory/msearch "reduce bank conflicts for FP8 B operand" --kernel gemm --technique swizzle`)

### Step 5: IMPLEMENT

Modify kernel source. **One optimization per iteration.** Keep changes focused.

### Step 6: EVALUATE

Run eval again: `./eval.sh --kernel <YOUR_KERNEL> > eval.log 2>&1`

### Step 7: DECIDE

Do not use a single blind rule for every project.

First apply the project's declared objective profile from:
- `program_<kernel>.md`
- `.claude/03_PROJECT_SPECIFICATION.md`
- `/data/src/bwk/common/docs/factory_objective_profiles.md`

Decision order:
1. Did any hard gate fail?
2. Did the declared primary objective improve?
3. Did a secondary objective regress enough to make the change unacceptable?

Typical examples:
- Fixed-shape kernel: improved weighted latency on the declared shape set and all
  hard gates still pass → **KEEP**
- General-shape library: one size got faster but edge shapes/stride coverage
  regressed → **DISCARD**
- Numerical method: residual or convergence improved while runtime changed only
  modestly → **KEEP**
- Alternative arithmetic: semantics changed or exactness regressed → **DISCARD**

When the project does not define anything better yet, use the fallback rule:
- Improved >2% AND tests pass → **KEEP**
- Within noise (±2%) → **DISCARD**
- Regression → **DISCARD**
- Tests fail → attempt fix (max 3 tries), else discard

### Step 8: RECORD

Record the experiment in `factory_brain`:
```bash
python3 /data/src/bwk/common/memory/factory_brain.py experiment-add \
  --kernel <kernel> \
  --status <keep|discard> \
  --description "<real reason>" \
  --timestamp "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --commit "$(git rev-parse --short HEAD)" \
  --duration-us <us> \
  --vs-ref <ratio> \
  --sm-pct <pct> \
  --stall-math <pct> \
  --stall-wait <pct> \
  --stall-scoreboard <pct> \
  --stall-barrier <pct> \
  --top-stall <name>
```

Use decimal for `vs_ref` (e.g. `1.34`). Status is `keep` or `discard`.
TSV mirrors are deprecated and optional; if they still exist for compatibility,
the DB record is the source of truth.

In `description`, state the real reason:
- `keep: weighted benchmark +3.4%, coverage unchanged`
- `discard: only N=4096 improved, 63/65 regressed`
- `keep: residual 8x lower at same runtime`

**Multi-function projects:** include the op name in the description field
(e.g. `"gemv: vectorized loads, 1.76x cuBLAS"`).

### Step 9: UPDATE STATE

Update `docs/<kernel>_agent_state.md` with:
- Current best (commit, duration, vs reference, top stall)
- Updated diagnosis if bottleneck profile changed
- What you tried and why it worked/didn't

`factory_brain` is the structured source of truth for experiment rows.
`docs/<kernel>_agent_state.md` is the narrative memory that explains them.

**This file is your persistent memory.** If your context clears, the next
instance reads this to pick up where you left off. Write it thoroughly.

**Multi-function projects:** maintain per-op sections in your agent state so
progress on each function is tracked independently.

### Step 10: CHECK STOP CRITERIA

**HALT if ANY of these are true:**

1. **Plateau:** 5 consecutive discards with no improvement in the primary
   objective. You're spinning
   wheels and burning tokens. Write a halt note and stop.

2. **Noise floor:** Your last 3 kept results are all within 2% of each other.
   You've converged. Further iterations are noise.

3. **Exhausted playbook:** You've tried every relevant technique from
   `sm120_optimization_playbook.md` for your top stall type. There's nothing
   left in the vocabulary.

4. **Diminishing returns:** Each kept improvement is <1% for the last 3 keeps
   on the primary objective. The curve has flattened.

5. **Target met:** You've hit or exceeded the acceptance criteria in your
   project specification.

6. **Build/test stuck:** 3 consecutive iterations fail to compile or pass
   tests. Something is fundamentally broken — stop and write a note.

**When you halt:**
- Write a note in `for_foreman-claude/` explaining:
  - What you accomplished (best vs_ref, total kept/discarded)
  - Why you're stopping (which criterion)
  - What directions remain unexplored
  - Your recommendation for next steps
- Update `docs/<kernel>_agent_state.md` with final state
- Touch `.autokernel.<kernel>.halted` so the dashboard can show halt status
- **Do NOT keep iterating hoping something will change.** Tokens are finite.

## Multi-Function Strategy

For projects with N functions, don't optimize one function to perfection before
touching the others. Instead:

1. **First pass:** get all functions correct and benchmarked (baseline vs reference)
2. **Prioritize:** focus on the function with the worst vs_ref or highest impact
3. **Cycle:** when you hit a stop criterion on one function, move to the next
4. **The bar applies per-function:** every op must match or beat its reference
5. **Halt when all functions have hit a stop criterion**

Use `--func` to profile just the function you're working on — it's faster than
profiling all N functions every iteration.

## Token Conservation

**Every iteration costs tokens.** Be deliberate, not exploratory.

- **Read the playbook before coding.** The technique vocabulary exists so you
  don't reinvent approaches that already failed on this chip.
- **Don't retry dead ends.** If it's in the "What FAILED" section or
  "Universal Dead Ends," skip it. No exceptions.
- **Don't keep noise.** ±2% is not signal. Keeping noise pollutes your
  experiment history and wastes the next iteration analyzing a false improvement.
- **One change per iteration.** If you change two things, you don't know which
  one helped. Revert and try them separately.
- **Stop when you converge.** The playbook tells you what the ceiling looks
  like for each kernel type. If you're within 5% of a known ceiling, you're
  done unless you have a genuinely novel idea.
