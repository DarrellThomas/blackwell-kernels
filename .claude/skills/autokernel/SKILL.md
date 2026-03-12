---
name: autokernel
description: This skill should be used when the user asks to "start autokernel", "run autokernel", "resume autokernel", "optimize kernels", or wants to start or resume the profile-driven kernel optimization loop.
disable-model-invocation: true
argument-hint: [tag]
allowed-tools: Read, Write, Edit, Glob, Grep, Bash
---

# autokernel — Profile-Driven Kernel Optimization

Autonomous optimization loop for CUDA kernels on RTX 5090 (sm_120).
Profiles with ncu, identifies bottlenecks, implements targeted fixes, benchmarks, keeps or discards.

## Current State

**Git branch:** !`git branch --show-current`
**Existing autokernel branches:** !`git branch --list 'autokernel/*' | tr -d ' ' || echo "none"`
**Uncommitted changes:** !`git status --short | head -5 || echo "clean"`
**Results file:** !`test -f results.tsv && wc -l < results.tsv | awk '{print $1 - 1 " experiments logged"}' || echo "does not exist"`
**Last result:** !`test -f results.tsv && tail -1 results.tsv || echo "n/a"`
**Dashboard:** !`curl -s -o /dev/null -w "%{http_code}" http://localhost:8420/ 2>/dev/null || echo "not running"`

## Dashboard Auto-Start

If the dashboard status above shows "not running" or anything other than "200", start it before doing anything else:

```bash
nohup python3 dashboard.py > /dev/null 2>&1 &
```

Confirm it's up: `curl -s -o /dev/null -w "%{http_code}" http://localhost:8420/`

Tell the user: "Dashboard started at http://localhost:8420"

## Decision Logic

Based on the state above, determine which scenario applies:

### Scenario A: Fresh Start
**Conditions:** No autokernel branch exists, or user provided a new tag via `$ARGUMENTS`.

Steps:
1. Agree on a tag. If user passed `$ARGUMENTS`, use that. Otherwise propose today's date (e.g., `mar12`).
2. Create branch: `git checkout -b autokernel/<tag>`
3. Read all in-scope files listed in `program.md` § Setup.
4. Initialize `results.tsv` with the header row:
   ```
   commit	duration_us	vs_sdpa	sm_pct	stall_math	stall_wait	stall_scoreboard	stall_barrier	top_stall	status	description
   ```
5. Create `logs/` directory: `mkdir -p logs`
6. Run baseline: `./eval.sh > eval.log 2>&1`
7. Record baseline in `results.tsv`.
8. Confirm setup with user, then begin the optimization loop per `program.md`.

### Scenario B: Resume on Same Branch
**Conditions:** Currently on an `autokernel/*` branch AND `results.tsv` exists with data.

Steps:
1. Read `results.tsv` to understand progress so far — how many iterations, what was kept, what was discarded, what the last experiment was.
2. Read `git log --oneline -10` to see recent commits.
3. Check for uncommitted changes (`git status`). If any:
   - If changes look like an in-progress optimization attempt, ask whether to commit or discard.
   - If changes are just log files, ignore.
4. Run a fresh eval to establish current kernel state:
   ```
   ./eval.sh > eval.log 2>&1
   ```
5. Report to user: "Resuming from iteration N. Last kept result was X us / Y.Yx SDPA. Current bottleneck: Z."
6. Continue the optimization loop per `program.md`.

### Scenario C: Resume on Wrong Branch
**Conditions:** NOT on an `autokernel/*` branch, but autokernel branches exist.

Steps:
1. List existing autokernel branches.
2. Ask user which branch to resume, or whether to start fresh.
3. `git checkout autokernel/<branch>` and proceed per Scenario B.

### Scenario D: Resume with Lost Results
**Conditions:** On an `autokernel/*` branch, but `results.tsv` is missing or empty.

Steps:
1. Reconstruct state from git history:
   ```
   git log --oneline
   ```
2. Initialize `results.tsv` with header row.
3. Run a baseline eval on current HEAD to establish the starting point.
4. Record baseline. Note in description: "resumed — prior results lost".
5. Continue the optimization loop per `program.md`.

## Core References

All detailed instructions for the optimization loop are in `program.md`. Read it fully before beginning.

Key files to read on startup:
- `program.md` — loop instructions, metrics, exit criteria
- `.claude/04_HARD_WON_LESSONS.md` — invariants and guardrails
- `docs/nvidia_blackwell_tuning_guide_sm120.md` — sm_120 hardware limits
- `csrc/attention/flash_attn_v2_sm120.cu` — the kernel being optimized
- `csrc/common/*.cuh` — shared primitives

## Quick Commands

```bash
# Full eval pipeline (build + test + bench + profile)
./eval.sh > eval.log 2>&1
grep -E "^(build|test|bench|profile|primary_|tsv_|top_)" eval.log

# Start dashboard
python3 dashboard.py &

# Check results
cat results.tsv | column -t -s $'\t'
```
