# Worker Communication — Factory Database

All coordination happens through the factory database. No markdown notes, no file
passing. Every agent — worker, foreman, ops, tester — reads and writes the same DB.

**Database location:** `/data/src/bwk/common/memory/research.db`
**CLI tool:** `python3 /data/src/bwk/common/memory/factory_brain.py <command>`
**Shortcut:** `fb <command>` (same thing)
**HTTP API:** `http://localhost:8421/api/...` (when memory server is running)

## Heartbeat — REQUIRED Every Iteration

**At the start of every optimization iteration, report your heartbeat:**

```bash
fb heartbeat <your-kernel-type> --task "short description of what you're doing" --job <your-job-id>
```

Examples:
```bash
fb heartbeat lu --task "exp31: testing NB=64 tiles" --job 1
fb heartbeat qr --task "profiling DSYRK integration" --job 2
```

**When you finish your work (met target or exhausted approaches):**
```bash
fb heartbeat <kernel> --task "met target 1.09x" --state complete --job <id>
```

This is how the factory knows you're alive. If your heartbeat goes stale (>30 min),
the system assumes you crashed and will restart you. It costs one line of bash per
iteration — do it first, before profiling or benchmarking.

**The heartbeat replaces the old `.alive` heartbeat files.** Do not touch `.alive` files.

## You Run 24/7

**You never stop unless foreman-claude or Darrell explicitly tells you to stop.**
Do not ask for permission to continue. Do not offer to "pause for the night."
Your job is continuous optimization.

If you hit a wall:
1. Document the dead end in your `agent_state.md`
2. Post a message to the DB if it's outside your scope (see below)
3. **Try a different approach and keep going**

**The bar: match or beat the reference on every metric.** This is an RTX 5090 —
170 SMs, 1792 GB/s bandwidth, 209.5 TFLOP/s peak. If you're below 1.0x on
anything, that's unfinished work, not a completed project.

## Experiment Tracking

**Every experiment must be recorded in `factory_brain`** — this is now the
source of truth. If you benchmark something, write a DB row:
```bash
python3 /data/src/bwk/common/memory/factory_brain.py experiment-add \
  --kernel <kernel> \
  --status <keep|discard> \
  --description "what changed and why" \
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
TSV files are deprecated compatibility artifacts only. If a TSV mirror exists,
it may be imported, but the database row is authoritative. **No DB row = invisible.**

## The Andon Cord — Database Messages

When something outside your scope is blocking you, **post a message to the database**.
This replaces the old `for_foreman-claude/` folder system. Messages are visible to
every agent immediately — no waiting for a patrol cycle.

### How to Post a Message

```bash
python3 /data/src/bwk/common/memory/factory_brain.py message-create \
    --from <your-kernel-type> \
    --subject "Short description of the issue" \
    --body "Details: what's wrong, impact, suggested fix" \
    --type <halt|blocker|question|info|research_request> \
    --priority <urgent|normal|low> \
    --job <job-id>  # optional: link to your job
```

**Message types:**
- `halt` — you're stopping because work is exhausted or blocked
- `blocker` — you can't proceed without help (missing primitive, broken tool)
- `question` — need clarification on spec or scope
- `info` — FYI (dead end worth sharing, cross-project finding)
- `research_request` — bounded request to the researcher; use this when you need targeted literature, prior factory history, or reference guidance before continuing

### Requesting Research

Use `research_request` when you need the researcher. Keep it bounded and job-linked:

```bash
fb message-create --from <kernel> --to researcher --job <id> \
    --subject "Research needed: <short topic>" \
    --body "Need: <exact question>. Constraints: <scope/shape/hardware>. Why blocked: <reason>. Deliverable: <what you need back>." \
    --type research_request --priority normal
```

Rules:
- Ask one bounded question per message.
- Tie the request to your current job when possible.
- Use `research_request` for pull-based researcher help, not `question` or `info`.
- When the answer lands, incorporate it into the work and resolve the loop.

### Before You Stop

**BEFORE YOU STOP FOR ANY REASON, do both of these:**

1. **Set heartbeat to complete:**
```bash
fb heartbeat <kernel> --task "reason for stopping" --state complete --job <id>
```

2. **Post a halt message with details:**
```bash
fb message-create --from <kernel> --subject "Halt: <reason>" \
    --body "Accomplished: <what>. Stopping because: <why>. Remaining: <what's left>" \
    --type halt --priority normal
```

### What to Post About

- **Missing infrastructure:** eval.sh, profile scripts, build system
- **Architecture concerns:** shared code changes, cross-project patterns
- **Specification questions:** ambiguity, conflicting guidance
- **Dead ends worth sharing:** approaches that fundamentally can't work
- **Tool/environment issues:** build failures, GPU contention

### Checking Messages

```bash
# See all open messages
python3 /data/src/bwk/common/memory/factory_brain.py messages --status open

# See messages for your job
python3 /data/src/bwk/common/memory/factory_brain.py messages --job <id>
```

## Checking Your Job State

Every workpiece has a job record in the database. Check your current state:

```bash
# List all jobs (see where yours is)
python3 /data/src/bwk/common/memory/factory_brain.py jobs

# See your job's full history
python3 /data/src/bwk/common/memory/factory_brain.py job-history <id>
```

Your job has a **state** (e.g., `hw_optimizing`, `testing`, `shipped`) and a
**phase** (ideation, development, validation, rework, quality, shipping, terminal).
The foreman or ops updates your job state as work progresses.

## Checking Your Spec

Your job record may include a `spec` field with your detailed specification.
Query it via the HTTP API or ask the foreman for your current spec.

## Research Memory Search

The database also indexes all research briefs, reference docs, playbooks, and
kernel source code. Use it when the playbook doesn't have what you need.

```bash
# Semantic search
/data/src/bwk/common/memory/msearch "reduce shared memory bank conflicts" --kernel gemm

# Full-text search (exact keywords)
/data/src/bwk/common/memory/msearch --fts "ldmatrix_x4" -k 3

# Filter by stall type or technique
/data/src/bwk/common/memory/msearch "improve occupancy" --stall not_selected
```

**When to use it:**
- The playbook doesn't cover your specific stall/technique combination
- You want to see how another project solved a similar problem
- You need a specific PTX instruction or code pattern

**If msearch says "server not running"**, post a `blocker` message — the foreman
or ops will restart it.

## Your Authority

**You own your project.** You have full authority to:
- Build tests, diagnostics, tools, and harnesses you need
- Try novel approaches, unconventional architectures, radical experiments
- Change your kernel API if it enables better performance
- Add new files, headers, test cases — whatever advances the objective

If you're pausing to ask "should I try X?" — just try it. Document what you learn.

## The Principle

> Any worker can stop the line. If the architecture isn't right, that's a
> foreman problem, not a worker problem. Surface it through the database.
