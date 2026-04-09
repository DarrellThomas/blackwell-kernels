# Phase: Shipping

Your kernel passed all quality gates. It's being shipped to the primitives shelf.

## What Happens Automatically

The watchdog handles shelf shipping only after the work has been closed out cleanly:
1. Source work is committed in the project repo
2. The job/result version is bumped
3. The shelf copy is updated from the committed source
4. The primitives manifest in the DB is updated
5. The job transitions to `shipped`

## Source-Control Closure

A job is not truly shipped until source control is closed out. Before shipping:
1. Run `git status --short` in the project repo
2. Stage only the files that belong to the job
3. Commit them with a job-specific message
4. Record the resulting commit SHA in the DB-facing notes/experiment trail
5. If the worktree is too dirty to do this safely, stop and post a blocker/info message instead of pretending the job is done

A shipped job should have a reproducible commit, not just a dirty worktree plus a green DB row.

## Version Bumps

If this kernel was **previously shipped** and is coming back through after a
compliance pass or rework, it gets a **version bump**. The primitives DB
tracks version numbers — each reship increments the version so consumers
know they need to update. Check the current version before shipping:
```bash
fb primitives --kernel <kernel>
```

## Your Responsibility

Before the watchdog ships, verify:

### BLAS Signature Compliance
Every shipped primitive MUST support the full BLAS signature:
```
C = alpha * op(A) * op(B) + beta * C   with lda, ldb, ldc strides
```

Without `alpha`, `beta`, `lda`, `ldc`, the kernel cannot operate on sub-matrices —
which is what EVERY factorization algorithm and EVERY Octave caller does.

### Post-Ship Verification
After the watchdog ships, verify the shelf copy matches your source:
```bash
diff csrc/<kernel>/<kernel>_sm120.cu /data/src/bwk/common/csrc/primitives/<kernel>/<kernel>_sm120.cu
```

If they differ, the ship failed silently. Post a blocker message.

### Consumer Notification
If other projects depend on your kernel (check the DB):
```bash
msearch "uses <kernel>" --kernel all -k 5
```

Post an info message so consumers know to update:
```bash
fb message-create --from <kernel> --subject "Shipped updated <kernel> to shelf" --type info
```

## After Shipping

Your job is complete. Update your agent state with final metrics:
```bash
# Final state in agent_state.md
docs/<kernel>_agent_state.md
```

The job will be marked `shipped` (terminal state). Well done.
