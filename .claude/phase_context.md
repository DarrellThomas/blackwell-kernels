# Phase: Rework

Your job was sent back for rework. Something failed during validation.

## First: Check What Failed

```bash
fb messages --job <your-job-id> --status open
fb job-history <your-job-id>
/data/src/bwk/common/memory/msearch "<kernel> failure root cause" --kernel <kernel> -k 5
```

The watchdog posted a message with the failure details. Read it carefully. Rework now requires a research checkpoint against the DB before you start changing code.

If your own checkpoint is not enough, escalate with a bounded researcher pull:
```bash
fb message-create --from <kernel> --to researcher --job <your-job-id> \
    --subject "Research needed: <failure mode>" \
    --body "Need: <exact question>. Failing case: <shape/input>. Constraints: <ABI/hardware/accuracy>. Deliverable: <what answer would unblock you>." \
    --type research_request
```

## Common Failure Modes

### Edge Test Failure
Your kernel crashes or gives wrong answers on non-standard inputs. Check:
- Non-tile-aligned sizes (63, 65, 127, 129 — the sizes between your tile boundaries)
- Degenerate inputs (0x0, 1x1, Mx1, 1xN)
- Sub-matrix access with lda > M (this is how LAPACK calls you)
- Alpha/beta scaling edge cases (alpha=0 must skip A*B, beta=0 must skip reading C)

### Lint Failure
Code quality issues. Run the linter to see what's wrong:
```bash
python3 /data/src/bwk/common/scripts/lint_cuda.py csrc/<kernel>/<kernel>_sm120.cu
```

### Test Failure
Basic correctness failure. Your kernel produces wrong numerical results.
Compare against reference at failing sizes:
```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=python python3 tests/test_<kernel>.py -v
```

## Fix Protocol

1. Read the failure message — understand exactly what failed
2. Write a failing test that reproduces the issue
3. Fix the kernel code
4. Run the full compliance test:
   ```bash
   CUDA_VISIBLE_DEVICES=1 python3 /data/src/bwk/common/scripts/blas_compliance.py \
       python/blackwell_kernels/<module>.py <function> --op <op>
   ```
5. Verify ALL tests pass (not just the one that failed)
6. Record the result in factory_brain: `fb experiment-add --kernel <kernel> --job <id> --status keep --description "rework: fixed <what>" ...`

## When Fixed

Set job state to signal rework is complete:
```bash
fb job-update <id> --state rework_complete --by <kernel> --reason "fixed: <what>"
```

The watchdog will re-run validation automatically.

## If You Can't Fix It

If the failure is fundamental (design limitation, not a bug):
1. Document the limitation in `docs/<kernel>_agent_state.md`
2. Post a message: `fb message-create --from <kernel> --subject "Rework blocked: <why>" --type blocker`
3. Do NOT mark rework_complete if the issue persists
