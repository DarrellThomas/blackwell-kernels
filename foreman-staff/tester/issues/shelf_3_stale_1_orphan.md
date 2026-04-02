# Primitives shelf: 3 stale files + 1 orphan

## What's broken

The primitives shelf (`common/csrc/primitives/`) is out of sync with worktree sources.

### Stale (worktree has newer version)

| File | Shelf hash | Worktree hash | Source |
|------|-----------|---------------|--------|
| batched_gemm_sm120.cu | aae0a911 | f982ac1e | linalg/csrc/linalg/ |
| gemm_f32_sm120.cu | 0266d7c8 | a460d3d6 | linalg/csrc/linalg/ |
| syrk_f32_sm120.cu | 009e7bb3 | 926a0b73 | linalg/csrc/linalg/ |

### Orphan (no worktree source found)

| File | Shelf hash | Notes |
|------|-----------|-------|
| trmm_f32_sm120.cu | 6d48c69c | No matching source in gemm/ or linalg/ csrc dirs |

## Severity

**Infrastructure** — Consumers building against `common/csrc/primitives/` are
using stale kernel source. The orphaned `trmm_f32_sm120.cu` may have been
renamed or removed in the worktree.

## Action needed

1. Reship the 3 stale files: `verify-primitives.sh --fix`
2. Investigate trmm_f32_sm120.cu orphan — was it renamed or intentionally removed?
