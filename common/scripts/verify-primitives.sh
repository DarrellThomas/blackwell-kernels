#!/bin/bash
# verify-primitives.sh — Check shipped primitives against the database manifest
#
# The database (research.db) is the source of truth for what's shipped.
# This script queries the DB — no JSON files, no guessing.
#
# Modes:
#   ./verify-primitives.sh                    # verify shelf matches DB manifest
#   ./verify-primitives.sh --check-wt         # compare shelf to worktrees (informational)
#   ./verify-primitives.sh --ship FILE [DIR]  # ship a file to the shelf (updates DB)
#   ./verify-primitives.sh --list             # show all shipped primitives from DB
#
# No --fix flag. Shipping is always explicit via --ship.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
COMMON_DIR="$REPO_ROOT/common"
export TRANSFORMERS_NO_TF=1
export TF_CPP_MIN_LOG_LEVEL=3

MEMORY="$COMMON_DIR/memory"
MODE="${1:-check}"

case "$MODE" in

check|"")
    python3 -c "
import sys; sys.path.insert(0, '$MEMORY')
from factory_brain import ResearchMemory
mem = ResearchMemory()
results = mem.verify_shelf()
ok = sum(1 for r in results if r['status'] == 'ok')
modified = sum(1 for r in results if r['status'] == 'modified')
untracked = sum(1 for r in results if r['status'] == 'untracked')
print('=== Primitives Shelf Verification (from DB) ===')
print()
for r in results:
    if r['status'] == 'ok':
        print(f'  OK         {r[\"path\"]:<40s}  v{r[\"version\"]}  hash={r[\"hash\"]}')
    elif r['status'] == 'modified':
        print(f'  MODIFIED   {r[\"path\"]:<40s}  v{r[\"version\"]}  shelf={r[\"file_hash\"]}  db={r[\"manifest_hash\"]}')
    else:
        print(f'  UNTRACKED  {r[\"path\"]:<40s}  hash={r[\"file_hash\"]}')
print()
print(f'Summary: {ok} ok, {modified} modified, {untracked} untracked')
if untracked:
    print(f'  Ship untracked files with: verify-primitives.sh --ship /path/to/file.cu [subdir]')
mem.close()
" 2>/dev/null
    ;;

--check-wt)
    echo "=== Worktree vs Shelf Comparison ==="
    SHELF="$COMMON_DIR/csrc/primitives"
    for f in "$SHELF"/gemm/*.cu "$SHELF"/linalg/*.cu "$SHELF"/qr/*.cu "$SHELF"/spmv/*.cu; do
        [ -f "$f" ] || continue
        name=$(basename "$f")
        shelf_hash=$(sha256sum "$f" | cut -c1-16)
        for wt in $REPO_ROOT/linalg/csrc $REPO_ROOT/gemm/csrc $REPO_ROOT/qr/csrc $REPO_ROOT/spmv/csrc; do
            match=$(find "$wt" -name "$name" 2>/dev/null | head -1)
            if [ -n "$match" ] && [ "$(sha256sum "$match" | cut -c1-16)" != "$shelf_hash" ]; then
                printf "  DIFFERS  %-35s  ← %s\n" "$name" "${match#${REPO_ROOT}/}"
                break
            fi
        done
    done
    ;;

--ship)
    SOURCE="${2:?Usage: verify-primitives.sh --ship FILE [subdir]}"
    SUBDIR="${3:-linalg}"
    python3 -c "
import sys; sys.path.insert(0, '$MEMORY')
from factory_brain import ResearchMemory
mem = ResearchMemory()
result = mem.ship_primitive('$SOURCE', '$SUBDIR', shipped_by='ops')
if result['action'] == 'unchanged':
    print(f'UNCHANGED: {result[\"shelf_path\"]} already at v{result[\"version\"]} (hash matches)')
else:
    blas = '✓' if result.get('has_blas_interface') else '✗'
    print(f'SHIPPED: {result[\"shelf_path\"]}')
    print(f'  Version: v{result[\"version\"]}')
    print(f'  Hash: {result[\"hash\"]}')
    print(f'  BLAS interface: {blas}')
    print(f'  Source: $SOURCE')
mem.close()
" 2>/dev/null
    ;;

--list)
    python3 -c "
import sys; sys.path.insert(0, '$MEMORY')
from factory_brain import ResearchMemory
mem = ResearchMemory()
prims = mem.get_primitives()
if not prims:
    print('No primitives registered. Ship files with --ship.')
else:
    print(f'{\"PATH\":<40s} {\"VER\":>4s} {\"HASH\":<18s} {\"BLAS\":>4s} {\"SHIPPED\":<20s} FROM')
    print('-' * 120)
    for p in prims:
        blas = '✓' if p['has_blas_interface'] else '✗'
        shipped = p['shipped_at'][:19] if p['shipped_at'] else '-'
        src = p.get('shipped_from', '-')
        if src and len(src) > 40:
            src = '...' + src[-37:]
        print(f'{p[\"shelf_path\"]:<40s} v{p[\"version\"]:>3d} {p[\"content_hash\"]:<18s} {blas:>4s} {shipped:<20s} {src}')
mem.close()
" 2>/dev/null
    ;;

*)
    echo "Usage:"
    echo "  verify-primitives.sh              # verify shelf matches DB"
    echo "  verify-primitives.sh --check-wt   # compare to worktrees"
    echo "  verify-primitives.sh --ship FILE  # ship a file (updates DB)"
    echo "  verify-primitives.sh --list       # list all primitives from DB"
    ;;
esac
