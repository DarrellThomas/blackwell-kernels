#!/usr/bin/env bash
# Simple replacement for the broken apply_patch helper.
# Reads a unified diff from stdin and applies it with patch -p0.
set -euo pipefail
TMP=$(mktemp)
trap 'rm -f "$TMP"' EXIT
cat > "$TMP"
patch -p0 --no-backup-if-mismatch < "$TMP"
