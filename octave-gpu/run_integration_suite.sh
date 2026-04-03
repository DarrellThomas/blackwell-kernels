#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROFILE="${BWK_INTEGRATION_PROFILE:-standard}"
RUN_BENCH=0

while (($#)); do
  case "$1" in
    --profile)
      PROFILE="${2:?missing profile}"
      shift 2
      ;;
    --bench)
      RUN_BENCH=1
      shift
      ;;
    *)
      echo "usage: $0 [--profile smoke|standard|heavy] [--bench]" >&2
      exit 2
      ;;
  esac
done

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13}"

OCTAVE_BIN="${OCTAVE_BIN:-octave-cli}"
LOG_DIR="$ROOT/results/integration"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$LOG_DIR"

PRELOAD_SO="$ROOT/lib/libbwk_blas.so"
PLUGIN_OCT="$ROOT/plugin/gpu_create.oct"

if ! command -v "$OCTAVE_BIN" >/dev/null 2>&1; then
  echo "missing octave-cli: set OCTAVE_BIN or install Octave" >&2
  exit 1
fi

echo "==> Building gpu_create.oct"
make -C "$ROOT/plugin" all

echo "==> Building libbwk_blas.so"
make -C "$ROOT/lib" all

echo "==> dnrm2_abi"
make -C "$ROOT/lib" test-dnrm2-abi | tee "$LOG_DIR/${STAMP}-dnrm2_abi.log"

echo "==> dgeqrf_abi"
make -C "$ROOT/lib" test-dgeqrf-abi | tee "$LOG_DIR/${STAMP}-dgeqrf_abi.log"

echo "==> dpotrs_abi"
make -C "$ROOT/lib" test-dpotrs-abi | tee "$LOG_DIR/${STAMP}-dpotrs_abi.log"

echo "==> dpotrs_octave"
make -C "$ROOT/lib" test-dpotrs-octave | tee "$LOG_DIR/${STAMP}-dpotrs_octave.log"

echo "==> dgetrs_abi"
make -C "$ROOT/lib" test-dgetrs-abi | tee "$LOG_DIR/${STAMP}-dgetrs_abi.log"

echo "==> dgetrs_octave"
make -C "$ROOT/lib" test-dgetrs-octave | tee "$LOG_DIR/${STAMP}-dgetrs_octave.log"

run_octave() {
  local name="$1"
  shift
  local log_file="$LOG_DIR/${STAMP}-${name}.log"
  echo "==> ${name}"
  (
    cd "$ROOT"
    "$@"
  ) | tee "$log_file"
}

run_octave linear_solve \
  python3 "$ROOT/tests/test_linear_solve.py"

run_octave norm \
  python3 "$ROOT/tests/test_norm.py"

run_octave preload \
  env LD_PRELOAD="$PRELOAD_SO" "$OCTAVE_BIN" --quiet \
  "$ROOT/benchmarks/test_preload_integration.m"

run_octave accuracy \
  "$OCTAVE_BIN" --quiet --path "$ROOT/plugin" --path "$ROOT/plugin/inst" \
  "$ROOT/benchmarks/accuracy_audit.m"

run_octave stress \
  "$OCTAVE_BIN" --quiet --path "$ROOT/plugin" --path "$ROOT/plugin/inst" \
  "$ROOT/benchmarks/stress_plugin.m"

run_octave stress_enhanced \
  env BWK_STRESS_PROFILE="$PROFILE" "$OCTAVE_BIN" --quiet \
  --path "$ROOT/plugin" --path "$ROOT/plugin/inst" \
  "$ROOT/benchmarks/stress_plugin_enhanced.m"

if [[ "$RUN_BENCH" -eq 1 ]]; then
  run_octave bench_plugin \
    "$OCTAVE_BIN" --quiet --path "$ROOT/plugin" --path "$ROOT/plugin/inst" \
    "$ROOT/benchmarks/bench_plugin.m"
fi

echo "==> Integration suite complete"
echo "Logs: $LOG_DIR/${STAMP}-*.log"
