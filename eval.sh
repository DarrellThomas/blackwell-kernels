#!/bin/bash
# Copyright (c) 2026 Darrell Thomas. MIT License. See LICENSE file.
#
# Autokernel evaluation script: build → test → benchmark → profile
#
# Usage:
#   ./eval.sh                        # attention kernel, full pipeline
#   ./eval.sh --kernel gemm          # GEMM kernel, full pipeline
#   ./eval.sh --kernel attention --quick    # attention, skip profiling
#   ./eval.sh --kernel gemm --profile      # GEMM, profile only
#
# All output uses machine-parseable "key: value" lines for the agent to grep.
# Exit codes: 0 = success, 1 = build fail, 2 = test fail, 3 = bench fail, 4 = profile fail

set -euo pipefail
cd "$(dirname "$0")"

# ─── ARGUMENT PARSING ─────────────────────────────────────────────────────────
KERNEL="attention"
MODE="full"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --kernel)   KERNEL="$2"; shift 2 ;;
        --quick)    MODE="quick"; shift ;;
        --profile)  MODE="profile"; shift ;;
        *)          echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ─── PER-KERNEL CONFIG ────────────────────────────────────────────────────────
case "$KERNEL" in
    attention)
        TEST_SCRIPT="tests/test_attention.py"
        BENCH_SCRIPT="benchmarks/bench_attention.py"
        PROFILE_SCRIPT="profiles/profile_v2.py"
        NCU_KERNEL_NAME="flash_attn_v2_kernel"
        ;;
    gemm)
        TEST_SCRIPT="tests/test_gemm.py"
        BENCH_SCRIPT="benchmarks/bench_gemm.py"
        PROFILE_SCRIPT="profiles/profile_gemm.py"
        NCU_KERNEL_NAME="bf16_gemm_kernel"
        ;;
    *)
        echo "Unknown kernel: $KERNEL"
        echo "Available kernels: attention, gemm"
        exit 1
        ;;
esac

RESULTS_DIR="results"
LOG_DIR="logs/${KERNEL}"
mkdir -p "$RESULTS_DIR" "$LOG_DIR"

export CUDA_VISIBLE_DEVICES=1
export CUDA_HOME=/usr/local/cuda-13
export PYTHONPATH=python

NCU="$CUDA_HOME/bin/ncu"

echo "=== KERNEL: $KERNEL ==="

# Key ncu metrics for bottleneck identification
NCU_METRICS="gpu__time_duration.sum"
NCU_METRICS+=",sm__throughput.avg.pct_of_peak_sustained_elapsed"
NCU_METRICS+=",sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed"
NCU_METRICS+=",smsp__warp_issue_stalled_long_scoreboard_per_warp_active.ratio"
NCU_METRICS+=",smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.ratio"
NCU_METRICS+=",smsp__warp_issue_stalled_wait_per_warp_active.ratio"
NCU_METRICS+=",smsp__warp_issue_stalled_barrier_per_warp_active.ratio"
NCU_METRICS+=",smsp__warp_issue_stalled_short_scoreboard_per_warp_active.ratio"
NCU_METRICS+=",smsp__warp_issue_stalled_lg_throttle_per_warp_active.ratio"
NCU_METRICS+=",smsp__warp_issue_stalled_not_selected_per_warp_active.ratio"
NCU_METRICS+=",l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum"
NCU_METRICS+=",l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum"

# ─── BUILD ─────────────────────────────────────────────────────────────────────
if [[ "$MODE" != "profile" ]]; then
    echo "=== BUILD ==="
    if python3 setup.py build_ext --inplace > build.log 2>&1; then
        echo "build: PASS"
    else
        echo "build: FAIL"
        echo "--- build.log tail ---"
        tail -20 build.log
        exit 1
    fi

    # ─── TEST ──────────────────────────────────────────────────────────────────
    echo "=== TEST ==="
    if python3 "$TEST_SCRIPT" > test.log 2>&1; then
        echo "test: PASS"
    else
        echo "test: FAIL"
        echo "--- test.log tail ---"
        tail -30 test.log
        exit 2
    fi

    # ─── BENCHMARK ─────────────────────────────────────────────────────────────
    echo "=== BENCHMARK ==="
    if python3 "$BENCH_SCRIPT" > bench.log 2>&1; then
        echo "bench: PASS"
        cat bench.log
        echo ""
        # Parse generic primary metrics (all bench scripts emit these)
        CUSTOM_MS=$(grep "^primary_custom_ms:" bench.log | awk '{print $2}' || true)
        VS_REF=$(grep "^primary_vs_ref:" bench.log | awk '{print $2}' | tr -d 'x' || true)
        if [[ -n "$CUSTOM_MS" ]]; then
            echo "primary_custom_ms: $CUSTOM_MS"
        fi
        if [[ -n "$VS_REF" ]]; then
            echo "primary_vs_ref: ${VS_REF}x"
        fi
    else
        echo "bench: FAIL"
        echo "--- bench.log tail ---"
        tail -20 bench.log
        exit 3
    fi
fi

# ─── PROFILE ───────────────────────────────────────────────────────────────────
if [[ "$MODE" != "quick" ]]; then
    echo "=== PROFILE ==="
    echo "(requires sudo for ncu GPU performance counters)"

    if sudo -E CUDA_VISIBLE_DEVICES=1 "$NCU" \
        --metrics "$NCU_METRICS" \
        --csv \
        --kernel-name "$NCU_KERNEL_NAME" \
        --launch-count 1 \
        python3 "$PROFILE_SCRIPT" > profile.log 2>&1; then

        echo "profile: PASS"
        echo ""

        # Parse ncu CSV using Python (handles quoted fields with embedded commas)
        python3 -c "
import csv, sys

metrics = {}
with open('profile.log') as f:
    for line in f:
        if line.startswith('\"') and not line.startswith('\"ID\"'):
            reader = csv.reader([line])
            for row in reader:
                if len(row) >= 15:
                    mname, munit, mval = row[12], row[13], row[14]
                    metrics[mname] = mval

# Map to friendly names and print
name_map = {
    'gpu__time_duration.sum': ('ncu_duration_ns', None),
    'sm__throughput.avg.pct_of_peak_sustained_elapsed': ('ncu_sm_throughput_pct', None),
    'sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed': ('ncu_tensor_pipe_pct', None),
    'smsp__warp_issue_stalled_long_scoreboard_per_warp_active.ratio': ('ncu_stall_long_scoreboard', None),
    'smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.ratio': ('ncu_stall_math_throttle', None),
    'smsp__warp_issue_stalled_wait_per_warp_active.ratio': ('ncu_stall_wait', None),
    'smsp__warp_issue_stalled_barrier_per_warp_active.ratio': ('ncu_stall_barrier', None),
    'smsp__warp_issue_stalled_short_scoreboard_per_warp_active.ratio': ('ncu_stall_short_scoreboard', None),
    'smsp__warp_issue_stalled_lg_throttle_per_warp_active.ratio': ('ncu_stall_lg_throttle', None),
    'smsp__warp_issue_stalled_not_selected_per_warp_active.ratio': ('ncu_stall_not_selected', None),
    'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum': ('ncu_bank_conflicts_store', None),
    'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum': ('ncu_bank_conflicts_load', None),
}

for mname, (friendly, _) in name_map.items():
    if mname in metrics:
        val = metrics[mname].replace(',', '')
        if friendly == 'ncu_duration_ns':
            # Also print in microseconds
            try:
                us = float(val) / 1000
                print(f'ncu_duration_us: {us:.1f}')
            except ValueError:
                print(f'ncu_duration_us: {val}')
        else:
            print(f'{friendly}: {val}')

# Bottleneck analysis: find top stall
print()
print('=== BOTTLENECK ANALYSIS ===')
stalls = {}
for mname, (friendly, _) in name_map.items():
    if 'stall_' in friendly and mname in metrics:
        try:
            stalls[friendly] = float(metrics[mname].replace(',', ''))
        except ValueError:
            pass

if stalls:
    ranked = sorted(stalls.items(), key=lambda x: -x[1])
    for i, (name, val) in enumerate(ranked):
        label = name.replace('ncu_stall_', '')
        pct = val * 100
        marker = ' <-- #1 BOTTLENECK' if i == 0 else ''
        print(f'  {label:25s} {pct:5.1f}%{marker}')
    print()
    top_name = ranked[0][0].replace('ncu_stall_', '')
    top_pct = ranked[0][1] * 100
    print(f'top_bottleneck: {top_name}')
    print(f'top_bottleneck_pct: {top_pct:.1f}')

# TSV-ready summary: matches results/*.tsv column format exactly
# Agent can grep for 'tsv_' lines and use values directly
print()
print('=== TSV-READY SUMMARY ===')
dur = float(metrics.get('gpu__time_duration.sum', '0').replace(',', '')) / 1000
sm = float(metrics.get('sm__throughput.avg.pct_of_peak_sustained_elapsed', '0').replace(',', ''))
s_math = stalls.get('ncu_stall_math_throttle', 0) * 100
s_wait = stalls.get('ncu_stall_wait', 0) * 100
s_long = stalls.get('ncu_stall_long_scoreboard', 0) * 100
s_short = stalls.get('ncu_stall_short_scoreboard', 0) * 100
s_barrier = stalls.get('ncu_stall_barrier', 0) * 100
s_scoreboard = s_long + s_short  # combined for TSV

print(f'tsv_duration_us: {dur:.1f}')
print(f'tsv_sm_pct: {sm:.1f}')
print(f'tsv_stall_math: {s_math:.1f}')
print(f'tsv_stall_wait: {s_wait:.1f}')
print(f'tsv_stall_scoreboard: {s_scoreboard:.1f}')
print(f'tsv_stall_barrier: {s_barrier:.1f}')
print(f'tsv_top_stall: {top_name}')
"
    else
        echo "profile: FAIL"
        echo "--- profile.log tail ---"
        tail -20 profile.log
        exit 4
    fi
fi

echo ""
echo "=== DONE ==="
