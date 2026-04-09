#!/bin/bash
# Copyright (c) 2026 Darrell Thomas. MIT License. See LICENSE file.
#
# Autokernel evaluation script: build → test → benchmark → profile
#
# Supports both single-function and multi-function projects.
# Multi-function projects define a FUNCTIONS array — the profile step
# loops through each function, profiling with ncu per-function.
#
# Usage:
#   ./eval.sh --kernel <name>              # full pipeline
#   ./eval.sh --kernel <name> --quick      # skip profiling
#   ./eval.sh --kernel <name> --profile    # profile only
#   ./eval.sh --kernel <name> --func <fn>  # profile single function only

set -euo pipefail
cd "$(dirname "$0")"

KERNEL=""
MODE="full"
SINGLE_FUNC=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --kernel)   KERNEL="$2"; shift 2 ;;
        --quick)    MODE="quick"; shift ;;
        --profile)  MODE="profile"; shift ;;
        --func)     SINGLE_FUNC="$2"; shift 2 ;;
        *)          echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ─── PROJECT-SPECIFIC KERNEL CONFIG ──────────────────────────────────────────
#
# Single-function project: set NCU_KERNEL_NAME to one kernel name.
# Multi-function project: set FUNCTIONS array with "op_name|ncu_kernel_name"
#   entries. The profile script must accept --op <op_name>.
#
# Examples:
#
#   # Single function (e.g., rmsnorm, dotproduct):
#   mykernel)
#       TEST_SCRIPT="tests/test_mykernel.py"
#       BENCH_SCRIPT="benchmarks/bench_mykernel.py"
#       PROFILE_SCRIPT="profiles/profile_mykernel.py"
#       NCU_KERNEL_NAME="mykernel_sm120"
#       ;;
#
#   # Multi-function (e.g., linalg):
#   mylib)
#       TEST_SCRIPT="tests/test_mylib.py"
#       BENCH_SCRIPT="benchmarks/bench_mylib.py"
#       PROFILE_SCRIPT="profiles/profile_mylib.py"   # must accept --op <name>
#       FUNCTIONS=(
#           "gemm|gemm_sm120"
#           "gemv|gemv_sm120"
#           "dot|dot_sm120"
#       )
#       ;;

case "$KERNEL" in
    qv8)
        TEST_SCRIPT="tests/test_qv8.py"
        BENCH_SCRIPT="benchmarks/bench_qv8.py"
        PROFILE_SCRIPT="profiles/profile_qv8.py"
        NCU_KERNEL_NAME="qv8_batch_simulate_kernel"
        ;;
    *)
        echo "Unknown kernel: $KERNEL"
        echo "Available kernels: qv8"
        exit 1
        ;;
esac

RESULTS_DIR="results"
LOG_DIR="logs/${KERNEL}"
mkdir -p "$RESULTS_DIR" "$LOG_DIR"

# ─── GPU SCHEDULING ──────────────────────────────────────────────────────────
SCHED_PATH="$(dirname "$0")/../common/scripts/gpu-sched.sh"
if [[ -f "$SCHED_PATH" ]]; then
    source "$SCHED_PATH"
    gpu_acquire 1
else
    echo "gpu-sched: not found at $SCHED_PATH, falling back to GPU 1" >&2
    export CUDA_VISIBLE_DEVICES=1
fi
export CUDA_HOME=/usr/local/cuda-13
export PYTHONPATH=python

NCU="$CUDA_HOME/bin/ncu"

HEARTBEAT_FILE=".autokernel.${KERNEL}.alive"
if [[ -f "$HEARTBEAT_FILE" ]]; then
    touch "$HEARTBEAT_FILE"
else
    date +%s > "$HEARTBEAT_FILE"
fi

echo "=== KERNEL: $KERNEL (GPU: $CUDA_VISIBLE_DEVICES) ==="

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

# ─── BUILD / TEST / BENCH ────────────────────────────────────────────────────
if [[ "$MODE" != "profile" ]]; then
    echo "=== BUILD ==="
    if python3 setup.py build_ext --inplace > build.log 2>&1; then
        echo "build: PASS"
    else
        echo "build: FAIL"
        tail -20 build.log
        exit 1
    fi

    echo "=== TEST ==="
    if python3 "$TEST_SCRIPT" > test.log 2>&1; then
        echo "test: PASS"
    else
        echo "test: FAIL"
        tail -30 test.log
        exit 2
    fi

    echo "=== BENCHMARK ==="
    if python3 "$BENCH_SCRIPT" > bench.log 2>&1; then
        echo "bench: PASS"
        cat bench.log
        echo ""
        CUSTOM_MS=$(grep "^primary_custom_ms:" bench.log | awk '{print $2}' || true)
        VS_REF=$(grep "^primary_vs_ref:" bench.log | awk '{print $2}' | tr -d 'x' || true)
        [[ -n "$CUSTOM_MS" ]] && echo "primary_custom_ms: $CUSTOM_MS"
        [[ -n "$VS_REF" ]] && echo "primary_vs_ref: ${VS_REF}x"
    else
        echo "bench: FAIL"
        tail -20 bench.log
        exit 3
    fi
fi

# ─── PROFILE ──────────────────────────────────────────────────────────────────
#
# Two modes:
#   1. Single-function: NCU_KERNEL_NAME is set, profile once
#   2. Multi-function: FUNCTIONS array is set, loop through each
#
if [[ "$MODE" != "quick" ]]; then
    # Determine which profiling mode to use
    if [[ -n "${FUNCTIONS+x}" ]] && [[ ${#FUNCTIONS[@]} -gt 0 ]]; then
        # ── Multi-function profiling ──────────────────────────────────────
        echo ""
        echo "=== PROFILE (${#FUNCTIONS[@]} functions) ==="

        # If --func specified, only profile that one function
        if [[ -n "$SINGLE_FUNC" ]]; then
            PROFILE_LIST=()
            for entry in "${FUNCTIONS[@]}"; do
                IFS='|' read -r op_name ncu_name <<< "$entry"
                if [[ "$op_name" == "$SINGLE_FUNC" ]]; then
                    PROFILE_LIST=("$entry")
                    break
                fi
            done
            if [[ ${#PROFILE_LIST[@]} -eq 0 ]]; then
                echo "Unknown function: $SINGLE_FUNC"
                echo "Available: $(printf '%s ' "${FUNCTIONS[@]}" | sed 's/|[^ ]* / /g')"
                exit 1
            fi
        else
            PROFILE_LIST=("${FUNCTIONS[@]}")
        fi

        for entry in "${PROFILE_LIST[@]}"; do
            IFS='|' read -r op_name ncu_name <<< "$entry"

            echo ""
            echo "--- PROFILE: $op_name (ncu kernel: $ncu_name) ---"

            PROFILE_LOG="$LOG_DIR/profile_${op_name}.csv"

            if sudo -E CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} "$NCU" \
                --metrics "$NCU_METRICS" --csv \
                --kernel-name "$ncu_name" --launch-count 1 \
                python3 "$PROFILE_SCRIPT" --op "$op_name" > "$PROFILE_LOG" 2>&1; then

                # Parse ncu CSV output
                SM_PCT=$(grep "sm__throughput.avg.pct_of_peak_sustained_elapsed" "$PROFILE_LOG" | grep -v "^==" | tail -1 | awk -F',' '{print $NF}' | tr -d '"' || echo "-")
                MATH_THR=$(grep "math_pipe_throttle" "$PROFILE_LOG" | grep -v "^==" | tail -1 | awk -F',' '{print $NF}' | tr -d '"' || echo "-")
                WAIT=$(grep "stalled_wait" "$PROFILE_LOG" | grep -v "^==" | tail -1 | awk -F',' '{print $NF}' | tr -d '"' || echo "-")
                SCOREBOARD=$(grep "long_scoreboard" "$PROFILE_LOG" | grep -v "^==" | tail -1 | awk -F',' '{print $NF}' | tr -d '"' || echo "-")
                BARRIER=$(grep "stalled_barrier" "$PROFILE_LOG" | grep -v "^==" | tail -1 | awk -F',' '{print $NF}' | tr -d '"' || echo "-")
                BANK_LD=$(grep "bank_conflicts.*op_ld" "$PROFILE_LOG" | grep -v "^==" | tail -1 | awk -F',' '{print $NF}' | tr -d '"' || echo "0")
                BANK_ST=$(grep "bank_conflicts.*op_st" "$PROFILE_LOG" | grep -v "^==" | tail -1 | awk -F',' '{print $NF}' | tr -d '"' || echo "0")

                # Determine top stall
                TOP_STALL="-"
                TOP_VAL=0
                for stall_pair in "math_throttle:$MATH_THR" "wait:$WAIT" "long_scoreboard:$SCOREBOARD" "barrier:$BARRIER"; do
                    s_name="${stall_pair%%:*}"
                    s_val="${stall_pair##*:}"
                    if [[ "$s_val" != "-" ]] && (( $(echo "$s_val > $TOP_VAL" | bc -l 2>/dev/null || echo 0) )); then
                        TOP_STALL="$s_name"
                        TOP_VAL="$s_val"
                    fi
                done

                echo "  SM%: $SM_PCT | math_throttle: $MATH_THR | wait: $WAIT | scoreboard: $SCOREBOARD | barrier: $BARRIER"
                echo "  bank_conflicts: ld=$BANK_LD st=$BANK_ST | top_stall: $TOP_STALL"

                # Output parseable lines for the worker's optimization loop
                echo "profile_${op_name}_sm_pct: $SM_PCT"
                echo "profile_${op_name}_stall_math: $MATH_THR"
                echo "profile_${op_name}_stall_wait: $WAIT"
                echo "profile_${op_name}_stall_scoreboard: $SCOREBOARD"
                echo "profile_${op_name}_stall_barrier: $BARRIER"
                echo "profile_${op_name}_top_stall: $TOP_STALL"

                echo "  profile: PASS ($op_name)"
            else
                echo "  profile: FAIL ($op_name)"
                tail -10 "$PROFILE_LOG" 2>/dev/null
            fi
        done

    elif [[ -n "${NCU_KERNEL_NAME+x}" ]] && [[ -n "$NCU_KERNEL_NAME" ]]; then
        # ── Single-function profiling ─────────────────────────────────────
        echo "=== PROFILE ==="
        if sudo -E CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} "$NCU" \
            --metrics "$NCU_METRICS" --csv \
            --kernel-name "$NCU_KERNEL_NAME" --launch-count 1 \
            python3 "$PROFILE_SCRIPT" > profile.log 2>&1; then
            echo "profile: PASS"

            SM_PCT=$(grep "sm__throughput.avg.pct_of_peak_sustained_elapsed" profile.log | grep -v "^==" | tail -1 | awk -F',' '{print $NF}' | tr -d '"' || echo "-")
            MATH_THR=$(grep "math_pipe_throttle" profile.log | grep -v "^==" | tail -1 | awk -F',' '{print $NF}' | tr -d '"' || echo "-")
            WAIT=$(grep "stalled_wait" profile.log | grep -v "^==" | tail -1 | awk -F',' '{print $NF}' | tr -d '"' || echo "-")
            SCOREBOARD=$(grep "long_scoreboard" profile.log | grep -v "^==" | tail -1 | awk -F',' '{print $NF}' | tr -d '"' || echo "-")
            BARRIER=$(grep "stalled_barrier" profile.log | grep -v "^==" | tail -1 | awk -F',' '{print $NF}' | tr -d '"' || echo "-")

            TOP_STALL="-"
            TOP_VAL=0
            for stall_pair in "math_throttle:$MATH_THR" "wait:$WAIT" "long_scoreboard:$SCOREBOARD" "barrier:$BARRIER"; do
                s_name="${stall_pair%%:*}"
                s_val="${stall_pair##*:}"
                if [[ "$s_val" != "-" ]] && (( $(echo "$s_val > $TOP_VAL" | bc -l 2>/dev/null || echo 0) )); then
                    TOP_STALL="$s_name"
                    TOP_VAL="$s_val"
                fi
            done

            echo "SM%: $SM_PCT | math_throttle: $MATH_THR | wait: $WAIT | scoreboard: $SCOREBOARD | barrier: $BARRIER | top_stall: $TOP_STALL"
            echo "profile_sm_pct: $SM_PCT"
            echo "profile_stall_math: $MATH_THR"
            echo "profile_stall_wait: $WAIT"
            echo "profile_stall_scoreboard: $SCOREBOARD"
            echo "profile_stall_barrier: $BARRIER"
            echo "profile_top_stall: $TOP_STALL"
        else
            echo "profile: FAIL"
            tail -20 profile.log
            exit 4
        fi
    else
        echo "WARNING: No NCU_KERNEL_NAME or FUNCTIONS defined — skipping profile"
    fi
fi

echo ""
echo "=== DONE ==="
