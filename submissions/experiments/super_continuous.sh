#!/bin/bash
# Parallel continuous iteration — keeps ~6 workers busy alongside overnight scripts.
# Target: total load ~ncpu, no idle cores, no oversubscription.

set -u
OUT=submissions/experiments/results/continuous
mkdir -p $OUT

TARGET_WORKERS=6   # slots we try to fill
NCPU=$(sysctl -n hw.ncpu)

log() { echo "[$(date +%H:%M:%S)] $*"; }

VARIANTS=(
    "placer_exp_v36"        # adaptive cycle
    "placer_exp_v51"        # per-net + adaptive (short-budget winner)
    "placer_submission_v4"  # full pipeline + per-net
)

# Short budgets for rapid experimentation; long budgets for convergence.
BUDGETS=(200 300 500 700)

BENCHES=(
    ibm01 ibm02 ibm03 ibm04 ibm06 ibm07 ibm08
    ibm09 ibm10 ibm11 ibm12 ibm13 ibm14
    ibm15 ibm16 ibm17 ibm18
)

i=0
while true; do
    # Count our current jobs
    my_jobs=$(jobs -r | wc -l | tr -d ' ')

    # Count total placer processes (all scripts)
    total=$(ps aux | grep "placer_exp\|placer_submission" | grep -v grep | wc -l | tr -d ' ')

    # System load; throttle if near NCPU+2
    load=$(uptime | grep -oE "average[s]?: [0-9.]+" | grep -oE "[0-9.]+" | head -1)
    load_int=${load%.*}
    [ -z "$load_int" ] && load_int=0

    if [ "$my_jobs" -lt "$TARGET_WORKERS" ] && [ "$load_int" -lt $((NCPU + 2)) ]; then
        v=${VARIANTS[$((RANDOM % ${#VARIANTS[@]}))]}
        b=${BENCHES[$((RANDOM % ${#BENCHES[@]}))]}
        budget=${BUDGETS[$((RANDOM % ${#BUDGETS[@]}))]}
        stamp=$(date +%H%M%S)_$RANDOM
        of="$OUT/${v}_${b}_${budget}s_${stamp}.log"

        log "START [$my_jobs/$TARGET_WORKERS, load=$load] $v $b ${budget}s"
        if [[ "$v" == "placer_submission_v4" ]]; then
            (PLACER_TOTAL_BUDGET=$budget PLACER_SOFT_BUDGET=$((budget * 3 / 5)) \
                PLACER_PARALLEL_WORKERS=0 \
                uv run evaluate submissions/experiments/$v.py -b $b > "$of" 2>&1
             r=$(grep -E "^proxy=" "$of" | tail -1)
             echo "[$(date +%H:%M:%S)] DONE $v $b: $r" >> /tmp/super_continuous.log) &
        else
            (uv run evaluate submissions/experiments/$v.py -b $b > "$of" 2>&1
             r=$(grep -E "^proxy=" "$of" | tail -1)
             echo "[$(date +%H:%M:%S)] DONE $v $b: $r" >> /tmp/super_continuous.log) &
        fi
        sleep 3
    else
        sleep 20
    fi
    i=$((i + 1))
done
