#!/bin/bash
# Continuous iteration — runs forever until killed.
# Rotates through variants + benchmarks to keep CPU busy all night.

set -u
OUT=submissions/experiments/results/continuous
mkdir -p $OUT

log() { echo "[$(date +%H:%M:%S)] $*"; }

# List of variants + benchmarks (rotates). Each variant-benchmark gets ONE run then we rotate.
VARIANTS=(
    "placer_exp_v36"
    "placer_exp_v51"
    "placer_submission_v4"
)
BUDGETS=(300 500 700)
BENCHES=(ibm01 ibm09 ibm11 ibm13 ibm15 ibm17 ibm18 ibm02 ibm04 ibm06 ibm10 ibm12 ibm14 ibm16)

i=0
while true; do
    v=${VARIANTS[$((i % ${#VARIANTS[@]}))]}
    b=${BENCHES[$((i % ${#BENCHES[@]}))]}
    budget=${BUDGETS[$((i % ${#BUDGETS[@]}))]}
    stamp=$(date +%H%M%S)
    of="$OUT/${v}_${b}_${budget}s_${stamp}.log"

    # Avoid over-sub: wait if load > 10
    while true; do
        load=$(uptime | awk -F'load average' '{print $2}' | awk -F',' '{print $1}' | awk '{print $2}')
        # Fallback parse
        load=$(uptime | grep -oE "averages?: [0-9.]+" | grep -oE "[0-9.]+" | head -1)
        if [ -z "$load" ]; then load=5; fi
        int_load=$(echo "$load" | cut -d. -f1)
        if [ "$int_load" -le 10 ]; then
            break
        fi
        sleep 30
    done

    log "START $v $b ${budget}s"
    if [[ "$v" == "placer_submission_v4" ]]; then
        PLACER_TOTAL_BUDGET=$budget PLACER_SOFT_BUDGET=$((budget * 3 / 5)) \
            PLACER_PARALLEL_WORKERS=0 \
            uv run evaluate submissions/experiments/$v.py -b $b > "$of" 2>&1
    else
        uv run evaluate submissions/experiments/$v.py -b $b > "$of" 2>&1
    fi
    result=$(grep -E "^proxy=" "$of" | tail -1)
    log "DONE $v $b: $result"
    i=$((i + 1))
done
