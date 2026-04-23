#!/bin/bash
# Long overnight: phases 2+ after current overnight.sh completes.
#
# Phase 2 (~3 hours): sub_v3 on all 17 benchmarks with 650s budget (adaptive + surrogate).
# Phase 3 (~2 hours): push key benchmarks harder — ibm01, ibm02, ibm09, ibm15, ibm17, ibm18.
# Phase 4 (~1 hour): misc variants on ibm01 for further exploration.

set -u
OUT=submissions/experiments/results/overnight
OUT2=submissions/experiments/results/overnight_v3
OUT3=submissions/experiments/results/overnight_v3_long
mkdir -p $OUT2 $OUT3

log() { echo "[$(date +%Y-%m-%d' '%H:%M:%S)] $*"; }

# Wait for current overnight.sh to complete (poll for parent process)
log "Waiting for current overnight.sh..."
while pgrep -f "overnight.sh" > /dev/null; do
    sleep 60
done
log "Current overnight complete. Starting phase 2 with sub_v3."

# Phase 2: sub_v3 on all 17
for bench in ibm01 ibm02 ibm03 ibm04 ibm06 ibm07 ibm08 ibm09 ibm10 ibm11 ibm12 ibm13 ibm14 ibm15 ibm16 ibm17 ibm18; do
    budget=650
    soft=450
    of="$OUT2/${bench}.log"
    log "START v3 $bench ${budget}s"
    PLACER_TOTAL_BUDGET=$budget PLACER_SOFT_BUDGET=$soft PLACER_PARALLEL_WORKERS=0 \
        uv run evaluate submissions/experiments/placer_submission_v3.py -b $bench \
        > "$of" 2>&1
    line=$(grep -E "^proxy=" "$of" 2>/dev/null | tail -1)
    log "DONE  v3 $bench $line"
done
log "Phase 2 complete."

# Phase 3: push key benchmarks harder with 1400s budget
for bench in ibm01 ibm09 ibm11 ibm15 ibm17 ibm18; do
    budget=1400
    soft=1000
    of="$OUT3/${bench}.log"
    log "START long $bench ${budget}s"
    PLACER_TOTAL_BUDGET=$budget PLACER_SOFT_BUDGET=$soft PLACER_PARALLEL_WORKERS=0 \
        uv run evaluate submissions/experiments/placer_submission_v3.py -b $bench \
        > "$of" 2>&1
    line=$(grep -E "^proxy=" "$of" 2>/dev/null | tail -1)
    log "DONE  long $bench $line"
done
log "All phases complete."
