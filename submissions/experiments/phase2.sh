#!/bin/bash
# Phase 2: placer_submission_v3 on all 17 at 650s each.
# Also covers the overnight failures (ibm10/12/14/17).
set -u
OUT=submissions/experiments/results/overnight_v3
mkdir -p $OUT
log() { echo "[$(date +%H:%M:%S)] $*" >> /tmp/phase2.log; }

for bench in ibm01 ibm09 ibm11 ibm02 ibm03 ibm04 ibm06 ibm07 ibm08 \
             ibm10 ibm12 ibm13 ibm14 ibm15 ibm16 ibm17 ibm18; do
    of="$OUT/${bench}.log"
    log "START v3 $bench 650s"
    PLACER_TOTAL_BUDGET=650 PLACER_SOFT_BUDGET=450 PLACER_PARALLEL_WORKERS=0 \
        uv run evaluate submissions/experiments/placer_submission_v3.py -b $bench > "$of" 2>&1
    log "DONE v3 $bench: $(grep '^proxy=' $of | tail -1)"
done
log "phase2 complete"
