#!/bin/bash
# Rerun the 3 overnight benchmarks that failed legalize under CPU contention.
# Sequential, 800s budget each. Total ~40min.
set -u
OUT=submissions/experiments/results/overnight
log() { echo "[$(date +%H:%M:%S)] $*" >> /tmp/rerun_failed.log; }

for bench in ibm10 ibm12 ibm14; do
    of="$OUT/${bench}.log"
    log "RERUN $bench"
    uv run evaluate submissions/experiments/placer_submission.py -b $bench > "$of" 2>&1
    log "DONE $bench: $(grep '^proxy=' $of | tail -1)"
done
log "rerun_failed complete"
