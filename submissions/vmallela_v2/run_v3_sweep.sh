#!/bin/bash
# v3 validation sweep. Writes to results_verified_v3/ so it does not clobber
# the submitted baseline at results_verified/.
#
# Single-run seed=42 per bench, 3300s budget, serial — same methodology as
# run_verified_sweep.sh but with the v3 evaluator + placer changes.
set -u

OUTDIR="submissions/vmallela_v2/results_verified_v3"
mkdir -p "$OUTDIR"

# Default to all 17; caller can pass a subset as $@.
if [ "$#" -eq 0 ]; then
  BENCHES=(ibm01 ibm02 ibm03 ibm04 ibm06 ibm07 ibm08 ibm09 ibm10 ibm11 ibm12 ibm13 ibm14 ibm15 ibm16 ibm17 ibm18)
else
  BENCHES=("$@")
fi

for b in "${BENCHES[@]}"; do
  logfile="$OUTDIR/${b}.log"
  if [ -f "$logfile" ] && grep -q "^proxy=" "$logfile"; then
    echo "=== $b already complete, skipping ==="
    continue
  fi
  echo "=== [$(date +'%H:%M:%S')] $b starting ==="
  ./submissions/vmallela_v2/run.sh -b "$b" 2>&1 | tee "$logfile"
  echo "=== [$(date +'%H:%M:%S')] $b done ==="
done

echo "=== ALL BENCHES DONE $(date) ==="
