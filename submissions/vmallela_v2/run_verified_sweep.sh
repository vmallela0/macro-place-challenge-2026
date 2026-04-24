#!/bin/bash
# Serial single-run sweep: each benchmark gets 3300s, run one at a time, seed=42.
# NEVER run in parallel. Load must stay low.
set -u

OUTDIR="submissions/vmallela_v2/results_verified"
mkdir -p "$OUTDIR"

BENCHES=(ibm01 ibm02 ibm03 ibm04 ibm06 ibm07 ibm08 ibm09 ibm10 ibm11 ibm12 ibm13 ibm14 ibm15 ibm16 ibm17 ibm18)

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
