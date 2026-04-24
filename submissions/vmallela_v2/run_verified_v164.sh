#!/bin/bash
# Serial sweep with the v164 config (sweep-winner): RATIOS=(0.05, 0.50, 0.15, 0.15, 0.15),
# GROW_FACTOR=1.3, INITIAL_CYCLE_DIVISOR=10.
# Each benchmark at 3300 s, one at a time, seed=42.
set -u

OUTDIR="submissions/vmallela_v2/results_verified_v164"
mkdir -p "$OUTDIR"

BENCHES=(ibm01 ibm02 ibm03 ibm04 ibm06 ibm07 ibm08 ibm09 ibm10 ibm11 ibm12 ibm13 ibm14 ibm15 ibm16 ibm17 ibm18)

for b in "${BENCHES[@]}"; do
  logfile="$OUTDIR/${b}.log"
  if [ -f "$logfile" ] && grep -q "^proxy=" "$logfile"; then
    echo "=== $b already complete, skipping ==="
    continue
  fi
  echo "=== [$(date +'%H:%M:%S')] $b starting ==="
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    PYTHONHASHSEED=42 PLACER_TOTAL_BUDGET=3300 PLACER_PARALLEL_WORKERS=0 \
    uv run evaluate submissions/experiments/placer_exp_v164.py -b "$b" 2>&1 | tee "$logfile"
  echo "=== [$(date +'%H:%M:%S')] $b done ==="
done

echo "=== ALL v164 BENCHES DONE $(date) ==="
