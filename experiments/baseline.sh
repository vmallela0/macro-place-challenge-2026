#!/bin/bash
# experiments/baseline.sh
# Run the ibm01/ibm06/ibm10 × seeds 42/43/44 baseline on the current HEAD.
# Writes results to experiments/results.csv with exp_id=baseline.
#
# Per-bench budget is configurable via BUDGET env (default 550s = fast iter).
# Total wall time at 550s × 3 benches × 3 seeds = ~90 minutes serial.
set -u

BUDGET="${BUDGET:-550}"
BENCHES=("${BENCHES[@]:-ibm01 ibm06 ibm10}")
SEEDS=("${SEEDS[@]:-42 43 44}")

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

for b in ${BENCHES[@]}; do
  for s in ${SEEDS[@]}; do
    "$REPO_ROOT/experiments/run_experiment.sh" baseline "$b" "$s" "$BUDGET"
  done
done

echo "[baseline] all runs complete. See experiments/results.csv"
