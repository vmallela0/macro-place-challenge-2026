#!/bin/bash
# experiments/run_experiment.sh
# Run one benchmark under a specific experiment configuration, log to results.csv.
#
# Usage:
#   experiments/run_experiment.sh <exp_id> <benchmark> <seed> <budget_s> [EXTRA_ENV=val ...]
#
# Examples:
#   experiments/run_experiment.sh baseline ibm01 42 550
#   experiments/run_experiment.sh exp6_sa_cd ibm01 42 550 PLACER_EXP6_SA=1 PLACER_SA_T0=0.001
#
# CSV schema (experiments/results.csv):
#   exp_id, branch, benchmark, seed, budget_s, final_cost, hpwl, density,
#   congestion, wall_clock_s, timestamp, notes
#
# The `notes` column captures the extra env vars as `K=V|K=V`.
set -u

if [ "$#" -lt 4 ]; then
  cat <<EOF
Usage: $0 <exp_id> <benchmark> <seed> <budget_s> [EXTRA_ENV=val ...]

  exp_id     Short identifier, e.g. baseline, exp6_sa_cd
  benchmark  e.g. ibm01, ibm06, ibm10
  seed       integer (42, 43, 44 for the standard 3-seed set)
  budget_s   PLACER_TOTAL_BUDGET seconds (550 = fast iter, 3300 = full)
EOF
  exit 2
fi

EXP_ID="$1"; BENCH="$2"; SEED="$3"; BUDGET="$4"; shift 4
EXTRAS=("$@")
# Also allow empty EXTRAS safely with set -u
EXTRAS=("${EXTRAS[@]:-}")

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

CSV="$REPO_ROOT/experiments/results.csv"
LOGDIR="$REPO_ROOT/experiments/logs/$EXP_ID"
mkdir -p "$LOGDIR"

BRANCH="$(git rev-parse --abbrev-ref HEAD)"
SHA="$(git rev-parse --short HEAD)"
LOGFILE="$LOGDIR/${BENCH}_s${SEED}_b${BUDGET}.log"

# Locked env matching submission run.sh.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED="$SEED"
export PLACER_SEED="$SEED"
export PLACER_TOTAL_BUDGET="$BUDGET"
export PLACER_PARALLEL_WORKERS=0

NOTES=""
for kv in "${EXTRAS[@]}"; do
  if [ -z "$kv" ]; then continue; fi
  k="${kv%%=*}"
  v="${kv#*=}"
  export "$k=$v"
  if [ -z "$NOTES" ]; then NOTES="$k=$v"; else NOTES="$NOTES|$k=$v"; fi
done

TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
T0=$(python3 -c 'import time; print(time.time())')

echo "[harness] exp=$EXP_ID bench=$BENCH seed=$SEED budget=$BUDGET branch=$BRANCH@$SHA notes='$NOTES'"
uv run evaluate submissions/vmallela_v2/placer.py -b "$BENCH" 2>&1 | tee "$LOGFILE" >/dev/null

T1=$(python3 -c 'import time; print(time.time())')
WALL=$(python3 -c "print(f'{$T1 - $T0:.1f}')")

# Parse final proxy line:
# "proxy=0.7914  (wl=0.073 den=0.550 cong=0.886)  VALID  [1030.46s]"
PROXY_LINE="$(grep -E '^proxy=' "$LOGFILE" | tail -1)"
if [ -z "$PROXY_LINE" ]; then
  COST=""; WL=""; DEN=""; CONG=""; STATUS="FAIL"
else
  COST="$(echo "$PROXY_LINE" | sed -E 's/^proxy=([0-9.]+).*/\1/')"
  WL="$(echo "$PROXY_LINE"    | sed -E 's/.*wl=([0-9.]+).*/\1/')"
  DEN="$(echo "$PROXY_LINE"   | sed -E 's/.*den=([0-9.]+).*/\1/')"
  CONG="$(echo "$PROXY_LINE"  | sed -E 's/.*cong=([0-9.]+).*/\1/')"
  STATUS="$(echo "$PROXY_LINE" | grep -oE 'VALID|INVALID' || echo UNKNOWN)"
fi

# Append to CSV (with header if first run).
if [ ! -f "$CSV" ]; then
  echo 'exp_id,branch,benchmark,seed,budget_s,final_cost,hpwl,density,congestion,wall_clock_s,timestamp,notes' > "$CSV"
fi
# Escape commas in notes
NOTES_ESC="$(echo "$NOTES" | sed 's/,/;/g')"
echo "$EXP_ID,$BRANCH@$SHA,$BENCH,$SEED,$BUDGET,$COST,$WL,$DEN,$CONG,$WALL,$TS,$NOTES_ESC" >> "$CSV"

echo "[harness] DONE cost=$COST status=$STATUS wall=${WALL}s -> $CSV"
