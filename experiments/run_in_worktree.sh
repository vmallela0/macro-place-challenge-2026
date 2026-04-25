#!/bin/bash
# experiments/run_in_worktree.sh
# Run one experiment from a worktree directory using main repo's .venv directly.
# Bypasses `uv run` (which reinstalls editable paths and breaks symlink trick).
#
# Usage:
#   experiments/run_in_worktree.sh <exp_id> <worktree_dir> <bench> <seed> <budget> [ENV=val ...]
#
# Worktree must already be set up:
#   git worktree add <worktree_dir> <branch>
#   cd <worktree_dir> && git submodule update --init external/MacroPlacement
#
# Appends to MAIN repo's experiments/results.csv.
set -u
if [ "$#" -lt 5 ]; then
  echo "Usage: $0 <exp_id> <worktree_dir> <bench> <seed> <budget_s> [ENV=val ...]"
  exit 2
fi

EXP_ID="$1"; WT="$2"; BENCH="$3"; SEED="$4"; BUDGET="$5"; shift 5
EXTRAS=("$@")
EXTRAS=("${EXTRAS[@]:-}")

MAIN_REPO="/Users/vmallela/Desktop/challenges/macro-place-challenge-2026"
PYBIN="$MAIN_REPO/.venv/bin/python"
CSV="$MAIN_REPO/experiments/results.csv"
LOGDIR="$MAIN_REPO/experiments/logs/$EXP_ID"
mkdir -p "$LOGDIR"

cd "$WT"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"
SHA="$(git rev-parse --short HEAD)"
LOGFILE="$LOGDIR/${BENCH}_s${SEED}_b${BUDGET}.log"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED="$SEED"
export PLACER_SEED="$SEED"
export PLACER_TOTAL_BUDGET="$BUDGET"
export PLACER_PARALLEL_WORKERS=0
export PYTHONPATH=.

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
echo "[wt-harness] exp=$EXP_ID wt=$WT branch=$BRANCH@$SHA bench=$BENCH seed=$SEED budget=$BUDGET notes='$NOTES'"

"$PYBIN" -m macro_place.evaluate submissions/vmallela_v2/placer.py -b "$BENCH" 2>&1 | tee "$LOGFILE" >/dev/null

T1=$(python3 -c 'import time; print(time.time())')
WALL=$(python3 -c "print(f'{$T1 - $T0:.1f}')")

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

if [ ! -f "$CSV" ]; then
  echo 'exp_id,branch,benchmark,seed,budget_s,final_cost,hpwl,density,congestion,wall_clock_s,timestamp,notes' > "$CSV"
fi
NOTES_ESC="$(echo "$NOTES" | sed 's/,/;/g')"
# Use flock to avoid CSV race when many parallel runs append.
(
  flock 200
  echo "$EXP_ID,$BRANCH@$SHA,$BENCH,$SEED,$BUDGET,$COST,$WL,$DEN,$CONG,$WALL,$TS,$NOTES_ESC" >> "$CSV"
) 200>"$CSV.lock"

echo "[wt-harness] DONE exp=$EXP_ID cost=$COST status=$STATUS wall=${WALL}s"
