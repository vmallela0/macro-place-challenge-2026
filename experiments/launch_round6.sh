#!/bin/bash
# Round 6: validate the safe stack on multi-bench + test FIXED fast_tournament.
set -u
H=/Users/vmallela/Desktop/challenges/macro-place-challenge-2026/experiments/run_in_worktree.sh
WT=/tmp/wt_master
PIDS=()

run_one() {
  local exp="$1"; local bench="$2"; local seed="$3"; local budget="$4"; shift 4
  ( "$H" "$exp" "$WT" "$bench" "$seed" "$budget" "$@" > "/tmp/r6_${exp}_${bench}_s${seed}_b${budget}.out" 2>&1 ) &
  PIDS+=($!)
  echo "[r6] launched $exp $bench seed=$seed budget=$budget"
}

# Track A: validated stack WITHOUT fast_tournament (safe baseline for round 6)
SAFE="PLACER_SA_T0=0.00005 PLACER_ESC_HARD_DESTROY=80"
run_one safe ibm01 42 1100 $SAFE
run_one safe ibm06 42 1100 $SAFE
run_one safe ibm06 43 1100 $SAFE
run_one safe ibm10 42 1100 $SAFE
run_one safe ibm17 42 1100 $SAFE

# Track B: same stack WITH FIXED fast_tournament — validate the fix on big benches
FAST="PLACER_SA_T0=0.00005 PLACER_ESC_HARD_DESTROY=80 PLACER_EXP_FAST_TOURNAMENT=1"
run_one fast ibm01 42 1100 $FAST
run_one fast ibm06 42 1100 $FAST
run_one fast ibm10 42 1100 $FAST
run_one fast ibm17 42 1100 $FAST

# Track C: fast_tournament fix sanity check on ibm17 alone
run_one fast_only ibm17 42 1100 PLACER_EXP_FAST_TOURNAMENT=1

echo "[r6] launched ${#PIDS[@]} jobs"
wait
echo "[r6] all done at $(date +%H:%M:%S)"
