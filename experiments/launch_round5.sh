#!/bin/bash
# Round 5: generalize the winning full stack across more benchmarks + try max-stack.
set -u
H=/Users/vmallela/Desktop/challenges/macro-place-challenge-2026/experiments/run_in_worktree.sh
WT=/tmp/wt_master
PIDS=()

run_one() {
  local exp="$1"; local bench="$2"; local seed="$3"; local budget="$4"; shift 4
  ( "$H" "$exp" "$WT" "$bench" "$seed" "$budget" "$@" > "/tmp/r5_${exp}_${bench}_s${seed}_b${budget}.out" 2>&1 ) &
  PIDS+=($!)
  echo "[r5] launched $exp $bench seed=$seed budget=$budget pid=$!"
}

# Full stack = fast_tournament + SA T0=5e-5 + escape destroy=80
STACK="PLACER_EXP_FAST_TOURNAMENT=1 PLACER_SA_T0=0.00005 PLACER_ESC_HARD_DESTROY=80"

# 3-seed ibm01 baseline-equivalent + 2 budget points (control)
run_one fullstack_ibm01_b1100 ibm01 42 1100 $STACK
run_one fullstack_ibm01_b1100 ibm01 43 1100 $STACK

# 3-seed ibm06 at 1100s (fair budget)
run_one fullstack_ibm06_b1100 ibm06 42 1100 $STACK
run_one fullstack_ibm06_b1100 ibm06 43 1100 $STACK

# 3-seed ibm10 at 1100s (fair-ish for 786 macros)
run_one fullstack_ibm10_b1100 ibm10 42 1100 $STACK

# ibm17 at 1100s (hardest — budget-starved)
run_one fullstack_ibm17_b1100 ibm17 42 1100 $STACK

# Max-stack on ibm01 — add plateau_8 + linesearch on top
MAXSTACK="$STACK PLACER_PLATEAU_N=8 PLACER_EXP7_LINESEARCH=1"
run_one maxstack_ibm01 ibm01 42 550 $MAXSTACK
run_one maxstack_ibm01 ibm01 43 550 $MAXSTACK
run_one maxstack_ibm01 ibm01 44 550 $MAXSTACK

# Try fast_tournament alone at higher budget on ibm01 (does it stack with longer budget?)
run_one fast_only_ibm01_b1100 ibm01 42 1100 PLACER_EXP_FAST_TOURNAMENT=1

echo "[r5] launched ${#PIDS[@]} jobs"
wait
echo "[r5] all done at $(date +%H:%M:%S)"
