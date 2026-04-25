#!/bin/bash
# Round 4: validate winning stacks at 3 seeds + test fast_tournament knob.
set -u
H=/Users/vmallela/Desktop/challenges/macro-place-challenge-2026/experiments/run_in_worktree.sh
WT=/tmp/wt_master
PIDS=()

run_one() {
  local exp="$1"; local seed="$2"; shift 2
  ( "$H" "$exp" "$WT" ibm01 "$seed" 550 "$@" > "/tmp/r4_${exp}_s${seed}.out" 2>&1 ) &
  PIDS+=($!)
  echo "[r4] launched $exp seed=$seed pid=$!"
}

# Multi-seed for the leading stack
run_one r4_sa5_destroy80 42 PLACER_SA_T0=0.00005 PLACER_ESC_HARD_DESTROY=80
run_one r4_sa5_destroy80 43 PLACER_SA_T0=0.00005 PLACER_ESC_HARD_DESTROY=80
run_one r4_sa5_destroy80 44 PLACER_SA_T0=0.00005 PLACER_ESC_HARD_DESTROY=80

# Test fast_tournament alone
run_one r4_fast_tournament 42 PLACER_EXP_FAST_TOURNAMENT=1
run_one r4_fast_tournament 43 PLACER_EXP_FAST_TOURNAMENT=1

# fast_tournament + best SA stack
run_one r4_fast_sa5_d80 42 PLACER_EXP_FAST_TOURNAMENT=1 PLACER_SA_T0=0.00005 PLACER_ESC_HARD_DESTROY=80
run_one r4_fast_sa5_d80 43 PLACER_EXP_FAST_TOURNAMENT=1 PLACER_SA_T0=0.00005 PLACER_ESC_HARD_DESTROY=80
run_one r4_fast_sa5_d80 44 PLACER_EXP_FAST_TOURNAMENT=1 PLACER_SA_T0=0.00005 PLACER_ESC_HARD_DESTROY=80

# Generalization spot-check: best stack on ibm06 + ibm10 (1 seed each)
( "$H" r4_sa5_d80_ibm06 "$WT" ibm06 42 550 PLACER_SA_T0=0.00005 PLACER_ESC_HARD_DESTROY=80 > /tmp/r4_sa5_d80_ibm06_s42.out 2>&1 ) &
PIDS+=($!)
( "$H" r4_sa5_d80_ibm10 "$WT" ibm10 42 550 PLACER_SA_T0=0.00005 PLACER_ESC_HARD_DESTROY=80 > /tmp/r4_sa5_d80_ibm10_s42.out 2>&1 ) &
PIDS+=($!)

echo "[r4] launched ${#PIDS[@]} jobs"
wait
echo "[r4] all done at $(date +%H:%M:%S)"
