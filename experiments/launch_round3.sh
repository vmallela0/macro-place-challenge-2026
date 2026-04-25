#!/bin/bash
set -u
H=/Users/vmallela/Desktop/challenges/macro-place-challenge-2026/experiments/run_in_worktree.sh
WT=/tmp/wt_master
PIDS=()

run_one() {
  local exp="$1"; local seed="$2"; shift 2
  ( "$H" "$exp" "$WT" ibm01 "$seed" 550 "$@" > "/tmp/r3_${exp}_s${seed}.out" 2>&1 ) &
  PIDS+=($!)
  echo "[r3] launched $exp seed=$seed pid=$!"
}

# Multi-seed validation of top 2 (kill the noise)
run_one r3_sa_t0_00005 42 PLACER_SA_T0=0.00005
run_one r3_sa_t0_00005 43 PLACER_SA_T0=0.00005
run_one r3_sa_t0_00005 44 PLACER_SA_T0=0.00005
run_one r3_sa_t0_00007 42 PLACER_SA_T0=0.00007
run_one r3_sa_t0_00007 43 PLACER_SA_T0=0.00007
run_one r3_sa_t0_00007 44 PLACER_SA_T0=0.00007

# Stacks (1 seed each — quick signal)
run_one r3_sa5_plat8       42 PLACER_SA_T0=0.00005 PLACER_PLATEAU_N=8
run_one r3_sa5_destroy80   42 PLACER_SA_T0=0.00005 PLACER_ESC_HARD_DESTROY=80
run_one r3_sa7_plat8       42 PLACER_SA_T0=0.00007 PLACER_PLATEAU_N=8
run_one r3_sa7_destroy80   42 PLACER_SA_T0=0.00007 PLACER_ESC_HARD_DESTROY=80

echo "[r3] launched ${#PIDS[@]} jobs"
wait
echo "[r3] all done at $(date +%H:%M:%S)"
