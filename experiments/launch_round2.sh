#!/bin/bash
set -u
H=/Users/vmallela/Desktop/challenges/macro-place-challenge-2026/experiments/run_in_worktree.sh
WT=/tmp/wt_master
PIDS=()

run_one() {
  local exp="$1"; shift
  ( "$H" "$exp" "$WT" ibm01 42 550 "$@" > "/tmp/r2_${exp}.out" 2>&1 ) &
  PIDS+=($!)
  echo "[r2] launched $exp pid=$!"
}

# === Cold SA sweep around T0=0.0001 winner ===
run_one r2_sa_t0_00003 PLACER_SA_T0=0.00003
run_one r2_sa_t0_00005 PLACER_SA_T0=0.00005
run_one r2_sa_t0_00007 PLACER_SA_T0=0.00007
run_one r2_sa_t0_00010 PLACER_SA_T0=0.0001
run_one r2_sa_t0_00015 PLACER_SA_T0=0.00015
run_one r2_sa_t0_00020 PLACER_SA_T0=0.0002

# === Stack SA winner with other knobs ===
run_one r2_sa_destroy80 PLACER_SA_T0=0.0001 PLACER_ESC_HARD_DESTROY=80
run_one r2_sa_destroy60 PLACER_SA_T0=0.0001 PLACER_ESC_HARD_DESTROY=60
run_one r2_sa_linesearch PLACER_SA_T0=0.0001 PLACER_EXP7_LINESEARCH=1

# === Other knobs in isolation ===
run_one r2_plateau_2 PLACER_PLATEAU_N=2
run_one r2_plateau_8 PLACER_PLATEAU_N=8

# === Control: bare master_grid (no env, should match exp0 fixes only) ===
run_one r2_control_baseline

echo "[r2] launched ${#PIDS[@]} jobs"
wait
echo "[r2] all done at $(date +%H:%M:%S)"
python3 /Users/vmallela/Desktop/challenges/macro-place-challenge-2026/experiments/analyze.py --bench ibm01 --budget 550 --exp r2_
