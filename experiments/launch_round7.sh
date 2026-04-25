#!/bin/bash
# Round 7: cooling rate sweep + soft escape sweep + super-cold SA + best-of-5
set -u
H=/Users/vmallela/Desktop/challenges/macro-place-challenge-2026/experiments/run_in_worktree.sh
WT=/tmp/wt_master
PIDS=()
run_one() {
  local exp="$1"; local bench="$2"; local seed="$3"; local budget="$4"; shift 4
  ( "$H" "$exp" "$WT" "$bench" "$seed" "$budget" "$@" > "/tmp/r7_${exp}_${bench}_s${seed}_b${budget}.out" 2>&1 ) &
  PIDS+=($!)
  echo "[r7] launched $exp $bench seed=$seed budget=$budget"
}

# Common stack: SA T0=5e-5 + escape destroy=80
COMMON="PLACER_SA_T0=0.00005 PLACER_ESC_HARD_DESTROY=80"

# === Group A: SA cooling rate sweep ===
run_one cool_999    ibm01 42 550 $COMMON PLACER_SA_COOLING=0.999
run_one cool_9999   ibm01 42 550 $COMMON PLACER_SA_COOLING=0.9999

# === Group B: super-cold SA ===
run_one ultracold_1e5 ibm01 42 550 PLACER_SA_T0=0.00001 PLACER_ESC_HARD_DESTROY=80
run_one ultracold_3e5 ibm01 42 550 PLACER_SA_T0=0.00003 PLACER_ESC_HARD_DESTROY=80

# === Group C: soft escape sweep ===
run_one s_destroy_60 ibm01 42 550 $COMMON PLACER_ESC_SOFT_DESTROY=60
run_one s_destroy_100 ibm01 42 550 $COMMON PLACER_ESC_SOFT_DESTROY=100

# === Group D: kitchen sink ===
run_one kitchen_sink ibm01 42 550 \
    PLACER_SA_T0=0.00005 PLACER_SA_COOLING=0.9999 \
    PLACER_ESC_HARD_DESTROY=80 PLACER_ESC_SOFT_DESTROY=60 \
    PLACER_PLATEAU_N=8

# === Group E: best-of-K — same config × 3 extra seeds (already have 42/43/44 from R3) ===
run_one bestof_s45 ibm01 45 550 $COMMON
run_one bestof_s46 ibm01 46 550 $COMMON
run_one bestof_s47 ibm01 47 550 $COMMON

echo "[r7] launched ${#PIDS[@]} jobs"
wait
echo "[r7] all done at $(date +%H:%M:%S)"
