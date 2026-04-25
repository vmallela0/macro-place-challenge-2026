#!/bin/bash
# experiments/v5_full_sweep.sh
# Full 17-bench × 1-seed sweep across all v5 branches.
# Designed for the 258 GB / 64-thread machine: launches up to MAX_PARALLEL
# benchmarks at once (default 16), waiting for slots to free as runs complete.
#
# Usage:
#   experiments/v5_full_sweep.sh [seed] [budget] [max_parallel]
#   experiments/v5_full_sweep.sh                  # defaults: seed=42, budget=3300, max_parallel=16
#   experiments/v5_full_sweep.sh 42 3300 32       # 32-way parallel (uses ~32 cores at 1 BLAS thread)
#
# Each branch needs a worktree at /tmp/wt_<branch> with submodule symlinked
# to MAIN_REPO/external/MacroPlacement (run experiments/v5_setup_worktrees.sh
# first).
#
# Logs: experiments/logs/<exp_id>/<bench>_s<seed>_b<budget>.log
# CSV:  experiments/results.csv (rows added per run)

set -u
SEED="${1:-42}"
BUDGET="${2:-3300}"
MAX_PARALLEL="${3:-16}"

BRANCHES=(
  "v5_cells_skip:v5_cells_skip:/tmp/wt_v5_cells:"
  "v5_escape_v2:v5_escape_v2:/tmp/wt_v5_escape:PLACER_ESC_K_REGIONS=4"
  "v5_surrogate_struct:v5_surrogate_struct:/tmp/wt_v5_surr:PLACER_SURR_STRUCTURED=1"
  "v5_warmstart:v5_warmstart:/tmp/wt_v5_warm:PLACER_WARMSTART=1"
  "v5_combined:v5_combined:/tmp/wt_v5_combined:PLACER_ESC_K_REGIONS=4 PLACER_SURR_STRUCTURED=1 PLACER_WARMSTART=1"
)
BENCHES=(ibm01 ibm02 ibm03 ibm04 ibm06 ibm07 ibm08 ibm09 ibm10 ibm11 ibm12 ibm13 ibm14 ibm15 ibm16 ibm17 ibm18)

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "[v5_sweep] seed=$SEED budget=$BUDGET max_parallel=$MAX_PARALLEL"
echo "[v5_sweep] branches: ${#BRANCHES[@]}, benches: ${#BENCHES[@]}, total runs: $(( ${#BRANCHES[@]} * ${#BENCHES[@]} ))"

# Build the full task list (exp_id, wt, bench, env vars).
TASKS=()
for entry in "${BRANCHES[@]}"; do
  IFS=':' read -r exp_id branch wt envs <<< "$entry"
  if [ ! -d "$wt" ]; then
    echo "[v5_sweep] WARN: worktree $wt missing for $exp_id; skipping"
    continue
  fi
  for bench in "${BENCHES[@]}"; do
    TASKS+=("$exp_id|$wt|$bench|$envs")
  done
done

echo "[v5_sweep] queued ${#TASKS[@]} tasks"

# Slot management via background jobs + jobs -p count.
run_task() {
  local task="$1"
  IFS='|' read -r exp_id wt bench envs <<< "$task"
  # Convert "K1=V1 K2=V2" into space-separated env tokens for run_in_worktree.sh.
  IFS=' ' read -ra env_arr <<< "$envs"
  bash "$REPO_ROOT/experiments/run_in_worktree.sh" "$exp_id" "$wt" "$bench" "$SEED" "$BUDGET" "${env_arr[@]:-}" \
    > "/tmp/v5_sweep_${exp_id}_${bench}.out" 2>&1
}

active=0
for task in "${TASKS[@]}"; do
  while [ "$active" -ge "$MAX_PARALLEL" ]; do
    wait -n   # wait for any background job to finish
    active=$((active - 1))
  done
  run_task "$task" &
  active=$((active + 1))
  IFS='|' read -r exp_id _ bench _ <<< "$task"
  echo "[v5_sweep] launched $exp_id $bench (active=$active)"
done

wait
echo "[v5_sweep] all done. Inspect $REPO_ROOT/experiments/results.csv"
