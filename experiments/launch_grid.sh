#!/bin/bash
# Launch all parallel experiments at once. Each runs ibm01 seed=42 budget=550.
# Outputs append to experiments/results.csv (with flock) and per-exp logs.
set -u

REPO=/Users/vmallela/Desktop/challenges/macro-place-challenge-2026
H="$REPO/experiments/run_in_worktree.sh"
PIDS=()

# Each entry: exp_id | worktree | extra env vars (space-separated)
declare -a JOBS=(
  "exp0_baseline       /tmp/wt_exp0   "
  "exp6_sa_t0_0003     /tmp/wt_exp6   PLACER_SA_T0=0.0003"
  "exp6_sa_t0_0010     /tmp/wt_exp6   PLACER_SA_T0=0.001"
  "exp7_linesearch     /tmp/wt_exp7   "
  "exp14_destroy_40    /tmp/wt_exp14  PLACER_ESC_HARD_DESTROY=40"
  "exp14_destroy_60    /tmp/wt_exp14  PLACER_ESC_HARD_DESTROY=60"
  "exp14_cand_120      /tmp/wt_exp14  PLACER_ESC_HARD_CAND=120"
)

for job in "${JOBS[@]}"; do
  read -r exp wt envs <<< "$job"
  envstr=()
  for kv in $envs; do envstr+=("$kv"); done
  echo "[grid] launching $exp (wt=$wt, env=$envs)"
  ( "$H" "$exp" "$wt" ibm01 42 550 "${envstr[@]}" > "/tmp/grid_${exp}.out" 2>&1 ) &
  PIDS+=($!)
done

echo "[grid] launched ${#PIDS[@]} jobs: ${PIDS[*]}"
wait
echo "[grid] all done at $(date +%H:%M:%S)"
python3 "$REPO/experiments/analyze.py" --bench ibm01 --budget 550
