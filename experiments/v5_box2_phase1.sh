#!/bin/bash
# experiments/v5_box2_phase1.sh — paired multi-seed sweep on the second box.
#
# Scientific design (see v5_box2 PROMPT_BOX2.md / handoff):
#   - First box owns seed 42 for v5_combined. Skip that.
#   - Run v5_combined at seeds {43,44,45,46} × 17 benches = 68 runs.
#   - Run v5_cluster  at seeds {42,43,44,45,46} × 17 benches = 85 runs.
#   - Total 153 single-threaded runs, randomized launch order, MAX_PARALLEL-way.
#
# Usage:
#   bash experiments/v5_box2_phase1.sh [budget=3300] [max_parallel=32]
#
# Worktrees needed:
#   /tmp/wt_v5_combined  (off origin/v5_combined)
#   /tmp/wt_v5_cluster   (off origin/v5_cluster)
# Each with experiments/run_in_worktree.sh patched to the correct MAIN_REPO.

set -u
BUDGET="${1:-3300}"
MAX_PARALLEL="${2:-32}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

COMBINED_ENVS="PLACER_ESC_K_REGIONS=4 PLACER_SURR_STRUCTURED=1 PLACER_WARMSTART=1"
CLUSTER_ENVS="$COMBINED_ENVS PLACER_ESC_CLUSTER_FRAC=0.3"

BENCHES=(ibm01 ibm02 ibm03 ibm04 ibm06 ibm07 ibm08 ibm09 ibm10 ibm11 ibm12 ibm13 ibm14 ibm15 ibm16 ibm17 ibm18)
COMBINED_SEEDS=(43 44 45 46)
CLUSTER_SEEDS=(42 43 44 45 46)

# Build full task list as exp_id|wt|bench|seed|envs lines, then shuffle.
TASKS=()
for SEED in "${COMBINED_SEEDS[@]}"; do
  for B in "${BENCHES[@]}"; do
    TASKS+=("v5_combined_box2|/tmp/wt_v5_combined|$B|$SEED|$COMBINED_ENVS")
  done
done
for SEED in "${CLUSTER_SEEDS[@]}"; do
  for B in "${BENCHES[@]}"; do
    TASKS+=("v5_cluster_box2|/tmp/wt_v5_cluster|$B|$SEED|$CLUSTER_ENVS")
  done
done

echo "[phase1] queued ${#TASKS[@]} tasks (expected 153)"
echo "[phase1] budget=${BUDGET}s max_parallel=$MAX_PARALLEL"

# Deterministic shuffle (seed 1) so a re-run launches in same order.
SHUFFLED=$(printf '%s\n' "${TASKS[@]}" | awk 'BEGIN{srand(1)} {print rand() "\t" $0}' | sort -k1,1n | cut -f2-)

active=0
launched=0
while IFS= read -r task; do
  IFS='|' read -r exp_id wt bench seed envs <<< "$task"
  while [ "$active" -ge "$MAX_PARALLEL" ]; do
    wait -n
    active=$((active - 1))
  done
  IFS=' ' read -ra env_arr <<< "$envs"
  bash "$wt/experiments/run_in_worktree.sh" "$exp_id" "$wt" "$bench" "$seed" "$BUDGET" "${env_arr[@]:-}" \
    > "/tmp/phase1_${exp_id}_${bench}_s${seed}.out" 2>&1 &
  active=$((active + 1))
  launched=$((launched + 1))
  echo "[phase1] $launched/${#TASKS[@]} launched: $exp_id $bench s=$seed (active=$active)"
done <<< "$SHUFFLED"

wait
echo "[phase1] DONE: 153 runs complete. Inspect $REPO_ROOT/experiments/results.csv"
