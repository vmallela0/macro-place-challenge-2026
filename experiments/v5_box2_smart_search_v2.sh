#!/bin/bash
# experiments/v5_box2_smart_search_v2.sh — focused single-config sweep on hard benches.
#
# Pivoted from 4-config breadth → 1-config × 4 seeds × 5 hard benches for
# better statistical power per config + coverage of all 5 hard benches.
# Goal: beat 1.0 mean by reducing hard-bench mean ~3%.
#
# Config: cluster30_plateau2 (most aggressive escape, theoretically biggest swing).
#   PLACER_ESC_K_REGIONS=4 + SURR_STRUCTURED=1 + WARMSTART=1 (= v5_combined)
#   + PLACER_ESC_CLUSTER_FRAC=0.3 (cluster-translate)
#   + PLACER_PLATEAU_N=2 (escape every 2 plateaus, vs default 4)
#   + PLACER_ESC_HARD_DESTROY=120 PLACER_ESC_HARD_CAND=80 (bigger LNS)
#
# 5 hard benches × 4 seeds = 20 runs.
# 8-way × 3300s × 3 batches = 9900s = 2.75h wall.
set -u
BUDGET="${1:-3300}"
MAX_PARALLEL="${2:-8}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

export MKL_ENABLE_INSTRUCTIONS=AVX2
export MKL_DEBUG_CPU_TYPE=4

ENVS="PLACER_ESC_K_REGIONS=4 PLACER_SURR_STRUCTURED=1 PLACER_WARMSTART=1 PLACER_ESC_CLUSTER_FRAC=0.3 PLACER_PLATEAU_N=2 PLACER_ESC_HARD_DESTROY=120 PLACER_ESC_HARD_CAND=80"
EXP_ID="ss_cluster30_plateau2"
WT="/tmp/wt_v5_cluster"
BENCHES=(ibm12 ibm15 ibm16 ibm17 ibm18)
SEEDS=(42 43 44 45)

TASKS=()
for SEED in "${SEEDS[@]}"; do
  for B in "${BENCHES[@]}"; do
    TASKS+=("$EXP_ID|$WT|$B|$SEED|$ENVS")
  done
done

echo "[ssv2] queued ${#TASKS[@]} tasks (expected 20)"
echo "[ssv2] config=$EXP_ID  budget=${BUDGET}s  max_parallel=$MAX_PARALLEL"

SHUFFLED=$(printf '%s\n' "${TASKS[@]}" | awk 'BEGIN{srand(4)} {print rand() "\t" $0}' | sort -k1,1n | cut -f2-)

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
    > "/tmp/ssv2_${bench}_s${seed}.out" 2>&1 &
  active=$((active + 1))
  launched=$((launched + 1))
  echo "[ssv2] $launched/${#TASKS[@]} launched: $bench s=$seed (active=$active)"
done <<< "$SHUFFLED"

wait
echo "[ssv2] DONE at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
