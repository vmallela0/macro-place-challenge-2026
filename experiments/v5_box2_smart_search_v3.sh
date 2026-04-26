#!/bin/bash
# experiments/v5_box2_smart_search_v3.sh — 5-way fallback after 8-way failed.
#
# 8-way (ssv2) also produced INVALID rows on hard benches:
#   ibm12 [legalize] 718s cost=inf → 1.6251 INVALID (187 overlaps)
# AVX-512 throttle still bites at 8 concurrent placers on Skylake-SP, pushing
# legalize past 660s budget cap. Drop further to 5-way.
#
# Plan: 5-way × 5 hard benches × 3 seeds = 15 runs, ~2.75h wall.
# Leaves ~30min for analysis + push to fit user's 4h budget.
#
# Config: cluster30_plateau2 (same as ssv2).
set -u
BUDGET="${1:-3300}"
MAX_PARALLEL="${2:-5}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

export MKL_ENABLE_INSTRUCTIONS=AVX2
export MKL_DEBUG_CPU_TYPE=4

ENVS="PLACER_ESC_K_REGIONS=4 PLACER_SURR_STRUCTURED=1 PLACER_WARMSTART=1 PLACER_ESC_CLUSTER_FRAC=0.3 PLACER_PLATEAU_N=2 PLACER_ESC_HARD_DESTROY=120 PLACER_ESC_HARD_CAND=80"
EXP_ID="ssv3_cluster30_plateau2"
WT="/tmp/wt_v5_cluster"
BENCHES=(ibm12 ibm15 ibm16 ibm17 ibm18)
SEEDS=(42 43 44)

TASKS=()
for SEED in "${SEEDS[@]}"; do
  for B in "${BENCHES[@]}"; do
    TASKS+=("$EXP_ID|$WT|$B|$SEED|$ENVS")
  done
done

echo "[ssv3] queued ${#TASKS[@]} tasks (expected 15)"
echo "[ssv3] config=$EXP_ID  budget=${BUDGET}s  max_parallel=$MAX_PARALLEL"

SHUFFLED=$(printf '%s\n' "${TASKS[@]}" | awk 'BEGIN{srand(5)} {print rand() "\t" $0}' | sort -k1,1n | cut -f2-)

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
    > "/tmp/ssv3_${bench}_s${seed}.out" 2>&1 &
  active=$((active + 1))
  launched=$((launched + 1))
  echo "[ssv3] $launched/${#TASKS[@]} launched: $bench s=$seed (active=$active)"
done <<< "$SHUFFLED"

wait
echo "[ssv3] DONE at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
