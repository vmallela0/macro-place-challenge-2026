#!/bin/bash
# experiments/v5_box2_smart_search.sh — 4h tight smart-search on hard benches.
#
# Constraint: ~4h total wall budget, then push & shutdown.
# Design: 4 configs × 3 hard benches × 2 seeds = 24 runs.
# 8-way × 3300s × 3 batches = 9900s = 2.75h. Leaves time for analysis + push.
#
# Configs informed by Explore agent's knob map + cost decomposition
# (cost = 1.0 wirelength + 0.5 density + 0.5 congestion; congestion dominates
# on hard benches). Each variant targets the congestion-aware escape phase.
set -u
BUDGET="${1:-3300}"
MAX_PARALLEL="${2:-8}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

export MKL_ENABLE_INSTRUCTIONS=AVX2
export MKL_DEBUG_CPU_TYPE=4

# Base envs (v5_combined: K_REGIONS=4 + SURR_STRUCTURED=1 + WARMSTART=1).
BASE="PLACER_ESC_K_REGIONS=4 PLACER_SURR_STRUCTURED=1 PLACER_WARMSTART=1"

# 4 configs to test on hard benches. Each variant targets a different
# hypothesis for moving cost from ~1.10–1.30 down toward 0.97 on hard benches.
declare -a CONFIGS=(
  "cluster30|$BASE PLACER_ESC_CLUSTER_FRAC=0.3"
  "cluster50|$BASE PLACER_ESC_CLUSTER_FRAC=0.5 PLACER_CLUSTER_MAX_SIZE=10"
  "cluster30_plateau2|$BASE PLACER_ESC_CLUSTER_FRAC=0.3 PLACER_PLATEAU_N=2 PLACER_ESC_HARD_DESTROY=120 PLACER_ESC_HARD_CAND=80"
  "cluster30_fasttourn16|$BASE PLACER_ESC_CLUSTER_FRAC=0.3 PLACER_EXP_FAST_TOURNAMENT=1 PLACER_FAST_TOURN_K=16"
)

BENCHES=(ibm12 ibm17 ibm18)
SEEDS=(42 43)
WT="/tmp/wt_v5_cluster"

TASKS=()
for ENTRY in "${CONFIGS[@]}"; do
  IFS='|' read -r LABEL ENVS <<< "$ENTRY"
  for SEED in "${SEEDS[@]}"; do
    for B in "${BENCHES[@]}"; do
      TASKS+=("ss_${LABEL}|$WT|$B|$SEED|$ENVS")
    done
  done
done

echo "[smart] queued ${#TASKS[@]} tasks (expected 24)"
echo "[smart] budget=${BUDGET}s max_parallel=$MAX_PARALLEL configs=${#CONFIGS[@]} benches=${#BENCHES[@]} seeds=${#SEEDS[@]}"

# Shuffle: mix configs/benches across batches for early visibility.
SHUFFLED=$(printf '%s\n' "${TASKS[@]}" | awk 'BEGIN{srand(3)} {print rand() "\t" $0}' | sort -k1,1n | cut -f2-)

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
    > "/tmp/smart_${exp_id}_${bench}_s${seed}.out" 2>&1 &
  active=$((active + 1))
  launched=$((launched + 1))
  echo "[smart] $launched/${#TASKS[@]} launched: $exp_id $bench s=$seed (active=$active)"
done <<< "$SHUFFLED"

wait
echo "[smart] DONE at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
