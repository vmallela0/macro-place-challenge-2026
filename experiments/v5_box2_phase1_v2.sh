#!/bin/bash
# experiments/v5_box2_phase1_v2.sh — Phase 1 retry with reduced concurrency.
#
# v1 launched 32-way and produced INVALID results (AVX-512 throttle exhausted
# legalize budget). v2 reduces to MAX_PARALLEL workers (default 16) and adds
# AVX2 BLAS hint to mitigate Skylake-SP all-cores throttling.
#
# Usage:
#   bash experiments/v5_box2_phase1_v2.sh [budget=3300] [max_parallel=16]
set -u
BUDGET="${1:-3300}"
MAX_PARALLEL="${2:-16}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# AVX2 BLAS hint: avoid AVX-512 frequency throttling on Skylake-SP.
export MKL_ENABLE_INSTRUCTIONS=AVX2
export MKL_DEBUG_CPU_TYPE=4

COMBINED_ENVS="PLACER_ESC_K_REGIONS=4 PLACER_SURR_STRUCTURED=1 PLACER_WARMSTART=1"
CLUSTER_ENVS="$COMBINED_ENVS PLACER_ESC_CLUSTER_FRAC=0.3"

BENCHES=(ibm01 ibm02 ibm03 ibm04 ibm06 ibm07 ibm08 ibm09 ibm10 ibm11 ibm12 ibm13 ibm14 ibm15 ibm16 ibm17 ibm18)
COMBINED_SEEDS=(43 44 45 46)
CLUSTER_SEEDS=(42 43 44 45 46)

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

echo "[phase1_v2] queued ${#TASKS[@]} tasks"
echo "[phase1_v2] budget=${BUDGET}s max_parallel=$MAX_PARALLEL MKL_AVX2=enabled"

# Deterministic shuffle for resumability + bench/seed mixing.
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
    > "/tmp/phase1v2_${exp_id}_${bench}_s${seed}.out" 2>&1 &
  active=$((active + 1))
  launched=$((launched + 1))
  if (( launched % 8 == 0 )); then
    echo "[phase1_v2] $launched/${#TASKS[@]} launched (active=$active)"
  fi
done <<< "$SHUFFLED"

wait
echo "[phase1_v2] DONE at $(date -u +%Y-%m-%dT%H:%M:%SZ): all 153 runs complete"
