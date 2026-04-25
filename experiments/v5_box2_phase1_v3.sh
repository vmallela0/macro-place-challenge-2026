#!/bin/bash
# experiments/v5_box2_phase1_v3.sh — Phase 1 retry, 8-way concurrency.
#
# Lessons from v1/v2 attempts on this 32-physical-core Skylake-SP box:
#   - 32-way (v1): Phase 1 produced INVALID rows — AVX-512 all-cores throttle
#     dropped per-process speed enough that legalize tournament couldn't find
#     a valid candidate within its 660s budget. Killed at 1086s wall.
#   - 16-way (ctest diagnostic): ibm10 still hung past budget — multi-process
#     at any concurrency may share the throttle issue (or just be 2-3x slower
#     per-thread than box1's Zen 5).
#   - Single-process: v5_escape_v2 ibm01 600s → 0.8071 VALID. Confirmed works.
#
# So we drop to 8-way (matches box1's tested config: 1 placer per ~4 physical
# cores), which avoids any AVX-512 throttle window. Wall is longer but
# results are valid.
#
# Design:
#   - v5_cluster only (the novel branch box1 isn't running): seeds {42,43,44,45}
#     × 17 benches = 68 runs.
#   - 8-way concurrent at 3300s = ~7.8h wall.
#   - MKL_AVX2 hint set as belt+suspenders.
#
# Usage: bash experiments/v5_box2_phase1_v3.sh [budget=3300] [max_parallel=8]
set -u
BUDGET="${1:-3300}"
MAX_PARALLEL="${2:-8}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

export MKL_ENABLE_INSTRUCTIONS=AVX2
export MKL_DEBUG_CPU_TYPE=4

CLUSTER_ENVS="PLACER_ESC_K_REGIONS=4 PLACER_SURR_STRUCTURED=1 PLACER_WARMSTART=1 PLACER_ESC_CLUSTER_FRAC=0.3"

BENCHES=(ibm01 ibm02 ibm03 ibm04 ibm06 ibm07 ibm08 ibm09 ibm10 ibm11 ibm12 ibm13 ibm14 ibm15 ibm16 ibm17 ibm18)
SEEDS=(42 43 44 45)

TASKS=()
for SEED in "${SEEDS[@]}"; do
  for B in "${BENCHES[@]}"; do
    TASKS+=("v5_cluster_box2|/tmp/wt_v5_cluster|$B|$SEED|$CLUSTER_ENVS")
  done
done

echo "[phase1_v3] queued ${#TASKS[@]} tasks (expected 68)"
echo "[phase1_v3] budget=${BUDGET}s max_parallel=$MAX_PARALLEL MKL_AVX2=enabled"

# Deterministic shuffle.
SHUFFLED=$(printf '%s\n' "${TASKS[@]}" | awk 'BEGIN{srand(2)} {print rand() "\t" $0}' | sort -k1,1n | cut -f2-)

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
    > "/tmp/phase1v3_${exp_id}_${bench}_s${seed}.out" 2>&1 &
  active=$((active + 1))
  launched=$((launched + 1))
  echo "[phase1_v3] $launched/${#TASKS[@]} launched: $bench s=$seed (active=$active)"
done <<< "$SHUFFLED"

wait
echo "[phase1_v3] DONE at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
