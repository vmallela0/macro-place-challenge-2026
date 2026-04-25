#!/bin/bash
# experiments/v5_box2_phase2.sh — v5_combined_pp on hard benches × 2 seeds.
#
# Phase 2: targeted, expensive variant. PARALLEL_WORKERS=4 → 5 cores per run.
# Hard benches only (ibm12, ibm17, ibm18) × seeds {43, 44} = 6 runs.
# 6 × 5 = 30 cores at peak (within 32 physical).
#
# Usage:
#   bash experiments/v5_box2_phase2.sh [budget=3300]
set -u
BUDGET="${1:-3300}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PP_ENVS="PLACER_ESC_K_REGIONS=4 PLACER_SURR_STRUCTURED=1 PLACER_WARMSTART=1 PLACER_PARALLEL_WORKERS=4"
WT="/tmp/wt_v5_combined"

BENCHES=(ibm12 ibm17 ibm18)
SEEDS=(43 44)

active=0
MAX_PARALLEL=6
for SEED in "${SEEDS[@]}"; do
  for B in "${BENCHES[@]}"; do
    while [ "$active" -ge "$MAX_PARALLEL" ]; do
      wait -n; active=$((active - 1))
    done
    IFS=' ' read -ra env_arr <<< "$PP_ENVS"
    bash "$WT/experiments/run_in_worktree.sh" v5_combined_pp_box2 "$WT" "$B" "$SEED" "$BUDGET" "${env_arr[@]:-}" \
      > "/tmp/phase2_pp_${B}_s${SEED}.out" 2>&1 &
    active=$((active + 1))
    echo "[phase2] launched pp $B s=$SEED (active=$active)"
  done
done
wait
echo "[phase2] DONE"
