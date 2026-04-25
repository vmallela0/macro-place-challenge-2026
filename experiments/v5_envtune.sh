#!/bin/bash
# experiments/v5_envtune.sh — env-only tuning sweep on top of a base v5 config.
#
# After the main sweep + winner seeded-verification, run this on the
# hardest benches with extra env tweaks to chase sub-1.0 mean.
#
# Usage:
#   experiments/v5_envtune.sh <base_exp_id> [bench_set] [budget] [max_parallel]
#
# Examples:
#   experiments/v5_envtune.sh v5_combined hard 1200 8     # ibm12,17,18 only
#   experiments/v5_envtune.sh v5_cluster all 1800 8       # all 17 benches
#
# bench_set: 'hard' = ibm12,17,18; 'med' = ibm10,12,14,15,16,17,18; 'all' = 17 IBM
#
# Each combo × N benches generates a labelled exp_id like:
#   <base>+plateau2,<base>+esc120, etc.
set -u
BASE="${1:?usage: $0 <base_exp_id> [bench_set] [budget] [max_parallel]}"
BENCH_SET="${2:-hard}"
BUDGET="${3:-1200}"
MAX_PARALLEL="${4:-8}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Map base exp_id to its worktree + base envs (must mirror v5_full_sweep.sh).
case "$BASE" in
  v5_cells_skip)        WT=/tmp/wt_v5_cells; BASE_ENVS="";;
  v5_escape_v2)         WT=/tmp/wt_v5_escape; BASE_ENVS="PLACER_ESC_K_REGIONS=4";;
  v5_surrogate_struct)  WT=/tmp/wt_v5_surr; BASE_ENVS="PLACER_SURR_STRUCTURED=1";;
  v5_warmstart)         WT=/tmp/wt_v5_warm; BASE_ENVS="PLACER_WARMSTART=1";;
  v5_combined)          WT=/tmp/wt_v5_combined; BASE_ENVS="PLACER_ESC_K_REGIONS=4 PLACER_SURR_STRUCTURED=1 PLACER_WARMSTART=1";;
  v5_cluster)           WT=/tmp/wt_v5_cluster; BASE_ENVS="PLACER_ESC_K_REGIONS=4 PLACER_SURR_STRUCTURED=1 PLACER_WARMSTART=1 PLACER_ESC_CLUSTER_FRAC=0.3";;
  *) echo "unknown base: $BASE"; exit 2;;
esac

case "$BENCH_SET" in
  hard) BENCHES=(ibm12 ibm17 ibm18);;
  med)  BENCHES=(ibm10 ibm12 ibm14 ibm15 ibm16 ibm17 ibm18);;
  all)  BENCHES=(ibm01 ibm02 ibm03 ibm04 ibm06 ibm07 ibm08 ibm09 ibm10 ibm11 ibm12 ibm13 ibm14 ibm15 ibm16 ibm17 ibm18);;
  *) echo "unknown bench_set: $BENCH_SET"; exit 2;;
esac

# Env tweaks to test on top of base. Each tuple: <label>|<extra_envs>
TWEAKS=(
  "plateau2|PLACER_PLATEAU_N=2"
  "esc120|PLACER_ESC_HARD_DESTROY=120 PLACER_ESC_HARD_CAND=80"
  "linesearch|PLACER_EXP7_LINESEARCH=1"
  "topk5|PLACER_SURR_TOPK=5"
  "sa1e4|PLACER_SA_T0=0.0001"
  "plateau2_esc120|PLACER_PLATEAU_N=2 PLACER_ESC_HARD_DESTROY=120 PLACER_ESC_HARD_CAND=80"
)

echo "[envtune] base=$BASE wt=$WT base_envs='$BASE_ENVS'"
echo "[envtune] bench_set=$BENCH_SET (${#BENCHES[@]} benches), budget=${BUDGET}s, parallel=$MAX_PARALLEL"
echo "[envtune] ${#TWEAKS[@]} tweaks × ${#BENCHES[@]} benches = $(( ${#TWEAKS[@]} * ${#BENCHES[@]} )) runs"

active=0
for tweak in "${TWEAKS[@]}"; do
  IFS='|' read -r label extra <<< "$tweak"
  exp_id="${BASE}+${label}"
  ALL_ENVS="$BASE_ENVS $extra"
  IFS=' ' read -ra env_arr <<< "$ALL_ENVS"
  for B in "${BENCHES[@]}"; do
    while [ "$active" -ge "$MAX_PARALLEL" ]; do
      wait -n; active=$((active - 1))
    done
    bash "$REPO_ROOT/experiments/run_in_worktree.sh" "$exp_id" "$WT" "$B" 42 "$BUDGET" "${env_arr[@]:-}" \
      > "/tmp/envtune_${exp_id}_${B}.out" 2>&1 &
    active=$((active + 1))
    echo "[envtune] launched $exp_id $B (active=$active)"
  done
done
wait
echo "[envtune] done. inspect $REPO_ROOT/experiments/results.csv for ${BASE}+* rows"
