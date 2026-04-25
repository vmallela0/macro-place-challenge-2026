#!/bin/bash
# experiments/v5_post_sweep.sh — fully autonomous follow-on after the main v5 sweep.
#
# Runs three phases:
#   Phase A: optimized_v4 baseline at seed 42 (17 runs) + v5_cluster at seed 42 (17 runs)
#            interleaved at 8-way parallel (~4h)
#   Phase B: pick winner across all 7 candidates (5 v5 + v5_cluster + optimized_v4)
#   Phase C: multi-seed verify (seeds 43+44, 17 benches each) on the winner (~4h)
#
# Triggered by tmux session "post" which polls until v5_full_sweep.sh exits.
# Usage:
#   experiments/v5_post_sweep.sh [budget] [max_parallel]
set -u
BUDGET="${1:-3300}"
MAX_PARALLEL="${2:-8}"

REPO=/home/vedu_mallela/macro-place-challenge-2026
HARNESS="/tmp/wt_v5_combined/experiments/run_in_worktree.sh"
PHASE_LOG="/tmp/v5_post.out"

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { echo "[$(ts)] $*" | tee -a "$PHASE_LOG"; }

# === Phase A: v4 baseline + v5_cluster at seed 42 (interleaved, 8-way) ===

log "== Phase A: v4 baseline + v5_cluster, seed 42, ${BUDGET}s, ${MAX_PARALLEL}-way =="

BENCHES=(ibm01 ibm02 ibm03 ibm04 ibm06 ibm07 ibm08 ibm09 ibm10 ibm11 ibm12 ibm13 ibm14 ibm15 ibm16 ibm17 ibm18)

# Build interleaved task list: alternate v4 and v5_cluster benches so they
# finish at roughly the same time, not 17-of-each in batches.
TASKS_A=()
for B in "${BENCHES[@]}"; do
  TASKS_A+=("baseline_v4_s42|/tmp/wt_optimized_v4|$B|")
  TASKS_A+=("v5_cluster|/tmp/wt_v5_cluster|$B|PLACER_ESC_K_REGIONS=4 PLACER_SURR_STRUCTURED=1 PLACER_WARMSTART=1 PLACER_ESC_CLUSTER_FRAC=0.3")
done

active=0
for task in "${TASKS_A[@]}"; do
  while [ "$active" -ge "$MAX_PARALLEL" ]; do
    wait -n; active=$((active - 1))
  done
  IFS='|' read -r exp_id wt bench envs <<< "$task"
  IFS=' ' read -ra env_arr <<< "$envs"
  bash "$HARNESS" "$exp_id" "$wt" "$bench" 42 "$BUDGET" "${env_arr[@]:-}" \
    > "/tmp/post_${exp_id}_${bench}.out" 2>&1 &
  active=$((active + 1))
  log "  launched $exp_id $bench (active=$active)"
done
wait
log "== Phase A done =="

# === Phase B: pick winner across all candidates ===

log "== Phase B: picking winner =="
WINNER_LINE=$("$REPO/.venv/bin/python" - <<'PY'
import csv
from collections import defaultdict
CSV = "/home/vedu_mallela/macro-place-challenge-2026/experiments/results.csv"
rows = defaultdict(list)
with open(CSV) as fh:
    for r in csv.DictReader(fh):
        if "sanity" in r["exp_id"]: continue
        try: c = float(r["final_cost"])
        except: continue
        rows[r["exp_id"]].append(c)
candidates = [(e, sum(c)/len(c)) for e, c in rows.items() if len(c) == 17 and "_s4" not in e]
if not candidates:
    print("NONE"); raise SystemExit
e, m = min(candidates, key=lambda t: t[1])
print(f"{e} {m:.4f}")
PY
)
log "  winner: $WINNER_LINE"
WINNER="${WINNER_LINE%% *}"

if [ "$WINNER" = "NONE" ] || [ -z "$WINNER" ]; then
  log "ERROR: no winner could be determined; stopping"
  exit 1
fi

# Map winner back to (worktree, env vars).
case "$WINNER" in
  baseline_v4_s42)      WT=/tmp/wt_optimized_v4; ENVS="";;
  v5_cells_skip)        WT=/tmp/wt_v5_cells; ENVS="";;
  v5_escape_v2)         WT=/tmp/wt_v5_escape; ENVS="PLACER_ESC_K_REGIONS=4";;
  v5_surrogate_struct)  WT=/tmp/wt_v5_surr; ENVS="PLACER_SURR_STRUCTURED=1";;
  v5_warmstart)         WT=/tmp/wt_v5_warm; ENVS="PLACER_WARMSTART=1";;
  v5_combined)          WT=/tmp/wt_v5_combined; ENVS="PLACER_ESC_K_REGIONS=4 PLACER_SURR_STRUCTURED=1 PLACER_WARMSTART=1";;
  v5_cluster)           WT=/tmp/wt_v5_cluster; ENVS="PLACER_ESC_K_REGIONS=4 PLACER_SURR_STRUCTURED=1 PLACER_WARMSTART=1 PLACER_ESC_CLUSTER_FRAC=0.3";;
  *) log "ERROR: unknown winner $WINNER"; exit 2;;
esac

log "  worktree=$WT envs='$ENVS'"

# === Phase C: multi-seed verify (seeds 43, 44) ===

log "== Phase C: multi-seed verify ${WINNER} on seeds 43, 44 =="
active=0
for SEED in 43 44; do
  for B in "${BENCHES[@]}"; do
    while [ "$active" -ge "$MAX_PARALLEL" ]; do
      wait -n; active=$((active - 1))
    done
    IFS=' ' read -ra env_arr <<< "$ENVS"
    bash "$HARNESS" "${WINNER}_s${SEED}" "$WT" "$B" "$SEED" "$BUDGET" "${env_arr[@]:-}" \
      > "/tmp/post_${WINNER}_s${SEED}_${B}.out" 2>&1 &
    active=$((active + 1))
    log "  launched ${WINNER}_s${SEED} $B (active=$active)"
  done
done
wait
log "== Phase C done =="

# === Push v5_cluster branch if it competes (and it has results) ===
if [ "$WINNER" = "v5_cluster" ]; then
  log "v5_cluster won — pushing branch to origin"
  git -C /tmp/wt_v5_cluster push origin v5_cluster 2>&1 | tee -a "$PHASE_LOG"
fi

log "== ALL POST-SWEEP PHASES COMPLETE =="
log "Inspect $REPO/experiments/results.csv for full data"
log "Run: python $REPO/experiments/per_bench_picker.py for per-bench breakdown"
