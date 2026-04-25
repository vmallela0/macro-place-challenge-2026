#!/bin/bash
# experiments/run_winner_seeds.sh — multi-seed verification of the v5 sweep winner.
#
# After the main sweep (seed 42) completes, picks the winning v5 branch from
# experiments/results.csv (lowest mean over 17 VALID benches), then runs all
# 17 benches × 2 extra seeds (43, 44) on that branch's worktree.
#
# Usage:
#   experiments/run_winner_seeds.sh [budget] [max_parallel]
#   experiments/run_winner_seeds.sh                    # defaults: 3300s, 8-way
#   experiments/run_winner_seeds.sh 1800 4             # cheaper alternative
#
# Wall: 34 runs × budget / max_parallel. At 8-way × 3300s ≈ 4 hours.
set -u
BUDGET="${1:-3300}"
MAX_PARALLEL="${2:-8}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CSV="$REPO_ROOT/experiments/results.csv"

if [ ! -f "$CSV" ]; then
  echo "[winner_seeds] no results.csv — run the main sweep first"; exit 1
fi

# Pick winner via the picker script's brain — lowest mean over 17 VALID rows.
WINNER_LINE=$("$REPO_ROOT/.venv/bin/python" - "$CSV" <<'PY'
import csv, sys
from collections import defaultdict
rows = defaultdict(list)
with open(sys.argv[1]) as fh:
    for r in csv.DictReader(fh):
        if "sanity" in r["exp_id"]: continue
        try: c = float(r["final_cost"])
        except: continue
        rows[r["exp_id"]].append(c)
candidates = [(e, sum(c)/len(c)) for e, c in rows.items() if len(c) == 17]
if not candidates:
    print("NONE")
else:
    e, m = min(candidates, key=lambda t: t[1])
    print(f"{e} {m:.4f}")
PY
)

if [ "$WINNER_LINE" = "NONE" ]; then
  echo "[winner_seeds] no branch with all 17 VALID yet — wait for sweep to complete"; exit 1
fi

WINNER="${WINNER_LINE%% *}"
WINNER_MEAN="${WINNER_LINE##* }"
echo "[winner_seeds] winner: $WINNER  seed-42 mean: $WINNER_MEAN"

# Map exp_id back to (worktree, env vars). MUST match v5_full_sweep.sh.
case "$WINNER" in
  v5_cells_skip)        WT=/tmp/wt_v5_cells; ENVS="";;
  v5_escape_v2)         WT=/tmp/wt_v5_escape; ENVS="PLACER_ESC_K_REGIONS=4";;
  v5_surrogate_struct)  WT=/tmp/wt_v5_surr; ENVS="PLACER_SURR_STRUCTURED=1";;
  v5_warmstart)         WT=/tmp/wt_v5_warm; ENVS="PLACER_WARMSTART=1";;
  v5_combined)          WT=/tmp/wt_v5_combined; ENVS="PLACER_ESC_K_REGIONS=4 PLACER_SURR_STRUCTURED=1 PLACER_WARMSTART=1";;
  v5_cluster)           WT=/tmp/wt_v5_cluster; ENVS="PLACER_ESC_K_REGIONS=4 PLACER_SURR_STRUCTURED=1 PLACER_WARMSTART=1 PLACER_ESC_CLUSTER_FRAC=0.3";;
  *) echo "[winner_seeds] unknown winner: $WINNER"; exit 2;;
esac

if [ ! -d "$WT" ]; then echo "[winner_seeds] worktree $WT missing"; exit 2; fi
echo "[winner_seeds] worktree: $WT  envs: $ENVS"
echo "[winner_seeds] launching 17 benches × seeds 43,44 = 34 runs at ${MAX_PARALLEL}-way × ${BUDGET}s"

BENCHES=(ibm01 ibm02 ibm03 ibm04 ibm06 ibm07 ibm08 ibm09 ibm10 ibm11 ibm12 ibm13 ibm14 ibm15 ibm16 ibm17 ibm18)

active=0
for SEED in 43 44; do
  for B in "${BENCHES[@]}"; do
    while [ "$active" -ge "$MAX_PARALLEL" ]; do
      wait -n; active=$((active - 1))
    done
    IFS=' ' read -ra env_arr <<< "$ENVS"
    bash "$REPO_ROOT/experiments/run_in_worktree.sh" "${WINNER}_s${SEED}" "$WT" "$B" "$SEED" "$BUDGET" "${env_arr[@]:-}" \
      > "/tmp/winner_${WINNER}_s${SEED}_${B}.out" 2>&1 &
    active=$((active + 1))
    echo "[winner_seeds] launched ${WINNER}_s${SEED} $B (active=$active)"
  done
done

wait
echo "[winner_seeds] done. inspect $CSV for ${WINNER}_s43 and ${WINNER}_s44 rows"
