#!/bin/bash
# Option C: SP basin-hop with 4-worker per-hop pool ("mini-portfolio per hop").
# Hard-first order: ibm15 first (we have prior baseline 1.1332 on this same
# bench from the 900s × 1-worker smoke — direct comparison). If ibm15
# rejects, kill and ship A.
#
# Per-bench wall: 1100s portfolio × 8 workers + 30s Lap + 1 SP hop ×
#   (4 workers × 900s in parallel = 900s wall) ≈ 2030s ≈ 34 min.
# 17 benches × 34 min ≈ 9.6h. Hard timeout 2700s/bench gives margin.

set -u
cd "$(dirname "$0")/.."

OUT="/tmp/v7_optionC_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"

WORKER_BUDGET=${WORKER_BUDGET:-1800}
HARD_TIMEOUT_S=${HARD_TIMEOUT_S:-2700}

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8

export PLACER_TOTAL_BUDGET=$WORKER_BUDGET
export PLACER_V6_WORKERS=8
export PLACER_V6_GPU_WORKERS=1
export PLACER_V6_CONSENSUS=1
export PLACER_V6_CONSENSUS_REFINE=120
export PLACER_V6_CONSENSUS_K=16
export PLACER_SA_T0=0.00005
export PLACER_ESC_HARD_DESTROY=80

export PLACER_V7_LAPLACIAN=1
export PLACER_V7_LAPLACIAN_PASSES=2
export PLACER_V7_LAPLACIAN_BUDGET_FRAC=0.04

# Option C: SP basin-hop with 4-worker pool per hop.
export PLACER_V7_BASIN_HOPS=1                     # 1 hop (per smoke; more time per hop didn't help)
export PLACER_V7_BASIN_HOP_BUDGET=900             # 900s per worker
export PLACER_V7_BASIN_HOP_RESERVE=950            # leaves 850s for portfolio
export PLACER_V7_BASIN_PERTURB=sp
export PLACER_V7_SP_N_SWAPS=3
export PLACER_V7_BASIN_HOP_AUTO=999.0             # force-only, no auto
export PLACER_V7_SP_MULTI_WORKERS=4                # ← Option C: 4 parallel workers per hop

# Adam disabled (smooth-vs-exact divergence proven structural)
export PLACER_V7_ADAM=0

export PLACER_V6_SAVE_PLACEMENT="$OUT/{name}.npy"

# Hard-first order so a strikeout on ibm15 stops us early.
# Order rationale:
#   ibm15: prior 1-worker SP smoke gave 1.1332 baseline + 1.1909 hop (rejected).
#          If 4-worker SP doesn't beat 1.1332, kill the sweep.
#   ibm17, ibm18: very-hard benches (v4 ≥ 1.29).
#   ibm14, ibm12, ibm16: medium-hard (v4 1.13-1.18).
#   ibm13, ibm04, 06-11: easies — if hard benches succeed, easies wrap up.
BENCHES="ibm15 ibm17 ibm18 ibm14 ibm12 ibm16 ibm13 ibm04 ibm06 ibm07 ibm08 ibm09 ibm10 ibm11 ibm01 ibm02 ibm03"

echo "v7 Option C: SP basin-hop, 4-worker per-hop pool, hard-first" > "$OUT/sweep.log"
echo "  started: $(date)" >> "$OUT/sweep.log"
echo "  results dir: $OUT" >> "$OUT/sweep.log"
echo "  Adam: DISABLED. Basin-hop: 1 hop × 4 workers × 900s." >> "$OUT/sweep.log"
echo "  order: $BENCHES" >> "$OUT/sweep.log"
echo "" >> "$OUT/sweep.log"

echo "benchmark,proxy_cost,wirelength_cost,density_cost,congestion_cost,overlap_count,wall_clock_s,exit_code,timestamp" \
  > "$OUT/results.csv"

for b in $BENCHES; do
  echo "=== $b: started $(date) ===" | tee -a "$OUT/sweep.log"
  t_start=$(date +%s)

  .venv/bin/python -m macro_place.evaluate \
    submissions/vmallela_v7/placer.py --benchmark "$b" \
    > "$OUT/${b}.log" 2>&1 &
  cmd_pid=$!
  ( sleep "$HARD_TIMEOUT_S"; if kill -0 $cmd_pid 2>/dev/null; then
      echo "  TIMEOUT after ${HARD_TIMEOUT_S}s; killing $cmd_pid" \
        | tee -a "$OUT/sweep.log"
      pkill -P $cmd_pid 2>/dev/null
      kill -TERM $cmd_pid 2>/dev/null
      sleep 5
      kill -KILL $cmd_pid 2>/dev/null
    fi ) &
  killer_pid=$!
  wait $cmd_pid 2>/dev/null
  rc=$?
  kill $killer_pid 2>/dev/null
  wait $killer_pid 2>/dev/null

  t_end=$(date +%s)
  elapsed=$((t_end - t_start))
  echo "  exit code: $rc, wall: ${elapsed}s" | tee -a "$OUT/sweep.log"

  line=$(grep -E "^proxy=" "$OUT/${b}.log" | tail -1)
  proxy=$(echo "$line" | sed -E 's/.*proxy=([0-9.]+).*/\1/' | head -c 12)
  wl=$(echo "$line"    | sed -E 's/.*wl=([0-9.]+).*/\1/'    | head -c 12)
  den=$(echo "$line"   | sed -E 's/.*den=([0-9.]+).*/\1/'   | head -c 12)
  cong=$(echo "$line"  | sed -E 's/.*cong=([0-9.]+).*/\1/'  | head -c 12)
  if echo "$line" | grep -q "VALID"; then
    overlaps=0
  else
    overlaps=$(grep -E "overlaps=" "$OUT/${b}.log" | tail -1 | sed -E 's/.*overlaps=([0-9]+).*/\1/' | head -c 8)
    [ -z "$overlaps" ] && overlaps="?"
  fi

  echo "${b},${proxy:-NA},${wl:-NA},${den:-NA},${cong:-NA},${overlaps:-NA},${elapsed},${rc},$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    >> "$OUT/results.csv"

  hop_status=$(grep -E "(ACCEPTED|hops accepted|stays |basin-hop done)" "$OUT/${b}.log" | tail -2 | tr '\n' ' | ')
  echo "  proxy=${proxy:-NA} overlaps=${overlaps:-NA} | $hop_status" \
    | tee -a "$OUT/sweep.log"

  if [ -f "$OUT/${b}.npy" ]; then
    .venv/bin/python scripts/v6_placement_plot.py "$b" "$OUT/${b}.npy" \
      "$OUT/${b}.png" >> "$OUT/sweep.log" 2>&1 || \
      echo "  plot $OUT/${b}.png failed" | tee -a "$OUT/sweep.log"
    .venv/bin/python scripts/v6_placement_plot.py "$b" "$OUT/${b}.npy" \
      "assets/v7_${b}.png" >> "$OUT/sweep.log" 2>&1 || true
  else
    echo "  (no .npy saved for ${b}; skipping plot)" | tee -a "$OUT/sweep.log"
  fi
done

echo "" >> "$OUT/sweep.log"
echo "v7 Option C finished: $(date)" >> "$OUT/sweep.log"

.venv/bin/python scripts/v7_results_to_readme.py "$OUT/results.csv" \
  >> "$OUT/sweep.log" 2>&1 || true

echo "DONE" >> "$OUT/sweep.log"
