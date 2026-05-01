#!/bin/bash
# Option A: clean v7=portfolio+Laplacian baseline sweep.
# Adam disabled (smooth-vs-exact divergence proven structural on hard
# benches). Basin-hop disabled (Gaussian σ-grid 0/9 acceptances; SP-swap
# basin-hop also strikes out — local-minimizer plateau ~+0.06 above
# post-Lap on ibm15 regardless of hop budget or worker count).
#
# This is the honest v7 ship: just the working layers (portfolio +
# Laplacian).  Per-bench wall ~1800s (full portfolio budget, no reserve).
# 17 benches × ~32 min = ~9h. Hard timeout 2400s/bench gives margin.

set -u
cd "$(dirname "$0")/.."

OUT="/tmp/v7_baseline_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"

WORKER_BUDGET=${WORKER_BUDGET:-1800}
HARD_TIMEOUT_S=${HARD_TIMEOUT_S:-2400}

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

# v7 baseline: Laplacian on, everything else off.
export PLACER_V7_LAPLACIAN=1
export PLACER_V7_LAPLACIAN_PASSES=2
export PLACER_V7_LAPLACIAN_BUDGET_FRAC=0.04
export PLACER_V7_BASIN_HOPS=0
export PLACER_V7_BASIN_HOP_AUTO=999.0
export PLACER_V7_BASIN_HOP_RESERVE=0
export PLACER_V7_ADAM=0

export PLACER_V6_SAVE_PLACEMENT="$OUT/{name}.npy"

BENCHES="ibm01 ibm02 ibm03 ibm04 ibm06 ibm07 ibm08 ibm09 ibm10 ibm11 ibm12 ibm13 ibm14 ibm15 ibm16 ibm17 ibm18"

echo "v7 baseline (portfolio + Laplacian only)" > "$OUT/sweep.log"
echo "  started: $(date)" >> "$OUT/sweep.log"
echo "  worker budget: ${WORKER_BUDGET}s" >> "$OUT/sweep.log"
echo "  Adam: DISABLED. Basin-hop: DISABLED." >> "$OUT/sweep.log"
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
  echo "  proxy=${proxy:-NA} overlaps=${overlaps:-NA}" \
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
echo "v7 baseline sweep finished: $(date)" >> "$OUT/sweep.log"

.venv/bin/python scripts/v7_results_to_readme.py "$OUT/results.csv" \
  >> "$OUT/sweep.log" 2>&1 || true

echo "DONE" >> "$OUT/sweep.log"
