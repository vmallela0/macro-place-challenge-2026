#!/bin/bash
# Single-v4 + Hessian full 17-bench compliant sweep.
# Validated config: ibm15 → 1.0924 in 3394s wall (Δ -0.011 vs v4).
#
# Architecture:
#   v4 single worker × 2300s (matches v4's 3300s baseline at 70% time)
#   Laplacian ~30s
#   Hessian Phase 4.6: 8 candidates × 1000s parallel
#   Wall ≤ 3600s with 200s margin
#
# 17 benches × ~57 min = ~16h. ETA finish ~10:30 AM PDT next day.

set -u
cd "$(dirname "$0")/.."

OUT="/tmp/v7_singlev4_sweep_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"

WORKER_BUDGET=${WORKER_BUDGET:-2300}
# Bash timeout 3700s = 100s slack for PNG generation AFTER the placer
# finishes. The PLACER itself runs in ≤ 3600s (validated on ibm17:
# 3571s wall placer time, then plot generation takes ~30s); the
# competition rule is on the algorithm wall time, not bookkeeping.
HARD_TIMEOUT_S=${HARD_TIMEOUT_S:-3700}

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Single v4 worker (no portfolio diversity, no consensus).
export PLACER_TOTAL_BUDGET=$WORKER_BUDGET
export PLACER_V6_WORKERS=1
export PLACER_V6_GPU_WORKERS=0
export PLACER_V6_CONSENSUS=0
export PLACER_SA_T0=0.00005
export PLACER_ESC_HARD_DESTROY=80

export PLACER_V7_LAPLACIAN=1
export PLACER_V7_LAPLACIAN_PASSES=2
export PLACER_V7_LAPLACIAN_BUDGET_FRAC=0.04
export PLACER_V7_BASIN_HOPS=0
export PLACER_V7_BASIN_HOP_AUTO=999.0
export PLACER_V7_BASIN_HOP_RESERVE=0
export PLACER_V7_ADAM=0
export PLACER_V7_EVICT=0
export PLACER_V7_SINKHORN=0

# Phase 4.6 Hessian
export PLACER_V7_HESSIAN=1
export PLACER_V7_HESSIAN_STEPS="0.02,-0.02,0.05,-0.05"
export PLACER_V7_HESSIAN_BUDGET=1000
export PLACER_V7_HESSIAN_LANCZOS=50
export PLACER_V7_HESSIAN_MAX_ITERS=1

export PLACER_V6_SAVE_PLACEMENT="$OUT/{name}.npy"

# Hard-first ordering
BENCHES="ibm15 ibm17 ibm18 ibm12 ibm14 ibm16 ibm13 ibm04 ibm06 ibm07 ibm08 ibm09 ibm10 ibm11 ibm01 ibm02 ibm03"

echo "v7 single-v4 + Hessian full 17-bench sweep" > "$OUT/sweep.log"
echo "  started: $(date)" >> "$OUT/sweep.log"
echo "  results dir: $OUT" >> "$OUT/sweep.log"
echo "  config: 2300s v4 + 30s Lap + 1000s Hessian = ~3380s wall" >> "$OUT/sweep.log"
echo "  order: $BENCHES" >> "$OUT/sweep.log"
echo "  ibm15 smoke result: 1.0924 (Δ -0.011 vs v4)" >> "$OUT/sweep.log"
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

  hess_status=$(grep -E "(HESSIAN WIN|hessian: λ|hessian: no candidate)" "$OUT/${b}.log" | tail -2 | tr '\n' ' | ')
  echo "  proxy=${proxy:-NA} overlaps=${overlaps:-NA} | $hess_status" \
    | tee -a "$OUT/sweep.log"

  if [ -f "$OUT/${b}.npy" ]; then
    .venv/bin/python scripts/v6_placement_plot.py "$b" "$OUT/${b}.npy" \
      "$OUT/${b}.png" >> "$OUT/sweep.log" 2>&1 || true
    .venv/bin/python scripts/v6_placement_plot.py "$b" "$OUT/${b}.npy" \
      "assets/v7_${b}.png" >> "$OUT/sweep.log" 2>&1 || true
  fi
done

echo "" >> "$OUT/sweep.log"
echo "v7 single-v4 + Hessian sweep finished: $(date)" >> "$OUT/sweep.log"
.venv/bin/python scripts/v7_results_to_readme.py "$OUT/results.csv" \
  >> "$OUT/sweep.log" 2>&1 || true
echo "DONE" >> "$OUT/sweep.log"
