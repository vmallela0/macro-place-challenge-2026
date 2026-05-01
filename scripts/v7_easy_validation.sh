#!/bin/bash
# Validate easy-bench Hessian gain. ibm01 + ibm09 (smallest, v4 ~0.78 each).
# Same architecture as the production sweep (single-v4 + Hessian, 2300+1000s).
# Each bench ~30-45 min wall.
#
# Decision criteria after both finish:
#   - If avg Δ ≥ -0.008 below v4: easy benches are over-performing → sub-1.0 likely
#   - If avg Δ -0.003 to -0.008: matches projection → mean ~1.009 → sub-1.0 possible (40%)
#   - If avg Δ < -0.003: easies under-performing → mean ~1.012-1.015 → sub-1.0 unlikely

set -u
cd "$(dirname "$0")/.."

OUT="/tmp/v7_easy_validate_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"

WORKER_BUDGET=2300
HARD_TIMEOUT_S=3700

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8

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

export PLACER_V7_HESSIAN=1
export PLACER_V7_HESSIAN_STEPS="0.02,-0.02,0.05,-0.05"
export PLACER_V7_HESSIAN_BUDGET=1000
export PLACER_V7_HESSIAN_LANCZOS=50
export PLACER_V7_HESSIAN_MAX_ITERS=1

export PLACER_V6_SAVE_PLACEMENT="$OUT/{name}.npy"

BENCHES="ibm01 ibm09"

echo "v7 easy-bench Hessian validation" > "$OUT/sweep.log"
echo "  started: $(date)" >> "$OUT/sweep.log"
echo "  benches: $BENCHES (v4: ibm01=0.7803, ibm09=0.7785)" >> "$OUT/sweep.log"
echo "" >> "$OUT/sweep.log"

echo "benchmark,proxy_cost,wall_clock_s,exit_code" > "$OUT/results.csv"

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

  line=$(grep -E "^proxy=" "$OUT/${b}.log" | tail -1)
  proxy=$(echo "$line" | sed -E 's/.*proxy=([0-9.]+).*/\1/' | head -c 12)
  echo "${b},${proxy:-NA},${elapsed},${rc}" >> "$OUT/results.csv"

  hess_status=$(grep -E "(HESSIAN WIN|hessian: λ|hessian: no candidate)" "$OUT/${b}.log" | tail -2 | tr '\n' ' | ')
  echo "  proxy=${proxy:-NA} wall=${elapsed}s | $hess_status" \
    | tee -a "$OUT/sweep.log"
done

echo "" | tee -a "$OUT/sweep.log"
echo "VERDICT" | tee -a "$OUT/sweep.log"
cat "$OUT/results.csv" | tee -a "$OUT/sweep.log"
echo "DONE" >> "$OUT/sweep.log"
