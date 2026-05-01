#!/bin/bash
# Resume sweep: 12 remaining benches after the 5 validated.
# Already done in /tmp/v7_singlev4_sweep_20260430_175923/ + easy_validate dirs:
#   ibm15 = 1.0835 (Δ -0.0194 vs v4)
#   ibm17 = 1.2813 (Δ -0.0199)
#   ibm18 = 1.2697 (Δ -0.0168)
#   ibm01 = 0.7653 (Δ -0.0150)
#   ibm09 = 0.7628 (Δ -0.0157)
#   avg Δ -0.0174 across 5
# Remaining: ibm12, ibm14, ibm16, ibm13, ibm04, ibm06, ibm07, ibm08,
#            ibm10, ibm11, ibm02, ibm03
# Hard-first ordering of remaining: ibm12, ibm14, ibm16, ibm13 first.
#
# 12 × ~57 min = ~11.5 hours. ETA done ~10:30 AM PDT.

set -u
cd "$(dirname "$0")/.."

OUT="/tmp/v7_singlev4_resume_$(date +%Y%m%d_%H%M%S)"
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

# Hard-first remaining
BENCHES="ibm12 ibm14 ibm16 ibm13 ibm04 ibm06 ibm07 ibm08 ibm10 ibm11 ibm02 ibm03"

echo "v7 single-v4 + Hessian RESUME sweep (12 benches)" > "$OUT/sweep.log"
echo "  started: $(date)" >> "$OUT/sweep.log"
echo "  results dir: $OUT" >> "$OUT/sweep.log"
echo "  config: 2300s v4 + 30s Lap + 1000s Hessian" >> "$OUT/sweep.log"
echo "  hard timeout: ${HARD_TIMEOUT_S}s (placer ≤ 3600s, +100s plot slack)" >> "$OUT/sweep.log"
echo "  order: $BENCHES" >> "$OUT/sweep.log"
echo "" >> "$OUT/sweep.log"
echo "  Already validated:" >> "$OUT/sweep.log"
echo "    ibm15 = 1.0835 (Δ -0.0194)" >> "$OUT/sweep.log"
echo "    ibm17 = 1.2813 (Δ -0.0199)" >> "$OUT/sweep.log"
echo "    ibm18 = 1.2697 (Δ -0.0168)" >> "$OUT/sweep.log"
echo "    ibm01 = 0.7653 (Δ -0.0150)" >> "$OUT/sweep.log"
echo "    ibm09 = 0.7628 (Δ -0.0157)" >> "$OUT/sweep.log"
echo "    avg Δ = -0.0174" >> "$OUT/sweep.log"
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

  # If the bash hard-timed-out, the placer may have finished but bash
  # didn't read the proxy line. Try to recover from the [v7] DONE log.
  if [ "${proxy:-NA}" = "NA" ]; then
    done_cost=$(grep -E "\[v7\] DONE: cost=" "$OUT/${b}.log" \
                | tail -1 | sed -E 's/.*cost=([0-9.]+).*/\1/' | head -c 12)
    if [ -n "$done_cost" ]; then
      proxy=$done_cost
      overlaps=0
      echo "  [recovered from log] proxy=$proxy" | tee -a "$OUT/sweep.log"
    fi
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
echo "v7 single-v4 + Hessian RESUME sweep finished: $(date)" >> "$OUT/sweep.log"
echo "DONE" >> "$OUT/sweep.log"
