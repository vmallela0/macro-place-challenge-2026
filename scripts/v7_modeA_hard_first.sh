#!/bin/bash
# Mode A — hard benches first, then easy. Adam HPWL+density only
# (congestion DISABLED). Pivot triggered by ibm12 Path C+ trajectory:
# step-25 exact-cost spiked +76% above baseline (1.193 → 2.099),
# proving the frozen-routing congestion surrogate is the dominant
# liar in the smooth proxy.
#
# Order: ibm12 ibm13 ibm14 ibm15 ibm16 ibm17 ibm18 ibm04 ibm06
#        ibm07 ibm08 ibm09 ibm10 ibm11
# (skipping ibm01-03; they ran clean in /tmp/v7_pathC_20260429_152926/
# and Adam was rejected on all three — Mode A won't change that since
# post-Laplacian on easies is already near-optimal.)

set -u
cd "$(dirname "$0")/.."

OUT="/tmp/v7_modeA_$(date +%Y%m%d_%H%M%S)"
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

export PLACER_V7_LAPLACIAN=1
export PLACER_V7_LAPLACIAN_PASSES=2
export PLACER_V7_LAPLACIAN_BUDGET_FRAC=0.04
export PLACER_V7_BASIN_HOPS=0
export PLACER_V7_BASIN_HOP_AUTO=999.0
export PLACER_V7_BASIN_HOP_RESERVE=0

# Mode A: HPWL + Density only (congestion DISABLED).
export PLACER_V7_ADAM=1
export PLACER_V7_ADAM_STEPS=100
export PLACER_V7_ADAM_LR_FRAC=0.02
export PLACER_V7_ADAM_SOFT_ONLY=1
export PLACER_V7_ADAM_INERTIA=1.0
export PLACER_V7_K_DENS_FRAC=0.10
export PLACER_V7_K_CONG_FRAC=0.05
export PLACER_V7_ADAM_SNAPSHOT_EVERY=10
export PLACER_V7_ADAM_VALIDATE_EVERY=25
export PLACER_V7_ADAM_ENABLE_DENS=1
export PLACER_V7_ADAM_ENABLE_CONG=0   # ← Mode A trigger

export PLACER_V6_SAVE_PLACEMENT="$OUT/{name}.npy"

BENCHES="ibm12 ibm13 ibm14 ibm15 ibm16 ibm17 ibm18 ibm04 ibm06 ibm07 ibm08 ibm09 ibm10 ibm11"

echo "v7 Mode A (HPWL+density Adam, no congestion) — hard first" > "$OUT/sweep.log"
echo "  started: $(date)" >> "$OUT/sweep.log"
echo "  results dir: $OUT" >> "$OUT/sweep.log"
echo "  Adam: steps=${PLACER_V7_ADAM_STEPS} K_d=${PLACER_V7_K_DENS_FRAC} (cong DISABLED)" >> "$OUT/sweep.log"
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
  adam_status=$(grep -E "ADAM WIN|adam: rejected" "$OUT/${b}.log" | tail -1)
  echo "  proxy=${proxy:-NA} overlaps=${overlaps:-NA} | $adam_status" \
    | tee -a "$OUT/sweep.log"

  if [ -f "$OUT/${b}.npy" ]; then
    .venv/bin/python scripts/v6_placement_plot.py "$b" "$OUT/${b}.npy" \
      "$OUT/${b}.png" >> "$OUT/sweep.log" 2>&1 || \
      echo "  plot $OUT/${b}.png failed" | tee -a "$OUT/sweep.log"
    .venv/bin/python scripts/v6_placement_plot.py "$b" "$OUT/${b}.npy" \
      "assets/v7_${b}.png" >> "$OUT/sweep.log" 2>&1 || \
      echo "  plot assets/v7_${b}.png failed" | tee -a "$OUT/sweep.log"
  else
    echo "  (no .npy saved for ${b}; skipping plot)" | tee -a "$OUT/sweep.log"
  fi
done

echo "" >> "$OUT/sweep.log"
echo "v7 Mode A finished: $(date)" >> "$OUT/sweep.log"

.venv/bin/python scripts/v7_results_to_readme.py "$OUT/results.csv" \
  >> "$OUT/sweep.log" 2>&1 || \
  echo "  (v7_results_to_readme.py missing; skipping README update)" \
    | tee -a "$OUT/sweep.log"

echo "DONE" >> "$OUT/sweep.log"
