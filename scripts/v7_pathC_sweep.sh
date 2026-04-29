#!/bin/bash
# Path C: Adam-only sweep (no basin-hop). Triggered after the ibm15 mini-grid
# confirmed 0/9 hop accepts — basin-hop's reduced-pipeline local minimizer
# can't recover from any meaningful perturbation within 300s.
#
# Stack:
#   v6 portfolio (8 workers + GPU CD + consensus)
#   → Layer 1: Laplacian soft-resolve
#   → Layer 3: Adam Phase 4.5 with vectorized LSE-HPWL + Segmented CVaR (sniper
#     congestion top-2%) + GradNorm component balancing
#   → strict-improvement gate via compute_proxy_cost
#
# Per-bench output: <OUT>/<bench>.log, <OUT>/<bench>.png, <OUT>/<bench>.npy
# Auto-updates submissions/vmallela_v7/README.md with the per-bench table.

set -u
cd "$(dirname "$0")/.."

WORKER_BUDGET=${WORKER_BUDGET:-1800}
HARD_TIMEOUT_S=${HARD_TIMEOUT_S:-2400}
RESULTS_DIR_BASE=${RESULTS_DIR_BASE:-/tmp}

OUT="$RESULTS_DIR_BASE/v7_pathC_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"

# Locked env (matches submissions/vmallela_v7/run.sh).
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

# v7: Laplacian on, basin-hop OFF (Path C pivot).
export PLACER_V7_LAPLACIAN=1
export PLACER_V7_LAPLACIAN_PASSES=2
export PLACER_V7_LAPLACIAN_BUDGET_FRAC=0.04
export PLACER_V7_BASIN_HOPS=0
export PLACER_V7_BASIN_HOP_AUTO=999.0     # never auto-trigger
export PLACER_V7_BASIN_HOP_RESERVE=0      # no reserve; portfolio uses full budget

# v7 Adam Phase 4.5 ON. 300 steps, segmented α (density 10%, congestion 2%).
# soft_only=1: hards held proximal (T3 lever available via SOFT_ONLY=0 + INERTIA).
export PLACER_V7_ADAM=1
export PLACER_V7_ADAM_STEPS=300
export PLACER_V7_ADAM_LR_FRAC=0.02
export PLACER_V7_ADAM_SOFT_ONLY=1
export PLACER_V7_ADAM_INERTIA=1.0
export PLACER_V7_K_DENS_FRAC=0.10
export PLACER_V7_K_CONG_FRAC=0.02

export PLACER_V6_SAVE_PLACEMENT="$OUT/{name}.npy"

BENCHES="ibm01 ibm02 ibm03 ibm04 ibm06 ibm07 ibm08 ibm09 ibm10 ibm11 ibm12 ibm13 ibm14 ibm15 ibm16 ibm17 ibm18"

echo "v7 Path C (Adam-only) sweep" > "$OUT/sweep.log"
echo "  started: $(date)" >> "$OUT/sweep.log"
echo "  worker budget: ${WORKER_BUDGET}s" >> "$OUT/sweep.log"
echo "  hard timeout: ${HARD_TIMEOUT_S}s" >> "$OUT/sweep.log"
echo "  results dir: $OUT" >> "$OUT/sweep.log"
echo "  Adam: steps=${PLACER_V7_ADAM_STEPS} K_d=${PLACER_V7_K_DENS_FRAC} K_c=${PLACER_V7_K_CONG_FRAC}" >> "$OUT/sweep.log"
echo "  Basin-hop: DISABLED" >> "$OUT/sweep.log"
echo "" >> "$OUT/sweep.log"

echo "benchmark,proxy_cost,wirelength_cost,density_cost,congestion_cost,overlap_count,wall_clock_s,exit_code,timestamp" \
  > "$OUT/results.csv"

for b in $BENCHES; do
  echo "=== $b: started $(date) ===" | tee -a "$OUT/sweep.log"
  t_start=$(date +%s)

  # macOS lacks `timeout`; emulate.
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

  # Surface ADAM acceptance + key trajectory data
  adam_status=$(grep -E "ADAM WIN|adam: rejected" "$OUT/${b}.log" | tail -1)
  echo "  proxy=${proxy:-NA} overlaps=${overlaps:-NA} | $adam_status" \
    | tee -a "$OUT/sweep.log"

  # Render placement plot. Goes to BOTH the run-archive dir and assets/.
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
echo "v7 Path C sweep finished: $(date)" >> "$OUT/sweep.log"

# Auto-update v7 README with per-bench table.
.venv/bin/python scripts/v7_results_to_readme.py "$OUT/results.csv" \
  >> "$OUT/sweep.log" 2>&1 || \
  echo "  (v7_results_to_readme.py missing; skipping README update)" \
    | tee -a "$OUT/sweep.log"

echo "DONE" >> "$OUT/sweep.log"
