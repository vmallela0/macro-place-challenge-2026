#!/bin/bash
# Deploy normalized electrostatic-density Hessian on the 5 high-room
# benches. Compare each to verified baseline.
#
# Validated on ibm12 in clean A/B: electro_norm 1.1422 vs CVaR 1.1449
# (Δ -0.0027) and vs verified 1.1557 (Δ -0.0135). Both clean wins.
#
# This sweep is the rolled-out version of the validated config.

set -u
cd "$(dirname "$0")/.."

OUT="/tmp/albania1_electro_norm_sweep_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"

WORKER_BUDGET=${WORKER_BUDGET:-2300}
HARD_TIMEOUT_S=${HARD_TIMEOUT_S:-3700}

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
export PLACER_V7_HESSIAN=1
export PLACER_V7_HESSIAN_BUDGET=1000
export PLACER_V7_HESSIAN_LANCZOS=100
export PLACER_V7_HESSIAN_TIKHONOV=1e-4
export PLACER_V7_HESSIAN_MAX_ITERS=1
export PLACER_V7_HALO_FRAC=0.0
export PLACER_V7_K_DENS_FRAC=0.10
export PLACER_V7_K_CONG_FRAC=0.05
export PLACER_V7_ORIENTATION_FLIP=1
export PLACER_V7_HESSIAN_HPWL_WEIGHT=1.0
export PLACER_V7_HESSIAN_DENS_WEIGHT=0.5
export PLACER_V7_HESSIAN_CONG=0   # cong-off (focused sweep showed cong hurts ibm06)
# THE VALIDATED CONFIG: normalized electrostatic, weight=0.5
export PLACER_V7_HESSIAN_ELECTROSTATIC=1
export PLACER_V7_HESSIAN_ELECTRO_NORM=1
export PLACER_V7_HESSIAN_ELECTRO_WEIGHT=0.5

BENCHES="ibm12 ibm06 ibm18 ibm07 ibm03"

echo "albania1 electro-norm validated sweep" > "$OUT/sweep.log"
echo "  started: $(date)" >> "$OUT/sweep.log"
echo "  results dir: $OUT" >> "$OUT/sweep.log"
echo "  config: ELECTROSTATIC=1, NORM=1, weight=0.5, cong-off" >> "$OUT/sweep.log"
echo "  benches: $BENCHES" >> "$OUT/sweep.log"
echo "  ETA: ~5h (5 runs × ~57 min)" >> "$OUT/sweep.log"
echo "" >> "$OUT/sweep.log"

echo "benchmark,proxy_cost,wirelength_cost,density_cost,congestion_cost,overlap_count,wall_clock_s,exit_code,timestamp" \
  > "$OUT/results.csv"

for b in $BENCHES; do
  echo "=== $b: started $(date) ===" | tee -a "$OUT/sweep.log"
  t_start=$(date +%s)
  log="$OUT/${b}.log"
  .venv/bin/python -u -m macro_place.evaluate \
    submissions/vmallela_v7/placer.py --benchmark "$b" > "$log" 2>&1 &
  cmd_pid=$!
  ( sleep "$HARD_TIMEOUT_S"
    kill -0 $cmd_pid 2>/dev/null && kill -KILL $cmd_pid 2>/dev/null ) &
  killer_pid=$!
  wait $cmd_pid 2>/dev/null
  rc=$?
  kill $killer_pid 2>/dev/null
  elapsed=$(($(date +%s) - t_start))

  line=$(grep -E "^proxy=" "$log" | tail -1)
  proxy=$(echo "$line" | sed -E 's/.*proxy=([0-9.]+).*/\1/' | head -c 12)
  wl=$(echo "$line"    | sed -E 's/.*wl=([0-9.]+).*/\1/'    | head -c 12)
  den=$(echo "$line"   | sed -E 's/.*den=([0-9.]+).*/\1/'   | head -c 12)
  cong=$(echo "$line"  | sed -E 's/.*cong=([0-9.]+).*/\1/'  | head -c 12)
  if echo "$line" | grep -q "VALID"; then overlaps=0
  else overlaps=$(grep -E "overlaps=" "$log" | tail -1 | sed -E 's/.*overlaps=([0-9]+).*/\1/' | head -c 8); [ -z "$overlaps" ] && overlaps="?"
  fi
  echo "${b},${proxy:-NA},${wl:-NA},${den:-NA},${cong:-NA},${overlaps:-NA},${elapsed},${rc},$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    >> "$OUT/results.csv"
  case "$b" in
    ibm03) base="0.9166" ;;
    ibm06) base="1.0546" ;;
    ibm07) base="1.0324" ;;
    ibm12) base="1.1557" ;;
    ibm18) base="1.2697" ;;
    *)     base="?" ;;
  esac
  if [ "$proxy" != "NA" ] && [ "$base" != "?" ]; then
    delta=$(awk -v p="$proxy" -v b="$base" 'BEGIN { printf "%+.4f", p-b }')
  else delta="?"
  fi
  echo "  $b electro-norm proxy=${proxy:-NA} (verified=$base, Δ=$delta)" \
    | tee -a "$OUT/sweep.log"
done

echo "" >> "$OUT/sweep.log"
echo "DONE" >> "$OUT/sweep.log"
