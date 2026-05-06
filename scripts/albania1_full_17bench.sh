#!/bin/bash
# Full 17-bench sweep with the validated config:
#   - MAX_ITERS=3 (iterative Hessian, 3 saddle crossings)
#   - HESSIAN_BUDGET=400 per iter (1200s total Hessian)
#   - electro-norm density (DREAMPlace-style global structure)
#   - cong-off (matches verified config; cong-aware Hessian was bench-noise)
#
# Validated on ibm06: MAX_ITERS=3 gave -0.024 total lift (vs typical
# -0.011 single-iter) — 2× the lift.
#
# Order: highest-room benches first so early results indicate trajectory.
# 17 benches × ~1h = ~17h total wall.

set -u
cd "$(dirname "$0")/.."

OUT="/tmp/albania1_full17_$(date +%Y%m%d_%H%M%S)"
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
# THE CONFIG:
export PLACER_V7_HESSIAN=1
export PLACER_V7_HESSIAN_BUDGET=400
export PLACER_V7_HESSIAN_MAX_ITERS=3
export PLACER_V7_HESSIAN_TOTAL_BUDGET=1300
export PLACER_V7_HESSIAN_LANCZOS=100
export PLACER_V7_HESSIAN_TIKHONOV=1e-4
export PLACER_V7_HALO_FRAC=0.0
export PLACER_V7_K_DENS_FRAC=0.10
export PLACER_V7_K_CONG_FRAC=0.05
export PLACER_V7_ORIENTATION_FLIP=1
export PLACER_V7_HESSIAN_HPWL_WEIGHT=1.0
export PLACER_V7_HESSIAN_DENS_WEIGHT=0.5
export PLACER_V7_HESSIAN_CONG=0
export PLACER_V7_HESSIAN_ELECTROSTATIC=1
export PLACER_V7_HESSIAN_ELECTRO_NORM=1
export PLACER_V7_HESSIAN_ELECTRO_WEIGHT=0.5

# High-room first → low-room last.
BENCHES="ibm12 ibm06 ibm18 ibm07 ibm03 ibm08 ibm15 ibm02 ibm04 ibm16 ibm17 ibm13 ibm10 ibm14 ibm01 ibm11 ibm09"

# Verified baselines for delta reporting (bash 3.2 compatible).
verified_for() {
  case "$1" in
    ibm01) echo "0.7653" ;;
    ibm02) echo "0.9482" ;;
    ibm03) echo "0.9166" ;;
    ibm04) echo "0.9287" ;;
    ibm06) echo "1.0546" ;;
    ibm07) echo "1.0324" ;;
    ibm08) echo "1.0291" ;;
    ibm09) echo "0.7628" ;;
    ibm10) echo "0.9492" ;;
    ibm11) echo "0.8013" ;;
    ibm12) echo "1.1557" ;;
    ibm13) echo "0.8757" ;;
    ibm14) echo "1.1070" ;;
    ibm15) echo "1.0835" ;;
    ibm16) echo "1.0435" ;;
    ibm17) echo "1.2813" ;;
    ibm18) echo "1.2697" ;;
    *)     echo "?" ;;
  esac
}

echo "albania1 full 17-bench sweep (MAX_ITERS=3 + electro-norm)" > "$OUT/sweep.log"
echo "  started: $(date)" >> "$OUT/sweep.log"
echo "  results dir: $OUT" >> "$OUT/sweep.log"
echo "  config: HESSIAN_BUDGET=400, MAX_ITERS=3, ELECTROSTATIC=1, NORM=1, weight=0.5, cong-off" >> "$OUT/sweep.log"
echo "  benches (high-room first): $BENCHES" >> "$OUT/sweep.log"
echo "  ETA: ~17h (17 runs × ~1h)" >> "$OUT/sweep.log"
echo "" >> "$OUT/sweep.log"

echo "benchmark,proxy_cost,wirelength_cost,density_cost,congestion_cost,overlap_count,wall_clock_s,exit_code,verified,delta,timestamp" \
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
  verified=$(verified_for "$b")
  if [ "$proxy" != "NA" ] && [ -n "$proxy" ]; then
    delta=$(awk -v p="$proxy" -v v="$verified" 'BEGIN { printf "%+.4f", p-v }')
  else delta="?"
  fi
  echo "${b},${proxy:-NA},${wl:-NA},${den:-NA},${cong:-NA},${overlaps:-NA},${elapsed},${rc},${verified},${delta},$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    >> "$OUT/results.csv"
  echo "  $b proxy=${proxy:-NA} (verified=$verified, Δ=$delta) cong=${cong:-NA} wall=${elapsed}s" \
    | tee -a "$OUT/sweep.log"

  # Running mean update
  mean=$(awk -F, 'NR>1 && $2!="NA" && $2!="" {s+=$2; n++} END {if(n) printf "%.4f", s/n}' "$OUT/results.csv")
  v_mean=$(awk -F, 'NR>1 && $9!="" {s+=$9; n++} END {if(n) printf "%.4f", s/n}' "$OUT/results.csv")
  d_mean=$(awk -v m="$mean" -v vm="$v_mean" 'BEGIN { printf "%+.4f", m-vm }')
  echo "  running mean: $mean (verified mean of completed: $v_mean, Δ=$d_mean)" \
    | tee -a "$OUT/sweep.log"
done

# Final summary
echo "" >> "$OUT/sweep.log"
echo "=== FINAL ===" >> "$OUT/sweep.log"
mean=$(awk -F, 'NR>1 && $2!="NA" && $2!="" {s+=$2; n++} END {if(n) printf "%.4f", s/n}' "$OUT/results.csv")
v_mean=$(awk -F, 'NR>1 && $9!="" {s+=$9; n++} END {if(n) printf "%.4f", s/n}' "$OUT/results.csv")
n_valid=$(awk -F, 'NR>1 && $2!="NA" && $2!="" {n++} END {print n+0}' "$OUT/results.csv")
echo "  $n_valid/17 valid benches" >> "$OUT/sweep.log"
echo "  albania1 mean: $mean" >> "$OUT/sweep.log"
echo "  verified mean (matched benches): $v_mean" >> "$OUT/sweep.log"
echo "  Δ mean: $(awk -v m=$mean -v vm=$v_mean 'BEGIN { printf "%+.4f", m-vm }')" >> "$OUT/sweep.log"
echo "DONE" >> "$OUT/sweep.log"
