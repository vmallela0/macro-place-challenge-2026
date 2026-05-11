#!/bin/bash
# zeus full-17 sweep — production config (matches albania1_full_17bench.sh)
# PLUS PLACER_V7_HESSIAN_CONG=1 + PLACER_V7_HESSIAN_RUDY=1.
#
# This is the production candidate. Fires only after the 3-bench A/B
# confirms positive signal.
#
# 17 benches × ~1h each, run in waves of 8 to fit 64 cores at ~8 worker
# procs per placer (PLACER_V6_WORKERS=1 but Hessian spawns its own pool).
# Total wall: ~3h with 8-wide parallelism.

set -u
cd "$(dirname "$0")/.."

OUT="/tmp/zeus_full17_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"

WORKER_BUDGET=${WORKER_BUDGET:-2300}
HARD_TIMEOUT_S=${HARD_TIMEOUT_S:-3700}
PARALLEL=${PARALLEL:-8}

# Common env — mirrors scripts/albania1_full_17bench.sh exactly.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
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
export PLACER_V7_HESSIAN_BUDGET=400
export PLACER_V7_HESSIAN_MAX_ITERS=3
export PLACER_V7_HESSIAN_TOTAL_BUDGET=1300
export PLACER_V7_HESSIAN_LANCZOS=100
export PLACER_V7_HESSIAN_TIKHONOV=1e-4
export PLACER_V7_HESSIAN_ADAPTIVE=1
export PLACER_V7_HESSIAN_ADAPTIVE_TOPK=1
export PLACER_V7_HALO_FRAC=0.0
export PLACER_V7_K_DENS_FRAC=0.10
export PLACER_V7_K_CONG_FRAC=0.05
export PLACER_V7_ORIENTATION_FLIP=1
export PLACER_V7_HESSIAN_HPWL_WEIGHT=1.0
export PLACER_V7_HESSIAN_DENS_WEIGHT=0.5
export PLACER_V7_HESSIAN_ELECTROSTATIC=1
export PLACER_V7_HESSIAN_ELECTRO_NORM=1
export PLACER_V7_HESSIAN_ELECTRO_WEIGHT=0.5
# zeus: cong-aware Hessian with differentiable RUDY routing demand.
export PLACER_V7_HESSIAN_CONG=1
export PLACER_V7_HESSIAN_CONG_WEIGHT=0.5
export PLACER_V7_HESSIAN_RUDY=1
export PLACER_V7_HESSIAN_RUDY_MARGIN=4
export PLACER_V7_HESSIAN_RUDY_MAX_WINDOW=64

BENCHES="ibm12 ibm06 ibm18 ibm07 ibm03 ibm08 ibm15 ibm02 ibm04 ibm16 ibm17 ibm13 ibm10 ibm14 ibm01 ibm11 ibm09"

verified_for() {
  case "$1" in
    ibm01) echo "0.7653" ;; ibm02) echo "0.9482" ;; ibm03) echo "0.9166" ;;
    ibm04) echo "0.9287" ;; ibm06) echo "1.0546" ;; ibm07) echo "1.0324" ;;
    ibm08) echo "1.0291" ;; ibm09) echo "0.7628" ;; ibm10) echo "0.9492" ;;
    ibm11) echo "0.8013" ;; ibm12) echo "1.1557" ;; ibm13) echo "0.8757" ;;
    ibm14) echo "1.1070" ;; ibm15) echo "1.0835" ;; ibm16) echo "1.0435" ;;
    ibm17) echo "1.2813" ;; ibm18) echo "1.2697" ;; *) echo "?" ;;
  esac
}

echo "zeus full-17 sweep (RUDY-differentiable cong-aware Hessian)" > "$OUT/sweep.log"
echo "  started: $(date)  out: $OUT  parallel: $PARALLEL" >> "$OUT/sweep.log"
echo "  budget: $WORKER_BUDGET/bench  timeout: $HARD_TIMEOUT_S" >> "$OUT/sweep.log"
echo "" >> "$OUT/sweep.log"

echo "benchmark,proxy_cost,wirelength_cost,density_cost,congestion_cost,overlap_count,wall_clock_s,exit_code,verified,delta,timestamp" \
  > "$OUT/results.csv"

run_one() {
  local b="$1"
  local log="$OUT/${b}.log"
  echo "  starting $b $(date)" | tee -a "$OUT/sweep.log"
  t_start=$(date +%s)
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
  if [ -n "$proxy" ] && [ "$proxy" != "NA" ]; then
    delta=$(awk -v p="$proxy" -v v="$verified" 'BEGIN { printf "%+.4f", p-v }')
  else delta="?"
  fi
  echo "${b},${proxy:-NA},${wl:-NA},${den:-NA},${cong:-NA},${overlaps:-NA},${elapsed},${rc},${verified},${delta},$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    >> "$OUT/results.csv"
  echo "  $b proxy=${proxy:-NA} (verified=$verified, Δ=$delta) cong=${cong:-NA} wall=${elapsed}s" \
    | tee -a "$OUT/sweep.log"
}

# Wave parallelization.
pids=()
for b in $BENCHES; do
  run_one "$b" &
  pids+=($!)
  if [ ${#pids[@]} -ge "$PARALLEL" ]; then
    wait "${pids[0]}"
    pids=("${pids[@]:1}")
  fi
done
for pid in "${pids[@]}"; do wait "$pid"; done

# Final summary
echo "" >> "$OUT/sweep.log"
echo "=== FINAL ===" >> "$OUT/sweep.log"
mean=$(awk -F, 'NR>1 && $2!="NA" && $2!="" {s+=$2; n++} END {if(n) printf "%.4f", s/n}' "$OUT/results.csv")
v_mean=$(awk -F, 'NR>1 && $9!="" {s+=$9; n++} END {if(n) printf "%.4f", s/n}' "$OUT/results.csv")
n_valid=$(awk -F, 'NR>1 && $2!="NA" && $2!="" {n++} END {print n+0}' "$OUT/results.csv")
echo "  $n_valid/17 valid; mean=$mean (verified=$v_mean Δ=$(awk -v m=$mean -v vm=$v_mean 'BEGIN { printf "%+.4f", m-vm }'))" \
  >> "$OUT/sweep.log"
echo "DONE" >> "$OUT/sweep.log"
echo "OUT_DIR=$OUT"
