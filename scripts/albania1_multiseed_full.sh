#!/bin/bash
# Multi-seed jammed-state deployment on the 5 high-room benches.
# Fires only if ibm12 validation showed min < 1.140 (Δ < -0.016 vs verified).
#
# Per bench: K=8 parallel seeds → take min. If jammed-state hypothesis
# holds, this gives ~0.010-0.020 per-bench improvement over verified.
#
# Sequential across benches; 8 seeds parallel within each.
# 5 benches × ~57 min = ~4.7 hours.

set -u
cd "$(dirname "$0")/.."

OUT="/tmp/albania1_multiseed_full_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"

WORKER_BUDGET=${WORKER_BUDGET:-2300}
HARD_TIMEOUT_S=${HARD_TIMEOUT_S:-3700}
N_SEEDS=${N_SEEDS:-8}

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
export PLACER_V7_HESSIAN_BUDGET=1000
export PLACER_V7_HESSIAN_LANCZOS=100
export PLACER_V7_HESSIAN_TIKHONOV=1e-4
export PLACER_V7_HESSIAN_MAX_ITERS=1
export PLACER_V7_HALO_FRAC=0.0
export PLACER_V7_K_DENS_FRAC=0.10
export PLACER_V7_ORIENTATION_FLIP=1
# Multi-seed test: cong-OFF for clean variance comparison.
export PLACER_V7_HESSIAN_CONG=0
export PLACER_V7_HESSIAN_AUTO_CONG=0

BENCHES="ibm06 ibm18 ibm07 ibm03 ibm12"

echo "albania1 multi-seed full sweep" > "$OUT/sweep.log"
echo "  started: $(date)" >> "$OUT/sweep.log"
echo "  results dir: $OUT" >> "$OUT/sweep.log"
echo "  benches: $BENCHES" >> "$OUT/sweep.log"
echo "  N_seeds per bench: $N_SEEDS (parallel)" >> "$OUT/sweep.log"
echo "  cong-off baseline; pure variance test" >> "$OUT/sweep.log"
echo "  ETA: ~4.7h (5 benches × ~57 min)" >> "$OUT/sweep.log"
echo "" >> "$OUT/sweep.log"

echo "benchmark,seed,proxy_cost,wirelength_cost,density_cost,congestion_cost,overlap_count,wall_clock_s,timestamp" \
  > "$OUT/results.csv"

for b in $BENCHES; do
  echo "=== $b: $(date) ===" | tee -a "$OUT/sweep.log"
  bench_t_start=$(date +%s)
  pids=()
  for s in $(seq 0 $((N_SEEDS-1))); do
    seed=$((42 + s))
    log="$OUT/${b}_seed${seed}.log"
    PLACER_BASE_SEED=$seed .venv/bin/python -u -m macro_place.evaluate \
      submissions/vmallela_v7/placer.py --benchmark "$b" \
      > "$log" 2>&1 &
    pids+=($!)
  done
  ( sleep "$HARD_TIMEOUT_S"
    for p in "${pids[@]}"; do
      if kill -0 $p 2>/dev/null; then
        kill -KILL $p 2>/dev/null
      fi
    done ) &
  killer_pid=$!
  for p in "${pids[@]}"; do wait $p 2>/dev/null; done
  kill $killer_pid 2>/dev/null
  bench_elapsed=$(($(date +%s) - bench_t_start))

  # Collect
  best_proxy=999.0
  for s in $(seq 0 $((N_SEEDS-1))); do
    seed=$((42 + s))
    log="$OUT/${b}_seed${seed}.log"
    line=$(grep -E "^proxy=" "$log" | tail -1)
    proxy=$(echo "$line" | sed -E 's/.*proxy=([0-9.]+).*/\1/' | head -c 12)
    wl=$(echo "$line"    | sed -E 's/.*wl=([0-9.]+).*/\1/'    | head -c 12)
    den=$(echo "$line"   | sed -E 's/.*den=([0-9.]+).*/\1/'   | head -c 12)
    cong=$(echo "$line"  | sed -E 's/.*cong=([0-9.]+).*/\1/'  | head -c 12)
    if echo "$line" | grep -q "VALID"; then
      overlaps=0
    else
      overlaps=$(grep -E "overlaps=" "$log" | tail -1 | sed -E 's/.*overlaps=([0-9]+).*/\1/' | head -c 8)
      [ -z "$overlaps" ] && overlaps="?"
    fi
    echo "${b},${seed},${proxy:-NA},${wl:-NA},${den:-NA},${cong:-NA},${overlaps:-NA},${bench_elapsed},$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      >> "$OUT/results.csv"
    if [ "$proxy" != "NA" ] && [ "$proxy" != "" ]; then
      best_proxy=$(awk -v a="$best_proxy" -v b="$proxy" 'BEGIN { print (b<a)?b:a }')
    fi
  done
  case "$b" in
    ibm03) verified="0.9166" ;;
    ibm06) verified="1.0546" ;;
    ibm07) verified="1.0324" ;;
    ibm12) verified="1.1557" ;;
    ibm18) verified="1.2697" ;;
    *)     verified="?" ;;
  esac
  delta=$(awk -v p="$best_proxy" -v v="$verified" 'BEGIN { printf "%+.4f", p-v }')
  echo "  $b best_proxy=$best_proxy (verified=$verified, Δ=$delta), $bench_elapsed s wall" \
    | tee -a "$OUT/sweep.log"
done

echo "" >> "$OUT/sweep.log"
echo "albania1 multi-seed full sweep finished: $(date)" >> "$OUT/sweep.log"
echo "DONE" >> "$OUT/sweep.log"
