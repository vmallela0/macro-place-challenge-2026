#!/bin/bash
# Cong-weight sensitivity test on ibm06 — find the optimal cong_weight
# given ibm06's deep saddle (λ_min ≈ -0.008 at w=0.5).
#
# Hypothesis: ibm06 was hurt by AUTO_CONG (w=1.5) because its saddle
# is already deep enough that boosting cong_weight overshoots. Test
# the full continuum {0.5, 0.75, 1.0, 1.25, 1.5} to find the optimum.
#
# 5 cong_weights × 1 bench = 5 parallel runs, each ~1h. Total wall ~1h.

set -u
cd "$(dirname "$0")/.."

OUT="/tmp/albania1_cong_weight_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"

WORKER_BUDGET=${WORKER_BUDGET:-2300}
HARD_TIMEOUT_S=${HARD_TIMEOUT_S:-3700}
BENCH=${BENCH:-ibm06}

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
export PLACER_V7_K_CONG_FRAC=0.05
export PLACER_V7_ORIENTATION_FLIP=1
export PLACER_V7_HESSIAN_HPWL_WEIGHT=1.0
export PLACER_V7_HESSIAN_DENS_WEIGHT=0.5
export PLACER_V7_HESSIAN_CONG=1
export PLACER_V7_HESSIAN_AUTO_CONG=0   # use explicit weight per run

WEIGHTS="0.0 0.5 0.75 1.0 1.25 1.5"

echo "albania1 cong-weight sensitivity test on $BENCH" > "$OUT/sweep.log"
echo "  started: $(date)" >> "$OUT/sweep.log"
echo "  results dir: $OUT" >> "$OUT/sweep.log"
echo "  bench: $BENCH" >> "$OUT/sweep.log"
echo "  weights: $WEIGHTS" >> "$OUT/sweep.log"
echo "  weight=0.0 = cong-off baseline (HPWL+density only)" >> "$OUT/sweep.log"
echo "" >> "$OUT/sweep.log"

echo "weight,proxy_cost,wirelength_cost,density_cost,congestion_cost,overlap_count,wall_clock_s,timestamp" \
  > "$OUT/results.csv"

# Launch all weights in parallel
pids=()
for w in $WEIGHTS; do
  log="$OUT/w${w}.log"
  echo "  launching weight=$w"
  if [ "$w" = "0.0" ]; then
    PLACER_V7_HESSIAN_CONG=0 \
      .venv/bin/python -u -m macro_place.evaluate \
      submissions/vmallela_v7/placer.py --benchmark "$BENCH" \
      > "$log" 2>&1 &
  else
    PLACER_V7_HESSIAN_CONG_WEIGHT=$w \
      .venv/bin/python -u -m macro_place.evaluate \
      submissions/vmallela_v7/placer.py --benchmark "$BENCH" \
      > "$log" 2>&1 &
  fi
  pids+=($!)
done

echo "  all $(echo $WEIGHTS | wc -w) weights launched. Waiting..." \
  | tee -a "$OUT/sweep.log"
t_start=$(date +%s)

( sleep "$HARD_TIMEOUT_S"
  for p in "${pids[@]}"; do
    if kill -0 $p 2>/dev/null; then
      kill -KILL $p 2>/dev/null
    fi
  done ) &
killer_pid=$!

for p in "${pids[@]}"; do wait $p 2>/dev/null; done
kill $killer_pid 2>/dev/null

elapsed=$(($(date +%s) - t_start))
echo "  all weights done in ${elapsed}s" | tee -a "$OUT/sweep.log"

# Collect results
for w in $WEIGHTS; do
  log="$OUT/w${w}.log"
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
  echo "${w},${proxy:-NA},${wl:-NA},${den:-NA},${cong:-NA},${overlaps:-NA},${elapsed},$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    >> "$OUT/results.csv"
  # Lambda_min from log
  lam=$(grep -E "λ_min=|lambda_min=|λ=" "$log" | tail -1 | sed -E 's/.*λ_min=([+-]?[0-9.e+-]+).*/\1/' | head -c 16)
  echo "  weight=$w proxy=${proxy:-NA} cong=${cong:-NA} λ_min=${lam:-?} overlaps=${overlaps:-NA}" \
    | tee -a "$OUT/sweep.log"
done

echo "" >> "$OUT/sweep.log"
echo "albania1 cong-weight sweep finished: $(date)" >> "$OUT/sweep.log"
echo "DONE" >> "$OUT/sweep.log"
