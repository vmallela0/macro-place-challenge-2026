#!/bin/bash
# Multi-seed jammed-state validation test on ibm12.
#
# Hypothesis: at 80% utilization, placement is in a jammed-solid
# regime where the energy landscape has many comparable local minima.
# Min of K samples beats single sample by σ·√(2 ln K) for Gaussian
# tails, and by K^(1/α) for heavier Lévy-α tails. We verify σ
# empirically.
#
# Cost-off, k=42..49 (8 seeds) in parallel. Each gets full 2300s
# v4 budget. Each runs the full v7 pipeline (v4 + Laplacian +
# Hessian, all with cong-off to keep variables clean).
#
# Compare to verified ibm12 = 1.1557.
# Expected with K=8 Gaussian: min ≈ 1.157 - 0.005·√(2·ln 8) = 1.147
# Expected with K=8 Lévy-α=1.5: min ≈ 1.157 - 0.005·8^(2/3) = 1.137
# Verified or better → jammed-state hypothesis confirmed.

set -u
cd "$(dirname "$0")/.."

OUT="/tmp/albania1_multiseed_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"

WORKER_BUDGET=${WORKER_BUDGET:-2300}
HARD_TIMEOUT_S=${HARD_TIMEOUT_S:-3700}
N_SEEDS=${N_SEEDS:-8}
BENCH=${BENCH:-ibm12}

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
# CONG-OFF: clean variance test on the v4+Lap+Hessian pipeline.
export PLACER_V7_HESSIAN_CONG=0
export PLACER_V7_HESSIAN_AUTO_CONG=0

echo "albania1 multi-seed jammed-state test on $BENCH" > "$OUT/sweep.log"
echo "  started: $(date)" >> "$OUT/sweep.log"
echo "  results dir: $OUT" >> "$OUT/sweep.log"
echo "  bench: $BENCH" >> "$OUT/sweep.log"
echo "  N_seeds: $N_SEEDS (42..$((42+N_SEEDS-1)))" >> "$OUT/sweep.log"
echo "  cong-off, full v7 pipeline. ETA: ~57min wall (parallel)." >> "$OUT/sweep.log"
echo "" >> "$OUT/sweep.log"

echo "seed,proxy_cost,wirelength_cost,density_cost,congestion_cost,overlap_count,wall_clock_s,exit_code,timestamp" \
  > "$OUT/results.csv"

# Launch all seeds in parallel
pids=()
for s in $(seq 0 $((N_SEEDS-1))); do
  seed=$((42 + s))
  log="$OUT/seed${seed}.log"
  echo "  launching seed=$seed"
  PLACER_BASE_SEED=$seed .venv/bin/python -u -m macro_place.evaluate \
    submissions/vmallela_v7/placer.py --benchmark "$BENCH" \
    > "$log" 2>&1 &
  pids+=($!)
done

echo "  all $N_SEEDS seeds launched (pids: ${pids[*]}). Waiting..." \
  | tee -a "$OUT/sweep.log"
t_start=$(date +%s)

# Hard timeout watcher
( sleep "$HARD_TIMEOUT_S"
  for p in "${pids[@]}"; do
    if kill -0 $p 2>/dev/null; then
      echo "  TIMEOUT: killing seed pid $p" | tee -a "$OUT/sweep.log"
      kill -KILL $p 2>/dev/null
    fi
  done ) &
killer_pid=$!

# Wait for all seeds
for p in "${pids[@]}"; do
  wait $p 2>/dev/null
done
kill $killer_pid 2>/dev/null

t_end=$(date +%s)
elapsed=$((t_end - t_start))
echo "  all seeds done in ${elapsed}s" | tee -a "$OUT/sweep.log"

# Collect results
for s in $(seq 0 $((N_SEEDS-1))); do
  seed=$((42 + s))
  log="$OUT/seed${seed}.log"
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
  echo "${seed},${proxy:-NA},${wl:-NA},${den:-NA},${cong:-NA},${overlaps:-NA},${elapsed},0,$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    >> "$OUT/results.csv"
  echo "  seed=$seed proxy=${proxy:-NA} cong=${cong:-NA} overlaps=${overlaps:-NA}" \
    | tee -a "$OUT/sweep.log"
done

# Compute statistics
echo "" >> "$OUT/sweep.log"
echo "=== Multi-seed jammed-state statistics ===" >> "$OUT/sweep.log"
.venv/bin/python -c "
import csv
import numpy as np
import sys
rows = list(csv.DictReader(open('$OUT/results.csv')))
proxies = [float(r['proxy_cost']) for r in rows
           if r['proxy_cost'] not in ('NA','')]
if not proxies:
    print('no valid proxies')
    sys.exit(0)
arr = np.array(proxies)
print(f'N={len(arr)} valid runs')
print(f'  mean   = {arr.mean():.4f}')
print(f'  std    = {arr.std():.4f}')
print(f'  min    = {arr.min():.4f}  (best seed)')
print(f'  median = {np.median(arr):.4f}')
print(f'  max    = {arr.max():.4f}')
print()
print(f'verified $BENCH = 1.1557')
print(f'min vs verified: Δ = {arr.min()-1.1557:+.4f}')
print(f'mean vs verified: Δ = {arr.mean()-1.1557:+.4f}')
print()
print(f'Predictions:')
print(f'  Gaussian min(K=8): mean - {arr.std():.4f}·√(2 ln 8) = {arr.mean()-arr.std()*np.sqrt(2*np.log(8)):.4f}')
" >> "$OUT/sweep.log" 2>&1

cat "$OUT/sweep.log" | tail -20
echo "" >> "$OUT/sweep.log"
echo "DONE" >> "$OUT/sweep.log"
