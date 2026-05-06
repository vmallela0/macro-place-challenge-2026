#!/bin/bash
# Test recursive bisection warm-start on ibm12.
# Two runs in parallel: with and without RECURSIVE_BISECT.
# Same code, same seed. Difference is whether v4 starts from .plc or
# from Fiedler recursive bisect.

set -u
cd "$(dirname "$0")/.."

OUT="/tmp/albania1_bisect_test_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"

WORKER_BUDGET=${WORKER_BUDGET:-2300}
HARD_TIMEOUT_S=${HARD_TIMEOUT_S:-3700}
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
export PLACER_V7_HESSIAN=1
export PLACER_V7_HESSIAN_BUDGET=1000
export PLACER_V7_HESSIAN_LANCZOS=100
export PLACER_V7_HESSIAN_TIKHONOV=1e-4
export PLACER_V7_HESSIAN_MAX_ITERS=1
export PLACER_V7_HALO_FRAC=0.0
export PLACER_V7_K_DENS_FRAC=0.10
export PLACER_V7_ORIENTATION_FLIP=0
export PLACER_V7_HESSIAN_HPWL_WEIGHT=1.0
export PLACER_V7_HESSIAN_DENS_WEIGHT=0.5
export PLACER_V7_HESSIAN_CONG=0
export PLACER_V7_HESSIAN_ELECTROSTATIC=0

echo "albania1 recursive-bisect warm-start test on $BENCH" > "$OUT/sweep.log"
echo "  started: $(date)" >> "$OUT/sweep.log"
echo "  results dir: $OUT" >> "$OUT/sweep.log"

echo "config,proxy_cost,wirelength_cost,density_cost,congestion_cost,overlap_count,wall_clock_s,timestamp" \
  > "$OUT/results.csv"

# Run both conditions in parallel
log_plc="$OUT/plc_init.log"
log_bisect="$OUT/bisect_init.log"

PLACER_V7_RECURSIVE_BISECT=0 \
.venv/bin/python -u -m macro_place.evaluate \
  submissions/vmallela_v7/placer.py --benchmark "$BENCH" > "$log_plc" 2>&1 &
pid_plc=$!

PLACER_V7_RECURSIVE_BISECT=1 \
.venv/bin/python -u -m macro_place.evaluate \
  submissions/vmallela_v7/placer.py --benchmark "$BENCH" > "$log_bisect" 2>&1 &
pid_bisect=$!

echo "  plc-init pid: $pid_plc, bisect-init pid: $pid_bisect" | tee -a "$OUT/sweep.log"
t_start=$(date +%s)

( sleep "$HARD_TIMEOUT_S"
  for p in $pid_plc $pid_bisect; do
    kill -0 $p 2>/dev/null && kill -KILL $p 2>/dev/null
  done ) &
killer_pid=$!

wait $pid_plc 2>/dev/null
wait $pid_bisect 2>/dev/null
kill $killer_pid 2>/dev/null

elapsed=$(($(date +%s) - t_start))
echo "  both done in ${elapsed}s" | tee -a "$OUT/sweep.log"

for label in plc bisect; do
  log="$OUT/${label}_init.log"
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
  echo "${label},${proxy:-NA},${wl:-NA},${den:-NA},${cong:-NA},${overlaps:-NA},${elapsed},$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    >> "$OUT/results.csv"
  echo "  $label proxy=${proxy:-NA} cong=${cong:-NA} overlaps=${overlaps:-NA}" \
    | tee -a "$OUT/sweep.log"
done

echo "" >> "$OUT/sweep.log"
echo "DONE" >> "$OUT/sweep.log"
