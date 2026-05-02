#!/bin/bash
# v8 — full 17-bench sweep with ARC + PT + Riemannian enabled.
# Adapted from scripts/slj2_full_sweep.sh:
#   - calls submissions/vmallela_v8/placer.py (which subclasses v7)
#   - sets PLACER_V8_ARC=1, PLACER_V8_REPLICA=1, PLACER_V8_RIEMANNIAN=1
#     (overridable via env)
#   - ibm15-first ordering (per v8 spec: stop early if ibm15 misses target)
set -u
cd "$(dirname "$0")/.."

OUT="/tmp/v8_sweep_$(date +%Y%m%d_%H%M%S)"
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
export PLACER_V6_GPU_WORKERS=${PLACER_V6_GPU_WORKERS:-1}
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

# Hessian gate must be on so v8's _hessian_escape_phase override is reached.
export PLACER_V7_HESSIAN=1
export PLACER_V7_HESSIAN_STEPS="0.02,-0.02,0.05,-0.05"
export PLACER_V7_HESSIAN_BUDGET=1000
export PLACER_V7_HESSIAN_LANCZOS=50
export PLACER_V7_HESSIAN_MAX_ITERS=1

# slj2 layers off — v8 replaces them with ARC.
export PLACER_SLJ2_TOPK=${PLACER_SLJ2_TOPK:-1}
export PLACER_SLJ2_MIRROR=${PLACER_SLJ2_MIRROR:-0}
export PLACER_SLJ2_POOL=${PLACER_SLJ2_POOL:-8}

# v8 phase gates (default all on; user can disable individually).
export PLACER_V8_ARC=${PLACER_V8_ARC:-1}
export PLACER_V8_REPLICA=${PLACER_V8_REPLICA:-1}
export PLACER_V8_RIEMANNIAN=${PLACER_V8_RIEMANNIAN:-1}

export PLACER_V6_SAVE_PLACEMENT="$OUT/{name}.npy"

# ibm15 first per v8 spec — halt early if it misses the cumulative target
BENCHES="ibm15 ibm17 ibm18 ibm12 ibm14 ibm16 ibm13 ibm04 ibm06 ibm07 ibm08 ibm09 ibm10 ibm11 ibm01 ibm02 ibm03"

{
  echo "v8 sweep — ARC + PT + Riemannian"
  echo "  started: $(date)"
  echo "  results dir: $OUT"
  echo "  config: budget=$WORKER_BUDGET timeout=${HARD_TIMEOUT_S}s"
  echo "  v8: arc=$PLACER_V8_ARC pt=$PLACER_V8_REPLICA riem=$PLACER_V8_RIEMANNIAN"
  echo "  pool=$PLACER_SLJ2_POOL"
  echo "  order: $BENCHES"
  echo
} > "$OUT/sweep.log"

echo "benchmark,proxy_cost,wirelength_cost,density_cost,congestion_cost,overlap_count,wall_clock_s,exit_code,timestamp" \
  > "$OUT/results.csv"

for b in $BENCHES; do
  echo "=== $b: started $(date) ===" | tee -a "$OUT/sweep.log"
  t_start=$(date +%s)

  .venv/bin/python -m macro_place.evaluate \
    submissions/vmallela_v8/placer.py --benchmark "$b" \
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

  v8_status=$(grep -E "(v8_phase_(win|keep|enter)|HESSIAN WIN|hessian:)" "$OUT/${b}.log" | tail -3 | tr '\n' ' | ')
  echo "  proxy=${proxy:-NA} overlaps=${overlaps:-NA} | $v8_status" \
    | tee -a "$OUT/sweep.log"

  if [ -f "$OUT/${b}.npy" ]; then
    .venv/bin/python scripts/v6_placement_plot.py "$b" "$OUT/${b}.npy" \
      "$OUT/${b}.png" >> "$OUT/sweep.log" 2>&1 || true
  fi

  # Per v8 spec: ibm15 is the leadoff. If ibm15 fails the cumulative
  # target (≤1.068 with all phases) AND v8 phases are all enabled,
  # halt before burning compute on the rest.
  if [ "$b" = "ibm15" ] \
     && [ "${PLACER_V8_HALT_ON_IBM15_MISS:-1}" = "1" ] \
     && [ "$PLACER_V8_ARC" = "1" ] && [ "$PLACER_V8_REPLICA" = "1" ] \
     && [ "$PLACER_V8_RIEMANNIAN" = "1" ]; then
    if awk -v p="$proxy" 'BEGIN{exit !(p+0 > 1.068)}' \
       || [ "$overlaps" != "0" ]; then
      echo "  ibm15 missed v8 cumulative target (≤1.068, ov=0); halting sweep" \
        | tee -a "$OUT/sweep.log"
      break
    fi
  fi
done

echo "" >> "$OUT/sweep.log"
echo "v8 sweep finished: $(date)" >> "$OUT/sweep.log"
echo "DONE" >> "$OUT/sweep.log"
