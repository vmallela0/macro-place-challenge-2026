#!/bin/bash
# albania1 step 1: A/B PLACER_V7_K_DENS_FRAC ∈ {0.10 control, 0.05 treatment}
# on 4 representative benches (ibm01, ibm15, ibm17, ibm08).
#
# Order interleaves bench × condition: ibm01-ctrl, ibm01-trt, ibm15-ctrl, ...
# Same-machine, same-seed; back-to-back same-bench eliminates drift across
# conditions. ETA: 8 runs × ~57 min = ~7.7 hours wall on M5 Pro.
#
# Results land at /tmp/albania1_cvar_ab_<TS>/results.csv with columns:
#   benchmark, condition, proxy_cost, wl, density, congestion,
#   overlap_count, wall_clock_s, exit_code, timestamp
#
# Pairs to compare: same bench, ctrl vs trt → Δ proxy.

set -u
cd "$(dirname "$0")/.."

OUT="/tmp/albania1_cvar_ab_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"

WORKER_BUDGET=${WORKER_BUDGET:-2300}
HARD_TIMEOUT_S=${HARD_TIMEOUT_S:-3700}

# Match v7_singlev4_full_sweep.sh's locked env exactly so the only
# difference between control and treatment is PLACER_V7_K_DENS_FRAC.
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
export PLACER_V7_HESSIAN_STEPS="0.02,-0.02,0.05,-0.05"
export PLACER_V7_HESSIAN_BUDGET=1000
export PLACER_V7_HESSIAN_LANCZOS=50
export PLACER_V7_HESSIAN_MAX_ITERS=1
# Step 1 isolates CVaR-tighten only; halo stays off, orientation flip
# stays on but doesn't affect Tier 1 proxy (Tier 2 sidecar only).
export PLACER_V7_HALO_FRAC=0.0
export PLACER_V7_ORIENTATION_FLIP=1

BENCHES="ibm01 ibm15 ibm17 ibm08"
CONDITIONS="ctrl trt"

echo "albania1 CVaR A/B sweep" > "$OUT/sweep.log"
echo "  started: $(date)" >> "$OUT/sweep.log"
echo "  results dir: $OUT" >> "$OUT/sweep.log"
echo "  benches: $BENCHES" >> "$OUT/sweep.log"
echo "  conditions: ctrl=PLACER_V7_K_DENS_FRAC=0.10, trt=0.05" >> "$OUT/sweep.log"
echo "  ETA: ~8h wall (8 runs × ~57 min)" >> "$OUT/sweep.log"
echo "" >> "$OUT/sweep.log"

echo "benchmark,condition,proxy_cost,wirelength_cost,density_cost,congestion_cost,overlap_count,wall_clock_s,exit_code,timestamp" \
  > "$OUT/results.csv"

for b in $BENCHES; do
  for c in $CONDITIONS; do
    if [ "$c" = "ctrl" ]; then
      export PLACER_V7_K_DENS_FRAC=0.10
    else
      export PLACER_V7_K_DENS_FRAC=0.05
    fi
    echo "=== $b/$c (k_dens=$PLACER_V7_K_DENS_FRAC): started $(date) ===" \
      | tee -a "$OUT/sweep.log"
    t_start=$(date +%s)

    log="$OUT/${b}_${c}.log"
    .venv/bin/python -m macro_place.evaluate \
      submissions/vmallela_v7/placer.py --benchmark "$b" \
      > "$log" 2>&1 &
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

    echo "${b},${c},${proxy:-NA},${wl:-NA},${den:-NA},${cong:-NA},${overlaps:-NA},${elapsed},${rc},$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      >> "$OUT/results.csv"
    echo "  $b/$c proxy=${proxy:-NA} overlaps=${overlaps:-NA} wall=${elapsed}s rc=$rc" \
      | tee -a "$OUT/sweep.log"
  done

  # After both conditions for this bench, log the delta inline.
  ctrl_proxy=$(awk -F, -v b="$b" '$1==b && $2=="ctrl" {print $3}' "$OUT/results.csv")
  trt_proxy=$(awk -F, -v b="$b" '$1==b && $2=="trt"  {print $3}' "$OUT/results.csv")
  if [ -n "$ctrl_proxy" ] && [ -n "$trt_proxy" ]; then
    delta=$(echo "$ctrl_proxy $trt_proxy" | awk '{printf "%+.4f", $1-$2}')
    echo "  >>> $b: ctrl=$ctrl_proxy trt=$trt_proxy Δ(ctrl-trt)=$delta" \
      | tee -a "$OUT/sweep.log"
  fi
done

echo "" >> "$OUT/sweep.log"
echo "albania1 CVaR A/B sweep finished: $(date)" >> "$OUT/sweep.log"

# Summary
echo "" >> "$OUT/sweep.log"
echo "=== Summary ===" >> "$OUT/sweep.log"
awk -F, 'NR>1 && $2=="ctrl" {ctrl[$1]=$3; next}
         NR>1 && $2=="trt"  {trt[$1]=$3}
         END {
           printf "%-8s %-8s %-8s %s\n", "bench", "ctrl", "trt", "Δ(ctrl-trt)";
           for (b in trt) printf "%-8s %-8s %-8s %+.4f\n", b, ctrl[b], trt[b], ctrl[b]-trt[b];
         }' "$OUT/results.csv" >> "$OUT/sweep.log"
echo "DONE" >> "$OUT/sweep.log"
