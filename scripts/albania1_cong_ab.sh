#!/bin/bash
# albania1 step 1' — A/B PLACER_V7_HESSIAN_CONG ∈ {0 baseline, 1 treatment}
# on 4 representative benches (ibm01, ibm15, ibm17, ibm08).
#
# This supersedes the k_dens A/B because we discovered congestion is
# 73% of proxy variance and was missing from the Hessian surrogate.
# Cong-on/off is the BREAKTHROUGH-validating comparison.
#
# Order interleaves bench × condition: ibm01-off, ibm01-on, ibm15-off, ...
# ETA: 8 runs × ~57 min = ~7.7 hours wall on M5 Pro.

set -u
cd "$(dirname "$0")/.."

OUT="/tmp/albania1_cong_ab_$(date +%Y%m%d_%H%M%S)"
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
export PLACER_V7_HALO_FRAC=0.0
export PLACER_V7_K_DENS_FRAC=0.10
export PLACER_V7_ORIENTATION_FLIP=1
# The variable under test: PLACER_V7_HESSIAN_CONG ∈ {0, 1}

BENCHES="ibm01 ibm15 ibm17 ibm08"
CONDITIONS="off on"

echo "albania1 congestion A/B sweep" > "$OUT/sweep.log"
echo "  started: $(date)" >> "$OUT/sweep.log"
echo "  results dir: $OUT" >> "$OUT/sweep.log"
echo "  benches: $BENCHES" >> "$OUT/sweep.log"
echo "  conditions: off=PLACER_V7_HESSIAN_CONG=0 (baseline), on=1 (treatment)" \
  >> "$OUT/sweep.log"
echo "  ETA: ~8h wall (8 runs × ~57 min)" >> "$OUT/sweep.log"
echo "" >> "$OUT/sweep.log"

echo "benchmark,condition,proxy_cost,wirelength_cost,density_cost,congestion_cost,overlap_count,wall_clock_s,exit_code,timestamp" \
  > "$OUT/results.csv"

for b in $BENCHES; do
  for c in $CONDITIONS; do
    if [ "$c" = "off" ]; then
      export PLACER_V7_HESSIAN_CONG=0
    else
      export PLACER_V7_HESSIAN_CONG=1
    fi
    echo "=== $b/$c (cong=$PLACER_V7_HESSIAN_CONG): started $(date) ===" \
      | tee -a "$OUT/sweep.log"
    t_start=$(date +%s)

    log="$OUT/${b}_${c}.log"
    .venv/bin/python -u -m macro_place.evaluate \
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
    echo "  $b/$c proxy=${proxy:-NA} cong=${cong:-NA} overlaps=${overlaps:-NA} wall=${elapsed}s rc=$rc" \
      | tee -a "$OUT/sweep.log"
  done

  off_p=$(awk -F, -v b="$b" '$1==b && $2=="off" {print $3}' "$OUT/results.csv")
  on_p=$(awk -F, -v b="$b" '$1==b && $2=="on" {print $3}' "$OUT/results.csv")
  off_c=$(awk -F, -v b="$b" '$1==b && $2=="off" {print $6}' "$OUT/results.csv")
  on_c=$(awk -F, -v b="$b" '$1==b && $2=="on" {print $6}' "$OUT/results.csv")
  if [ -n "$off_p" ] && [ -n "$on_p" ]; then
    delta_p=$(echo "$off_p $on_p" | awk '{printf "%+.4f", $1-$2}')
    delta_c=$(echo "$off_c $on_c" | awk '{printf "%+.4f", $1-$2}')
    echo "  >>> $b: proxy off=$off_p on=$on_p Δ(off-on)=$delta_p; cong off=$off_c on=$on_c Δ=$delta_c" \
      | tee -a "$OUT/sweep.log"
  fi
done

echo "" >> "$OUT/sweep.log"
echo "albania1 congestion A/B sweep finished: $(date)" >> "$OUT/sweep.log"
echo "" >> "$OUT/sweep.log"
echo "=== Summary ===" >> "$OUT/sweep.log"
awk -F, 'NR>1 && $2=="off" {pf[$1]=$3; cf[$1]=$6; next}
         NR>1 && $2=="on"  {pn[$1]=$3; cn[$1]=$6}
         END {
           printf "%-8s %-10s %-10s %-12s %-10s %-10s %-12s\n",
                  "bench", "p-off", "p-on", "Δp(off-on)", "c-off", "c-on", "Δc(off-on)";
           for (b in pn) printf "%-8s %-10s %-10s %+12.4f %-10s %-10s %+12.4f\n",
                                  b, pf[b], pn[b], pf[b]-pn[b], cf[b], cn[b], cf[b]-cn[b];
         }' "$OUT/results.csv" >> "$OUT/sweep.log"
echo "DONE" >> "$OUT/sweep.log"
