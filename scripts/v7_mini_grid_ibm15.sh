#!/bin/bash
# Mini-grid on ibm15 to find optimal (sigma_0, force_hops) for v7 basin-hop.
# 9 configs: sigma_0 in {0.05, 0.10, 0.20} x force_hops in {1, 2, 3}.
# Each config gets a fresh 1800s budget (1350s portfolio + 450s reserve).
#
# Per-config output: <OUT>/cfg_<sigma>_<hops>.log
# Aggregate: <OUT>/grid.csv  (sigma, hops, proxy, overlaps, wall, exit_code)
# Pick: lowest valid proxy in grid.csv -> echoed at end.

set -u
cd "$(dirname "$0")/.."

OUT="/tmp/v7_grid_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"

WORKER_BUDGET=1800
HARD_TIMEOUT_S=2700
RESERVE=450

# Locked env (matches submission run.sh).
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8

export PLACER_TOTAL_BUDGET=$WORKER_BUDGET
export PLACER_V6_WORKERS=8
export PLACER_V6_GPU_WORKERS=1
export PLACER_V6_CONSENSUS=1
export PLACER_V6_CONSENSUS_REFINE=120
export PLACER_V6_CONSENSUS_K=16
export PLACER_SA_T0=0.00005
export PLACER_ESC_HARD_DESTROY=80

# v7 fixed
export PLACER_V7_LAPLACIAN=1
export PLACER_V7_LAPLACIAN_PASSES=2
export PLACER_V7_LAPLACIAN_BUDGET_FRAC=0.04
export PLACER_V7_BASIN_HOP_BUDGET=300
export PLACER_V7_BASIN_HOP_RESERVE=$RESERVE

# Grid axes
SIGMAS=(0.05 0.10 0.20)
HOPS=(1 2 3)

echo "v7 mini-grid on ibm15" > "$OUT/grid.log"
echo "  started: $(date)" >> "$OUT/grid.log"
echo "  worker budget: ${WORKER_BUDGET}s (portfolio $((WORKER_BUDGET-RESERVE))s + reserve ${RESERVE}s)" >> "$OUT/grid.log"
echo "  axes: sigma=${SIGMAS[*]}, hops=${HOPS[*]}" >> "$OUT/grid.log"
echo "  results dir: $OUT" >> "$OUT/grid.log"
echo "" >> "$OUT/grid.log"

echo "sigma,hops,proxy_cost,overlap_count,wall_clock_s,exit_code,timestamp" \
  > "$OUT/grid.csv"

for sigma in "${SIGMAS[@]}"; do
  for hops in "${HOPS[@]}"; do
    cfg="cfg_s${sigma}_h${hops}"
    echo "=== $cfg: sigma=$sigma hops=$hops started $(date) ===" \
      | tee -a "$OUT/grid.log"
    t_start=$(date +%s)

    export PLACER_V7_BASIN_SIGMA0=$sigma
    export PLACER_V7_BASIN_HOPS=$hops

    .venv/bin/python -m macro_place.evaluate \
      submissions/vmallela_v7/placer.py --benchmark ibm15 \
      > "$OUT/${cfg}.log" 2>&1 &
    cmd_pid=$!
    ( sleep "$HARD_TIMEOUT_S"; if kill -0 $cmd_pid 2>/dev/null; then
        echo "  TIMEOUT after ${HARD_TIMEOUT_S}s; killing $cmd_pid" \
          | tee -a "$OUT/grid.log"
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

    line=$(grep -E "^proxy=" "$OUT/${cfg}.log" | tail -1)
    proxy=$(echo "$line" | sed -E 's/.*proxy=([0-9.]+).*/\1/' | head -c 12)
    if echo "$line" | grep -q "VALID"; then
      overlaps=0
    else
      overlaps=$(grep -E "overlaps=" "$OUT/${cfg}.log" | tail -1 \
                 | sed -E 's/.*overlaps=([0-9]+).*/\1/' | head -c 8)
      [ -z "$overlaps" ] && overlaps="?"
    fi

    echo "${sigma},${hops},${proxy:-NA},${overlaps:-NA},${elapsed},${rc},$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      >> "$OUT/grid.csv"
    echo "  proxy=${proxy:-NA} overlaps=${overlaps:-NA} wall=${elapsed}s rc=$rc" \
      | tee -a "$OUT/grid.log"
  done
done

echo "" >> "$OUT/grid.log"
echo "v7 mini-grid finished: $(date)" >> "$OUT/grid.log"

# Pick winner: lowest proxy among valid (overlap_count == 0).
WINNER=$(awk -F',' 'NR>1 && $4=="0" && $3!="NA" {print $0}' "$OUT/grid.csv" \
         | sort -t',' -k3,3g | head -1)
if [ -n "$WINNER" ]; then
  WIN_SIGMA=$(echo "$WINNER" | cut -d',' -f1)
  WIN_HOPS=$(echo "$WINNER" | cut -d',' -f2)
  WIN_PROXY=$(echo "$WINNER" | cut -d',' -f3)
  echo "" | tee -a "$OUT/grid.log"
  echo "WINNER: sigma=$WIN_SIGMA hops=$WIN_HOPS proxy=$WIN_PROXY" \
    | tee -a "$OUT/grid.log"
  echo "  use: PLACER_V7_BASIN_SIGMA0=$WIN_SIGMA PLACER_V7_BASIN_HOPS=$WIN_HOPS" \
    | tee -a "$OUT/grid.log"
else
  echo "" | tee -a "$OUT/grid.log"
  echo "WARNING: no valid configs; check grid.csv" | tee -a "$OUT/grid.log"
fi

echo "DONE" >> "$OUT/grid.log"
