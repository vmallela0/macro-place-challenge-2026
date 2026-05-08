#!/bin/bash
# albania2 Bet A: Phase 0 homotopy spreader as warm-start.
#
# Smoke test: improves .plc-init proxy by 6.3% on ibm06. This script
# runs the FULL pipeline on ibm06 with PHASE0={0,1} to measure whether
# the warm-start lift survives v4 + Hessian + Lap.

set -u
cd "$(dirname "$0")/.."

OUT="/tmp/albania2_phase0_ab_$(date +%Y%m%d_%H%M%S)"
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
export PLACER_V7_HESSIAN_ADAPTIVE=1
export PLACER_V7_HESSIAN_ADAPTIVE_TOPK=1
export PLACER_V7_SPECTRAL_CRITICALITY=0
export PLACER_V7_HESSIAN_KDIM_NEWTON=0
# Phase 0 homotopy parameters (smoke-tested on ibm06)
export PLACER_V7_PHASE0_ITERS=500
export PLACER_V7_PHASE0_STAGES=20
export PLACER_V7_PHASE0_LAMBDA_0=0.05
export PLACER_V7_PHASE0_LAMBDA_F=2.0
export PLACER_V7_PHASE0_LR_FRAC=0.001

echo "albania2 Phase 0 A/B (ibm06)" > "$OUT/sweep.log"
echo "  started: $(date)" >> "$OUT/sweep.log"
echo "  out: $OUT" >> "$OUT/sweep.log"
echo "" >> "$OUT/sweep.log"

run_one() {
    local arm="$1"
    local phase0="$2"
    local log="$OUT/${arm}.log"
    echo "=== ${arm} (PHASE0=${phase0}) starting $(date) ===" \
        | tee -a "$OUT/sweep.log"
    local t_start=$(date +%s)
    env PLACER_V7_PHASE0="$phase0" \
        .venv/bin/python -u -m macro_place.evaluate \
        submissions/vmallela_v7/placer.py --benchmark ibm06 \
        > "$log" 2>&1 &
    local pid=$!
    ( sleep "$HARD_TIMEOUT_S"
      kill -0 $pid 2>/dev/null && kill -KILL $pid 2>/dev/null ) &
    local watch_pid=$!
    wait $pid 2>/dev/null
    kill $watch_pid 2>/dev/null
    local elapsed=$(($(date +%s) - t_start))
    local line=$(grep -E "^proxy=" "$log" | tail -1)
    local proxy=$(echo "$line" | sed -E 's/.*proxy=([0-9.]+).*/\1/' | head -c 12)
    local cong=$(echo "$line" | sed -E 's/.*cong=([0-9.]+).*/\1/' | head -c 12)
    echo "  ${arm}: proxy=${proxy:-NA} cong=${cong:-NA} wall=${elapsed}s" \
        | tee -a "$OUT/sweep.log"
}

run_one no_phase0 0
run_one phase0 1

echo "" >> "$OUT/sweep.log"
echo "=== summary (ibm06, verified=1.0546) ===" | tee -a "$OUT/sweep.log"
for arm in no_phase0 phase0; do
    line=$(grep -E "^proxy=" "$OUT/${arm}.log" | tail -1)
    proxy=$(echo "$line" | sed -E 's/.*proxy=([0-9.]+).*/\1/' | head -c 12)
    delta=$(awk -v p="$proxy" 'BEGIN { printf "%+.4f", p-1.0546 }')
    echo "  ${arm}: proxy=$proxy Δ=$delta" | tee -a "$OUT/sweep.log"
done
echo "DONE" >> "$OUT/sweep.log"
