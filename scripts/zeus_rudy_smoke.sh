#!/bin/bash
# zeus smoke — verify differentiable RUDY path doesn't crash
# Single bench (ibm06), short budget, just confirms the wiring is correct.
# Uses the 0.9975 winning config + RUDY_ON.

set -u
cd "$(dirname "$0")/.."

OUT="/tmp/zeus_rudy_smoke_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"

# Production env (mirrors scripts/albania1_full_17bench.sh) plus shorter budget.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8

export PLACER_TOTAL_BUDGET=${PLACER_TOTAL_BUDGET:-360}
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
export PLACER_V7_ADAM=0
export PLACER_V7_EVICT=0
export PLACER_V7_SINKHORN=0
export PLACER_V7_HESSIAN=1
export PLACER_V7_HESSIAN_BUDGET=${PLACER_V7_HESSIAN_BUDGET:-120}
export PLACER_V7_HESSIAN_LANCZOS=100
export PLACER_V7_HESSIAN_TIKHONOV=1e-4
export PLACER_V7_HESSIAN_MAX_ITERS=1
export PLACER_V7_HESSIAN_ADAPTIVE=1
export PLACER_V7_HESSIAN_ADAPTIVE_TOPK=1
export PLACER_V7_ORIENTATION_FLIP=1
# Match 0.9975 winning config: electrostatic-normalized density, cong-off.
# Then SWITCH cong-on + RUDY on top of that for the smoke.
export PLACER_V7_HESSIAN_ELECTROSTATIC=1
export PLACER_V7_HESSIAN_ELECTRO_NORM=1
export PLACER_V7_HESSIAN_ELECTRO_WEIGHT=1.0
export PLACER_V7_HESSIAN_CONG=${PLACER_V7_HESSIAN_CONG:-1}
export PLACER_V7_HESSIAN_RUDY=${PLACER_V7_HESSIAN_RUDY:-1}
export PLACER_V7_HESSIAN_HPWL_WEIGHT=1.0
export PLACER_V7_HESSIAN_DENS_WEIGHT=0.5
export PLACER_V7_HESSIAN_CONG_WEIGHT=${PLACER_V7_HESSIAN_CONG_WEIGHT:-0.5}

BENCH=${BENCH:-ibm06}
echo "zeus_rudy_smoke" > "$OUT/sweep.log"
echo "  bench: $BENCH" >> "$OUT/sweep.log"
echo "  budget: total=$PLACER_TOTAL_BUDGET hess=$PLACER_V7_HESSIAN_BUDGET" >> "$OUT/sweep.log"
echo "  cong=$PLACER_V7_HESSIAN_CONG rudy=$PLACER_V7_HESSIAN_RUDY weight=$PLACER_V7_HESSIAN_CONG_WEIGHT" >> "$OUT/sweep.log"
echo "  started $(date)" >> "$OUT/sweep.log"

t_start=$(date +%s)
.venv/bin/python -u -m macro_place.evaluate \
  submissions/vmallela_v7/placer.py --benchmark "$BENCH" \
  > "$OUT/${BENCH}.log" 2>&1
rc=$?
t_end=$(date +%s)
echo "  exit_code=$rc elapsed=$((t_end - t_start))s" >> "$OUT/sweep.log"
tail -25 "$OUT/${BENCH}.log" >> "$OUT/sweep.log"
echo "OUT_DIR=$OUT"
