#!/bin/bash
# Hessian Phase 4.6 integration smoke on ibm15.
# Tests the full placer.py pipeline with PLACER_V7_HESSIAN=1 enabled.
# Expected: post-Hessian cost ≤ post-Lap cost − 0.005 (matching the
# standalone smoke's Δ=-0.0169).

set -u
cd "$(dirname "$0")/.."

OUT="/tmp/v7_hessian_p46_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8

export PLACER_TOTAL_BUDGET=1800
export PLACER_V6_WORKERS=8
export PLACER_V6_GPU_WORKERS=1
export PLACER_V6_CONSENSUS=1
export PLACER_V6_CONSENSUS_REFINE=120
export PLACER_V6_CONSENSUS_K=16
export PLACER_SA_T0=0.00005
export PLACER_ESC_HARD_DESTROY=80

# v7: Laplacian on, basin-hop off, Adam off.
export PLACER_V7_LAPLACIAN=1
export PLACER_V7_LAPLACIAN_PASSES=2
export PLACER_V7_LAPLACIAN_BUDGET_FRAC=0.04
export PLACER_V7_BASIN_HOPS=0
export PLACER_V7_BASIN_HOP_AUTO=999.0
export PLACER_V7_BASIN_HOP_RESERVE=0
export PLACER_V7_ADAM=0
export PLACER_V7_EVICT=0
export PLACER_V7_SINKHORN=0

# Phase 4.6: Hessian escape — 4 candidates × 300s, total ~5 min wall
export PLACER_V7_HESSIAN=1
export PLACER_V7_HESSIAN_STEPS="0.02,-0.02,0.05,-0.05"
export PLACER_V7_HESSIAN_BUDGET=300
export PLACER_V7_HESSIAN_LANCZOS=50

export PLACER_V6_SAVE_PLACEMENT="$OUT/{name}.npy"

echo "v7 Hessian Phase 4.6 smoke on ibm15" > "$OUT/sweep.log"
echo "  started: $(date)" >> "$OUT/sweep.log"
echo "  config: PLACER_V7_HESSIAN=1, steps=±0.02 ±0.05, budget 300s/cand" >> "$OUT/sweep.log"

t_start=$(date +%s)
.venv/bin/python -m macro_place.evaluate \
  submissions/vmallela_v7/placer.py --benchmark ibm15 \
  > "$OUT/ibm15.log" 2>&1
rc=$?
t_end=$(date +%s)
elapsed=$((t_end - t_start))

line=$(grep -E "^proxy=" "$OUT/ibm15.log" | tail -1)
proxy=$(echo "$line" | sed -E 's/.*proxy=([0-9.]+).*/\1/' | head -c 12)

echo "" | tee -a "$OUT/sweep.log"
echo "FINAL: proxy=${proxy:-NA} wall=${elapsed}s rc=$rc" \
  | tee -a "$OUT/sweep.log"
echo "" | tee -a "$OUT/sweep.log"
echo "Hessian trace:" | tee -a "$OUT/sweep.log"
grep -E "(\[v7\]|\[hessian|HESSIAN WIN|hessian err|after portfolio|laplacian: post|step=)" "$OUT/ibm15.log" \
  | tee -a "$OUT/sweep.log"
echo "DONE" >> "$OUT/sweep.log"
