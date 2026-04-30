#!/bin/bash
# Hessian Phase 4.6 COMPLIANT smoke on ibm15.
# Per COMPETITION.md: "All submissions must be under 1 hour end-to-end
# runtime (per benchmark)". Fits portfolio + Lap + Hessian in 3600s.
#
# Config: 2300s portfolio + 30s Lap + 1200s Hessian = ~3530s wall.
# 70s margin for legalize/setup/teardown.

set -u
cd "$(dirname "$0")/.."

OUT="/tmp/v7_hess_compliant_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# COMPLIANT BUDGET: total ≤ 3600s
export PLACER_TOTAL_BUDGET=2300       # portfolio gets 2300s × 8 workers in parallel
export PLACER_V6_WORKERS=8
export PLACER_V6_GPU_WORKERS=1
export PLACER_V6_CONSENSUS=1
export PLACER_V6_CONSENSUS_REFINE=120
export PLACER_V6_CONSENSUS_K=16
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

# Phase 4.6 Hessian: 4 candidates × 1200s parallel
export PLACER_V7_HESSIAN=1
export PLACER_V7_HESSIAN_STEPS="0.02,-0.02,0.05,-0.05"
export PLACER_V7_HESSIAN_BUDGET=1200
export PLACER_V7_HESSIAN_LANCZOS=50

export PLACER_V6_SAVE_PLACEMENT="$OUT/{name}.npy"

echo "v7 Hessian COMPLIANT smoke on ibm15 (≤ 3600s wall)" > "$OUT/sweep.log"
echo "  started: $(date)" >> "$OUT/sweep.log"
echo "  config: 2300s portfolio + 30s Lap + 1200s Hessian = ~3530s" >> "$OUT/sweep.log"

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
if [ "$elapsed" -gt 3600 ]; then
  echo "❌ NON-COMPLIANT: wall ${elapsed}s > 3600s cap" \
    | tee -a "$OUT/sweep.log"
else
  echo "✓ COMPLIANT: wall ${elapsed}s ≤ 3600s cap" | tee -a "$OUT/sweep.log"
fi
echo "Compare: v4 ibm15 = 1.1029" | tee -a "$OUT/sweep.log"
echo "         non-compliant 4745s smoke = 1.0757" | tee -a "$OUT/sweep.log"
echo "" | tee -a "$OUT/sweep.log"
grep -E "(\[v7\]|HESSIAN WIN|hessian:|after portfolio|laplacian: post|step=)" "$OUT/ibm15.log" \
  | tee -a "$OUT/sweep.log"
echo "DONE" >> "$OUT/sweep.log"
