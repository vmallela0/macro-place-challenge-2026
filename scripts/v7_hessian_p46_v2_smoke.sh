#!/bin/bash
# Phase 4.6 v2 smoke: bump candidate budget 300→1200s.
# Standalone confirmation showed 1200s/candidate gives Δ=-0.017 on ibm15;
# 300s/candidate produced no lift. Fix: full pipeline budget per candidate.
#
# Wall: 1500s portfolio + 30s Lap + 1200s candidates parallel = ~2730s = 45 min.

set -u
cd "$(dirname "$0")/.."

OUT="/tmp/v7_hessian_p46v2_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Bump total budget to 3300 (full competition cap).
export PLACER_TOTAL_BUDGET=3300
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

# Phase 4.6 v2: 4 candidates × 1200s parallel
export PLACER_V7_HESSIAN=1
export PLACER_V7_HESSIAN_STEPS="0.02,-0.02,0.05,-0.05"
export PLACER_V7_HESSIAN_BUDGET=1200
export PLACER_V7_HESSIAN_LANCZOS=50

export PLACER_V6_SAVE_PLACEMENT="$OUT/{name}.npy"

echo "v7 Hessian Phase 4.6 v2 smoke on ibm15" > "$OUT/sweep.log"
echo "  started: $(date)" >> "$OUT/sweep.log"
echo "  config: total=3300s, candidate=1200s × 4 (× ±sign = 8)" >> "$OUT/sweep.log"

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
echo "Compare: standalone confirmation -0.017 / Phase 4.6 v1 +0.001" \
  | tee -a "$OUT/sweep.log"
echo "" | tee -a "$OUT/sweep.log"
echo "Hessian trace:" | tee -a "$OUT/sweep.log"
grep -E "(\[v7\]|HESSIAN WIN|hessian:|after portfolio|laplacian: post|step=)" "$OUT/ibm15.log" \
  | tee -a "$OUT/sweep.log"
echo "DONE" >> "$OUT/sweep.log"
