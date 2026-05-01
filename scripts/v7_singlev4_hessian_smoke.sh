#!/bin/bash
# Architecture pivot: single-worker v4 + Hessian (no v6 portfolio).
# Rationale: v4 at 3300s gets 1.103 on ibm15; v6 portfolio at 1800s × 8
# gets 1.12-1.13. Single-worker depth beats parallel diversity on hard
# benches. Plus, v6's consensus refine (120-250s) ate our budget on ibm17.
#
# Config:
#   v4 worker: 2300s budget (single, no parallel)
#   Laplacian: ~30s
#   Hessian: 8 candidates × 1000s parallel
# Wall projection: 2300 + 30 + 1000 + ~50 overhead = ~3380s (≤ 3600s, 220s margin).

set -u
cd "$(dirname "$0")/.."

OUT="/tmp/v7_singlev4_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Single v4 worker, no GPU, no consensus
export PLACER_TOTAL_BUDGET=2300
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

# Hessian Phase 4.6: 4 candidates × 1000s parallel (8 actual via ±sign)
export PLACER_V7_HESSIAN=1
export PLACER_V7_HESSIAN_STEPS="0.02,-0.02,0.05,-0.05"
export PLACER_V7_HESSIAN_BUDGET=1000
export PLACER_V7_HESSIAN_LANCZOS=50
export PLACER_V7_HESSIAN_MAX_ITERS=1

export PLACER_V6_SAVE_PLACEMENT="$OUT/{name}.npy"

echo "v7 single-v4 + Hessian smoke on ibm15" > "$OUT/sweep.log"
echo "  started: $(date)" >> "$OUT/sweep.log"
echo "  config: 2300s v4 + 1000s Hessian = ~3380s wall (≤3600s)" >> "$OUT/sweep.log"

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
  echo "❌ NON-COMPLIANT: wall ${elapsed}s > 3600s" | tee -a "$OUT/sweep.log"
else
  echo "✓ COMPLIANT: wall ${elapsed}s ≤ 3600s" | tee -a "$OUT/sweep.log"
fi
echo "Compare:" | tee -a "$OUT/sweep.log"
echo "  v4 baseline (3300s):           1.1029" | tee -a "$OUT/sweep.log"
echo "  v6 portfolio + Hessian (2050+1200, prior sweep): 1.1138" | tee -a "$OUT/sweep.log"
echo "  v6 portfolio + Hessian (2300+1200, smoke over cap): 1.1014" | tee -a "$OUT/sweep.log"
echo "" | tee -a "$OUT/sweep.log"
grep -E "(\[v7\]|HESSIAN WIN|hessian:|after portfolio|laplacian: post|step=)" "$OUT/ibm15.log" \
  | tee -a "$OUT/sweep.log"
echo "DONE" >> "$OUT/sweep.log"
