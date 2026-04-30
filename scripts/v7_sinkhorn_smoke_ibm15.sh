#!/bin/bash
# Rapid #3: Sinkhorn optimal-transport eviction on ibm15.
# Globally optimal soft → cell assignment via entropy-regularized OT.
# Cost = current cong[cell] + α·dist²(soft, cell). Sinkhorn finds the
# transport plan that minimizes this; apply by argmax. Strict accept.
#
# Distinct from Rapid #2 (greedy eviction) because:
#  - Sinkhorn solves the JOINT global OT problem, not soft-by-soft.
#  - Pairings emerge naturally (e.g., two softs swap if globally optimal).
#  - O(n_soft × n_cells × iter) compute; ~few seconds for ibm15.

set -u
cd "$(dirname "$0")/.."

OUT="/tmp/v7_sinkhorn_$(date +%Y%m%d_%H%M%S)"
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

export PLACER_V7_LAPLACIAN=1
export PLACER_V7_LAPLACIAN_PASSES=2
export PLACER_V7_LAPLACIAN_BUDGET_FRAC=0.04

# Disable other layers
export PLACER_V7_BASIN_HOPS=0
export PLACER_V7_BASIN_HOP_AUTO=999.0
export PLACER_V7_BASIN_HOP_RESERVE=0
export PLACER_V7_ADAM=0
export PLACER_V7_EVICT=0

# Phase 4.8: Sinkhorn OT
export PLACER_V7_SINKHORN=1
export PLACER_V7_SINKHORN_ALPHA=0.5    # weight on |p - q|² distance term
export PLACER_V7_SINKHORN_EPS=0.05     # entropy regularization
export PLACER_V7_SINKHORN_ITERS=50     # Sinkhorn iterations

export PLACER_V6_SAVE_PLACEMENT="$OUT/{name}.npy"

echo "v7 RAPID #3: Sinkhorn OT on ibm15" > "$OUT/sweep.log"
echo "  started: $(date)" >> "$OUT/sweep.log"
echo "  config: α=0.5 ε=0.05 iters=50" >> "$OUT/sweep.log"

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
echo "Sinkhorn trace:" | tee -a "$OUT/sweep.log"
grep -E "(\[v7\]|\[sinkhorn|SINKHORN WIN|after portfolio|laplacian: post)" \
  "$OUT/ibm15.log" | tee -a "$OUT/sweep.log"
echo "DONE" >> "$OUT/sweep.log"
