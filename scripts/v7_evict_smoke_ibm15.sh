#!/bin/bash
# Rapid #2: Top-K congestion eviction on ibm15.
# Greedy directed escape: identify hot cells, evict softs from them to
# coolest cell within radius R, validate via exact compute_proxy_cost.
# Strict-improvement gate.
#
# Distinct from CD because:
#  - Search focused on hot-tail softs only (not all softs)
#  - Direction informed by current cong gradient (not random)
#  - All evictions in one batch pass, then re-evaluate cong; barriers
#    don't block the eviction since each move is exact-cost validated.

set -u
cd "$(dirname "$0")/.."

OUT="/tmp/v7_evict_$(date +%Y%m%d_%H%M%S)"
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
export PLACER_V7_SINKHORN=0

# Phase 4.7: Top-K eviction
export PLACER_V7_EVICT=1
export PLACER_V7_EVICT_TOP_K=0.05
export PLACER_V7_EVICT_RADIUS=5
export PLACER_V7_EVICT_PASSES=3
# (no max_per_pass cap — let it try all hot softs)

export PLACER_V6_SAVE_PLACEMENT="$OUT/{name}.npy"

echo "v7 RAPID #2: Top-K congestion eviction on ibm15" > "$OUT/sweep.log"
echo "  started: $(date)" >> "$OUT/sweep.log"
echo "  config: top_k=5%, R=5 cells, 3 passes" >> "$OUT/sweep.log"

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
echo "Baselines on ibm15:" | tee -a "$OUT/sweep.log"
echo "  v4: 1.1029" | tee -a "$OUT/sweep.log"
echo "  post-Lap (recent smokes): 1.1332-1.1476" | tee -a "$OUT/sweep.log"
echo "  Lévy (Rapid #1): TBD" | tee -a "$OUT/sweep.log"

echo "" | tee -a "$OUT/sweep.log"
echo "Eviction trace:" | tee -a "$OUT/sweep.log"
grep -E "(\[v7\]|\[evict|EVICT WIN|after portfolio|laplacian: post)" \
  "$OUT/ibm15.log" | tee -a "$OUT/sweep.log"
echo "DONE" >> "$OUT/sweep.log"
