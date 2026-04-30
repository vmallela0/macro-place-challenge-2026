#!/bin/bash
# T2 SP basin-hop smoke test on ibm15 (the worst v6 regressor).
# Validates: (1) SP perturbation produces feasible state, (2) the
# reduced single-worker pipeline can converge from SP-decoded state,
# (3) post-Laplacian baseline is matched or beaten.
#
# Single-bench, ~30 min wall.

set -u
cd "$(dirname "$0")/.."

OUT="/tmp/v7_sp_smoke_$(date +%Y%m%d_%H%M%S)"
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

# T2: SP basin-hop config
export PLACER_V7_BASIN_HOPS=2                 # force 2 hops
export PLACER_V7_BASIN_HOP_BUDGET=300         # 300s per hop = 600s reserve
export PLACER_V7_BASIN_HOP_RESERVE=600        # leaves 1200s for portfolio
export PLACER_V7_BASIN_PERTURB=sp             # SP swap, not Gaussian
export PLACER_V7_SP_N_SWAPS=3                 # 3 adjacent swaps per hop
export PLACER_V7_BASIN_HOP_AUTO=999.0         # never auto-trigger; force-only

# Adam disabled (smooth-vs-exact divergence proven structural)
export PLACER_V7_ADAM=0

export PLACER_V6_SAVE_PLACEMENT="$OUT/{name}.npy"

echo "v7 T2 SP smoke on ibm15" > "$OUT/sweep.log"
echo "  started: $(date)" >> "$OUT/sweep.log"
echo "  config: SP swap k=3, 2 hops × 300s, portfolio 1200s × 8 workers" >> "$OUT/sweep.log"

t_start=$(date +%s)
.venv/bin/python -m macro_place.evaluate \
  submissions/vmallela_v7/placer.py --benchmark ibm15 \
  > "$OUT/ibm15.log" 2>&1
rc=$?
t_end=$(date +%s)
elapsed=$((t_end - t_start))

line=$(grep -E "^proxy=" "$OUT/ibm15.log" | tail -1)
proxy=$(echo "$line" | sed -E 's/.*proxy=([0-9.]+).*/\1/' | head -c 12)

echo "" >> "$OUT/sweep.log"
echo "FINAL: proxy=${proxy:-NA} wall=${elapsed}s rc=$rc" \
  | tee -a "$OUT/sweep.log"
echo "v4 ibm15 baseline: 1.1029" | tee -a "$OUT/sweep.log"
echo "v6 ibm15 (smoke): 1.131" | tee -a "$OUT/sweep.log"

# Surface key SP signals
echo "" | tee -a "$OUT/sweep.log"
echo "Hop trace:" | tee -a "$OUT/sweep.log"
grep -E "(\[v7.hop|ACCEPT|reject|SP swap)" "$OUT/ibm15.log" \
  | tee -a "$OUT/sweep.log"

echo "DONE" >> "$OUT/sweep.log"
