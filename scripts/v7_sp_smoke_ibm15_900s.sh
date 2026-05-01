#!/bin/bash
# Option B: SP smoke ibm15 with single 900s hop (vs the prior 300s × 2).
# Tests whether the gap-narrowing trend is time-per-hop bottlenecked
# (suggests scaling hop budget) or local-minimizer-strength bottlenecked
# (need full-portfolio-per-hop = Option C).
#
# Single 900s hop. Total wall ~30 min (850s portfolio + 30s Laplacian +
# 900s SP hop + overhead).

set -u
cd "$(dirname "$0")/.."

OUT="/tmp/v7_sp_smoke900_$(date +%Y%m%d_%H%M%S)"
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

# T2 SP basin-hop: SINGLE 900s hop (3x prior 300s).
export PLACER_V7_BASIN_HOPS=1
export PLACER_V7_BASIN_HOP_BUDGET=900
export PLACER_V7_BASIN_HOP_RESERVE=950   # leaves 850s for portfolio
export PLACER_V7_BASIN_PERTURB=sp
export PLACER_V7_SP_N_SWAPS=3
export PLACER_V7_BASIN_HOP_AUTO=999.0

# Adam disabled (broken)
export PLACER_V7_ADAM=0

export PLACER_V6_SAVE_PLACEMENT="$OUT/{name}.npy"

echo "v7 T2 SP smoke: 1 hop × 900s on ibm15" > "$OUT/sweep.log"
echo "  started: $(date)" >> "$OUT/sweep.log"
echo "  config: SP swap k=3, 1 hop × 900s, portfolio 850s × 8 workers" >> "$OUT/sweep.log"

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
echo "Prior smoke (300s × 2): final=1.1476, hop1=1.2116, hop2=1.1935" \
  | tee -a "$OUT/sweep.log"

echo "" | tee -a "$OUT/sweep.log"
echo "Hop trace:" | tee -a "$OUT/sweep.log"
grep -E "(\[v7\.hop|ACCEPT|reject|SP swap|after portfolio|laplacian: post)" "$OUT/ibm15.log" \
  | tee -a "$OUT/sweep.log"

echo "DONE" >> "$OUT/sweep.log"
