#!/bin/bash
# Rapid experiment #1: Lévy-flight basin-hop on ibm15.
# Heavy-tailed α-stable noise (α=1.5) replaces Gaussian. Most jumps small,
# ~5% are big topology-crossing moves. Provably better mixing for
# multimodal landscapes (Pavlyukevich 2007).
#
# 2 hops × 300s, single worker (no need for multi-worker since the smoke
# question is "do heavy-tailed jumps land in better basins than Gaussian
# σ=0.10 hops" — we already proved diversity doesn't help at this minimizer).

set -u
cd "$(dirname "$0")/.."

OUT="/tmp/v7_levy_$(date +%Y%m%d_%H%M%S)"
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

# Lévy basin-hop config
export PLACER_V7_BASIN_HOPS=2
export PLACER_V7_BASIN_HOP_BUDGET=300
export PLACER_V7_BASIN_HOP_RESERVE=600
export PLACER_V7_BASIN_PERTURB=levy
export PLACER_V7_LEVY_ALPHA=1.5
export PLACER_V7_BASIN_SIGMA0=0.05    # smaller than Gaussian since tails are heavy
export PLACER_V7_BASIN_HOP_AUTO=999.0

# Adam disabled
export PLACER_V7_ADAM=0

export PLACER_V6_SAVE_PLACEMENT="$OUT/{name}.npy"

echo "v7 RAPID #1: Lévy-flight basin-hop on ibm15" > "$OUT/sweep.log"
echo "  started: $(date)" >> "$OUT/sweep.log"
echo "  α=1.5 σ_0=0.05·canvas, 2 hops × 300s" >> "$OUT/sweep.log"

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
echo "  Gaussian σ=0.10/h=1: 1.1277 (rejected, fell back to post-Lap)" | tee -a "$OUT/sweep.log"
echo "  SP k=3 1-worker × 900s: 1.1332 (rejected)" | tee -a "$OUT/sweep.log"
echo "  SP k=3 4-worker × 900s: 1.1340 (rejected)" | tee -a "$OUT/sweep.log"

echo "" | tee -a "$OUT/sweep.log"
echo "Hop trace:" | tee -a "$OUT/sweep.log"
grep -E "(\[v7.hop|ACCEPT|reject|LEVY|after portfolio|laplacian: post)" "$OUT/ibm15.log" \
  | tee -a "$OUT/sweep.log"
echo "DONE" >> "$OUT/sweep.log"
