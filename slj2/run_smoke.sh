#!/bin/bash
# slj2 smoke — single placer run on ibm15 with the upgraded env.
# Compares to:
#   v7 dev-box reference: 1.0835  (target if we close the c4d gap)
#   v7 on c4d (lsj smoke): 1.1087  (what the un-upgraded pipeline does here)
# Passing bar: proxy < 1.0987 (i.e. beats lsj-c4d by > 0.01).
set -u
cd "$(dirname "$0")/.."

OUT="/tmp/slj2_smoke_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
echo "$OUT" > slj2/smoke_dir.txt

# Mirror scripts/slj2_full_sweep.sh env
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8

export PLACER_TOTAL_BUDGET=2000
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

export PLACER_V7_HESSIAN=1
export PLACER_V7_HESSIAN_STEPS="0.02,-0.02,0.05,-0.05"
export PLACER_V7_HESSIAN_BUDGET=700
export PLACER_V7_HESSIAN_LANCZOS=50
export PLACER_V7_HESSIAN_MAX_ITERS=2
export PLACER_V7_HESSIAN_TOTAL_BUDGET=1500

export PLACER_SLJ2_TOPK=2
export PLACER_SLJ2_MIRROR=1
export PLACER_SLJ2_POOL=16

export PLACER_V6_SAVE_PLACEMENT="$OUT/{name}.npy"

t_start=$(date +%s)
echo "slj2 smoke start $(date -u +%FT%TZ) on ibm15" > "$OUT/smoke.log"
echo "  topk=2  mirror=1  pool=16  v4_budget=2000  hess_budget=700×2  hess_total=1500" >> "$OUT/smoke.log"

.venv/bin/python -m macro_place.evaluate \
  submissions/vmallela_v7/placer.py --benchmark ibm15 \
  >> "$OUT/smoke.log" 2>&1
rc=$?

t_end=$(date +%s)
elapsed=$((t_end - t_start))

line=$(grep -E "^proxy=" "$OUT/smoke.log" | tail -1)
proxy=$(echo "$line" | sed -E 's/.*proxy=([0-9.]+).*/\1/' | head -c 12)
if echo "$line" | grep -q "VALID"; then
  overlaps=0
else
  overlaps=$(grep -E "overlaps=" "$OUT/smoke.log" | tail -1 | sed -E 's/.*overlaps=([0-9]+).*/\1/' | head -c 8)
  [ -z "$overlaps" ] && overlaps="?"
fi

{
  echo "----- slj2 smoke result -----"
  echo "proxy=$proxy"
  echo "overlaps=$overlaps"
  echo "wall=${elapsed}s"
  echo "exit=$rc"
  echo "lsj-c4d ref: 1.1087    dev-box v7 ref: 1.0835"
} | tee -a "$OUT/smoke.log"

# Pass bar: overlaps=0, proxy < 1.0987 (better than lsj-c4d by 0.01+),
# wall ≤ 3700, rc=0
if awk -v p="$proxy" 'BEGIN{exit !(p+0 < 1.0987)}' \
   && [ "$overlaps" = "0" ] && [ "$elapsed" -le 3700 ] && [ "$rc" -eq 0 ]; then
  echo "SLJ2_SMOKE_VERDICT=PASS" | tee -a "$OUT/smoke.log"
else
  echo "SLJ2_SMOKE_VERDICT=FAIL" | tee -a "$OUT/smoke.log"
fi
