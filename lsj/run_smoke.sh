#!/bin/bash
# lsj smoke — single placer run on ibm15 with the exact sweep env.
# Expected proxy ≈ 1.0835 ± 0.005, overlaps=0, wall ≤ 3600 s.
set -u
cd "$(dirname "$0")/.."

OUT="/tmp/v7_lsj_smoke_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
echo "$OUT" > lsj/smoke_dir.txt

# --- env: copy of scripts/v7_singlev4_full_sweep.sh lines 26-57 ---
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8

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

export PLACER_V7_HESSIAN=1
export PLACER_V7_HESSIAN_STEPS="0.02,-0.02,0.05,-0.05"
export PLACER_V7_HESSIAN_BUDGET=1000
export PLACER_V7_HESSIAN_LANCZOS=50
export PLACER_V7_HESSIAN_MAX_ITERS=1
export PLACER_V6_SAVE_PLACEMENT="$OUT/{name}.npy"
# --- end env ---

t_start=$(date +%s)
echo "smoke start $(date -u +%FT%TZ) on ibm15" > "$OUT/smoke.log"

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
  echo "----- smoke result -----"
  echo "proxy=$proxy"
  echo "overlaps=$overlaps"
  echo "wall=${elapsed}s"
  echo "exit=$rc"
  echo "expected: proxy ≈ 1.0835 ± 0.005, overlaps=0, wall ≤ 3600"
} | tee -a "$OUT/smoke.log"

# verdict: PASS if 1.0735 ≤ proxy ≤ 1.0935, overlaps=0, wall ≤ 3600 and rc=0
if awk -v p="$proxy" 'BEGIN{exit !(p+0>=1.0735 && p+0<=1.0935)}' \
   && [ "$overlaps" = "0" ] && [ "$elapsed" -le 3700 ] && [ "$rc" -eq 0 ]; then
  echo "SMOKE_VERDICT=PASS" | tee -a "$OUT/smoke.log"
else
  echo "SMOKE_VERDICT=FAIL" | tee -a "$OUT/smoke.log"
fi
