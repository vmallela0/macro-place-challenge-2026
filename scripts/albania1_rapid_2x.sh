#!/bin/bash
# Rapid two-experiment parallel test:
#   A. K=8 multi-seed with electro-norm + cong-off on ibm12
#   B. MAX_ITERS=3 (iterative Hessian) with electro-norm on ibm06
#
# 9 placers total in parallel. ETA ~1.5h.

set -u
cd "$(dirname "$0")/.."

OUT="/tmp/albania1_rapid_2x_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT/multiseed_ibm12" "$OUT/iter3_ibm06"

WORKER_BUDGET=${WORKER_BUDGET:-2300}
HARD_TIMEOUT_S=${HARD_TIMEOUT_S:-3700}

# Common locked env
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8

export PLACER_TOTAL_BUDGET=$WORKER_BUDGET
export PLACER_V6_WORKERS=1
export PLACER_V6_GPU_WORKERS=0
export PLACER_V6_CONSENSUS=0
export PLACER_SA_T0=0.00005
export PLACER_ESC_HARD_DESTROY=80
export PLACER_V7_LAPLACIAN=1
export PLACER_V7_LAPLACIAN_PASSES=2
export PLACER_V7_LAPLACIAN_BUDGET_FRAC=0.04
export PLACER_V7_BASIN_HOPS=0
export PLACER_V7_HESSIAN=1
export PLACER_V7_HESSIAN_LANCZOS=100
export PLACER_V7_HESSIAN_TIKHONOV=1e-4
export PLACER_V7_HALO_FRAC=0.0
export PLACER_V7_K_DENS_FRAC=0.10
export PLACER_V7_ORIENTATION_FLIP=0
export PLACER_V7_HESSIAN_HPWL_WEIGHT=1.0
export PLACER_V7_HESSIAN_DENS_WEIGHT=0.5
export PLACER_V7_HESSIAN_CONG=0
# Validated config: electro-norm @ weight=0.5
export PLACER_V7_HESSIAN_ELECTROSTATIC=1
export PLACER_V7_HESSIAN_ELECTRO_NORM=1
export PLACER_V7_HESSIAN_ELECTRO_WEIGHT=0.5

echo "albania1 rapid 2x experiments" > "$OUT/sweep.log"
echo "  started: $(date)" >> "$OUT/sweep.log"
echo "  exp A: K=8 multi-seed + electro-norm on ibm12" >> "$OUT/sweep.log"
echo "  exp B: MAX_ITERS=3 + electro-norm on ibm06" >> "$OUT/sweep.log"

# === Launch experiment A (K=8 multi-seed ibm12) ===
echo "=== exp A: K=8 multi-seed ibm12 (electro-norm) ===" | tee -a "$OUT/sweep.log"
A_pids=()
for s in $(seq 0 7); do
  seed=$((42 + s))
  log="$OUT/multiseed_ibm12/seed${seed}.log"
  PLACER_BASE_SEED=$seed PLACER_V7_HESSIAN_BUDGET=1000 PLACER_V7_HESSIAN_MAX_ITERS=1 \
  .venv/bin/python -u -m macro_place.evaluate \
    submissions/vmallela_v7/placer.py --benchmark ibm12 > "$log" 2>&1 &
  A_pids+=($!)
done
echo "  exp A pids: ${A_pids[*]}" | tee -a "$OUT/sweep.log"

# === Launch experiment B (MAX_ITERS=3 ibm06) ===
echo "=== exp B: MAX_ITERS=3 ibm06 (electro-norm) ===" | tee -a "$OUT/sweep.log"
log_B="$OUT/iter3_ibm06/result.log"
PLACER_V7_HESSIAN_BUDGET=400 PLACER_V7_HESSIAN_MAX_ITERS=3 \
PLACER_V7_HESSIAN_TOTAL_BUDGET=1300 \
.venv/bin/python -u -m macro_place.evaluate \
  submissions/vmallela_v7/placer.py --benchmark ibm06 > "$log_B" 2>&1 &
B_pid=$!
echo "  exp B pid: $B_pid" | tee -a "$OUT/sweep.log"

t_start=$(date +%s)

# Hard timeout
( sleep "$HARD_TIMEOUT_S"
  for p in "${A_pids[@]}" $B_pid; do
    kill -0 $p 2>/dev/null && kill -KILL $p 2>/dev/null
  done ) &
killer_pid=$!

# Wait for all
for p in "${A_pids[@]}" $B_pid; do wait $p 2>/dev/null; done
kill $killer_pid 2>/dev/null

elapsed=$(($(date +%s) - t_start))
echo "  all done in ${elapsed}s" | tee -a "$OUT/sweep.log"

# === Collect exp A results ===
echo "" >> "$OUT/sweep.log"
echo "=== exp A results (K=8 multi-seed ibm12, electro-norm) ===" \
  | tee -a "$OUT/sweep.log"
A_csv="$OUT/multiseed_ibm12/results.csv"
echo "seed,proxy_cost,cong" > "$A_csv"
for s in $(seq 0 7); do
  seed=$((42 + s))
  log="$OUT/multiseed_ibm12/seed${seed}.log"
  line=$(grep -E "^proxy=" "$log" | tail -1)
  proxy=$(echo "$line" | sed -E 's/.*proxy=([0-9.]+).*/\1/' | head -c 12)
  cong=$(echo "$line"  | sed -E 's/.*cong=([0-9.]+).*/\1/'  | head -c 12)
  echo "${seed},${proxy:-NA},${cong:-NA}" >> "$A_csv"
  echo "  seed=$seed proxy=${proxy:-NA} cong=${cong:-NA}" \
    | tee -a "$OUT/sweep.log"
done
A_min=$(awk -F, 'NR>1 && $2!="NA" && $2!="" {if (m=="" || $2<m) m=$2} END {print m+0}' "$A_csv")
A_mean=$(awk -F, 'NR>1 && $2!="NA" && $2!="" {s+=$2; n++} END {if(n) printf "%.4f", s/n}' "$A_csv")
echo "  K=8 min=$A_min, mean=$A_mean (verified ibm12=1.1557)" | tee -a "$OUT/sweep.log"
A_dmin=$(awk -v m="$A_min" 'BEGIN { printf "%+.4f", m-1.1557 }')
echo "  Δmin vs verified: $A_dmin" | tee -a "$OUT/sweep.log"

# === Collect exp B result ===
echo "" >> "$OUT/sweep.log"
echo "=== exp B result (MAX_ITERS=3 ibm06, electro-norm) ===" | tee -a "$OUT/sweep.log"
line=$(grep -E "^proxy=" "$log_B" | tail -1)
B_proxy=$(echo "$line" | sed -E 's/.*proxy=([0-9.]+).*/\1/' | head -c 12)
B_cong=$(echo "$line"  | sed -E 's/.*cong=([0-9.]+).*/\1/'  | head -c 12)
echo "  proxy=$B_proxy cong=$B_cong (verified ibm06=1.0546)" | tee -a "$OUT/sweep.log"
B_d=$(awk -v p="$B_proxy" 'BEGIN { printf "%+.4f", p-1.0546 }')
echo "  Δ vs verified: $B_d" | tee -a "$OUT/sweep.log"
# Show Hessian iterations
echo "  Hessian iterations:" | tee -a "$OUT/sweep.log"
grep -E "hessian iter|HESSIAN WIN" "$log_B" | sed 's/^/    /' | tee -a "$OUT/sweep.log"

echo "" >> "$OUT/sweep.log"
echo "DONE" >> "$OUT/sweep.log"
