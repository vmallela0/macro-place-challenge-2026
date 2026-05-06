#!/bin/bash
# Clean A/B test: identical post-Lap state, two Hessian variants.
#
# Step 1: run v4 + Laplacian once with SAVE_POST_LAP.
# Step 2: run Hessian-only with CVaR density (LOAD_POST_LAP).
# Step 3: run Hessian-only with electrostatic density (LOAD_POST_LAP).
# Compare proxy outcomes — pure Hessian effect, zero v4 noise.
#
# Total wall: ~38 min v4+Lap + 2 × ~17 min Hessian (sequential) = ~70 min.

set -u
cd "$(dirname "$0")/.."

OUT="/tmp/albania1_electro_clean_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"

WORKER_BUDGET=${WORKER_BUDGET:-2300}
HARD_TIMEOUT_S=${HARD_TIMEOUT_S:-3700}
BENCH=${BENCH:-ibm12}

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
export PLACER_V7_HESSIAN_BUDGET=1000
export PLACER_V7_HESSIAN_LANCZOS=100
export PLACER_V7_HESSIAN_TIKHONOV=1e-4
export PLACER_V7_HESSIAN_MAX_ITERS=1
export PLACER_V7_HALO_FRAC=0.0
export PLACER_V7_K_DENS_FRAC=0.10
export PLACER_V7_ORIENTATION_FLIP=0   # disable for cleaner comparison
export PLACER_V7_HESSIAN_HPWL_WEIGHT=1.0
export PLACER_V7_HESSIAN_DENS_WEIGHT=0.5
export PLACER_V7_HESSIAN_CONG=0   # cong-off for clean A/B

POST_LAP_FILE="$OUT/post_lap_${BENCH}.npy"

echo "albania1 electro CLEAN A/B on $BENCH" > "$OUT/sweep.log"
echo "  started: $(date)" >> "$OUT/sweep.log"
echo "  results dir: $OUT" >> "$OUT/sweep.log"
echo "" >> "$OUT/sweep.log"

# === Step 1: v4 + Laplacian, save post-Lap ===
echo "=== Step 1: v4+Lap (save post-Lap state) ===" | tee -a "$OUT/sweep.log"
t1=$(date +%s)
log_save="$OUT/v4lap.log"
PLACER_V7_SAVE_POST_LAP="$POST_LAP_FILE" \
PLACER_V7_HESSIAN=0 \
.venv/bin/python -u -m macro_place.evaluate \
  submissions/vmallela_v7/placer.py --benchmark "$BENCH" \
  > "$log_save" 2>&1
rc1=$?
t1e=$(($(date +%s) - t1))
echo "  step 1 rc=$rc1 wall=${t1e}s" | tee -a "$OUT/sweep.log"
if [ ! -f "$POST_LAP_FILE" ]; then
  echo "  ERROR: post-Lap file not saved; aborting" | tee -a "$OUT/sweep.log"
  exit 1
fi
post_lap_cost=$(grep -E "saved post-Lap" "$log_save" | tail -1 | sed -E 's/.*cost=([0-9.]+).*/\1/')
echo "  post-Lap cost: $post_lap_cost" | tee -a "$OUT/sweep.log"

# === Step 2: Hessian-only with CVaR (load post-Lap) ===
echo "" >> "$OUT/sweep.log"
echo "=== Step 2: CVaR Hessian (load post-Lap) ===" | tee -a "$OUT/sweep.log"
t2=$(date +%s)
log_cvar="$OUT/cvar_hess.log"
PLACER_V7_LOAD_POST_LAP="$POST_LAP_FILE" \
PLACER_V7_HESSIAN_ELECTROSTATIC=0 \
.venv/bin/python -u -m macro_place.evaluate \
  submissions/vmallela_v7/placer.py --benchmark "$BENCH" \
  > "$log_cvar" 2>&1
t2e=$(($(date +%s) - t2))
cvar_proxy=$(grep -E "^proxy=" "$log_cvar" | tail -1 | sed -E 's/.*proxy=([0-9.]+).*/\1/' | head -c 12)
echo "  step 2 (CVaR) proxy=$cvar_proxy wall=${t2e}s" | tee -a "$OUT/sweep.log"

# === Step 3: Hessian-only with electrostatic (load post-Lap) ===
echo "" >> "$OUT/sweep.log"
echo "=== Step 3: Electrostatic Hessian (load post-Lap) ===" | tee -a "$OUT/sweep.log"
t3=$(date +%s)
log_electro="$OUT/electro_hess.log"
PLACER_V7_LOAD_POST_LAP="$POST_LAP_FILE" \
PLACER_V7_HESSIAN_ELECTROSTATIC=1 \
PLACER_V7_HESSIAN_ELECTRO_WEIGHT=1.0 \
.venv/bin/python -u -m macro_place.evaluate \
  submissions/vmallela_v7/placer.py --benchmark "$BENCH" \
  > "$log_electro" 2>&1
t3e=$(($(date +%s) - t3))
electro_proxy=$(grep -E "^proxy=" "$log_electro" | tail -1 | sed -E 's/.*proxy=([0-9.]+).*/\1/' | head -c 12)
echo "  step 3 (electro) proxy=$electro_proxy wall=${t3e}s" | tee -a "$OUT/sweep.log"

# === Summary ===
echo "" >> "$OUT/sweep.log"
echo "=== Clean A/B summary ===" | tee -a "$OUT/sweep.log"
echo "  bench: $BENCH" | tee -a "$OUT/sweep.log"
echo "  post-Lap baseline cost: $post_lap_cost" | tee -a "$OUT/sweep.log"
echo "  CVaR Hessian:    proxy=$cvar_proxy" | tee -a "$OUT/sweep.log"
echo "  Electro Hessian: proxy=$electro_proxy" | tee -a "$OUT/sweep.log"
if [ -n "$cvar_proxy" ] && [ -n "$electro_proxy" ]; then
  delta=$(awk -v c="$cvar_proxy" -v e="$electro_proxy" 'BEGIN { printf "%+.4f", e-c }')
  echo "  electro - CVaR = $delta" | tee -a "$OUT/sweep.log"
  case "$BENCH" in
    ibm12) verified="1.1557" ;;
    ibm06) verified="1.0546" ;;
    ibm18) verified="1.2697" ;;
    ibm07) verified="1.0324" ;;
    ibm03) verified="0.9166" ;;
    *)     verified="?" ;;
  esac
  if [ "$verified" != "?" ]; then
    cd_v=$(awk -v p="$cvar_proxy" -v v="$verified" 'BEGIN { printf "%+.4f", p-v }')
    ed_v=$(awk -v p="$electro_proxy" -v v="$verified" 'BEGIN { printf "%+.4f", p-v }')
    echo "  CVaR vs verified ($verified): $cd_v" | tee -a "$OUT/sweep.log"
    echo "  Electro vs verified ($verified): $ed_v" | tee -a "$OUT/sweep.log"
  fi
fi
echo "" >> "$OUT/sweep.log"
echo "DONE" >> "$OUT/sweep.log"
