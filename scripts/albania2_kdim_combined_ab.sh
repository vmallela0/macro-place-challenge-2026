#!/bin/bash
# albania2 stage-2: K-dim Newton + Combined arms.
#
# Pre-requisite: spectral A/B (albania2_spectral_ab.sh) must have
# finished. We reuse its baseline + spectral results and add two more
# arms here so we get the full 4-arm ablation:
#
#   baseline (S=0,K=0)  <- from stage 1
#   spectral (S=1,K=0)  <- from stage 1
#   kdim     (S=0,K=1)  <- this script
#   combined (S=1,K=1)  <- this script
#
# 4 placers in parallel ({ibm06, ibm12} × {kdim, combined}). ~1h.
#
# Decision: which arm achieves the lowest mean across the 2 benches?
#   - If combined < spectral and combined < kdim   → ship S=1,K=1
#   - If best is one of {spectral, kdim}           → ship that one
#   - If all > baseline                            → fall back, both fail

set -u
cd "$(dirname "$0")/.."

OUT="/tmp/albania2_kdim_combined_ab_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"

WORKER_BUDGET=${WORKER_BUDGET:-2300}
HARD_TIMEOUT_S=${HARD_TIMEOUT_S:-3700}

# Common env (matches albania1 winning sweep + albania2 stage-1)
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
export PLACER_V7_HESSIAN_BUDGET=400
export PLACER_V7_HESSIAN_MAX_ITERS=3
export PLACER_V7_HESSIAN_TOTAL_BUDGET=1300
export PLACER_V7_HESSIAN_LANCZOS=100
export PLACER_V7_HESSIAN_TIKHONOV=1e-4
export PLACER_V7_HALO_FRAC=0.0
export PLACER_V7_K_DENS_FRAC=0.10
export PLACER_V7_K_CONG_FRAC=0.05
export PLACER_V7_ORIENTATION_FLIP=1
export PLACER_V7_HESSIAN_HPWL_WEIGHT=1.0
export PLACER_V7_HESSIAN_DENS_WEIGHT=0.5
export PLACER_V7_HESSIAN_CONG=0
export PLACER_V7_HESSIAN_ELECTROSTATIC=1
export PLACER_V7_HESSIAN_ELECTRO_NORM=1
export PLACER_V7_HESSIAN_ELECTRO_WEIGHT=0.5
export PLACER_V7_HESSIAN_ADAPTIVE=1
export PLACER_V7_HESSIAN_ADAPTIVE_TOPK=3   # K=3 eigvecs

echo "albania2 stage-2: kdim + combined" > "$OUT/sweep.log"
echo "  started: $(date)" >> "$OUT/sweep.log"
echo "  out: $OUT" >> "$OUT/sweep.log"
echo "  benches: ibm06, ibm12" >> "$OUT/sweep.log"
echo "  arms: kdim (K=1,S=0), combined (K=1,S=1)" >> "$OUT/sweep.log"
echo "" >> "$OUT/sweep.log"

run_one() {
    local bench="$1"
    local arm="$2"
    local log="$OUT/${arm}_${bench}.log"
    case "$arm" in
        kdim)
            env PLACER_V7_HESSIAN_KDIM_NEWTON=1 \
                PLACER_V7_HESSIAN_KDIM_TRUST=0.10 \
                PLACER_V7_SPECTRAL_CRITICALITY=0 \
                .venv/bin/python -u -m macro_place.evaluate \
                submissions/vmallela_v7/placer.py --benchmark "$bench" \
                > "$log" 2>&1
            ;;
        combined)
            env PLACER_V7_HESSIAN_KDIM_NEWTON=1 \
                PLACER_V7_HESSIAN_KDIM_TRUST=0.10 \
                PLACER_V7_SPECTRAL_CRITICALITY=1 \
                PLACER_V7_SPECTRAL_GAIN=0.5 \
                .venv/bin/python -u -m macro_place.evaluate \
                submissions/vmallela_v7/placer.py --benchmark "$bench" \
                > "$log" 2>&1
            ;;
    esac
}

echo "=== launching 4 runs in parallel ===" | tee -a "$OUT/sweep.log"
PIDS=()
for bench in ibm06 ibm12; do
    for arm in kdim combined; do
        echo "  starting ${arm}_${bench}" | tee -a "$OUT/sweep.log"
        run_one "$bench" "$arm" &
        PIDS+=($!)
    done
done

( sleep "$HARD_TIMEOUT_S"
  for p in "${PIDS[@]}"; do
    kill -0 $p 2>/dev/null && kill -KILL $p 2>/dev/null
  done ) &
WATCH_PID=$!

t_start=$(date +%s)
for p in "${PIDS[@]}"; do wait $p 2>/dev/null; done
kill $WATCH_PID 2>/dev/null
elapsed=$(($(date +%s) - t_start))
echo "  all 4 done in ${elapsed}s" | tee -a "$OUT/sweep.log"

# Collect results
echo "" >> "$OUT/sweep.log"
echo "=== results ===" | tee -a "$OUT/sweep.log"
echo "arm,bench,proxy,wl,den,cong,verified,delta" > "$OUT/results.csv"
for bench in ibm06 ibm12; do
    case "$bench" in
        ibm06) verified=1.0546 ;;
        ibm12) verified=1.1557 ;;
    esac
    for arm in kdim combined; do
        log="$OUT/${arm}_${bench}.log"
        line=$(grep -E "^proxy=" "$log" | tail -1)
        proxy=$(echo "$line" | sed -E 's/.*proxy=([0-9.]+).*/\1/' | head -c 12)
        wl=$(echo "$line"    | sed -E 's/.*wl=([0-9.]+).*/\1/'    | head -c 12)
        den=$(echo "$line"   | sed -E 's/.*den=([0-9.]+).*/\1/'   | head -c 12)
        cong=$(echo "$line"  | sed -E 's/.*cong=([0-9.]+).*/\1/'  | head -c 12)
        delta=$(awk -v p="$proxy" -v v="$verified" 'BEGIN { printf "%+.4f", p-v }')
        echo "${arm},${bench},${proxy:-NA},${wl:-NA},${den:-NA},${cong:-NA},${verified},${delta}" \
            >> "$OUT/results.csv"
        echo "  ${arm}_${bench}: proxy=$proxy verified=$verified Δ=$delta" \
            | tee -a "$OUT/sweep.log"
    done
done

echo "" >> "$OUT/sweep.log"
echo "=== summary ===" | tee -a "$OUT/sweep.log"
for arm in kdim combined; do
    m=$(awk -F, -v a="$arm" 'NR>1 && $1==a && $3!="NA" {s+=$3; n++} END {if(n) printf "%.4f", s/n}' "$OUT/results.csv")
    echo "  ${arm} mean: $m" | tee -a "$OUT/sweep.log"
done
echo "DONE" >> "$OUT/sweep.log"
