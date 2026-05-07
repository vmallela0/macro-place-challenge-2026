#!/bin/bash
# albania2 stage-3: spectral gain sweep.
#
# Test whether stage-1's spectral regression is from gain magnitude
# (gain=0.5 too aggressive) or from wrong signal direction.
#
#   gain=+0.1: weak boost of spectrally-critical nets
#   gain=-0.1: weak demote (negative test — if THIS wins, the signal
#              direction is wrong and we should invert)
#
# 2 gains × 2 benches = 4 runs in parallel. ~1h.

set -u
cd "$(dirname "$0")/.."

OUT="/tmp/albania2_spectral_gain_sweep_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"

WORKER_BUDGET=${WORKER_BUDGET:-2300}
HARD_TIMEOUT_S=${HARD_TIMEOUT_S:-3700}

# Common env (matches stage-1)
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
export PLACER_V7_HESSIAN_ADAPTIVE_TOPK=3
export PLACER_V7_SPECTRAL_CRITICALITY=1
export PLACER_V7_HESSIAN_KDIM_NEWTON=0   # isolate spectral effect

echo "albania2 stage-3: spectral gain sweep" > "$OUT/sweep.log"
echo "  started: $(date)" >> "$OUT/sweep.log"
echo "  out: $OUT" >> "$OUT/sweep.log"

run_one() {
    local bench="$1"
    local gain="$2"
    local arm="g${gain}"   # e.g. g+0.1, g-0.1
    local log="$OUT/${arm}_${bench}.log"
    env PLACER_V7_SPECTRAL_GAIN="$gain" \
        .venv/bin/python -u -m macro_place.evaluate \
        submissions/vmallela_v7/placer.py --benchmark "$bench" \
        > "$log" 2>&1
}

echo "=== launching 4 runs ===" | tee -a "$OUT/sweep.log"
PIDS=()
for bench in ibm06 ibm12; do
    for gain in 0.1 -0.1; do
        echo "  starting g${gain}_${bench}" | tee -a "$OUT/sweep.log"
        run_one "$bench" "$gain" &
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

# Collect
echo "" >> "$OUT/sweep.log"
echo "=== results ===" | tee -a "$OUT/sweep.log"
echo "gain,bench,proxy,wl,den,cong,verified,delta" > "$OUT/results.csv"
for bench in ibm06 ibm12; do
    case "$bench" in
        ibm06) verified=1.0546 ;;
        ibm12) verified=1.1557 ;;
    esac
    for gain in 0.1 -0.1; do
        log="$OUT/g${gain}_${bench}.log"
        line=$(grep -E "^proxy=" "$log" | tail -1)
        proxy=$(echo "$line" | sed -E 's/.*proxy=([0-9.]+).*/\1/' | head -c 12)
        wl=$(echo "$line"    | sed -E 's/.*wl=([0-9.]+).*/\1/'    | head -c 12)
        den=$(echo "$line"   | sed -E 's/.*den=([0-9.]+).*/\1/'   | head -c 12)
        cong=$(echo "$line"  | sed -E 's/.*cong=([0-9.]+).*/\1/'  | head -c 12)
        delta=$(awk -v p="$proxy" -v v="$verified" 'BEGIN { printf "%+.4f", p-v }')
        echo "${gain},${bench},${proxy:-NA},${wl:-NA},${den:-NA},${cong:-NA},${verified},${delta}" \
            >> "$OUT/results.csv"
        echo "  g${gain}_${bench}: proxy=$proxy verified=$verified Δ=$delta" \
            | tee -a "$OUT/sweep.log"
    done
done

echo "" >> "$OUT/sweep.log"
echo "=== summary ===" | tee -a "$OUT/sweep.log"
for gain in 0.1 -0.1; do
    m=$(awk -F, -v g="$gain" 'NR>1 && $1==g && $3!="NA" {s+=$3; n++} END {if(n) printf "%.4f", s/n}' "$OUT/results.csv")
    echo "  gain=${gain} mean: $m" | tee -a "$OUT/sweep.log"
done
echo "DONE" >> "$OUT/sweep.log"
