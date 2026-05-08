#!/bin/bash
# albania2 Bet A: full 17-bench sweep with PHASE0=1.
#
# DO NOT LAUNCH UNTIL PHASE 0 ibm06 A/B SHOWS A WIN.
#
# Runs 4 placers in parallel through 17 benches in batches of 4.
# Total ~4.5h on dev box.

set -u
cd "$(dirname "$0")/.."

OUT="/tmp/albania2_phase0_full17_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"

WORKER_BUDGET=${WORKER_BUDGET:-2300}
HARD_TIMEOUT_S=${HARD_TIMEOUT_S:-3700}

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Validated config + Phase 0 ON
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
export PLACER_V7_HESSIAN_ADAPTIVE_TOPK=1
export PLACER_V7_SPECTRAL_CRITICALITY=0
export PLACER_V7_HESSIAN_KDIM_NEWTON=0
# THE BIG BET: Phase 0 homotopy spreader as warm-start
export PLACER_V7_PHASE0=1
export PLACER_V7_PHASE0_ITERS=500
export PLACER_V7_PHASE0_STAGES=20
export PLACER_V7_PHASE0_LAMBDA_0=0.05
export PLACER_V7_PHASE0_LAMBDA_F=2.0
export PLACER_V7_PHASE0_LR_FRAC=0.001

BENCHES="ibm12 ibm06 ibm18 ibm07 ibm03 ibm08 ibm15 ibm02 ibm04 ibm16 ibm17 ibm13 ibm10 ibm14 ibm01 ibm11 ibm09"

verified_for() {
    case "$1" in
        ibm01) echo "0.7653" ;; ibm02) echo "0.9482" ;; ibm03) echo "0.9166" ;;
        ibm04) echo "0.9287" ;; ibm06) echo "1.0546" ;; ibm07) echo "1.0324" ;;
        ibm08) echo "1.0291" ;; ibm09) echo "0.7628" ;; ibm10) echo "0.9492" ;;
        ibm11) echo "0.8013" ;; ibm12) echo "1.1557" ;; ibm13) echo "0.8757" ;;
        ibm14) echo "1.1070" ;; ibm15) echo "1.0835" ;; ibm16) echo "1.0435" ;;
        ibm17) echo "1.2813" ;; ibm18) echo "1.2697" ;;
        *) echo "?" ;;
    esac
}

echo "albania2 Phase 0 full-17 sweep" > "$OUT/sweep.log"
echo "  started: $(date)" >> "$OUT/sweep.log"
echo "  benches: $BENCHES" >> "$OUT/sweep.log"
echo "  parallel batches of 4" >> "$OUT/sweep.log"
echo "" >> "$OUT/sweep.log"

echo "benchmark,proxy_cost,wirelength_cost,density_cost,congestion_cost,overlap_count,wall_clock_s,exit_code,verified,delta,timestamp" \
    > "$OUT/results.csv"

run_one() {
    local b="$1"
    local log="$OUT/${b}.log"
    local t_start=$(date +%s)
    .venv/bin/python -u -m macro_place.evaluate \
        submissions/vmallela_v7/placer.py --benchmark "$b" \
        > "$log" 2>&1 &
    local pid=$!
    ( sleep "$HARD_TIMEOUT_S"
      kill -0 $pid 2>/dev/null && kill -KILL $pid 2>/dev/null ) &
    local watch_pid=$!
    wait $pid 2>/dev/null
    local rc=$?
    kill $watch_pid 2>/dev/null
    local elapsed=$(($(date +%s) - t_start))
    local line=$(grep -E "^proxy=" "$log" | tail -1)
    local proxy=$(echo "$line" | sed -E 's/.*proxy=([0-9.]+).*/\1/' | head -c 12)
    local wl=$(echo "$line" | sed -E 's/.*wl=([0-9.]+).*/\1/' | head -c 12)
    local den=$(echo "$line" | sed -E 's/.*den=([0-9.]+).*/\1/' | head -c 12)
    local cong=$(echo "$line" | sed -E 's/.*cong=([0-9.]+).*/\1/' | head -c 12)
    if echo "$line" | grep -q "VALID"; then ovl=0
    else ovl=$(grep -E "overlaps=" "$log" | tail -1 | sed -E 's/.*overlaps=([0-9]+).*/\1/' | head -c 8); [ -z "$ovl" ] && ovl="?"
    fi
    local verified=$(verified_for "$b")
    local delta="?"
    if [ -n "$proxy" ] && [ "$proxy" != "NA" ]; then
        delta=$(awk -v p="$proxy" -v v="$verified" 'BEGIN { printf "%+.4f", p-v }')
    fi
    echo "${b},${proxy:-NA},${wl:-NA},${den:-NA},${cong:-NA},${ovl:-NA},${elapsed},${rc},${verified},${delta},$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        >> "$OUT/results.csv"
    echo "  $b proxy=${proxy:-NA} (verified=$verified, Δ=$delta) wall=${elapsed}s" \
        | tee -a "$OUT/sweep.log"
}

# Run in batches of 4
batch=()
for b in $BENCHES; do
    batch+=("$b")
    if [ ${#batch[@]} -eq 4 ]; then
        echo "=== batch: ${batch[*]} starting $(date) ===" | tee -a "$OUT/sweep.log"
        for bn in "${batch[@]}"; do run_one "$bn" & done
        wait
        batch=()
    fi
done
# Flush remaining
if [ ${#batch[@]} -gt 0 ]; then
    echo "=== final batch: ${batch[*]} starting $(date) ===" | tee -a "$OUT/sweep.log"
    for bn in "${batch[@]}"; do run_one "$bn" & done
    wait
fi

echo "" >> "$OUT/sweep.log"
echo "=== FINAL ===" | tee -a "$OUT/sweep.log"
mean=$(awk -F, 'NR>1 && $2!="NA" {s+=$2; n++} END {if(n) printf "%.4f", s/n}' "$OUT/results.csv")
v_mean=$(awk -F, 'NR>1 {s+=$9; n++} END {if(n) printf "%.4f", s/n}' "$OUT/results.csv")
n_valid=$(awk -F, 'NR>1 && $2!="NA" {n++} END {print n+0}' "$OUT/results.csv")
echo "  $n_valid/17 valid"
echo "  proxy mean: $mean (vs verified mean $v_mean, Δ=$(awk -v m=$mean -v vm=$v_mean 'BEGIN{printf "%+.4f", m-vm}'))" \
    | tee -a "$OUT/sweep.log"
echo "DONE" >> "$OUT/sweep.log"
