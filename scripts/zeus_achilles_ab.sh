#!/bin/bash
# zeus_achilles_ab — 17-bet FULL-PIPELINE A/B sweep across 3 screening benches.
#
# Bets B1..B12 are diverse mathematical mechanisms (see research/ZEUS_BETS.md).
# Each bet is one arm of a multi-way A/B; baseline = verified 0.9975 config.
#
# Step 1: 17 bet-arms × 3 screening benches (ibm06, ibm12, ibm15) =
#         51 full-v7 runs. Wave width 16. ~3 waves × 45min = ~2.5h wall.
# Step 2 happens in zeus_achilles_autopilot.sh: parses results, picks top-N,
#         runs full-17 sweep.
#
# Output:
#   $OUT/results.csv     stage,arm,benchmark,proxy,wl,den,cong,overlaps,wall,rc,delta
#   $OUT/sweep.log       human-readable progress
#   $OUT/<arm>_<bench>.log  per-run placer log

set -u
cd "$(dirname "$0")/.."

OUT="/tmp/zeus_achilles_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
ln -sfn "$OUT" "$HOME/zeus_runs/latest_achilles_ab" 2>/dev/null || true

WORKER_BUDGET=${WORKER_BUDGET:-2700}        # full v7 wall budget per placer (45 min)
WAVE_WIDTH=${WAVE_WIDTH:-16}                # parallel placers
ARMS="${ARMS:-baseline yoshida replica l1_cong linf_cong nesterov dmc jko free_energy smc rg catastrophe neb yoshida_replica dmc_smc rg_nesterov hmc_full}"
BENCHES="${BENCHES:-ibm06 ibm12 ibm15}"

# Common env — production 0.9975 config (electrostatic-norm + cong-OFF default).
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=42 CUBLAS_WORKSPACE_CONFIG=:4096:8
export PLACER_TOTAL_BUDGET=$WORKER_BUDGET
export PLACER_V6_WORKERS=1 PLACER_V6_GPU_WORKERS=0 PLACER_V6_CONSENSUS=0
export PLACER_SA_T0=0.00005 PLACER_ESC_HARD_DESTROY=80
export PLACER_V7_LAPLACIAN=1 PLACER_V7_LAPLACIAN_PASSES=2
export PLACER_V7_LAPLACIAN_BUDGET_FRAC=0.04
export PLACER_V7_BASIN_HOPS=0
export PLACER_V7_HESSIAN=1
export PLACER_V7_HESSIAN_BUDGET=400
export PLACER_V7_HESSIAN_LANCZOS=100
export PLACER_V7_HESSIAN_TIKHONOV=1e-4
export PLACER_V7_HESSIAN_MAX_ITERS=3
export PLACER_V7_HESSIAN_ADAPTIVE=1
export PLACER_V7_HESSIAN_ADAPTIVE_TOPK=1
export PLACER_V7_HALO_FRAC=0.0
export PLACER_V7_K_DENS_FRAC=0.10
export PLACER_V7_K_CONG_FRAC=0.05
export PLACER_V7_ORIENTATION_FLIP=1
export PLACER_V7_HESSIAN_HPWL_WEIGHT=1.0
export PLACER_V7_HESSIAN_DENS_WEIGHT=0.5
export PLACER_V7_HESSIAN_ELECTROSTATIC=1
export PLACER_V7_HESSIAN_ELECTRO_NORM=1
export PLACER_V7_HESSIAN_ELECTRO_WEIGHT=0.5

declare -A VERIFIED
VERIFIED[ibm01]="0.7653"; VERIFIED[ibm02]="0.9482"; VERIFIED[ibm03]="0.9166"
VERIFIED[ibm04]="0.9287"; VERIFIED[ibm06]="1.0546"; VERIFIED[ibm07]="1.0324"
VERIFIED[ibm08]="1.0291"; VERIFIED[ibm09]="0.7628"; VERIFIED[ibm10]="0.9492"
VERIFIED[ibm11]="0.8013"; VERIFIED[ibm12]="1.1557"; VERIFIED[ibm13]="0.8757"
VERIFIED[ibm14]="1.1070"; VERIFIED[ibm15]="1.0835"; VERIFIED[ibm16]="1.0435"
VERIFIED[ibm17]="1.2813"; VERIFIED[ibm18]="1.2697"

echo "zeus_achilles_ab" > "$OUT/sweep.log"
echo "  out: $OUT" >> "$OUT/sweep.log"
echo "  benches: $BENCHES" >> "$OUT/sweep.log"
echo "  arms:    $ARMS" >> "$OUT/sweep.log"
echo "  wave_w:  $WAVE_WIDTH" >> "$OUT/sweep.log"
echo "  budget:  ${WORKER_BUDGET}s/placer" >> "$OUT/sweep.log"
echo "  started: $(date)" >> "$OUT/sweep.log"
echo "stage,arm,benchmark,proxy,wl,density,congestion,overlaps,wall_s,exit_code,delta" \
  > "$OUT/results.csv"

# ────────────────────────────────────────────────────────────────────────
# Arm → env-var fragment mapping.
# Each arm is a SINGLE-mechanism or COMBO bet (see research/ZEUS_BETS.md).
# ────────────────────────────────────────────────────────────────────────
arm_env() {
    case "$1" in
        baseline)
            echo ""
            ;;
        yoshida)
            # B1: 4th-order symplectic HMC integrator.
            echo "PLACER_V7_HESSIAN_HMC_K=6 PLACER_V7_HESSIAN_HMC_TRAJ=8 PLACER_V7_HESSIAN_HMC_INTEGRATOR=yoshida4 PLACER_V7_HESSIAN_HMC_L=8"
            ;;
        replica)
            # B2: HMC + farthest-point diverse subset.
            echo "PLACER_V7_HESSIAN_HMC_K=6 PLACER_V7_HESSIAN_HMC_TRAJ=24 PLACER_V7_HESSIAN_HMC_REPLICA_KEEP=8 PLACER_V7_HESSIAN_HMC_L=12"
            ;;
        l1_cong)
            # B3: L1-sparse cong aggregator.
            echo "PLACER_V7_HESSIAN_CONG=1 PLACER_V7_HESSIAN_CONG_AGG=l1 PLACER_V7_HESSIAN_CONG_WEIGHT=0.5 PLACER_V7_HESSIAN_RUDY=1 PLACER_V7_HESSIAN_RUDY_MARGIN=4 PLACER_V7_HESSIAN_RUDY_MAX_WINDOW=64"
            ;;
        linf_cong)
            # B3: L-infinity (smooth-max) cong aggregator.
            echo "PLACER_V7_HESSIAN_CONG=1 PLACER_V7_HESSIAN_CONG_AGG=linf PLACER_V7_HESSIAN_CONG_LINF_TAU=30.0 PLACER_V7_HESSIAN_CONG_WEIGHT=0.5 PLACER_V7_HESSIAN_RUDY=1"
            ;;
        nesterov)
            # B4: Nesterov-ODE-RK4 Phase 0 optimizer.
            echo "PLACER_V7_PHASE0_OPTIMIZER=nesterov_ode PLACER_V7_PHASE0_NESTEROV_LR_MULT=0.5 PLACER_V7_PHASE0_NESTEROV_T_INIT=5.0 PLACER_V7_PHASE0_NESTEROV_CAP_FRAC=0.05"
            ;;
        dmc)
            # B5: Diffusion Monte Carlo walkers.
            echo "PLACER_V7_HESSIAN_DMC_WALKERS=16 PLACER_V7_HESSIAN_DMC_STEPS=25 PLACER_V7_HESSIAN_DMC_TAU=0.5 PLACER_V7_HESSIAN_DMC_BETA=1.0 PLACER_V7_HESSIAN_DMC_INIT_JITTER=8.0 PLACER_V7_HESSIAN_DMC_KEEP=4"
            ;;
        jko)
            # B6: JKO post-Adam refinement.
            echo "PLACER_V7_PHASE0_JKO_STEPS=8 PLACER_V7_PHASE0_JKO_TAU=5.0 PLACER_V7_PHASE0_JKO_ALPHA=0.5 PLACER_V7_PHASE0_JKO_EPS=10.0 PLACER_V7_PHASE0_JKO_SINK_ITERS=30"
            ;;
        free_energy)
            # B7: Gaussian-smoothed proxy at Hessian phase.
            echo "PLACER_V7_HESSIAN_FREE_ENERGY=1 PLACER_V7_HESSIAN_FE_SIGMA=5.0 PLACER_V7_HESSIAN_FE_K=4"
            ;;
        smc)
            # B8: SMC tempered sampler.
            echo "PLACER_V7_HESSIAN_SMC_N=16 PLACER_V7_HESSIAN_SMC_STAGES=8 PLACER_V7_HESSIAN_SMC_JITTER=3.0 PLACER_V7_HESSIAN_SMC_MCMC_SIGMA=2.0 PLACER_V7_HESSIAN_SMC_KEEP=4"
            ;;
        rg)
            # B9: RG net-length curriculum.
            echo "PLACER_V7_PHASE0_RG_CURRICULUM=1 PLACER_V7_PHASE0_RG_SIGMA_0=0.05 PLACER_V7_PHASE0_RG_SIGMA_INF=10.0 PLACER_V7_PHASE0_RG_BBOX_EVERY=5"
            ;;
        catastrophe)
            # B10: Catastrophe-fold cubic unfolding.
            echo "PLACER_V7_HESSIAN_CATASTROPHE_K=4 PLACER_V7_HESSIAN_CATASTROPHE_H_FRAC=0.005 PLACER_V7_HESSIAN_CATASTROPHE_CAP=0.15"
            ;;
        neb)
            # B11: NEB minimum-energy path. Needs >=2 candidates available;
            # combine with HMC to ensure pool.
            echo "PLACER_V7_HESSIAN_HMC_K=4 PLACER_V7_HESSIAN_HMC_TRAJ=8 PLACER_V7_HESSIAN_NEB=1 PLACER_V7_HESSIAN_NEB_IMAGES=7 PLACER_V7_HESSIAN_NEB_ITERS=20 PLACER_V7_HESSIAN_NEB_LR=0.3"
            ;;
        yoshida_replica)
            # B1+B2 combo
            echo "PLACER_V7_HESSIAN_HMC_K=6 PLACER_V7_HESSIAN_HMC_TRAJ=24 PLACER_V7_HESSIAN_HMC_INTEGRATOR=yoshida4 PLACER_V7_HESSIAN_HMC_L=8 PLACER_V7_HESSIAN_HMC_REPLICA_KEEP=6"
            ;;
        dmc_smc)
            # B5+B8 combo: two population samplers
            echo "PLACER_V7_HESSIAN_DMC_WALKERS=16 PLACER_V7_HESSIAN_DMC_STEPS=20 PLACER_V7_HESSIAN_DMC_KEEP=4 PLACER_V7_HESSIAN_SMC_N=12 PLACER_V7_HESSIAN_SMC_STAGES=6 PLACER_V7_HESSIAN_SMC_KEEP=4"
            ;;
        rg_nesterov)
            # B4+B9 combo: Phase 0 stack
            echo "PLACER_V7_PHASE0_OPTIMIZER=nesterov_ode PLACER_V7_PHASE0_NESTEROV_LR_MULT=0.5 PLACER_V7_PHASE0_RG_CURRICULUM=1 PLACER_V7_PHASE0_RG_SIGMA_0=0.05 PLACER_V7_PHASE0_RG_SIGMA_INF=10.0"
            ;;
        hmc_full)
            # B1+B2+B5+B8+B10+B11 combo: all the diverse Hessian-phase generators
            echo "PLACER_V7_HESSIAN_HMC_K=6 PLACER_V7_HESSIAN_HMC_TRAJ=16 PLACER_V7_HESSIAN_HMC_INTEGRATOR=yoshida4 PLACER_V7_HESSIAN_HMC_REPLICA_KEEP=4 PLACER_V7_HESSIAN_DMC_WALKERS=12 PLACER_V7_HESSIAN_DMC_KEEP=4 PLACER_V7_HESSIAN_SMC_N=8 PLACER_V7_HESSIAN_SMC_KEEP=2 PLACER_V7_HESSIAN_CATASTROPHE_K=3 PLACER_V7_HESSIAN_NEB=1"
            ;;
        *) echo "ERROR_unknown_arm_$1" ;;
    esac
}

# ────────────────────────────────────────────────────────────────────────
# Run one (arm, bench) full-v7 placer; capture results.
# ────────────────────────────────────────────────────────────────────────
run_arm_bench() {
    local arm="$1" b="$2"
    local log="$OUT/${arm}_${b}.log"
    local extra
    extra=$(arm_env "$arm")
    if [[ "$extra" == ERROR_* ]]; then
        echo "  ERROR: $extra" | tee -a "$OUT/sweep.log"; return
    fi
    local t_start=$(date +%s)
    env $extra \
        .venv/bin/python -u -m macro_place.evaluate \
        submissions/vmallela_v7/placer.py --benchmark "$b" \
        > "$log" 2>&1
    local rc=$?
    local elapsed=$(($(date +%s) - t_start))
    # Parse final cost + components from log.
    local proxy wl den cong ovlp
    # Final line is the canonical printout: "proxy=X (wl=Y den=Z cong=W) VALID/INVALID [T s]"
    local final_line
    final_line=$(grep -E "proxy=[0-9.]+\s+\(wl=" "$log" 2>/dev/null | tail -1)
    proxy=$(echo "$final_line" | sed -E 's/.*proxy=([0-9.]+).*/\1/')
    wl=$(echo    "$final_line" | sed -E 's/.*wl=([0-9.]+).*/\1/')
    den=$(echo   "$final_line" | sed -E 's/.*den=([0-9.]+).*/\1/')
    cong=$(echo  "$final_line" | sed -E 's/.*cong=([0-9.]+).*/\1/')
    # Overlap from "VALID" or "INVALID overlaps=N"
    if echo "$final_line" | grep -q "VALID"; then ovlp=0
    elif echo "$final_line" | grep -q "INVALID overlaps="; then
        ovlp=$(echo "$final_line" | sed -E 's/.*INVALID overlaps=([0-9]+).*/\1/')
    else ovlp=NA; fi
    # Delta vs verified.
    local v=${VERIFIED[$b]:-}
    local delta=""
    if [ -n "$v" ] && [ -n "$proxy" ] && [ "$proxy" != "NA" ]; then
        delta=$(awk -v p="$proxy" -v v="$v" 'BEGIN { printf "%+.4f", p - v }')
    fi
    echo "ab,${arm},${b},${proxy:-NA},${wl:-NA},${den:-NA},${cong:-NA},${ovlp:-NA},$elapsed,$rc,$delta" \
      >> "$OUT/results.csv"
    echo "  ${arm}/${b}: proxy=${proxy:-NA} Δ=${delta:-NA} ovlp=${ovlp:-NA} wall=${elapsed}s rc=$rc" \
      | tee -a "$OUT/sweep.log"
}

# ────────────────────────────────────────────────────────────────────────
# Run all (arm, bench) pairs in waves of WAVE_WIDTH.
# ────────────────────────────────────────────────────────────────────────
echo "" >> "$OUT/sweep.log"
echo "=== Step 1: 17-bet × 3-bench A/B ($WAVE_WIDTH parallel) ===" \
  | tee -a "$OUT/sweep.log"

declare -a job_queue
for arm in $ARMS; do
    for b in $BENCHES; do
        job_queue+=("$arm $b")
    done
done
n_jobs=${#job_queue[@]}
echo "  total jobs: $n_jobs" | tee -a "$OUT/sweep.log"

i=0
while [ $i -lt $n_jobs ]; do
    pids=()
    end=$((i + WAVE_WIDTH))
    [ $end -gt $n_jobs ] && end=$n_jobs
    echo "  --- wave: jobs [$i, $end) ---" | tee -a "$OUT/sweep.log"
    for ((j = i; j < end; j++)); do
        # Parse "arm bench" from queue entry.
        arm_b=(${job_queue[$j]})
        run_arm_bench "${arm_b[0]}" "${arm_b[1]}" &
        pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p"; done
    i=$end
done

echo "  --- DONE ($(date)) ---" | tee -a "$OUT/sweep.log"
echo "  results: $OUT/results.csv" | tee -a "$OUT/sweep.log"
