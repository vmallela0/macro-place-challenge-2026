#!/bin/bash
# zeus_supreme_ab — full-17 × 4-arm Hessian clean A/B.
#
# Step 1: v4 + Laplacian save for ALL 17 benches in parallel
#         (17 placers, ~1 core each, ~40 min wall on the slowest bench).
# Step 2: Hessian-only A/B for 17 × 4 = 68 placers loaded from each
#         bench's saved post-Lap state, run in waves of WAVE_WIDTH
#         (default 30) to keep ~64 cores filled without thrashing.
#
# Arms — full 2×2 factorial:
#   base       PLACER_V7_HESSIAN_RUDY=0  HESSIAN_HMC_K=0
#   rudy       PLACER_V7_HESSIAN_RUDY=1  HESSIAN_HMC_K=0
#   hmc        PLACER_V7_HESSIAN_RUDY=0  HESSIAN_HMC_K=6  HMC_TRAJ=8
#   rudy_hmc   PLACER_V7_HESSIAN_RUDY=1  HESSIAN_HMC_K=6  HMC_TRAJ=8
#
# Output:
#   $OUT/results.csv    stage,arm,benchmark,proxy,wl,den,cong,overlaps,wall,rc,delta
#   $OUT/sweep.log      human-readable progress
#   $OUT/post_lap_<bench>.npy  saved post-Lap state per bench (re-used per arm)
#   $OUT/<arm>_<bench>.log    placer log per (arm, bench)

set -u
cd "$(dirname "$0")/.."

OUT="/tmp/zeus_supreme_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"

WORKER_BUDGET=${WORKER_BUDGET:-2300}        # v4+Lap budget per bench
WAVE_WIDTH=${WAVE_WIDTH:-30}                # parallel placers in Step 2
ARMS="${ARMS:-base rudy hmc rudy_hmc}"
BENCHES="${BENCHES:-ibm01 ibm02 ibm03 ibm04 ibm06 ibm07 ibm08 ibm09 ibm10 ibm11 ibm12 ibm13 ibm14 ibm15 ibm16 ibm17 ibm18}"

# Common env — production 0.9975 config.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=42 CUBLAS_WORKSPACE_CONFIG=:4096:8
export PLACER_TOTAL_BUDGET=$WORKER_BUDGET
export PLACER_V6_WORKERS=1 PLACER_V6_GPU_WORKERS=0 PLACER_V6_CONSENSUS=0
export PLACER_SA_T0=0.00005 PLACER_ESC_HARD_DESTROY=80
export PLACER_V7_LAPLACIAN=1 PLACER_V7_LAPLACIAN_PASSES=2
export PLACER_V7_LAPLACIAN_BUDGET_FRAC=0.04
export PLACER_V7_BASIN_HOPS=0
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

echo "zeus_supreme_ab" > "$OUT/sweep.log"
echo "  out: $OUT" >> "$OUT/sweep.log"
echo "  benches: $BENCHES" >> "$OUT/sweep.log"
echo "  arms:    $ARMS" >> "$OUT/sweep.log"
echo "  wave_w:  $WAVE_WIDTH" >> "$OUT/sweep.log"
echo "  started: $(date)" >> "$OUT/sweep.log"
echo "stage,arm,benchmark,proxy,wl,density,congestion,overlaps,wall_s,exit_code,delta" \
  > "$OUT/results.csv"

# ────────────────────────────────────────────────────────────────────────
# Arm → env-var fragment mapping.
# ────────────────────────────────────────────────────────────────────────
arm_env() {
    case "$1" in
        base) echo "PLACER_V7_HESSIAN_CONG=0 PLACER_V7_HESSIAN_RUDY=0 PLACER_V7_HESSIAN_HMC_K=0 PLACER_V7_HESSIAN_HMC_TRAJ=0" ;;
        rudy) echo "PLACER_V7_HESSIAN_CONG=1 PLACER_V7_HESSIAN_RUDY=1 PLACER_V7_HESSIAN_CONG_WEIGHT=0.5 PLACER_V7_HESSIAN_RUDY_MARGIN=4 PLACER_V7_HESSIAN_RUDY_MAX_WINDOW=64 PLACER_V7_HESSIAN_HMC_K=0 PLACER_V7_HESSIAN_HMC_TRAJ=0" ;;
        hmc)  echo "PLACER_V7_HESSIAN_CONG=0 PLACER_V7_HESSIAN_RUDY=0 PLACER_V7_HESSIAN_HMC_K=6 PLACER_V7_HESSIAN_HMC_TRAJ=8 PLACER_V7_HESSIAN_HMC_L=12 PLACER_V7_HESSIAN_HMC_STEP=0.5" ;;
        rudy_hmc) echo "PLACER_V7_HESSIAN_CONG=1 PLACER_V7_HESSIAN_RUDY=1 PLACER_V7_HESSIAN_CONG_WEIGHT=0.5 PLACER_V7_HESSIAN_RUDY_MARGIN=4 PLACER_V7_HESSIAN_RUDY_MAX_WINDOW=64 PLACER_V7_HESSIAN_HMC_K=6 PLACER_V7_HESSIAN_HMC_TRAJ=8 PLACER_V7_HESSIAN_HMC_L=12 PLACER_V7_HESSIAN_HMC_STEP=0.5" ;;
        *) echo "ERROR_unknown_arm_$1" ;;
    esac
}

# ────────────────────────────────────────────────────────────────────────
# Step 1: v4 + Lap save, ALL 17 benches in parallel.
# ────────────────────────────────────────────────────────────────────────
echo "=== Step 1: v4 + Laplacian save (17 benches in parallel) ===" \
  | tee -a "$OUT/sweep.log"

declare -A POST_LAP
for b in $BENCHES; do POST_LAP[$b]="$OUT/post_lap_${b}.npy"; done

run_v4lap() {
    local b="$1"
    local log="$OUT/v4lap_${b}.log"
    local pl_file="${POST_LAP[$b]}"
    local t_start=$(date +%s)
    PLACER_V7_SAVE_POST_LAP="$pl_file" \
    PLACER_V7_HESSIAN=0 \
    .venv/bin/python -u -m macro_place.evaluate \
        submissions/vmallela_v7/placer.py --benchmark "$b" \
        > "$log" 2>&1
    rc=$?
    local elapsed=$(($(date +%s) - t_start))
    local pl_cost
    pl_cost=$(grep -E "saved post-Lap" "$log" 2>/dev/null | tail -1 \
                | sed -E 's/.*cost=([0-9.]+).*/\1/')
    echo "v4lap,-,${b},${pl_cost:-NA},,,,,$elapsed,$rc," >> "$OUT/results.csv"
    echo "  v4lap $b: post-Lap cost=${pl_cost:-NA} wall=${elapsed}s rc=$rc" \
      | tee -a "$OUT/sweep.log"
}

pids=()
for b in $BENCHES; do
    run_v4lap "$b" &
    pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done

echo "  --- Step 1 done ($(date)) ---" | tee -a "$OUT/sweep.log"

# How many post-Lap files actually exist?
n_pl=0
for b in $BENCHES; do
    if [ -f "${POST_LAP[$b]}" ]; then n_pl=$((n_pl+1));
    else echo "  WARN: ${POST_LAP[$b]} not saved; will skip $b" \
           | tee -a "$OUT/sweep.log"; fi
done
echo "  post-Lap saves: $n_pl / $(echo $BENCHES | wc -w)" \
  | tee -a "$OUT/sweep.log"

# ────────────────────────────────────────────────────────────────────────
# Step 2: 17 × 4 = 68 Hessian-only runs from saved post-Lap.
# Waves of WAVE_WIDTH (default 30) to stay within ~64 cores including
# the per-placer pool overhead.
# ────────────────────────────────────────────────────────────────────────
echo "" >> "$OUT/sweep.log"
echo "=== Step 2: Hessian-only A/B (wave width $WAVE_WIDTH) ===" \
  | tee -a "$OUT/sweep.log"

run_hess() {
    local arm="$1" b="$2"
    local log="$OUT/${arm}_${b}.log"
    local pl_file="${POST_LAP[$b]}"
    if [ ! -f "$pl_file" ]; then
        echo "hess,${arm},${b},NA,,,,,0,127," >> "$OUT/results.csv"
        echo "  skip $arm/$b: no post-Lap file" | tee -a "$OUT/sweep.log"
        return
    fi
    local extra
    extra=$(arm_env "$arm")
    if [[ "$extra" == ERROR_* ]]; then
        echo "  ERROR: $extra" | tee -a "$OUT/sweep.log"; return
    fi
    local t_start=$(date +%s)
    # NB: arm_env may emit env vars whose values contain '='. We pass via
    # `env` rather than `export` to keep per-arm scoping.
    env $extra \
        PLACER_V7_LOAD_POST_LAP="$pl_file" \
        PLACER_V7_HESSIAN=1 \
        .venv/bin/python -u -m macro_place.evaluate \
        submissions/vmallela_v7/placer.py --benchmark "$b" \
        > "$log" 2>&1
    local rc=$?
    local elapsed=$(($(date +%s) - t_start))
    local line
    line=$(grep -E "^proxy=" "$log" | tail -1)
    local proxy wl den cong overlaps verified delta
    proxy=$(echo "$line" | sed -E 's/.*proxy=([0-9.]+).*/\1/' | head -c 12)
    wl=$(echo    "$line" | sed -E 's/.*wl=([0-9.]+).*/\1/'    | head -c 12)
    den=$(echo   "$line" | sed -E 's/.*den=([0-9.]+).*/\1/'   | head -c 12)
    cong=$(echo  "$line" | sed -E 's/.*cong=([0-9.]+).*/\1/'  | head -c 12)
    if echo "$line" | grep -q "VALID"; then overlaps=0; else overlaps="?"; fi
    verified=${VERIFIED[$b]:-?}
    if [ -n "$proxy" ] && [ "$verified" != "?" ]; then
        delta=$(awk -v p="$proxy" -v v="$verified" 'BEGIN { printf "%+.4f", p-v }')
    else delta="?"
    fi
    echo "hess,${arm},${b},${proxy:-NA},${wl:-NA},${den:-NA},${cong:-NA},${overlaps:-NA},${elapsed},${rc},${delta}" \
      >> "$OUT/results.csv"
    echo "  hess $arm/$b: proxy=${proxy:-NA} Δ=$delta cong=${cong:-NA} ovlp=$overlaps wall=${elapsed}s" \
      | tee -a "$OUT/sweep.log"
}

# Build job list: (arm, bench) pairs, ordered so heavy arms (rudy_hmc,
# hmc) are spread across waves to balance load.
jobs=()
for b in $BENCHES; do
    for arm in $ARMS; do
        jobs+=("$arm:$b")
    done
done

# Run waves of WAVE_WIDTH parallel jobs.
total_jobs=${#jobs[@]}
echo "  $total_jobs Hessian runs queued; running in waves of $WAVE_WIDTH" \
  | tee -a "$OUT/sweep.log"

active_pids=()
for spec in "${jobs[@]}"; do
    arm="${spec%%:*}"
    b="${spec##*:}"
    run_hess "$arm" "$b" &
    active_pids+=($!)
    if [ ${#active_pids[@]} -ge $WAVE_WIDTH ]; then
        wait "${active_pids[0]}"
        active_pids=("${active_pids[@]:1}")
    fi
done
for p in "${active_pids[@]}"; do wait "$p"; done

echo "  --- Step 2 done ($(date)) ---" | tee -a "$OUT/sweep.log"

# ────────────────────────────────────────────────────────────────────────
# Summary table: arm × bench Δ matrix + per-arm mean.
# ────────────────────────────────────────────────────────────────────────
echo "" >> "$OUT/sweep.log"
.venv/bin/python -u <<EOF | tee -a "$OUT/sweep.log"
import csv
rows = [r for r in csv.DictReader(open('$OUT/results.csv'))
        if r['stage'] == 'hess']
arms = sorted(set(r['arm'] for r in rows),
              key=lambda a: ['base','rudy','hmc','rudy_hmc'].index(a)
              if a in ['base','rudy','hmc','rudy_hmc'] else 99)
benches = sorted(set(r['benchmark'] for r in rows))
print()
print("=== Δ vs verified (lower=better) ===")
print(f"{'arm':10}" + ' '.join(f'{b:>8}' for b in benches) + '   mean')
for a in arms:
    cells, ds = [], []
    for b in benches:
        m = [r for r in rows if r['arm']==a and r['benchmark']==b]
        if not m or m[0]['delta'] in ('?','NA',''):
            cells.append('     N/A')
        else:
            cells.append(f"{m[0]['delta']:>+8}")
            try: ds.append(float(m[0]['delta']))
            except ValueError: pass
    mean = sum(ds)/len(ds) if ds else float('nan')
    print(f"{a:10}" + ' '.join(cells) + f"   {mean:+.4f}")

print()
print("=== proxy mean (raw, lower=better) ===")
print(f"{'arm':10}  mean_proxy")
for a in arms:
    ps = []
    for r in rows:
        if r['arm'] != a: continue
        try: ps.append(float(r['proxy']))
        except (ValueError, KeyError): pass
    if ps:
        print(f"{a:10}  {sum(ps)/len(ps):.4f}  (n={len(ps)})")
    else:
        print(f"{a:10}  N/A")
EOF

echo "DONE" >> "$OUT/sweep.log"
echo "OUT_DIR=$OUT"
