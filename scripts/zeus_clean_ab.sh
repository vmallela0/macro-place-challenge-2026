#!/bin/bash
# zeus clean A/B — post-Lap save/load eliminates v4 SA noise between arms.
#
# For each bench:
#   Step 1: run v4 + Laplacian once, save post-Lap state to .npy.
#   Step 2: run Hessian-only baseline (electro-norm, cong-OFF).
#   Step 3: run Hessian-only zeus    (electro-norm, cong-ON + RUDY).
# Both Step 2 and Step 3 start from the SAME post-Lap state, so the
# delta is the pure Hessian-phase effect with zero Phase-1 variance.
#
# Benches: ibm06 (high room), ibm12 (highest room), ibm15 (regressing).
# Wall: ~30 min v4+Lap (3 benches parallel) + ~10 min Hessian-only
# (6 placers parallel) ≈ 45 min total.

set -u
cd "$(dirname "$0")/.."

OUT="/tmp/zeus_clean_ab_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"

WORKER_BUDGET=${WORKER_BUDGET:-2300}
HARD_TIMEOUT_S=${HARD_TIMEOUT_S:-3700}
BENCHES="${BENCHES:-ibm06 ibm12 ibm15}"

# Common env (production 0.9975 config).
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
VERIFIED[ibm06]="1.0546"; VERIFIED[ibm12]="1.1557"; VERIFIED[ibm15]="1.0835"
VERIFIED[ibm18]="1.2697"; VERIFIED[ibm07]="1.0324"; VERIFIED[ibm03]="0.9166"

echo "zeus_clean_ab" > "$OUT/sweep.log"
echo "  out: $OUT" >> "$OUT/sweep.log"
echo "  benches: $BENCHES" >> "$OUT/sweep.log"
echo "  started: $(date)" >> "$OUT/sweep.log"

echo "stage,arm,benchmark,proxy,wl,density,congestion,overlaps,wall_s,exit_code,delta" \
  > "$OUT/results.csv"

# ────────────────────────────────────────────────────────────────────────
# Step 1: v4 + Lap for each bench, save post-Lap state (parallel).
# ────────────────────────────────────────────────────────────────────────
echo "=== Step 1: v4 + Laplacian (saving post-Lap) ===" | tee -a "$OUT/sweep.log"
declare -A POST_LAP_FILE
for b in $BENCHES; do
  POST_LAP_FILE[$b]="$OUT/post_lap_${b}.npy"
done

run_v4lap() {
  local b="$1"
  local log="$OUT/v4lap_${b}.log"
  local pl_file="${POST_LAP_FILE[$b]}"
  local t_start=$(date +%s)
  PLACER_V7_SAVE_POST_LAP="$pl_file" \
  PLACER_V7_HESSIAN=0 \
  .venv/bin/python -u -m macro_place.evaluate \
    submissions/vmallela_v7/placer.py --benchmark "$b" \
    > "$log" 2>&1
  rc=$?
  local elapsed=$(($(date +%s) - t_start))
  local pl_cost=$(grep -E "saved post-Lap" "$log" | tail -1 \
                    | sed -E 's/.*cost=([0-9.]+).*/\1/')
  echo "v4lap,-,${b},${pl_cost:-NA},,,,,$elapsed,$rc," \
    >> "$OUT/results.csv"
  echo "  v4lap $b: post-Lap cost=${pl_cost:-NA} wall=${elapsed}s" \
    | tee -a "$OUT/sweep.log"
}

pids=()
for b in $BENCHES; do
  run_v4lap "$b" &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done

# Verify all post-Lap files exist.
for b in $BENCHES; do
  if [ ! -f "${POST_LAP_FILE[$b]}" ]; then
    echo "  ERROR: ${POST_LAP_FILE[$b]} not saved; will skip $b in step 2" \
      | tee -a "$OUT/sweep.log"
  fi
done

# ────────────────────────────────────────────────────────────────────────
# Step 2: Hessian-only A/B from saved post-Lap (parallel).
# ────────────────────────────────────────────────────────────────────────
echo "" >> "$OUT/sweep.log"
echo "=== Step 2: Hessian-only A/B from post-Lap ===" | tee -a "$OUT/sweep.log"

run_hess_arm() {
  local arm="$1" b="$2"
  local log="$OUT/${arm}_${b}.log"
  local pl_file="${POST_LAP_FILE[$b]}"
  if [ ! -f "$pl_file" ]; then return; fi
  local extra=""
  case "$arm" in
    base)
      extra="PLACER_V7_HESSIAN_CONG=0 PLACER_V7_HESSIAN_RUDY=0 PLACER_V7_HESSIAN_HMC_K=0 PLACER_V7_HESSIAN_HMC_TRAJ=0"
      ;;
    rudy)
      extra="PLACER_V7_HESSIAN_CONG=1 PLACER_V7_HESSIAN_RUDY=1 PLACER_V7_HESSIAN_CONG_WEIGHT=0.5 PLACER_V7_HESSIAN_RUDY_MARGIN=4 PLACER_V7_HESSIAN_RUDY_MAX_WINDOW=64 PLACER_V7_HESSIAN_HMC_K=0 PLACER_V7_HESSIAN_HMC_TRAJ=0"
      ;;
    rudy_hmc)
      extra="PLACER_V7_HESSIAN_CONG=1 PLACER_V7_HESSIAN_RUDY=1 PLACER_V7_HESSIAN_CONG_WEIGHT=0.5 PLACER_V7_HESSIAN_RUDY_MARGIN=4 PLACER_V7_HESSIAN_RUDY_MAX_WINDOW=64 PLACER_V7_HESSIAN_HMC_K=6 PLACER_V7_HESSIAN_HMC_TRAJ=16 PLACER_V7_HESSIAN_HMC_L=12 PLACER_V7_HESSIAN_HMC_STEP=0.5"
      ;;
  esac
  echo "  starting $arm/$b at $(date)" | tee -a "$OUT/sweep.log"
  local t_start=$(date +%s)
  env $extra \
    PLACER_V7_LOAD_POST_LAP="$pl_file" \
    .venv/bin/python -u -m macro_place.evaluate \
    submissions/vmallela_v7/placer.py --benchmark "$b" \
    > "$log" 2>&1
  rc=$?
  local elapsed=$(($(date +%s) - t_start))
  line=$(grep -E "^proxy=" "$log" | tail -1)
  proxy=$(echo "$line" | sed -E 's/.*proxy=([0-9.]+).*/\1/' | head -c 12)
  wl=$(echo "$line"    | sed -E 's/.*wl=([0-9.]+).*/\1/'    | head -c 12)
  den=$(echo "$line"   | sed -E 's/.*den=([0-9.]+).*/\1/'   | head -c 12)
  cong=$(echo "$line"  | sed -E 's/.*cong=([0-9.]+).*/\1/'  | head -c 12)
  if echo "$line" | grep -q "VALID"; then overlaps=0; else overlaps="?"; fi
  base=${VERIFIED[$b]:-?}
  if [ -n "$proxy" ] && [ "$base" != "?" ]; then
    delta=$(awk -v p="$proxy" -v v="$base" 'BEGIN { printf "%+.4f", p-v }')
  else delta="?"
  fi
  echo "hess,${arm},${b},${proxy:-NA},${wl:-NA},${den:-NA},${cong:-NA},${overlaps:-NA},${elapsed},${rc},${delta}" \
    >> "$OUT/results.csv"
  echo "  done $arm/$b: proxy=${proxy:-NA} Δ=$delta cong=${cong:-NA} ovlp=${overlaps:-?} wall=${elapsed}s" \
    | tee -a "$OUT/sweep.log"
}

ARMS=${ARMS:-"base rudy"}
pids=()
for b in $BENCHES; do
  for arm in $ARMS; do
    run_hess_arm "$arm" "$b" &
    pids+=($!)
  done
done
for p in "${pids[@]}"; do wait "$p"; done

# ────────────────────────────────────────────────────────────────────────
# Summary
# ────────────────────────────────────────────────────────────────────────
echo "" >> "$OUT/sweep.log"
.venv/bin/python -u <<EOF | tee -a "$OUT/sweep.log"
import csv
rows = [r for r in csv.DictReader(open('$OUT/results.csv'))
        if r['stage'] == 'hess']
benches = sorted(set(r['benchmark'] for r in rows))
arms = sorted(set(r['arm'] for r in rows))
header = '       ' + '   '.join(f'{b:>8}' for b in benches) + '    mean'
print(header)
for a in arms:
    cells, ds = [], []
    for b in benches:
        m = [r for r in rows if r['arm']==a and r['benchmark']==b]
        if not m: cells.append('     N/A'); continue
        r = m[0]
        cells.append(f"{r['delta']:>+8}")
        try: ds.append(float(r['delta']))
        except: pass
    md = sum(ds)/len(ds) if ds else float('nan')
    print(f"{a:7}" + '   '.join(cells) + f'   {md:+.4f}')
EOF

echo "DONE" >> "$OUT/sweep.log"
echo "OUT_DIR=$OUT"
