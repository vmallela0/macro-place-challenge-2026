#!/bin/bash
# zeus RUDY A/B — three benches, two arms each, all in parallel.
#
# Arms:
#   baseline : production 0.9975 config (electrostatic-norm, cong-OFF)
#   rudy     : same + cong-ON with differentiable RUDY routing demand
#
# Benches: ibm06 (high-room, win), ibm12 (highest-room, win), ibm15 (regress).
# Budget: 1800s per bench (Phase 1 ~1300s + Laplacian ~70s + Hessian ~400s).
# Total wall: ~30 min with 64 cores (6 placers parallel × ~6-8 worker procs each).

set -u
cd "$(dirname "$0")/.."

OUT="/tmp/zeus_rudy_ab_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
BENCHES="${BENCHES:-ibm06 ibm12 ibm15}"
BUDGET="${BUDGET:-1800}"

# Common env (matches scripts/albania1_full_17bench.sh).
common_env=(
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONHASHSEED=42
  PLACER_TOTAL_BUDGET=$BUDGET
  PLACER_V6_WORKERS=1 PLACER_V6_GPU_WORKERS=0 PLACER_V6_CONSENSUS=0
  PLACER_SA_T0=0.00005 PLACER_ESC_HARD_DESTROY=80
  PLACER_V7_LAPLACIAN=1 PLACER_V7_LAPLACIAN_PASSES=2
  PLACER_V7_LAPLACIAN_BUDGET_FRAC=0.04
  PLACER_V7_BASIN_HOPS=0 PLACER_V7_BASIN_HOP_AUTO=999.0
  PLACER_V7_ADAM=0 PLACER_V7_EVICT=0 PLACER_V7_SINKHORN=0
  PLACER_V7_HESSIAN=1
  PLACER_V7_HESSIAN_BUDGET=400
  PLACER_V7_HESSIAN_LANCZOS=100
  PLACER_V7_HESSIAN_TIKHONOV=1e-4
  PLACER_V7_HESSIAN_MAX_ITERS=3
  PLACER_V7_HESSIAN_ADAPTIVE=1
  PLACER_V7_HESSIAN_ADAPTIVE_TOPK=1
  PLACER_V7_ORIENTATION_FLIP=1
  PLACER_V7_HESSIAN_ELECTROSTATIC=1
  PLACER_V7_HESSIAN_ELECTRO_NORM=1
  PLACER_V7_HESSIAN_ELECTRO_WEIGHT=0.5
  PLACER_V7_HESSIAN_HPWL_WEIGHT=1.0
  PLACER_V7_HESSIAN_DENS_WEIGHT=0.5
)

# Verified per-bench baselines (from albania1_full17/results.csv).
declare -A VERIFIED
VERIFIED[ibm06]="1.0546"
VERIFIED[ibm12]="1.1557"
VERIFIED[ibm15]="1.0835"

run_arm() {
  local arm="$1" bench="$2"
  local log="$OUT/${arm}_${bench}.log"
  local extra_env=""
  case "$arm" in
    baseline)
      extra_env="PLACER_V7_HESSIAN_CONG=0 PLACER_V7_HESSIAN_RUDY=0"
      ;;
    rudy)
      extra_env="PLACER_V7_HESSIAN_CONG=1 PLACER_V7_HESSIAN_RUDY=1 PLACER_V7_HESSIAN_CONG_WEIGHT=0.5 PLACER_V7_HESSIAN_RUDY_MARGIN=4 PLACER_V7_HESSIAN_RUDY_MAX_WINDOW=64"
      ;;
    *) echo "unknown arm $arm"; return 1;;
  esac
  echo "  starting $arm/$bench at $(date)" | tee -a "$OUT/sweep.log"
  t_start=$(date +%s)
  env ${common_env[@]} $extra_env \
    .venv/bin/python -u -m macro_place.evaluate \
    submissions/vmallela_v7/placer.py --benchmark "$bench" \
    > "$log" 2>&1
  rc=$?
  t_end=$(date +%s)
  elapsed=$((t_end - t_start))

  line=$(grep -E "^proxy=" "$log" | tail -1)
  proxy=$(echo "$line" | sed -E 's/.*proxy=([0-9.]+).*/\1/' | head -c 12)
  wl=$(echo "$line"    | sed -E 's/.*wl=([0-9.]+).*/\1/'    | head -c 12)
  den=$(echo "$line"   | sed -E 's/.*den=([0-9.]+).*/\1/'   | head -c 12)
  cong=$(echo "$line"  | sed -E 's/.*cong=([0-9.]+).*/\1/'  | head -c 12)
  if echo "$line" | grep -q "VALID"; then overlaps=0; else overlaps="?"; fi
  base=${VERIFIED[$bench]}
  delta=$(awk -v p="${proxy:-1}" -v b="$base" 'BEGIN { printf "%+.4f", p-b }')

  echo "${arm},${bench},${proxy:-NA},${wl:-NA},${den:-NA},${cong:-NA},${overlaps:-NA},${elapsed},${rc},${delta}" \
    >> "$OUT/results.csv"
  echo "  done $arm/$bench: proxy=${proxy:-NA} Δ=$delta cong=${cong:-NA} ovlp=${overlaps:-?} wall=${elapsed}s" \
    | tee -a "$OUT/sweep.log"
}

echo "zeus_rudy_ab" > "$OUT/sweep.log"
echo "  out: $OUT" >> "$OUT/sweep.log"
echo "  benches: $BENCHES" >> "$OUT/sweep.log"
echo "  budget: $BUDGET" >> "$OUT/sweep.log"
echo "  started: $(date)" >> "$OUT/sweep.log"

echo "arm,benchmark,proxy,wl,density,congestion,overlaps,wall_s,exit_code,delta" \
  > "$OUT/results.csv"

# Fan out: 6 jobs in parallel (3 benches × 2 arms).
pids=()
for b in $BENCHES; do
  for arm in baseline rudy; do
    run_arm "$arm" "$b" &
    pids+=($!)
  done
done

# Wait for all.
fails=0
for pid in "${pids[@]}"; do
  wait "$pid" || fails=$((fails + 1))
done

echo "" >> "$OUT/sweep.log"
echo "=== summary ===" >> "$OUT/sweep.log"
.venv/bin/python -u <<EOF | tee -a "$OUT/sweep.log"
import csv
rows = list(csv.DictReader(open('$OUT/results.csv')))
arms = sorted(set(r['arm'] for r in rows))
benches = sorted(set(r['benchmark'] for r in rows))
print('     ', '  '.join(f'{b:>8}' for b in benches), '  mean Δ')
for a in arms:
    line = []
    deltas = []
    for b in benches:
        for r in rows:
            if r['arm'] == a and r['benchmark'] == b:
                line.append(f"{r['delta']:>+8}")
                try: deltas.append(float(r['delta']))
                except: pass
                break
        else:
            line.append('   N/A')
    md = sum(deltas) / len(deltas) if deltas else float('nan')
    print(f'{a:7}', '  '.join(line), f'  {md:+.4f}')
print(f'\\nfailures: $fails / ${#pids[@]}')
EOF

echo "DONE" >> "$OUT/sweep.log"
echo "OUT_DIR=$OUT"
