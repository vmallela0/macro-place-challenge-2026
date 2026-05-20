#!/bin/bash
# superhero_mac_pipeline.sh — A/B test gravity_drop vs default init on Mac.
#
# Pipeline (per bench):
#   1. gravity_drop produces an init JSON.
#   2. grav_polish runs CD twice: from gravity init AND from default init.
#   3. We tabulate (benchmark, cost_grav, cost_default, delta).
#
# Designed for an 18-core Mac. CD runs single-threaded, so we parallelize
# across (bench, arm) pairs. With 6 benches × 2 arms = 12 procs, we leave
# plenty of headroom.

set -u
cd "$(dirname "$0")/.."

OUT="${OUT:-/tmp/zeus_mac_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT/grav" "$OUT/polish"

# Default subset: smallest 4 benches by hard-macro count (fast feedback).
# Override with: BENCHES="ibm06 ibm01 ibm09 ibm02" ./superhero_mac_pipeline.sh
BENCHES=(${BENCHES:-ibm06 ibm01 ibm09 ibm02})
CD_TIME=${CD_TIME:-90}                # seconds per CD arm
LEGALIZE_ITERS=${LEGALIZE_ITERS:-0}   # _push_apart iters between gravity and CD (grav arm only)
GRAV_ITERS=${GRAV_ITERS:-500}
SEED=${SEED:-42}

echo "superhero_mac_pipeline"
echo "  out:      $OUT"
echo "  benches:  ${BENCHES[*]}"
echo "  CD_TIME:  ${CD_TIME}s per arm × 2 arms per bench"
echo "  started:  $(date)"

# ------- Stage 1: gravity_drop (parallel across benches) -------
echo
echo "=== Stage 1: gravity_drop ($(date)) ==="
pids=()
for b in "${BENCHES[@]}"; do
  nice .venv/bin/python -u submissions/vmallela_v7/gravity_drop.py \
    --benchmark "$b" --output "$OUT/grav/${b}.json" \
    --n-iters "$GRAV_ITERS" --dt 0.03 \
    --k-spring 0.3 --k-repel 2.0 --repel-range 2.0 \
    --gravity-max 0.05 --damping 0.93 --seed "$SEED" \
    > "$OUT/grav/${b}.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
echo "=== Stage 1 done ($(date)) ==="

# ------- Stage 2: A/B polish (parallel across (bench, arm) pairs) -------
echo
echo "=== Stage 2: A/B polish ($(date)) ==="
# Each python proc keeps OMP/etc. threads at 1 so CD doesn't oversubscribe.
pids=()
for b in "${BENCHES[@]}"; do
  if [ ! -f "$OUT/grav/${b}.json" ]; then
    echo "  WARN: $b skipped (no grav init)"
    continue
  fi
  # Arm A: grav init
  nice env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONHASHSEED=42 \
    .venv/bin/python -u submissions/vmallela_v7/grav_polish.py \
    --benchmark "$b" --grav-init "$OUT/grav/${b}.json" \
    --output "$OUT/polish/${b}_grav.json" \
    --cd-time "$CD_TIME" --seed "$SEED" --arms grav \
    --legalize-iters "$LEGALIZE_ITERS" \
    > "$OUT/polish/${b}_grav.log" 2>&1 &
  pids+=($!)
  # Arm B: default init
  nice env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONHASHSEED=42 \
    .venv/bin/python -u submissions/vmallela_v7/grav_polish.py \
    --benchmark "$b" --grav-init "$OUT/grav/${b}.json" \
    --output "$OUT/polish/${b}_default.json" \
    --cd-time "$CD_TIME" --seed "$SEED" --arms default \
    > "$OUT/polish/${b}_default.log" 2>&1 &
  pids+=($!)
done
echo "  launched ${#pids[@]} polish procs"
for p in "${pids[@]}"; do wait "$p"; done
echo "=== Stage 2 done ($(date)) ==="

# ------- Final summary -------
echo
echo "=== Final A/B summary ==="
.venv/bin/python3 - "$OUT" "${BENCHES[@]}" << 'PYEOF'
import json, sys, glob
out_dir = sys.argv[1]
benches = sys.argv[2:]
print(f"{'bench':6}  {'grav':>8}  {'default':>8}  {'delta':>9}  {'winner':>8}")
g_costs = []
d_costs = []
for b in benches:
    g = d = None
    fg = f"{out_dir}/polish/{b}_grav.json"
    fd = f"{out_dir}/polish/{b}_default.json"
    try:
        gj = json.load(open(fg))
        g = float(gj["arms"]["grav"]["final_cost"])
    except Exception:
        pass
    try:
        dj = json.load(open(fd))
        d = float(dj["arms"]["default"]["final_cost"])
    except Exception:
        pass
    if g is None or d is None:
        print(f"  {b:6}: incomplete (g={g}, d={d})")
        continue
    winner = "GRAV" if g < d else "DEF"
    print(f"  {b:6}: {g:>8.4f}  {d:>8.4f}  {g-d:>+9.4f}  {winner:>8}")
    g_costs.append(g)
    d_costs.append(d)
if g_costs:
    gm = sum(g_costs) / len(g_costs)
    dm = sum(d_costs) / len(d_costs)
    print(f"\nmean over {len(g_costs)} benches: grav={gm:.4f}  default={dm:.4f}  Δ={gm-dm:+.4f}")
PYEOF
