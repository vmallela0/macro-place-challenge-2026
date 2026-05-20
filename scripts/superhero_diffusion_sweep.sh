#!/bin/bash
# superhero_diffusion_sweep.sh — λ sweep × benches in parallel, all on Mac.
# Build diffusion init at each λ, then CD-polish. Default arm runs once per bench.

set -u
cd "$(dirname "$0")/.."

OUT="${OUT:-/tmp/zeus_diff_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT/init" "$OUT/polish"

BENCHES=(${BENCHES:-ibm06 ibm01 ibm02 ibm09})
LAMBDAS=(${LAMBDAS:-500 1000 2000 5000})
CD_TIME=${CD_TIME:-180}
SEED=${SEED:-42}

echo "superhero_diffusion_sweep"
echo "  out:      $OUT"
echo "  benches:  ${BENCHES[*]}"
echo "  lambdas:  ${LAMBDAS[*]}"
echo "  CD_TIME:  ${CD_TIME}s"
echo "  started:  $(date)"

# ------- Stage 1: build all diffusion inits in parallel -------
echo
echo "=== Stage 1: build diffusion inits (parallel) ==="
pids=()
for b in "${BENCHES[@]}"; do
  for lam in "${LAMBDAS[@]}"; do
    nice .venv/bin/python -u submissions/vmallela_v7/diffusion_init.py \
      --benchmark "$b" --output "$OUT/init/${b}_l${lam}.json" \
      --alpha 1e-3 --legalize-iters 0 --no-scale-canvas \
      --prior-lambda "$lam" --prior-source default \
      > "$OUT/init/${b}_l${lam}.log" 2>&1 &
    pids+=($!)
  done
done
for p in "${pids[@]}"; do wait "$p"; done
echo "=== Stage 1 done ($(date)) ==="
ls "$OUT/init/" | wc -l
echo

# ------- Stage 2: polish each (bench, lambda) and run default once per bench -------
echo "=== Stage 2: CD polish ==="
pids=()
for b in "${BENCHES[@]}"; do
  for lam in "${LAMBDAS[@]}"; do
    init_f="$OUT/init/${b}_l${lam}.json"
    [ -f "$init_f" ] || { echo "  skip $b l$lam (no init)"; continue; }
    nice env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONHASHSEED=42 \
      .venv/bin/python -u submissions/vmallela_v7/grav_polish.py \
      --benchmark "$b" --grav-init "$init_f" \
      --output "$OUT/polish/${b}_l${lam}.json" \
      --cd-time "$CD_TIME" --seed "$SEED" --legalize-iters 0 --arms grav \
      > "$OUT/polish/${b}_l${lam}.log" 2>&1 &
    pids+=($!)
  done
  # One default arm per bench (re-uses any init JSON for the file path but only runs "default" arm)
  nice env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONHASHSEED=42 \
    .venv/bin/python -u submissions/vmallela_v7/grav_polish.py \
    --benchmark "$b" --grav-init "$OUT/init/${b}_l${LAMBDAS[0]}.json" \
    --output "$OUT/polish/${b}_default.json" \
    --cd-time "$CD_TIME" --seed "$SEED" --legalize-iters 0 --arms default \
    > "$OUT/polish/${b}_default.log" 2>&1 &
  pids+=($!)
done
echo "  launched ${#pids[@]} polish procs at $(date)"
for p in "${pids[@]}"; do wait "$p"; done
echo "=== Stage 2 done ($(date)) ==="

# ------- Final summary -------
echo
echo "=== Final per-bench summary ==="
.venv/bin/python3 - "$OUT" "${BENCHES[@]}" -- "${LAMBDAS[@]}" << 'PYEOF'
import json, sys, glob, os
argv = sys.argv[1:]
sep = argv.index("--")
out_dir = argv[0]
benches = argv[1:sep]
lambdas = argv[sep+1:]
print(f"\n{'bench':6}  {'default':>8}", end="")
for lam in lambdas: print(f"  {'λ='+lam:>9}", end="")
print(f"  best_λ  best_proxy  Δ_vs_default")
g_minus_d = []
for b in benches:
    try:
        d = float(json.load(open(f"{out_dir}/polish/{b}_default.json"))["arms"]["default"]["final_cost"])
    except FileNotFoundError:
        d = None
    print(f"  {b:6}: {d if d is None else f'{d:>8.4f}'}", end="")
    best_p = None; best_lam = None
    for lam in lambdas:
        try:
            p = float(json.load(open(f"{out_dir}/polish/{b}_l{lam}.json"))["arms"]["grav"]["final_cost"])
            print(f"  {p:>9.4f}", end="")
            if best_p is None or p < best_p: best_p, best_lam = p, lam
        except FileNotFoundError:
            print(f"  {'-':>9}", end="")
    if best_p is not None and d is not None:
        delta = best_p - d
        winner = "DIFF" if delta < 0 else "DEF"
        print(f"  λ={best_lam:>4}  {best_p:.4f}  {delta:+.4f}  {winner}")
        g_minus_d.append(delta)
    else:
        print()
if g_minus_d:
    print(f"\nmean Δ (diff_best - default) = {sum(g_minus_d)/len(g_minus_d):+.4f} "
          f"over {len(g_minus_d)} benches  (negative = diffusion wins)")
PYEOF
