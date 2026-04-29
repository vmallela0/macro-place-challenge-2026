#!/bin/bash
# Overnight v6 sweep across all 17 IBM benchmarks.
#
# For each bench, runs:
#   PLACER_TOTAL_BUDGET=$WORKER_BUDGET (default 1800s)
#   PLACER_V6_WORKERS=8 (1 GPU + 7 CPU)
#   PLACER_V6_CONSENSUS=1 (graft + refine)
#   wrapped in `timeout $HARD_TIMEOUT_S` (default 2700s) so a stuck
#   worker can't stall the rest of the sweep.
#
# Writes per-bench logs to /tmp/v6_overnight_<TIMESTAMP>/<bench>.log
# Writes a summary CSV to /tmp/v6_overnight_<TIMESTAMP>/results.csv
# After the sweep completes, calls scripts/v6_results_to_readme.py to
# update submissions/vmallela_v6/README.md with a results table.

set -u
cd "$(dirname "$0")/.."

WORKER_BUDGET=${WORKER_BUDGET:-1800}
HARD_TIMEOUT_S=${HARD_TIMEOUT_S:-2700}
RESULTS_DIR_BASE=${RESULTS_DIR_BASE:-/tmp}

OUT="$RESULTS_DIR_BASE/v6_overnight_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"

# Locked env (matches submissions/vmallela_v6/run.sh).
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=42
export PLACER_TOTAL_BUDGET=$WORKER_BUDGET
export PLACER_V6_WORKERS=8
export PLACER_V6_GPU_WORKERS=1
export PLACER_V6_CONSENSUS=1
export PLACER_V6_CONSENSUS_REFINE=120
export PLACER_V6_CONSENSUS_K=16
export PLACER_SA_T0=0.00005
export PLACER_ESC_HARD_DESTROY=80
# Save the final placement per bench so the post-bench plotter can render
# a static visualization. Placements go in $OUT/<bench>.npy.
export PLACER_V6_SAVE_PLACEMENT="$OUT/{name}.npy"

BENCHES="ibm01 ibm02 ibm03 ibm04 ibm06 ibm07 ibm08 ibm09 ibm10 ibm11 ibm12 ibm13 ibm14 ibm15 ibm16 ibm17 ibm18"

echo "v6 overnight sweep" > "$OUT/sweep.log"
echo "  started: $(date)" >> "$OUT/sweep.log"
echo "  worker budget: ${WORKER_BUDGET}s" >> "$OUT/sweep.log"
echo "  hard timeout: ${HARD_TIMEOUT_S}s" >> "$OUT/sweep.log"
echo "  results dir: $OUT" >> "$OUT/sweep.log"
echo "" >> "$OUT/sweep.log"

# Header for results CSV.
echo "benchmark,proxy_cost,wirelength_cost,density_cost,congestion_cost,overlap_count,wall_clock_s,exit_code,timestamp" \
  > "$OUT/results.csv"

for b in $BENCHES; do
  echo "=== $b: started $(date) ===" | tee -a "$OUT/sweep.log"
  t_start=$(date +%s)

  # macOS doesn't ship a `timeout` binary; emulate with background process
  # + sleep + kill on overrun. Use the venv's python directly (avoids
  # `uv run` reinstalling editable package in worker subprocesses).
  .venv/bin/python -m macro_place.evaluate \
    submissions/vmallela_v6/placer.py --benchmark "$b" \
    > "$OUT/${b}.log" 2>&1 &
  cmd_pid=$!
  ( sleep "$HARD_TIMEOUT_S"; if kill -0 $cmd_pid 2>/dev/null; then
      echo "  TIMEOUT after ${HARD_TIMEOUT_S}s; killing $cmd_pid" \
        | tee -a "$OUT/sweep.log"
      pkill -P $cmd_pid 2>/dev/null
      kill -TERM $cmd_pid 2>/dev/null
      sleep 5
      kill -KILL $cmd_pid 2>/dev/null
    fi ) &
  killer_pid=$!
  wait $cmd_pid 2>/dev/null
  rc=$?
  # Cancel the killer if cmd finished first.
  kill $killer_pid 2>/dev/null
  wait $killer_pid 2>/dev/null

  t_end=$(date +%s)
  elapsed=$((t_end - t_start))
  echo "  exit code: $rc, wall: ${elapsed}s" | tee -a "$OUT/sweep.log"

  # Parse the final "proxy=0.xxxx (wl=... den=... cong=...)" line.
  # Example: "proxy=0.7969  (wl=0.074 den=0.556 cong=0.890)  VALID  [180.46s]"
  line=$(grep -E "^proxy=" "$OUT/${b}.log" | tail -1)
  proxy=$(echo "$line" | sed -E 's/.*proxy=([0-9.]+).*/\1/' | head -c 12)
  wl=$(echo "$line"    | sed -E 's/.*wl=([0-9.]+).*/\1/'    | head -c 12)
  den=$(echo "$line"   | sed -E 's/.*den=([0-9.]+).*/\1/'   | head -c 12)
  cong=$(echo "$line"  | sed -E 's/.*cong=([0-9.]+).*/\1/'  | head -c 12)
  if echo "$line" | grep -q "VALID"; then
    overlaps=0
  else
    overlaps=$(grep -E "overlaps=" "$OUT/${b}.log" | tail -1 | sed -E 's/.*overlaps=([0-9]+).*/\1/' | head -c 8)
    [ -z "$overlaps" ] && overlaps="?"
  fi

  echo "${b},${proxy:-NA},${wl:-NA},${den:-NA},${cong:-NA},${overlaps:-NA},${elapsed},${rc},$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    >> "$OUT/results.csv"
  echo "  proxy=${proxy:-NA} overlaps=${overlaps:-NA}" | tee -a "$OUT/sweep.log"

  # Render placement visualization. Goes to BOTH the run dir (archived
  # alongside the .npy) and assets/ (committed to the repo).
  if [ -f "$OUT/${b}.npy" ]; then
    .venv/bin/python scripts/v6_placement_plot.py "$b" "$OUT/${b}.npy" \
      "$OUT/${b}.png" >> "$OUT/sweep.log" 2>&1 || \
      echo "  plot $OUT/${b}.png failed" | tee -a "$OUT/sweep.log"
    .venv/bin/python scripts/v6_placement_plot.py "$b" "$OUT/${b}.npy" \
      "assets/v6_${b}.png" >> "$OUT/sweep.log" 2>&1 || \
      echo "  plot assets/v6_${b}.png failed" | tee -a "$OUT/sweep.log"
  else
    echo "  (no .npy saved for ${b}; skipping plot)" | tee -a "$OUT/sweep.log"
  fi
done

echo "" >> "$OUT/sweep.log"
echo "v6 main sweep finished: $(date)" >> "$OUT/sweep.log"
echo "results CSV: $OUT/results.csv" >> "$OUT/sweep.log"

# Post-process step 1: build markdown table and update v6 README.
.venv/bin/python scripts/v6_results_to_readme.py "$OUT/results.csv" \
  >> "$OUT/sweep.log" 2>&1

# Post-process step 2: render diagnostic convergence GIFs for any
# benchmark whose final proxy >= 1.0. These show push-apart -> legalize
# -> CD with snapshots so we can see WHERE the placer is getting stuck
# on the hard benches. Lighter than the full pipeline (single CPU
# worker, 60s CD budget, no consensus/LNS) — diagnostic only, NOT used
# for the reported score.
echo "" >> "$OUT/sweep.log"
echo "=== diagnostic GIFs for hard benches (proxy >= 1.0) ===" \
  | tee -a "$OUT/sweep.log"
GIF_BUDGET=${GIF_BUDGET:-60}
HARD_BENCHES=$(awk -F',' 'NR > 1 && $2 != "NA" && $2 + 0 >= 1.0 {print $1}' \
                "$OUT/results.csv")
if [ -z "$HARD_BENCHES" ]; then
  echo "  (no benches >= 1.0 — skipping diagnostic GIFs)" \
    | tee -a "$OUT/sweep.log"
else
  for hb in $HARD_BENCHES; do
    echo "  $hb: rendering diagnostic gif (${GIF_BUDGET}s CD budget)..." \
      | tee -a "$OUT/sweep.log"
    .venv/bin/python scripts/make_v6_gif.py "$hb" "$GIF_BUDGET" \
      "$OUT/${hb}.gif" >> "$OUT/sweep.log" 2>&1 || \
      echo "    gif $OUT/${hb}.gif failed" | tee -a "$OUT/sweep.log"
    .venv/bin/python scripts/make_v6_gif.py "$hb" "$GIF_BUDGET" \
      "assets/v6_${hb}.gif" >> "$OUT/sweep.log" 2>&1 || \
      echo "    gif assets/v6_${hb}.gif failed" | tee -a "$OUT/sweep.log"
  done
fi

echo "DONE" >> "$OUT/sweep.log"
