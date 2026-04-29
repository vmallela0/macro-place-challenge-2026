#!/bin/bash
# Wait for the running sweep (PID $WAIT_PID) to finish, then generate
# diagnostic GIFs for any benchmark whose final proxy >= $THRESHOLD.
# Used as a chained post-sweep step when the main sweep was already
# running before scripts/v6_overnight_sweep.sh got the GIF block added.
#
# Usage: WAIT_PID=<pid> RESULTS_DIR=<dir> ./v6_post_sweep_gifs.sh

set -u
cd "$(dirname "$0")/.."

WAIT_PID=${WAIT_PID:?"WAIT_PID env var required (PID of the running sweep)"}
RESULTS_DIR=${RESULTS_DIR:?"RESULTS_DIR env var required"}
THRESHOLD=${THRESHOLD:-1.0}
GIF_BUDGET=${GIF_BUDGET:-60}

echo "[post_gif] waiting for sweep PID=$WAIT_PID to finish..." >&2
while kill -0 "$WAIT_PID" 2>/dev/null; do
  sleep 30
done
echo "[post_gif] PID=$WAIT_PID exited at $(date)" >&2
sleep 5

CSV="$RESULTS_DIR/results.csv"
if [ ! -f "$CSV" ]; then
  echo "[post_gif] ERROR: $CSV not found" >&2
  exit 1
fi

echo "[post_gif] reading $CSV..."
HARD=$(awk -F',' -v t="$THRESHOLD" \
  'NR > 1 && $2 != "NA" && $2 + 0 >= t {print $1}' "$CSV")

if [ -z "$HARD" ]; then
  echo "[post_gif] no benches >= $THRESHOLD; nothing to render."
  exit 0
fi

echo "[post_gif] hard benches (proxy >= $THRESHOLD): $HARD"
echo "[post_gif] rendering with CD budget ${GIF_BUDGET}s each..."

for hb in $HARD; do
  echo "[post_gif] $hb starting at $(date)"
  .venv/bin/python scripts/make_v6_gif.py "$hb" "$GIF_BUDGET" \
    "$RESULTS_DIR/${hb}.gif" 2>&1 | tail -10
  .venv/bin/python scripts/make_v6_gif.py "$hb" "$GIF_BUDGET" \
    "assets/v6_${hb}.gif" 2>&1 | tail -3
done

echo "[post_gif] done at $(date)"
ls -la "$RESULTS_DIR"/*.gif 2>/dev/null | head -20
