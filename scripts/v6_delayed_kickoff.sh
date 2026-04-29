#!/bin/bash
# Sleep until $TARGET_TIME (default "19:10" today), then run the v6
# overnight sweep. Used to pre-arm the sweep so the user can leave their
# Mac alone and come back to results.
#
# Usage: scripts/v6_delayed_kickoff.sh [TARGET_TIME]
#   TARGET_TIME: HH:MM (24h, today). Default 19:10.
#
# The sweep itself sleeps the rest of the wall time on its own. This
# wrapper just stalls the START.

set -u
cd "$(dirname "$0")/.."

TARGET_TIME=${1:-19:10}

now=$(date +%s)
# Compute target as today HH:MM:00 in local time.
target_today=$(date -j -f '%Y-%m-%d %H:%M:%S' \
    "$(date '+%Y-%m-%d') ${TARGET_TIME}:00" +%s 2>/dev/null)

if [ -z "$target_today" ] || [ "$target_today" -le "$now" ]; then
    # If target time is in the past today, schedule for tomorrow.
    target_today=$(date -j -v+1d -f '%Y-%m-%d %H:%M:%S' \
        "$(date -v+1d '+%Y-%m-%d') ${TARGET_TIME}:00" +%s)
fi

delay=$((target_today - now))
echo "[v6_kickoff] now: $(date)"
echo "[v6_kickoff] target: $(date -r $target_today)"
echo "[v6_kickoff] sleeping ${delay}s ($((delay/60))m $((delay%60))s)..."

# Wake the Mac periodically while we wait. caffeinate -t prevents sleep
# for the duration. Plus a parallel caffeinate keeps display awake.
caffeinate -i -m -s -t "$delay" &
CAFFEINATE_PID=$!

sleep "$delay"

# Make sure caffeinate stays alive through the actual sweep too.
kill $CAFFEINATE_PID 2>/dev/null
caffeinate -i -m -s &
CAFFEINATE_PID=$!
disown $CAFFEINATE_PID 2>/dev/null

echo "[v6_kickoff] starting sweep at $(date) (caffeinate PID=$CAFFEINATE_PID)"
./scripts/v6_overnight_sweep.sh

echo "[v6_kickoff] sweep finished at $(date)"
echo "[v6_kickoff] killing caffeinate $CAFFEINATE_PID"
kill $CAFFEINATE_PID 2>/dev/null
