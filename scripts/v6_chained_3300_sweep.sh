#!/bin/bash
# Wait for the current 1800s sweep (PID $WAIT_PID) to finish, then run
# a 3300s/worker re-sweep matching v4's submitted budget exactly. This
# is the apples-to-apples comparison.
#
# After the 3300s sweep finishes, this script:
#   1. Re-runs scripts/v6_results_to_readme.py on the 3300s CSV
#      (overwrites the previously-inserted "Per-benchmark sweep results"
#      section in submissions/vmallela_v6/README.md)
#   2. Does NOT auto-commit/push — user reviews the diff first.

set -u
cd "$(dirname "$0")/.."

WAIT_PID=${WAIT_PID:-32680}
WORKER_BUDGET=${WORKER_BUDGET:-3300}
HARD_TIMEOUT_S=${HARD_TIMEOUT_S:-4500}

echo "[chain] waiting for sweep PID=$WAIT_PID to finish..." >&2
echo "[chain] target start time: when PID=$WAIT_PID exits" >&2

# Poll for the wait PID. Use kill -0 to test liveness.
while kill -0 "$WAIT_PID" 2>/dev/null; do
  sleep 60
done
echo "[chain] PID=$WAIT_PID exited at $(date)" >&2

# Brief grace period to ensure the previous sweep's child processes
# have fully cleaned up.
sleep 30

# Kick off the 3300s sweep. Same script with different env.
echo "[chain] starting 3300s sweep at $(date)" >&2
WORKER_BUDGET=$WORKER_BUDGET HARD_TIMEOUT_S=$HARD_TIMEOUT_S \
  ./scripts/v6_overnight_sweep.sh

echo "[chain] 3300s sweep finished at $(date)" >&2
echo "[chain] DONE. Review submissions/vmallela_v6/README.md and commit" >&2
echo "[chain] when satisfied. Logs in latest /tmp/v6_overnight_*/" >&2
