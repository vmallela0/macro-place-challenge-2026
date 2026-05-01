#!/bin/bash
# lsj — launch the full 17-bench sweep AND the auto-push watcher as detached
# daemons. Both survive logout (setsid + nohup). Exits immediately after
# starting the daemons.
#
# Outputs:
#   /tmp/v7_singlev4_sweep_<TS>/sweep.log   — placer wall log
#   lsj/watcher.log                          — per-row push log
#   lsj/sweep_dir.txt                        — points at the active /tmp dir

set -eu
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

# Refuse to run on the wrong branch (we'd push commits to the wrong place).
branch=$(git rev-parse --abbrev-ref HEAD)
if [ "$branch" != "lsj" ]; then
  echo "must be on the 'lsj' branch (currently '$branch')" >&2
  exit 1
fi

if pgrep -af "v7_singlev4_full_sweep.sh" >/dev/null; then
  echo "a sweep is already running:" >&2
  pgrep -af "v7_singlev4_full_sweep.sh" >&2
  exit 1
fi

TS=$(date +%Y%m%d_%H%M%S)
SWEEP_DIR="/tmp/v7_singlev4_sweep_${TS}"
mkdir -p "$SWEEP_DIR"
echo "$SWEEP_DIR" > lsj/sweep_dir.txt

# 1) Sweep — note the sweep script ignores the sweep_dir arg and creates its
#    own /tmp dir, so we run it and resolve the actual dir via pgrep/ls.
echo "launching sweep at $TS ..."
setsid nohup bash scripts/v7_singlev4_full_sweep.sh \
  > "${SWEEP_DIR}/launcher.log" 2>&1 < /dev/null &
SWEEP_PID=$!
echo "sweep pid: $SWEEP_PID"

# 2) Resolve the real sweep dir (the sweep script creates its own /tmp/v7_singlev4_sweep_<ITS_TS>)
#    Poll for up to 60s for a sweep.log to appear.
ACTUAL_DIR=""
for _ in $(seq 1 60); do
  d=$(ls -1dt /tmp/v7_singlev4_sweep_* 2>/dev/null | head -1 || true)
  if [ -n "$d" ] && [ -f "$d/sweep.log" ]; then
    ACTUAL_DIR="$d"
    break
  fi
  sleep 1
done
if [ -z "$ACTUAL_DIR" ]; then
  echo "could not find sweep dir after 60s — aborting" >&2
  exit 1
fi
echo "$ACTUAL_DIR" > lsj/sweep_dir.txt
echo "sweep dir: $ACTUAL_DIR"

# 3) Watcher — detached, polls the sweep's results.csv
echo "launching watcher ..."
setsid nohup bash lsj/watcher.sh "$ACTUAL_DIR" \
  > /dev/null 2>&1 < /dev/null &
WATCHER_PID=$!
echo "watcher pid: $WATCHER_PID"

echo
echo "Both processes detached. Tail logs:"
echo "  tail -f $ACTUAL_DIR/sweep.log"
echo "  tail -f lsj/watcher.log"
echo "  cat lsj/sweep_dir.txt"
