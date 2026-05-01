#!/bin/bash
# slj2 — launch the full 17-bench upgraded sweep AND auto-push watcher
# as detached daemons. Survives logout (setsid + nohup).
set -eu
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

branch=$(git rev-parse --abbrev-ref HEAD)
if [ "$branch" != "slj2" ]; then
  echo "must be on the 'slj2' branch (currently '$branch')" >&2
  exit 1
fi

if pgrep -af "(slj2_full_sweep|v7_singlev4_full_sweep)\.sh" >/dev/null; then
  echo "a sweep is already running:" >&2
  pgrep -af "(slj2_full_sweep|v7_singlev4_full_sweep)\.sh" >&2
  exit 1
fi

echo "launching slj2 sweep ..."
TMP="/tmp/slj2_pre_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$TMP"
setsid nohup bash scripts/slj2_full_sweep.sh \
  > "${TMP}/launcher.log" 2>&1 < /dev/null &
SWEEP_PID=$!
echo "sweep pid: $SWEEP_PID"

ACTUAL_DIR=""
for _ in $(seq 1 60); do
  d=$(ls -1dt /tmp/slj2_sweep_* 2>/dev/null | head -1 || true)
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
echo "$ACTUAL_DIR" > slj2/sweep_dir.txt
echo "sweep dir: $ACTUAL_DIR"

# Watcher: same pattern as lsj/watcher.sh but under slj2/ data dir,
# pushes commits to whichever branch is checked out.
echo "launching watcher ..."
setsid nohup bash lsj/watcher.sh "$ACTUAL_DIR" slj2 \
  > /dev/null 2>&1 < /dev/null &
WATCHER_PID=$!
echo "watcher pid: $WATCHER_PID"

echo
echo "Both processes detached. Tail logs:"
echo "  tail -f $ACTUAL_DIR/sweep.log"
echo "  tail -f slj2/watcher.log"
