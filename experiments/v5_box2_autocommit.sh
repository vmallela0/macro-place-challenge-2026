#!/bin/bash
# experiments/v5_box2_autocommit.sh — periodic commit/push of results.csv during sweeps.
# Runs in tmux session 'autocommit'. Commits every INTERVAL_S if CSV grew, then pushes.
set -u
INTERVAL_S="${1:-1800}"   # default 30 min
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

LAST_COUNT=$(wc -l < experiments/results.csv)
echo "[autocommit] starting at row count $LAST_COUNT, interval ${INTERVAL_S}s"

while true; do
  sleep "$INTERVAL_S"
  CUR=$(wc -l < experiments/results.csv)
  if [ "$CUR" -gt "$LAST_COUNT" ]; then
    NEW=$((CUR - LAST_COUNT))
    TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    git add experiments/results.csv
    git -c user.email="vedu.mallela@gmail.com" -c user.name="Vedu Mallela" \
      commit -m "v5_box2: +$NEW rows ($CUR total) @ $TS" 2>&1 | tail -2
    git push origin v5_box2_work 2>&1 | tail -2
    LAST_COUNT=$CUR
    echo "[autocommit] pushed +$NEW rows at $TS"
  else
    echo "[autocommit] no new rows at $(date -u +%H:%M:%SZ); CSV still at $CUR"
  fi
done
