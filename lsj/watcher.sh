#!/bin/bash
# lsj watcher — polls the active sweep's results.csv every 60s. On each new
# row it copies the row + PNG into the lsj branch and pushes a commit.
#
# Usage: watcher.sh <sweep_dir>
#   sweep_dir = the /tmp/v7_singlev4_sweep_TIMESTAMP/ created by
#               scripts/v7_singlev4_full_sweep.sh

set -u

SWEEP_DIR="${1:?usage: watcher.sh <sweep_dir> [data_dir=lsj]}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="${2:-lsj}"
SRC_CSV="${SWEEP_DIR}/results.csv"
DST_CSV="${REPO}/${DATA_DIR}/results.csv"
PNG_DIR="${REPO}/${DATA_DIR}/png"
LOG="${REPO}/${DATA_DIR}/watcher.log"
mkdir -p "$PNG_DIR"
BRANCH=$(cd "$REPO" && git rev-parse --abbrev-ref HEAD)

EXPECTED=17
START_TS=$(date +%s)
MAX_WALL=$((20 * 3600))  # 20h safety cap

echo "[$(date -u +%FT%TZ)] watcher start, sweep=$SWEEP_DIR" >> "$LOG"

# Wait for the sweep to write its header.
for _ in $(seq 1 60); do
  [ -f "$SRC_CSV" ] && break
  sleep 5
done
if [ ! -f "$SRC_CSV" ]; then
  echo "[$(date -u +%FT%TZ)] FATAL: $SRC_CSV did not appear" >> "$LOG"
  exit 1
fi

last_seen=0   # data rows we've already pushed (not counting header)

while true; do
  cur=$(wc -l < "$SRC_CSV" 2>/dev/null || echo 0)
  cur_data=$(( cur > 0 ? cur - 1 : 0 ))

  if [ "$cur_data" -gt "$last_seen" ]; then
    # process each new row
    new=$((last_seen + 1))
    while [ "$new" -le "$cur_data" ]; do
      # nth data row (header is row 1, so data row N is line N+1)
      row=$(sed -n "$((new + 1))p" "$SRC_CSV")
      bench=$(echo "$row" | cut -d, -f1)
      proxy=$(echo "$row" | cut -d, -f2)
      overlaps=$(echo "$row" | cut -d, -f6)
      wall=$(echo "$row" | cut -d, -f7)
      echo "[$(date -u +%FT%TZ)] new row #$new: $bench proxy=$proxy overlaps=$overlaps wall=${wall}s" >> "$LOG"

      # append to lsj CSV (header already present)
      echo "$row" >> "$DST_CSV"

      # copy PNG if present (sweep writes both /tmp/.../bench.png and assets/v7_bench.png)
      src_png="${SWEEP_DIR}/${bench}.png"
      asset_png="${REPO}/assets/v7_${bench}.png"
      dst_png="${PNG_DIR}/${bench}.png"
      if [ -f "$src_png" ]; then
        cp -f "$src_png" "$dst_png"
      elif [ -f "$asset_png" ]; then
        cp -f "$asset_png" "$dst_png"
      fi

      # regen README table
      "${REPO}/.venv/bin/python" "${REPO}/${DATA_DIR}/update_readme.py" >> "$LOG" 2>&1 || true

      # commit + push to whichever branch is checked out
      cd "$REPO"
      git add "${DATA_DIR}/results.csv" "$dst_png" README.md 2>>"$LOG"
      msg="${BRANCH}: ${bench} proxy=${proxy} overlaps=${overlaps} wall=${wall}s"
      if git commit -m "$msg" >> "$LOG" 2>&1; then
        for attempt in 1 2 3; do
          if git push origin "$BRANCH" >> "$LOG" 2>&1; then
            echo "[$(date -u +%FT%TZ)] pushed $bench to $BRANCH (attempt $attempt)" >> "$LOG"
            break
          fi
          echo "[$(date -u +%FT%TZ)] push attempt $attempt failed, retrying" >> "$LOG"
          sleep 10
        done
      else
        echo "[$(date -u +%FT%TZ)] nothing to commit for $bench (skipping push)" >> "$LOG"
      fi

      new=$((new + 1))
    done
    last_seen=$cur_data
  fi

  if [ "$last_seen" -ge "$EXPECTED" ]; then
    echo "[$(date -u +%FT%TZ)] all $EXPECTED rows seen — watcher exit clean" >> "$LOG"
    break
  fi

  now=$(date +%s)
  if [ $((now - START_TS)) -gt "$MAX_WALL" ]; then
    echo "[$(date -u +%FT%TZ)] safety cap hit (${MAX_WALL}s) — watcher exit" >> "$LOG"
    break
  fi

  sleep 60
done
