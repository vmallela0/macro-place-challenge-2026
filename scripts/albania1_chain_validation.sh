#!/bin/bash
# Wait for the current k_dens A/B to finish, then kick off cong
# validation. Caffeinated to keep Mac awake.

set -u
cd "$(dirname "$0")/.."

# Find the most recent k_dens A/B sweep dir
PRIOR=$(ls -td /tmp/albania1_cvar_ab_* 2>/dev/null | head -1)
echo "[$(date)] watching prior A/B: $PRIOR" >> /tmp/albania1_chain.log

if [ -n "$PRIOR" ]; then
  while true; do
    if grep -q "DONE" "$PRIOR/sweep.log" 2>/dev/null; then
      break
    fi
    # Also break if no python process is running and sweep.log hasn't
    # been updated in 5 min (sweep died silently).
    if ! pgrep -fa "albania1_cvar_ab.sh" >/dev/null 2>&1; then
      mtime=$(stat -f %m "$PRIOR/sweep.log" 2>/dev/null || echo 0)
      now=$(date +%s)
      if [ $((now - mtime)) -gt 300 ]; then
        echo "[$(date)] prior sweep appears dead (5min stale); proceeding" \
          >> /tmp/albania1_chain.log
        break
      fi
    fi
    sleep 60
  done
fi

echo "[$(date)] prior A/B done; launching cong validation" >> /tmp/albania1_chain.log
exec ./scripts/albania1_cong_validation.sh
