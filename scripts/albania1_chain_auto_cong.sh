#!/bin/bash
# Wait for the focused cong sweep to finish, then kick off AUTO_CONG sweep.

set -u
cd "$(dirname "$0")/.."

PRIOR=$(ls -td /tmp/albania1_focused_cong_* 2>/dev/null | head -1)
echo "[$(date)] auto_cong_chain: watching focused sweep: $PRIOR" >> /tmp/albania1_chain.log

if [ -n "$PRIOR" ]; then
  while ! grep -q "DONE" "$PRIOR/sweep.log" 2>/dev/null; do
    sleep 60
  done
fi

echo "[$(date)] auto_cong_chain: focused sweep done; launching AUTO_CONG" \
  >> /tmp/albania1_chain.log
exec ./scripts/albania1_auto_cong_sweep.sh
