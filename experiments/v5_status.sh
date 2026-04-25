#!/bin/bash
# experiments/v5_status.sh — quick status of v5_* placer runs.
# Shows last log line per branch + alive PIDs.

set -u
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_BASE="$REPO_ROOT/experiments/logs"

BRANCHES=(v5_cells_skip v5_escape_v2 v5_surrogate_struct v5_warmstart v5_combined)

echo
printf "%-22s  %s\n" "branch" "last_log_line"
printf "%-22s  %s\n" "----------------------" "----------------------------------------"
for b in "${BRANCHES[@]}"; do
  log="$LOG_BASE/$b/ibm01_s42_b3300.log"
  if [ -f "$log" ]; then
    last=$(grep -E "^\[|cost=|proxy=|warmstart" "$log" | tail -1 | cut -c1-80)
    [ -z "$last" ] && last=$(tail -1 "$log" | cut -c1-80)
  else
    last="(no log)"
  fi
  printf "%-22s  %s\n" "$b" "$last"
done

echo
n=$(pgrep -lf "macro_place.evaluate" | wc -l | tr -d ' ')
echo "alive placer processes: $n"
pgrep -lf "macro_place.evaluate" | awk '{print "  pid="$1}' | head -10
echo
load=$(uptime | awk -F'load averages?:' '{print $2}' | xargs)
mem=$(vm_stat 2>/dev/null | head -3)
echo "load: $load"
df -h / | tail -1 | awk '{print "disk: " $4 " avail (" $5 ")"}'
