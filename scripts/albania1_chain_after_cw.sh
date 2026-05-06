#!/bin/bash
# Wait for cong_weight sweep on ibm06 to finish, then decide:
# - If optimal w ≈ 0.75: validates saddle-depth rule. Deploy
#   AUTO_LAMBDA_SCAN across the other high-room benches.
# - If optimal w = 0.5: saddle-depth rule wrong. Halt; user decides.

set -u
cd "$(dirname "$0")/.."

PRIOR=$(ls -td /tmp/albania1_cong_weight_2* 2>/dev/null | head -1)
echo "[$(date)] cw-decision: watching $PRIOR" >> /tmp/albania1_chain.log
[ -z "$PRIOR" ] && exit 1

while ! grep -q "DONE" "$PRIOR/sweep.log" 2>/dev/null; do
  sleep 60
done

# Find optimal weight (min proxy across non-NA rows)
OPT_W=$(awk -F, 'NR>1 && $2!="NA" && $2!="" {if ($2 < min || min=="") {min=$2; w=$1}} END {print w}' "$PRIOR/results.csv")
OPT_PROXY=$(awk -F, -v w="$OPT_W" 'NR>1 && $1==w {print $2}' "$PRIOR/results.csv")
W05_PROXY=$(awk -F, '$1=="0.5" {print $2}' "$PRIOR/results.csv")

echo "[$(date)] cw-decision: optimal w=$OPT_W (proxy=$OPT_PROXY); w=0.5 was $W05_PROXY" \
  >> /tmp/albania1_chain.log

# Decision: if optimal w in {0.75, 1.0} AND beats w=0.5, validate saddle-depth rule
case "$OPT_W" in
  0.75|1.0)
    BEAT=$(awk -v o="$OPT_PROXY" -v b="$W05_PROXY" 'BEGIN { print (o < b) ? 1 : 0 }')
    if [ "$BEAT" = "1" ]; then
      echo "[$(date)] cw-decision: VALIDATED (w=$OPT_W beats w=0.5) → deploy AUTO_LAMBDA_SCAN" \
        >> /tmp/albania1_chain.log
      # Deploy AUTO_LAMBDA_SCAN sweep on the other 4 high-room benches
      export PLACER_V7_HESSIAN_AUTO_LAMBDA_SCAN=1
      exec ./scripts/albania1_focused_cong.sh
    fi
    ;;
esac

echo "[$(date)] cw-decision: NOT VALIDATED — halt" >> /tmp/albania1_chain.log
exit 0
