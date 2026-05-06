#!/bin/bash
# Wait for electro smoke to finish, then decide:
#   - If electro proxy < cvar proxy on ibm12: launch focused electro sweep
#   - Else halt; user decides

set -u
cd "$(dirname "$0")/.."

PRIOR=$(ls -td /tmp/albania1_electro_smoke_2* 2>/dev/null | head -1)
echo "[$(date)] electro-decision: watching $PRIOR" >> /tmp/albania1_chain.log
[ -z "$PRIOR" ] && exit 1

while ! grep -q "DONE" "$PRIOR/sweep.log" 2>/dev/null; do
  sleep 60
done

CVAR=$(awk -F, '$1=="cvar" {print $2}' "$PRIOR/results.csv")
ELECTRO=$(awk -F, '$1=="electro" {print $2}' "$PRIOR/results.csv")
echo "[$(date)] electro-decision: cvar=$CVAR electro=$ELECTRO" \
  >> /tmp/albania1_chain.log

# Decision
if [ "$ELECTRO" = "NA" ] || [ "$ELECTRO" = "" ]; then
  echo "[$(date)] electro-decision: electro FAILED (NA) — halt" \
    >> /tmp/albania1_chain.log
  exit 0
fi

WIN=$(awk -v e="$ELECTRO" -v c="$CVAR" 'BEGIN { print (e < c) ? 1 : 0 }')
if [ "$WIN" = "1" ]; then
  echo "[$(date)] electro-decision: electro WINS ($ELECTRO < $CVAR) → focused sweep" \
    >> /tmp/albania1_chain.log
  exec ./scripts/albania1_electro_focused.sh
else
  echo "[$(date)] electro-decision: electro LOSES/ties ($ELECTRO ≥ $CVAR) — halt" \
    >> /tmp/albania1_chain.log
  exit 0
fi
