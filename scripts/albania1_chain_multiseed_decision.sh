#!/bin/bash
# Wait for ibm12 multi-seed validation; if min < 1.140 (Δ < -0.016 vs
# verified 1.1557), launch the multi-seed full sweep on remaining
# 4 high-room benches. Otherwise halt with explanation.

set -u
cd "$(dirname "$0")/.."

PRIOR=$(ls -td /tmp/albania1_multiseed_2* 2>/dev/null | head -1)
echo "[$(date)] multiseed-decision: watching $PRIOR" >> /tmp/albania1_chain.log

if [ -z "$PRIOR" ]; then
  for _ in $(seq 1 60); do
    sleep 60
    PRIOR=$(ls -td /tmp/albania1_multiseed_2* 2>/dev/null | head -1)
    [ -n "$PRIOR" ] && break
  done
fi
[ -z "$PRIOR" ] && exit 1

while ! grep -q "DONE" "$PRIOR/sweep.log" 2>/dev/null; do
  sleep 60
done

# Read min proxy from results.csv
MIN_PROXY=$(awk -F, 'NR>1 && $2!="NA" && $2!="" {if (min=="" || $2<min) min=$2} END {print min+0}' "$PRIOR/results.csv")
echo "[$(date)] multiseed-decision: ibm12 min proxy = $MIN_PROXY" \
  >> /tmp/albania1_chain.log

# Decision threshold: 1.140 = verified 1.1557 - 0.016 (Gaussian K=8 prediction)
GE_THRESHOLD=$(awk -v m="$MIN_PROXY" 'BEGIN { print (m < 1.140) ? 1 : 0 }')

if [ "$GE_THRESHOLD" = "1" ]; then
  echo "[$(date)] multiseed-decision: WIN (min=$MIN_PROXY < 1.140) → launch full multi-seed" \
    >> /tmp/albania1_chain.log
  exec ./scripts/albania1_multiseed_full.sh
else
  echo "[$(date)] multiseed-decision: NEUTRAL/LOSS (min=$MIN_PROXY ≥ 1.140) → halt" \
    >> /tmp/albania1_chain.log
  exit 0
fi
