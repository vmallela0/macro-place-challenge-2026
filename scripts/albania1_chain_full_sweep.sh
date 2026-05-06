#!/bin/bash
# Stage 3 of the autonomous overnight pipeline.
#
# After cong validation finishes, analyze whether cong-on is a real
# win, then kick off the full 17-bench sweep with the winning config.
#
# Decision rule: average of (cong_off - cong_on) across the validation
# benches > 0.01 → run full cong-on sweep at default config.
# < 0.005 → run cong-weight=1.0 A/B variant for further testing.
# Otherwise → log decision and exit (user will decide in the morning).

set -u
cd "$(dirname "$0")/.."

# Wait for cong validation to finish.
PRIOR=$(ls -td /tmp/albania1_cong_validation_* 2>/dev/null | head -1)
echo "[$(date)] stage3: watching $PRIOR" >> /tmp/albania1_chain.log

if [ -z "$PRIOR" ]; then
  # cong validation hasn't started yet; wait for it.
  for _ in $(seq 1 60); do
    sleep 60
    PRIOR=$(ls -td /tmp/albania1_cong_validation_* 2>/dev/null | head -1)
    [ -n "$PRIOR" ] && break
  done
fi
if [ -z "$PRIOR" ]; then
  echo "[$(date)] stage3: validation never started; exiting" >> /tmp/albania1_chain.log
  exit 1
fi
while ! grep -q "DONE" "$PRIOR/sweep.log" 2>/dev/null; do
  sleep 60
done

# Compute average proxy delta for each bench: (off - on)
DELTA=$(awk -F, '
  NR==1 {next}
  $2=="0" {off[$1]=$3; next}
  $2=="1" {on[$1]=$3; next}
  END {
    sum=0; n=0;
    for (b in on) {
      if (b in off && off[b] != "NA" && on[b] != "NA") {
        sum += off[b] - on[b]; n++;
      }
    }
    if (n > 0) printf "%.4f\n", sum/n; else print "NA";
  }' "$PRIOR/results.csv")

echo "[$(date)] stage3: cong delta (off-on) avg = $DELTA" \
  >> /tmp/albania1_chain.log

# Decision
if [ "$DELTA" = "NA" ]; then
  echo "[$(date)] stage3: insufficient data; exiting" \
    >> /tmp/albania1_chain.log
  exit 0
fi

# bash awk-style numeric comparison
GE_001=$(awk -v d="$DELTA" 'BEGIN { print (d >= 0.01) ? 1 : 0 }')
GE_0005=$(awk -v d="$DELTA" 'BEGIN { print (d >= 0.005) ? 1 : 0 }')

if [ "$GE_001" = "1" ]; then
  echo "[$(date)] stage3: cong-on WINS (Δ=$DELTA ≥ 0.01) → full 17-bench cong sweep" \
    >> /tmp/albania1_chain.log
  exec ./scripts/v7_singlev4_full_sweep.sh
elif [ "$GE_0005" = "1" ]; then
  echo "[$(date)] stage3: cong-on neutral (Δ=$DELTA in [0.005,0.01)) → trying cong_weight=1.0" \
    >> /tmp/albania1_chain.log
  export PLACER_V7_HESSIAN_CONG_WEIGHT=1.0
  exec ./scripts/albania1_cong_validation.sh
else
  echo "[$(date)] stage3: cong-on inconclusive or regresses (Δ=$DELTA < 0.005) → halting" \
    >> /tmp/albania1_chain.log
  exit 0
fi
