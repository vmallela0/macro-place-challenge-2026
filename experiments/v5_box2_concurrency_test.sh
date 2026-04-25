#!/bin/bash
# experiments/v5_box2_concurrency_test.sh — diagnostic for concurrency-induced legalize failure.
# Usage:
#   bash experiments/v5_box2_concurrency_test.sh <max_parallel> <budget> [bench=ibm10] [extra_envs...]
set -u
MAX="${1:-16}"
BUDGET="${2:-300}"
BENCH="${3:-ibm10}"
shift 3 2>/dev/null || true
EXTRA_ENVS="$*"
WT="/tmp/wt_v5_escape"
SEEDS=(50 51 52 53 54 55 56 57 60 61 62 63 64 65 66 67 70 71 72 73 74 75 76 77 80 81 82 83 84 85 86 87)

echo "[ctest] max_parallel=$MAX budget=${BUDGET}s bench=$BENCH extra_envs='$EXTRA_ENVS'"
mkdir -p /tmp/ctest_outs
rm -f /tmp/ctest_outs/*

active=0
for SEED in "${SEEDS[@]:0:$MAX}"; do
  while [ "$active" -ge "$MAX" ]; do
    wait -n; active=$((active - 1))
  done
  IFS=' ' read -ra env_arr <<< "PLACER_ESC_K_REGIONS=4 $EXTRA_ENVS"
  bash "$WT/experiments/run_in_worktree.sh" ctest_max${MAX} "$WT" "$BENCH" "$SEED" "$BUDGET" "${env_arr[@]:-}" \
    > "/tmp/ctest_outs/${BENCH}_s${SEED}.out" 2>&1 &
  active=$((active + 1))
done
wait
echo "[ctest] DONE"
echo "=== valid count ==="
grep -c VALID /tmp/ctest_outs/*.out | grep -v ":0$" | head
echo "=== invalid count ==="
grep -c INVALID /tmp/ctest_outs/*.out | grep -v ":0$" | head
echo "=== fail count ==="
grep -c FAIL /tmp/ctest_outs/*.out | grep -v ":0$" | head
