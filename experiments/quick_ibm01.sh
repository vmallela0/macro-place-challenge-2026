#!/bin/bash
# experiments/quick_ibm01.sh
# Fastest possible iteration loop — one ibm01 run at 550s, seed 42.
# Usage: experiments/quick_ibm01.sh <exp_id> [EXTRA_ENV=val ...]
# ~10 minutes. For A/B on a single idea.
set -u
if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <exp_id> [EXTRA_ENV=val ...]"
  exit 2
fi
EXP="$1"; shift
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
"$REPO_ROOT/experiments/run_experiment.sh" "$EXP" ibm01 42 550 "$@"
