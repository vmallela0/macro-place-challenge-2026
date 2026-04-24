#!/bin/bash
set -e

# Locked environment for reproducible single-run seed=42 placement.
# Do not override these unless you are measuring jitter or hardware sensitivity.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=42

export PLACER_TOTAL_BUDGET=${PLACER_TOTAL_BUDGET:-3300}
export PLACER_PARALLEL_WORKERS=0

exec uv run evaluate submissions/vmallela_v2/placer.py "$@"
