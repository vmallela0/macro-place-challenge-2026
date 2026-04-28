#!/bin/bash
set -e

# v6-gpu locked environment.
# Each worker runs single-threaded BLAS to avoid contention; we get parallelism
# from the multi-process portfolio (PLACER_V6_WORKERS) plus the MLX GPU on one
# worker.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=42

# Total wall-clock budget per benchmark (each worker uses up to this).
export PLACER_TOTAL_BUDGET=${PLACER_TOTAL_BUDGET:-3300}

# Portfolio config.
# 18-core M5 Pro: 8 workers (8 cores) leaves 10 cores for OS+MLX/grader.
# 16-core EPYC grader: same defaults are safe.
export PLACER_V6_WORKERS=${PLACER_V6_WORKERS:-8}
export PLACER_V6_GPU_WORKERS=${PLACER_V6_GPU_WORKERS:-1}

# v4-tuned operator defaults inside each worker.
export PLACER_SA_T0=${PLACER_SA_T0:-0.00005}
export PLACER_ESC_HARD_DESTROY=${PLACER_ESC_HARD_DESTROY:-80}

# T3.4 consensus warm-start (enabled by default — robust to OpenROAD Tier-2).
export PLACER_V6_CONSENSUS=${PLACER_V6_CONSENSUS:-1}
export PLACER_V6_CONSENSUS_REFINE=${PLACER_V6_CONSENSUS_REFINE:-180}
export PLACER_V6_CONSENSUS_K=${PLACER_V6_CONSENSUS_K:-16}

exec uv run evaluate submissions/vmallela_v6/placer.py "$@"
