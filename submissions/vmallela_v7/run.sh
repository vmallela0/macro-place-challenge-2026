#!/bin/bash
set -e

# v7-combinatorial locked environment. Inherits v6's defaults plus
# v7-specific knobs.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8

export PLACER_TOTAL_BUDGET=${PLACER_TOTAL_BUDGET:-3300}
export PLACER_V6_WORKERS=${PLACER_V6_WORKERS:-8}
export PLACER_V6_GPU_WORKERS=${PLACER_V6_GPU_WORKERS:-1}
export PLACER_V6_CONSENSUS=${PLACER_V6_CONSENSUS:-1}
export PLACER_V6_CONSENSUS_REFINE=${PLACER_V6_CONSENSUS_REFINE:-180}
export PLACER_V6_CONSENSUS_K=${PLACER_V6_CONSENSUS_K:-16}
export PLACER_SA_T0=${PLACER_SA_T0:-0.00005}
export PLACER_ESC_HARD_DESTROY=${PLACER_ESC_HARD_DESTROY:-80}

# v7-specific
export PLACER_V7_LAPLACIAN=${PLACER_V7_LAPLACIAN:-1}
export PLACER_V7_LAPLACIAN_PASSES=${PLACER_V7_LAPLACIAN_PASSES:-2}
export PLACER_V7_LAPLACIAN_BUDGET_FRAC=${PLACER_V7_LAPLACIAN_BUDGET_FRAC:-0.04}
# 0 = auto-trigger basin-hop only when post-Laplacian cost ≥ 1.0
# >0 = force N hops regardless. σ_0 is conservative (0.10·canvas_diag)
# since aggressive perturbation tends to push softs outside any
# reachable basin within the per-hop budget.
export PLACER_V7_BASIN_HOPS=${PLACER_V7_BASIN_HOPS:-0}
export PLACER_V7_BASIN_SIGMA0=${PLACER_V7_BASIN_SIGMA0:-0.10}
export PLACER_V7_BASIN_HOP_BUDGET=${PLACER_V7_BASIN_HOP_BUDGET:-300}

exec uv run evaluate submissions/vmallela_v7/placer.py "$@"
