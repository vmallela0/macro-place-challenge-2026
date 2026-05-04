# slj2 — algorithmic + CUDA-aware upgrades over v7

(Was the slj2-branch README before merging upstream. Preserved here so the
top-level README.md can be the canonical competition README.)

Branch `slj2` is a **cumulative upgrade** of the `vmallela_v7` macro placer,
targeting the actual competition grader hardware (per COMPETITION.md):

> All submissions will be evaluated on a **AMD EPYC 9655P with 16 cores +
> 100 GB of memory and an NVIDIA RTX 6000 Ada 48 GB.**

## Why this branch exists

The original v7 placer's Hessian phase used `mps else cpu` device dispatch
— it ran on Apple Metal on the Mac dev box but fell through to CPU on
anything else (including the grader, which has CUDA). The published
17-bench mean of **1.0003** was a Mac/MPS number; running v7 unmodified
on a CPU-only host (this `c4d-standard-16` validation box) lands at
~1.10 on `ibm15` — about 0.022 above the dev-box ref.

slj2 fixes this with two kinds of changes:

1. **Add a CUDA branch to the device dispatch** (`placer.py`). Now the
   Hessian phase prefers CUDA when available, MPS on Mac dev, CPU as a
   last resort. The grader's RTX 6000 Ada is now actually used.
2. **Algorithmic upgrades that help on any device** — top-k eigvecs,
   discrete-symmetry candidates, multi-iteration Hessian — making the
   saddle-escape phase more thorough so we don't have to rely on any
   single eigvec being "good enough" under any one backend's FP rounding.

We also flip `PLACER_V6_GPU_WORKERS` from `0` → `1` in the slj2 sweep
config so the v6 portfolio uses the grader's GPU.

## Layered upgrades (env-gated; baseline values = v7 behavior)

```
PLACER_SLJ2_TOPK=2     top-k smallest eigenvalues, not just λ_min
                       → 16 escape directions in the negative-curvature
                         subspace instead of 8 along one direction
                       → eigvec orthogonality (real-symmetric H) means
                         each direction is mathematically distinct

PLACER_SLJ2_MIRROR=1   add 3 discrete-symmetry candidates
                       → x-mirror, y-mirror, 180° rotation of softs
                       → each lands the placement in a *different basin*
                         the SA loop can't reach via local swaps

PLACER_SLJ2_POOL=16    parallelism cap for candidates
                       → c4d-standard-16 has 16 vCPUs; running
                         (8 topk + 3 mirror + …) candidates in parallel
                         doesn't oversubscribe

PLACER_V7_HESSIAN_MAX_ITERS=2   re-run the saddle escape from the
PLACER_V7_HESSIAN_TOTAL_BUDGET=1500   best post-Hessian state, until
                                      λ_min ≥ ε or budget exhausted
```

To make room for the second Hessian iteration we cut v4 from 2300 s →
2000 s. Diminishing returns on cycle 14+ (smoke #2 saw only -0.001 lift
per cycle by then), so the trade is small in v4 cost and big in
escape-attempt depth.

Strict-improvement gating against the **exact** (non-smooth) proxy is
preserved end-to-end — the algorithm cannot make a placement worse.
