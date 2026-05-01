# slj2 — algorithmic + CUDA-aware upgrades over v7

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

## What's different in slj2

Three layered upgrades to v7's Hessian saddle-escape phase, all gated by
env vars (set to baseline values → identical behavior to v7):

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

## Live results

<!--SLJ2:START-->
_Live results — slj2 algorithm on c4d-standard-16. 0 / 17 complete._

| Bench | slj2 proxy | v4 baseline | dev-box v7 | Δ vs v4 | Δ vs dev-box | Overlaps | Wall (s) | PNG |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| ibm01 | _pending_ | 0.7803 | 0.7653 | — | — | — | — | — |
| ibm02 | _pending_ | 0.9737 | 0.9482 | — | — | — | — | — |
| ibm03 | _pending_ | 0.9254 | 0.9166 | — | — | — | — | — |
| ibm04 | _pending_ | 0.9345 | 0.9287 | — | — | — | — | — |
| ibm06 | _pending_ | 1.0755 | 1.0546 | — | — | — | — | — |
| ibm07 | _pending_ | 1.0432 | 1.0324 | — | — | — | — | — |
| ibm08 | _pending_ | 1.0550 | 1.0291 | — | — | — | — | — |
| ibm09 | _pending_ | 0.7785 | 0.7628 | — | — | — | — | — |
| ibm10 | _pending_ | 0.9625 | 0.9492 | — | — | — | — | — |
| ibm11 | _pending_ | 0.8191 | 0.8013 | — | — | — | — | — |
| ibm12 | _pending_ | 1.1764 | 1.1557 | — | — | — | — | — |
| ibm13 | _pending_ | 0.8906 | 0.8757 | — | — | — | — | — |
| ibm14 | _pending_ | 1.1337 | 1.1070 | — | — | — | — | — |
| ibm15 | _pending_ | 1.1029 | 1.0835 | — | — | — | — | — |
| ibm16 | _pending_ | 1.0771 | 1.0435 | — | — | — | — | — |
| ibm17 | _pending_ | 1.3012 | 1.2813 | — | — | — | — | — |
| ibm18 | _pending_ | 1.2865 | 1.2697 | — | — | — | — | — |

_Sweep has not started — table will populate as each bench finishes._
<!--SLJ2:END-->

## Files in this branch

- `slj2/results.csv` — appended one row per bench
- `slj2/png/<bench>.png` — placement plot per completed bench
- `slj2/update_readme.py` — regenerates the table above from `results.csv`
- `slj2/run_smoke.sh` — single-bench smoke (ibm15) under the slj2 env
- `slj2/run_pipeline.sh` — launches sweep + watcher as detached daemons
- `lsj/watcher.sh` — branch-aware auto-pusher (used by both lsj and slj2)
- `scripts/slj2_full_sweep.sh` — driver for the full 17-bench sweep
- `submissions/vmallela_v7/_hessian_escape.py` — adds `slj2_topk_candidates`
  and `slj2_mirror_candidates` (env-gated; baseline behavior preserved)
- `submissions/vmallela_v7/placer.py` — `_hessian_escape_phase` honors
  `PLACER_SLJ2_TOPK`, `PLACER_SLJ2_MIRROR`, `PLACER_SLJ2_POOL`

## Reproducing locally

```bash
git clone https://github.com/vmallela0/macro-place-challenge-2026
cd macro-place-challenge-2026
git checkout slj2
git submodule update --init external/MacroPlacement

# Pinned to dev-box uv.lock for python 3.11 (CPU torch — no CUDA libs)
uv venv .venv
.venv/bin/pip install torch==2.10.0 \
  --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install numpy==2.4.2 scipy==1.17.0 matplotlib==3.10.8 \
  threadpoolctl tqdm absl-py
.venv/bin/pip install --no-deps -e .

# smoke (~57 min) — passes if proxy < 1.0987 (lsj-c4d - 0.01)
bash slj2/run_smoke.sh

# full sweep + auto-push (detached, ~16 h)
bash slj2/run_pipeline.sh
```
