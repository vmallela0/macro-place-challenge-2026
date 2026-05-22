# `vmallela_v6` — shared infrastructure (legacy GPU + portfolio)

This directory contains the v6 GPU + multi-process-portfolio placer
(mean **1.0184** on 17 IBM benchmarks at 1800 s / worker × 8 workers).
It is **not** the active submission, but it provides the portfolio runner,
GPU coordinate descent, Hungarian-LNS, and consensus warm-start that the v7
pipeline draws on — the v7 placer imports `_portfolio.py`, `_gpu_cd.py`,
`_hungarian_lns.py`, `_consensus.py`, `_torch_eval.py`, and `_mlx_eval.py`
from here at runtime.

v7 disables v6's portfolio at submission time (`PLACER_V6_WORKERS=1`,
`PLACER_V6_CONSENSUS=0`) because a single deep run with the same total
compute beats 8 shallow workers on the harder benchmarks.

**The current submission is `submissions/vmallela_v7/`.**
See [`../vmallela_v7/README.md`](../vmallela_v7/README.md) for the algorithm,
math derivation, and grader-verified results.
