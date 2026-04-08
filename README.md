# Macro Placement Challenge — vmallela's Submission Fork

> 👋 **Judges & reviewers:** This is **vmallela's submission** to the [Partcl/HRT Macro Placement Challenge](https://github.com/partcleda/macro-place-challenge-2026).
> The actual algorithm and submission docs live in [`submissions/vmallela/`](submissions/vmallela/).
> The original competition README is preserved at [`COMPETITION.md`](COMPETITION.md).

## Quick links

| What | Where |
|------|-------|
| **The submission code** | [`submissions/vmallela/placer.py`](submissions/vmallela/placer.py) |
| **Submission README** (approach overview, reproduction guide) | [`submissions/vmallela/README.md`](submissions/vmallela/README.md) |
| **Experiment log** (what we tried, what worked, what didn't) | [`submissions/vmallela/EXPERIMENTS.md`](submissions/vmallela/EXPERIMENTS.md) |

## TL;DR — Method: "Incremental CD with Parallel Restarts"

| Metric | Value |
|--------|-------|
| **Average proxy cost (17 IBM benchmarks)** | **1.4159** |
| **Validity** | All 17 VALID, zero overlaps |
| **Method** | Multi-restart coordinate descent on the **real** proxy cost |
| **Key technique** | Custom incremental evaluator (~300× speedup over PlacementCost) |
| **Parallelism** | 15 worker processes for restart phase |

## Why this beats DREAMPlace-style methods

The competition's proxy cost is `1.0 × wirelength + 0.5 × density + 0.5 × congestion`,
and **congestion is ~55% of the actual score**. DREAMPlace and similar analytical
placers optimize HPWL+density (a smooth approximation of ~45% of the cost) because
they need a differentiable objective for gradient descent. They cannot directly
optimize congestion.

Our approach: build a custom **incremental evaluator** that mirrors `PlacementCost`'s
exact wirelength + density + congestion computation but supports O(1ms) updates on
single-macro moves. With ~4ms per move evaluation (vs ~1.3s for the full evaluator),
coordinate descent on the *real* proxy cost becomes productive. Then run **15
parallel CD workers** from perturbed starting points to explore multiple local
optima of the true objective function.

## Reproducing the result

```bash
# Single benchmark:
uv run evaluate submissions/vmallela/placer.py --benchmark ibm01

# All 17 IBM benchmarks (~12 hours total):
uv run evaluate submissions/vmallela/placer.py --all
```

Per-benchmark runtime ranges from ~18 minutes (small benchmarks that converge fast)
to ~60 minutes (large benchmarks that use the full budget for parallel restart
exploration).

## Result vs Leaderboard

```
Rank   Team                                          Score
─────────────────────────────────────────────────────────────
  1    "MTK" (DreamPlace++)                          1.3998
  2    "UT Austin" (DREAMPlace Analytical)           1.4076
  3    THIS SUBMISSION (Incremental CD + Restarts)   1.4159   ← us
  4    "BakaBobo" (Spread+Refine)                    1.4403
  5    "Convex Optimization" (UWaterloo Student)     1.4556
  —    RePlAce (baseline)                            1.4578
```

We beat the next non-DREAMPlace entry (BakaBobo) by **0.024**, beat the RePlAce
baseline by **0.042**, and trail the DREAMPlace variants by **0.008-0.016**.

## What's NOT in our pipeline (and why)

We tried and rejected several approaches that other teams use. The full reasoning
is in [`submissions/vmallela/EXPERIMENTS.md`](submissions/vmallela/EXPERIMENTS.md):

- ❌ **Simulated annealing** — gave only 0.000009 improvement on ibm01, ate time
  better spent on parallel restarts
- ❌ **Spectral / Quadratic / Nesterov global placement** — lost the legalization
  tournament by 22-24% on ibm01; the benchmark's initial placement is already
  near-optimal and global re-placement hurts more than helps
- ❌ **DREAMPlace-style HPWL+density gradient** — optimizes the wrong objective
  (HPWL is only ~10% of the proxy cost)

## Honest caveats

The full caveats are in the submission README, but the short version:

1. **Run-to-run variance ~±0.001-0.005** on individual benchmarks due to
   parallel CD worker timing being wall-clock-bounded
2. **`TOTAL_TIME_LIMIT = 3300`** (55 min) leaves a safety margin under the
   1-hour competition limit; the slowest benchmark (ibm10) runs in ~3570s solo
3. **All 17 benchmarks pass official overlap verification** with zero overlaps

---

For the original Partcl/HRT competition README (rules, prizes, evaluation
methodology, leaderboard, etc.), see [`COMPETITION.md`](COMPETITION.md).
