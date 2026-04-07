# Submission: Incremental CD with Parallel Restarts

**Author:** vmallela
**Reported proxy cost:** 1.4156 (average across 17 IBM ICCAD04 benchmarks)
**All benchmarks:** VALID, zero overlaps

## TL;DR

Multi-restart coordinate descent that optimizes the **exact** proxy cost
(including the congestion term, which is 55% of the score) using a custom
incremental evaluator with ~300× speedup over the official `PlacementCost`.

The key insight: with the incremental evaluator, single-macro move evaluation
costs ~4ms instead of ~1.3s, which makes coordinate descent on the *real* proxy
cost productive — DREAMPlace-style global placers must use a smooth approximation
(HPWL + density) because they need a differentiable objective for gradient descent.
We trade global optimization for being able to optimize the actual scoring function.

## Pipeline

```
Phase 1: Push-apart pre-processing
  └─ 3 damping configurations (conservative / moderate / aggressive)

Phase 2: Multi-start legalization tournament
  ├─ 4 starting positions (3 push variants + raw initial)
  ├─ 30 orderings × 4 step sizes per start
  └─ Best legalized seed by real proxy cost wins

Phase 3: Coordinate descent with incremental evaluator
  ├─ 8-direction search per macro
  ├─ Delta schedule: 5.0 → 3.0 → 2.0 → ... → 0.02 (size-scaled)
  ├─ Connectivity-aware ordering (most-connected macros first)
  └─ Net-adjacent pairwise swap phase (looped to convergence)

Phase 4: Finite-difference gradient descent
  ├─ Probe each macro ±epsilon, compute numerical gradient on real cost
  ├─ Apply simultaneous update with overlap rejection
  └─ Diminishing learning rate with halving on rejection

Phase 5: Parallel multi-restart (15 worker processes)
  ├─ Each worker: own IncrementalEvaluator + perturbed starting position
  ├─ Each runs full CD + swap phase independently
  └─ Best result across all workers wins
```

## Why it beats HPWL-based methods

The competition's proxy cost is:
```
proxy = 1.0 × wirelength + 0.5 × density + 0.5 × congestion
```

In practice, this works out to roughly **10% wirelength + 35% density + 55% congestion**.
DREAMPlace and similar analytical placers optimize HPWL + density (which approximates
~45% of the cost). They can't optimize congestion directly because it's not smooth.

We optimize the *actual* function via the incremental evaluator. Every CD step
considers all three terms exactly. Every swap evaluates the real proxy cost.
The result is a placement that is locally optimal in the *true* objective, not
in a smooth approximation of it.

## File map

```
submissions/vmallela/
├── README.md         ← You are here
├── EXPERIMENTS.md    ← Full experiment log: what we tried, what worked, why
├── placer.py         ← The submission (single file, ~2770 lines)
├── placer_v2.py      ← Earlier variant kept for comparison (NOT submitted)
└── placer_backup.py  ← Pre-SA snapshot kept for reference (NOT submitted)
```

The submission is **`placer.py`** only. The other two files are local backups
preserved for transparency about the development history; they aren't in the
evaluation path.

## Reproducing the result

```bash
# Single benchmark:
uv run evaluate submissions/vmallela/placer.py --benchmark ibm01

# All 17 IBM benchmarks:
uv run evaluate submissions/vmallela/placer.py --all
```

The expected per-benchmark runtime on a 16-core machine is **~30-50 minutes**
(close to the 1-hour competition limit per benchmark). The pipeline uses the
full time budget for parallel CD restarts — it doesn't converge quickly because
we deliberately spend the budget exploring multiple basins of attraction.

## Key implementation details

### The IncrementalEvaluator (`IncrementalEvaluator` class in placer.py)

This is the core technical contribution. It mirrors `PlacementCost`'s
wirelength + density + congestion computation but supports O(1ms) incremental
updates when a single macro moves.

- **Wirelength:** maintains per-net HPWL, updates only affected nets on a move
- **Density:** maintains per-cell density grid, updates only affected cells
- **Congestion:** maintains routing + macro blockage arrays + their smoothed
  versions; on a move, unroutes the affected nets, re-routes them at the new
  position, updates macro blockage incrementally, and recomputes the smoothed
  congestion only for the affected region

Verified to within `< 1e-6` of the official `compute_proxy_cost` after 100+
moves (i.e., no precision drift). The float32/float64 type handling matches
PlacementCost exactly to avoid grid-cell-boundary mismatches.

### Multi-restart parallel CD (`_cd_worker`, `place()` Phase 5)

After the first CD + GD pass converges, the placer launches up to 15 worker
processes via `multiprocessing.Pool`. Each worker:

1. Loads its own `Benchmark` and `plc` instance (independent state)
2. Creates its own `IncrementalEvaluator`
3. Receives a starting position perturbed from `best_pos` (3-8 macros displaced)
4. Runs full CD + swap phase to convergence
5. Returns its best result

The main process waits for all workers and selects the best across all
restarts. With 15 cores, we get ~15× more exploration of the cost landscape
in the same wall-clock time.

### What's NOT in the pipeline

The file contains 6 functions that are defined but never called:

- `_quadratic_place` — Laplacian solve, lost legalization tournament by 23%
- `_centroid_place` — Jacobi iteration, similar to quadratic
- `_gradient_place` — FD gradient global placer, doesn't beat push-apart
- `_analytical_placement` — Spectral embedding + Nesterov, lost by 24%
- `_simulated_annealing` — SA on real proxy cost, marginal gain (0.000009),
  not worth the time it eats from CD restarts
- `_perturbation_phase` — Old perturbation loop, replaced by parallel restarts

These are kept in the file for reproducibility of the experiment log
(see `EXPERIMENTS.md`). They are not executed by `OptimalPlacer.place()`.

## Honest caveats

1. **Time-budget non-determinism:** the parallel CD workers exit when their
   wall-clock time budget runs out, so different machines (or different OS
   scheduler decisions) may produce slightly different final costs. We
   measured run-to-run variance of approximately ±0.001-0.005 on individual
   benchmarks. The reported 1.4156 average is the median of multiple runs.

2. **Time limit:** `TOTAL_TIME_LIMIT = 3300` (60 - 5 = 55 minutes) leaves a
   safety margin under the 1-hour competition limit. Two slow benchmarks
   (ibm10, ibm17) were solo-verified to run in 3570s and 3542s respectively
   on an Apple M-series Mac, both under 3600s. The competition machine
   (16-core EPYC 9655P) is faster and should have a comfortable margin.

3. **Overlap detection:** the placer uses `gap = 0.05` in its overlap
   checks (a small safety margin beyond the official `0.0040` overlap
   threshold). This sometimes rejects moves that would be technically legal
   but is the safer choice — all 17 benchmarks pass the official overlap
   verification with zero overlaps.

## Acknowledgments

- The IncrementalEvaluator was developed iteratively, with the final
  float32/float64 type-promotion handling guided by careful comparison
  against `PlacementCost` outputs at grid-cell boundaries.
- The multi-restart strategy was the result of observing that CD converged
  in ~200-400 seconds (well under the 3300 budget), leaving 2500+ seconds
  of unused compute time per benchmark.
- The decision to use the conservative push-apart variant as the default
  starting position came from the legalization tournament results — global
  re-placement methods (analytical, quadratic, spectral) all lost by 22-24%
  on ibm01, suggesting the benchmark's initial placement is already nearly
  optimal and our job is to refine it locally rather than discard it.

See `EXPERIMENTS.md` for the full experiment log including rejected approaches
and the data behind each design decision.
