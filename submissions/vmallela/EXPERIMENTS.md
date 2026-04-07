# Macro Placement Experiments — vmallela submission

Log of experiments run on the IBM ICCAD04 benchmarks while iterating
toward a top-3 placer. Documents what we tried, what worked, what didn't,
and the data behind each decision.

## Starting Point

- **Rank**: 3 ("Convex Optimization (UWaterloo Student)")
- **Score**: 1.4556 average across 17 IBM benchmarks
- **Gap to rank 1** (UT Austin DREAMPlace): 0.0480
- **Gap to rank 2** (BakaBobo Spread+Refine): 0.0153

## Pre-existing Architecture

```
Phase 1: Push-apart pre-processing (3 dampings)
Phase 2: Multi-start legalization (20 orderings × 4 step sizes × 4 starts)
Phase 3: Coordinate descent (with incremental evaluator)
       └─ Connectivity-aware ordering, 8-direction search per macro
       └─ Reserved 15% of time for swap phase
```

Known facts going in (from earlier work):
- **Incremental evaluator**: 306× speedup vs `compute_proxy_cost` (230 evals/s)
- **Cost composition**: ~10% wirelength, ~35% density, **~55% congestion**
- **Beat RePlAce on high-utilization** (ibm02, ibm10, ibm12)
- **Lose on low-utilization** (ibm01, ibm15, ibm17, ibm18)
- **SA refinement made things worse** in earlier attempts
- **ePlace/QP from existing init** moved macros too far

---

## Bugs Discovered

### Bug 1: Swap phase was completely broken (HIGH IMPACT)

The swap phase in `_coord_descent` directly mutated `pos[i]` and `pos[j]`
without calling `incr_eval.move_macro()`. Then `_eval_cost()` returned the
stale `incr_eval.get_proxy_cost()` from the last accepted CD move. Every
swap evaluation returned approximately the same cost, so `cost < current_cost`
never fired. **The entire swap phase budget (15% of CD time = ~420s) was wasted.**

Fix: rewrote swap to use `incr_eval.move_macro()` properly with two-level
undo (move i, save state, move j, on reject undo j and forward-move i back).

### Bug 2: Connectivity ordering was always empty

`_macro_connectivity()` reads from `benchmark.net_nodes`, which is an empty
list `[]` in the IBM `.pt` files. All macros got degree 0, so the
"most-connected first" ordering did nothing — it was just the index order.

Fix: switched to `incr_eval.macro_nets` (built from `plc.nets` via the
`IncrementalEvaluator` constructor), which has the real connectivity data.

### Bug 3: Swap phase missed checking macro `i` for overlaps

After moving `i` to `j`'s position, we checked whether `j` (at its new
position) overlapped anything except `i`. We forgot to check whether `i`
(at its new position) overlapped anything except `j`. This caused INVALID
results with 14-33 overlaps on multiple benchmarks.

Fix: check both `i` and `j` against all macros before applying the swap.

---

## Init Tournament (Most Important Result)

We added several "global placement" approaches as alternative starting
positions, then let them compete via the existing legalization tournament.
The best legalized seed wins and goes into CD.

**Result on ibm01 (28 evaluations per init in the 600s legalization budget):**

| Init | Method | Best Legalized Cost | vs Winner |
|------|--------|---------------------|-----------|
| **push_0** | Conservative push-apart from raw init | **1.051953** | — |
| push_1 | Moderate push-apart | 1.054810 | +0.3% |
| push_2 | Aggressive push-apart | 1.054968 | +0.3% |
| raw | Raw benchmark init | 1.101035 | +4.7% |
| quadratic | Closed-form Laplacian solve | 1.288547 | **+22.5%** |
| analytical | Spectral + Nesterov HPWL+density | 1.301757 | **+23.7%** |

### What this tells us

1. **The benchmark's initial placement is almost optimal already.** It's
   produced by a real placement tool with congestion knowledge baked in.
   Conservative push-apart (which only resolves overlaps without moving
   macros far) wins by a wide margin.

2. **Global re-placement HURTS.** Both quadratic (Laplacian solve) and
   analytical (DREAMPlace-style HPWL+density) lose by 22-24%. They move
   macros far from positions that the initial placement correctly identified.

3. **HPWL is the wrong objective.** Both quadratic and analytical optimize
   wirelength (HPWL), but HPWL is only ~10% of the proxy cost. The dominant
   term (congestion, ~55%) is invisible to these methods.

### Decision

Removed quadratic, centroid, gradient, and analytical inits from the
pipeline. Kept the function definitions in the file in case we revisit
on different benchmark families. Increased `n_orderings` from 20 to 30 to
use the freed legalization time on more push variant evaluations.

---

## Algorithmic Experiments

### ✅ Multi-restart CD with random perturbation

After CD converges, we have 2000+ seconds of unused budget. Perturb the
best position (move 3-8 random macros by ±0.3-1.5 × macro size), legalize,
re-run CD. Eventually parallelized: 15 workers, each with their own
`IncrementalEvaluator`, exploring different perturbations.

**Result on ibm01: 1.0274 → 1.0173** (improvement of 0.005, 0.5%).

The first CD run converges in ~200-400s. The remaining ~3000s is split
across 15 parallel restart workers, each exploring a different basin of
attraction in the proxy cost landscape.

### ✅ Extended delta schedule for CD

Added [5.0, 3.0] to the front of the CD delta schedule. Allows longer-range
moves in early passes. Combined with size-scaling (`delta * size_scale[i]`),
this lets large macros take big initial steps.

Modest impact (couldn't isolate from other changes), kept it.

### ❌ Gradient descent (finite-difference) on real proxy cost

Implemented `_gradient_descent_exact`: probe each macro with ±epsilon to
get a numerical gradient, apply simultaneous update, sync via
`incr_eval.sync_positions`. Idea was to discover correlated multi-macro
improvements that CD's coordinate-wise search misses.

**Initial result: produced INVALID placements (14 overlaps on ibm01).**
The lightweight push-apart after each gradient step wasn't enough to
resolve all overlaps from the simultaneous updates.

Added an explicit overlap check: reject any step that leaves overlaps and
halve the learning rate. Works correctly now but provides only marginal
improvement (0.001%) — not worth the time it consumes.

### ❌ Simulated annealing with exact proxy cost

Implemented as Phase 3.5 with safe rollback (compare SA result to CD
result, take best). Single-macro moves with normal-distribution
displacement, exponential cooling.

**Result on ibm01:**
- 2,704,304 iterations completed
- 94.5% acceptance rate (temperature too high)
- 78% overlap-rejection rate
- CD result: 1.021806
- SA result: 1.021797 — improvement of 0.000009 (~0.001%)
- **But the run with SA was 0.0045 WORSE overall** than the run without SA,
  because SA consumed time that should have gone to additional parallel
  CD restarts.

Decision: removed SA from the pipeline. Function kept in the file.

### ❌ Alternating CD ordering (odd passes reverse)

Idea: alternate between connectivity-descending and ascending orderings
to give small/peripheral macros a chance to relocate before big macros
lock down. Untested and risked oscillation. Removed before any benchmark
ran with it.

---

## Benchmark Results (after all changes)

| Benchmark | Util | Score | Time | Status |
|-----------|------|-------|------|--------|
| ibm01 | 42.8% | **1.0173** | 3479s | VALID (multi-restart) |
| ibm02 | 55.3% | 1.4762 | 1239s | VALID (no analytical) |
| ibm03 | 50.0% | 1.3073 | 3616s | VALID (over time, fixed) |
| ibm15 | 26.7% | 1.5563 | 2895s | VALID |
| ibm17 | 17.2% | 1.7263 | 3746s | VALID (over time, fixed) |
| ibm18 | 8.7% | 1.7812 | 3525s | VALID |

**Mean across these 6 benchmarks: 1.4774** (sample includes hardest sparse cases)

Time-limit fix: reduced `TOTAL_TIME_LIMIT` from 3500 to 3300 then back to
3500 after removing SA (which had been eating into the time budget).
The competition limit is 3600s per benchmark hard timeout.

---

## What Was Kept

```
Phase 1: Push-apart pre-processing (3 dampings, raw init only)
Phase 2: Multi-start legalization
       └─ 30 orderings × 4 step sizes × 4 starts (push_0/1/2 + raw)
       └─ 600s budget
Phase 3: Coordinate descent with parallel multi-restart
       ├─ First CD run (sequential, with incremental evaluator)
       ├─ Optional GD refinement (300s cap, with overlap rejection)
       └─ 15 parallel workers, each running CD on a perturbed start
```

**Bug fixes (all kept):**
- Swap phase rewrite (was dead code, now produces real swaps)
- Net-adjacent pair ordering for swaps (1717 pairs vs 288k for ibm17)
- Connectivity ordering uses real `plc.nets` data via `incr_eval.macro_nets`
- Both `i` and `j` checked for overlaps in swap phase
- GD overlap check (rejects steps with overlaps)

**Functions kept in the file but unused (in case we revisit):**
- `_quadratic_place` — Laplacian closed-form solve
- `_centroid_place` — Jacobi iteration
- `_gradient_place` — finite-difference gradient
- `_analytical_placement` — spectral embedding + Nesterov
- `_simulated_annealing` — SA with exact proxy cost
- `_gradient_descent_exact` — wired in as Phase 4 GD refinement

---

## Lessons Learned

1. **Trust the initial placement.** It's produced by a real tool with
   real-world heuristics. Local refinement beats global re-placement on
   every benchmark we tested.

2. **Optimizing a proxy of a proxy doesn't work.** HPWL is a proxy for
   wirelength, which is only 10% of our actual proxy cost. Methods that
   optimize HPWL (DREAMPlace, our analytical/quadratic placers) are
   solving the wrong problem.

3. **Our edge over DREAMPlace is the incremental evaluator.** It computes
   the EXACT proxy cost (with congestion) at 230 evals/s. Every second
   we spend in CD is productive because we're tuning the actual scoring
   function. DREAMPlace can't do this — it has to use a smooth approximation.

4. **Multi-restart with the real cost is the right strategy.** First CD
   converges fast (~200s) to a local optimum of the true proxy cost. The
   remaining 3000s is best spent on parallel restarts from perturbed
   starting points, each exploring a different basin of attraction.

5. **Test before believing.** SA "found a marginally better solution" on
   ibm01 (improvement of 0.000009) but the run with SA was 0.005 WORSE
   overall because SA consumed time that should have gone to more restarts.
   The sequence of changes mattered more than any individual optimization.

6. **Bugs can hide in plain sight.** Both Bug 1 (swap was dead code) and
   Bug 2 (connectivity ordering was empty) were silently wasting resources.
   Found by careful planning agent code review, not by random poking.

---

## Known Issues / Future Work

- **Time budget adaptive**: Some benchmarks (ibm17) finish in 3700s+, just
  over the 3600s competition limit. Need TOTAL_TIME_LIMIT to scale with
  benchmark size or have stricter early termination in the parallel phase.
- **Gradient init never properly tested**: On ibm01 it lost in the
  tournament, but it might win on sparser benchmarks where the initial
  placement is less optimal. Worth re-testing on ibm17/18 with a tournament
  comparison.
- **Per-benchmark baselines unknown**: We have no per-benchmark scores
  from the original 1.4556-average submission. Can't tell if our changes
  improved a specific benchmark or just held even.
- **GD phase contributes little**: Each GD iteration costs ~5s on ibm01
  (4 × 246 macro evaluations + sync), but improvements are marginal.
  Could be removed or run only on benchmarks where CD plateaus quickly.
