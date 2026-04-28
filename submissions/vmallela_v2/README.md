# vmallela_v2 — a coordinate-descent macro placer with incremental proxy-cost evaluation

Author: vmallela
Depends on: `submissions/vmallela/` (reuses its `IncrementalEvaluator`, push-apart, and legalizer)

## Overview

This submission is a search-based macro placer that optimizes the exact
ICCAD-style proxy cost

```
proxy = 1.0 · HPWL_norm + 0.5 · density_norm + 0.5 · congestion_norm
```

directly, without a smooth surrogate. The central enabling component is
an incremental cost evaluator (inherited from v1) that supports per-move
updates in O(affected pins + affected grid cells) rather than
recomputing the full cost from scratch. With that primitive, coordinate
descent on a single-macro move costs a few milliseconds instead of
~1 s, which makes local search on the non-smooth objective tractable
over a 1-hour wall-clock budget.

v2 differs from v1 along three axes. First, the inner-loop placement
tensor is expanded to cover both hard (physical macros, non-zero area,
overlap-constrained) and soft (std-cell clusters, zero-area,
overlap-unconstrained) modules; v1 committed only hard positions in its
final write-back. Second, the refinement loop replaces fixed per-phase
wall-clock budgets with an adaptive schedule that shrinks cycle
duration on plateau and grows it on improvement. Third, between
coordinate-descent cycles the placer runs a per-net weighted-median
pin-stepping pass that targets the HPWL component of the objective
directly — a classical result (Kahng and Tsay-Kuh, among others) that
is awkward to integrate with gradient-based analytical placers but
natural inside an exact-objective search loop.

## Problem setting

The benchmarks are the 17-instance IBM ICCAD 2004 suite plus four
commercial NG45 designs. Each benchmark supplies a grid-based canvas,
a netlist with hard macros and soft macro clusters (std cells
aggregated by a preprocessing step), and an initial floorplan in a
`.plc` file. A valid placement is one with zero overlaps above a
0.0040 threshold.

The objective is a weighted sum of three grid-based metrics:

- **HPWL_norm.** Sum of normalized half-perimeter wirelength across
  all nets, weighted by fanout. Linear in pin position, separable
  across x and y axes.
- **density_norm.** Sum of squared excess density over a macro-size
  grid, plus a scaled variance term. Not smooth across grid-cell
  boundaries.
- **congestion_norm.** Sum of the top-10% most congested edges in a
  smoothed routing-resource map, after accounting for macro
  blockages. Integer-grid function; discrete jumps when a macro
  crosses a cell boundary.

The proxy is not differentiable, so analytical placers (DREAMPlace,
RePlAce, variants) optimize a smooth approximation — HPWL plus a
bell-shaped density term — and accept whatever congestion the routing
step finds afterwards. Search-based placers (ours, simulated
annealing, several reinforcement-learning variants) can optimize the
exact sum but pay an evaluation cost per trial move.

## Algorithm

The pipeline has five phases. Phase budgets are derived from the
total wall-clock budget (default 3300 s; hard-capped at 3300 s in the
submission to leave 300 s for the competition harness's
post-placement validator and cost computation).

### Phase 1 — Push-apart preprocessing

A low-cost overlap resolver seeded by the benchmark's initial
placement. Three damping settings are run in parallel (conservative
0.4, moderate 0.6, aggressive 0.8 over 300/500/800 iterations). The
best result seeds Phase 2.

### Phase 2 — Legalization tournament

Given four candidate seeds (the three push-apart outputs plus the raw
initial placement), the placer runs 30 greedy orderings × 4 step-size
schedules against each seed, accepting moves that reduce the exact
proxy cost. This is an expensive step by design — a few percent of
the final cost is determined here by which basin the tournament picks.
Wall-clock budget: `max(60, min(600, T/5))`.

### Phase 3 — Hard-macro coordinate descent + LNS

A standard coordinate descent with an 8-direction probe per macro.
The search lattice is size-scaled: macros sized larger than a grid
cell probe in larger steps. Each probe is evaluated via the
incremental evaluator and accepted if the total proxy cost improves.
When CD plateaus, a large-neighborhood-search operator destroys
5–15 connected macros (via net-BFS) and repairs greedily; this
escapes local minima that single-macro moves cannot.

### Phase 4 — Soft-macro refinement (adaptive cycles)

This is the bulk of the budget (~60%). Each cycle interleaves five
operators in wall-clock proportion:

| Fraction | Operator                          | Purpose                                  |
|---------:|-----------------------------------|------------------------------------------|
|      5 % | Force-directed soft attraction    | Pull soft clusters toward net centroids  |
|     35 % | MLP-surrogate-ranked soft CD      | Prioritize probes predicted to help      |
|     15 % | Regular soft CD                   | Standard local search                    |
|     30 % | Soft large-neighborhood search    | Destroy/repair clusters                  |
|     15 % | Hard-CD polish + per-net HPWL step| Maintain hard positions, shrink HPWL     |

After each cycle, the placer compares the observed cost gain to two
thresholds and adjusts the next cycle's duration:

```
gain < 5e-5   → next_duration *= 0.7     # plateau: shorten
gain > 1e-2   → next_duration *= 1.1     # improving: lengthen
```

The phase terminates after four consecutive plateau cycles. In
practice, easy benchmarks stop at 5–15 cycles; hard benchmarks
(ibm17, ibm18) use 30 or more.

### Phase 5 — Inside the hard-CD polish: weighted-median pin stepping

In every cycle's hard-polish slot, a per-net HPWL pass runs before
the CD probes. Nets are visited in descending order of pin-count ×
weight. For each movable pin `p` on a net with other pins
`{q_1, …, q_k}` of weights `{w_1, …, w_k}`, the target position for
each axis is the `w`-weighted median of the `q_i` along that axis —
the classical 1-D Fermat-Weber solution for HPWL on that net. The
pin is stepped a fraction (default 0.5) toward the target, and the
move is accepted only if the full proxy cost improves (not just
HPWL). This is complementary to CD: CD searches an integer-lattice
neighborhood along 8 axis directions, whereas the weighted-median
target is continuous and generally off-axis.

## Key subsystems

### IncrementalEvaluator (inherited from v1)

A NumPy-backed evaluator that mirrors the official `PlacementCost`
semantics and maintains:

- per-net HPWL in a float64 array, updated on move by re-evaluating
  only the affected nets,
- a per-cell density grid, updated by subtracting the macro's old
  footprint and adding its new footprint,
- routing resource arrays plus their smoothed versions, updated by
  unrouting the affected nets at the old position and re-routing at
  the new position, then re-smoothing only the affected region.

Float32 macro positions and float64 aggregations match PlacementCost
exactly at grid-cell boundaries — this was the difficult part of v1,
and v2 does not modify it.

**Equivalence to batch PlacementCost is verified** in
`tests/test_evaluator_equivalence.py`: across ibm01 / ibm06 / ibm10,
100 random hard-macro moves per benchmark, the maximum observed
absolute difference between the incremental and batch evaluators is
2.75 × 10⁻⁷. The diff does not grow with problem size (ibm10 has
786 hard macros; its max diff is 2.64 × 10⁻⁷) nor with move count
(max is set in the first few moves and remains flat). This is
consistent with IEEE-754 summation-order rounding, not a logic bug.

### Soft position write-back

v1 computed but did not return soft-macro positions from its main
loop. v2 returns a `(num_hard + num_soft, 2)` placement tensor
covering both module types. Because the proxy cost weights HPWL,
density, and congestion equally in structure but the pin count is
dominated by the soft std-cell clusters, propagating the soft
optimization accounts for a meaningful fraction of the measured
improvement over v1.

### Adaptive cycle scheduler

Motivated by the observation in v1 that roughly half of each cycle's
wall-clock was spent in the plateau tail. The scheduler above is a
simple multiplicative adjustment with a hard floor (60 s) and a
four-strike termination. It is not claimed to be optimal; it is a
practical heuristic that avoids the two obvious failure modes (over-
and under-spending).

### MLP surrogate (`_soft_surrogate_v2.py`)

A 2-layer MLP that takes 11 features per probe candidate — current
proxy components, macro position and size, probe displacement, and
benchmark-level normalizations — and predicts whether the probe will
reduce cost. Training data is collected during regular CD cycles
(probe features paired with observed cost delta). The surrogate
ranks candidates for the next soft-CD slice; candidates predicted
unhelpful are skipped. Model weights persist across cycles so that
accumulated training carries forward. Backed by MPS on Apple Silicon
where available, CPU otherwise. The measured improvement in
controlled A/B tests is ~0.003 on the average cost (see
`EXPERIMENTS.md`, entry v19).

### Weighted-median pin stepping (`_per_net.py`)

Classical result: for a single net with pin weights `w_i`, the HPWL
is minimized in each axis by placing a free pin at the `w`-weighted
median of the other pins' positions along that axis (see e.g. Kahng
et al., *VLSI Physical Design*, Chapter 4, or the earlier result by
Tsay and Kuh for timing-driven placement). The pass here applies
this per-net for movable pins, gated by the full proxy cost (HPWL
reductions that cost density or congestion elsewhere are rejected).

## Per-benchmark results

Measured with `./run.sh --all` on a 10-core Apple Silicon MacBook Pro
under the locked environment in `run.sh` (OMP/MKL/BLAS all pinned to
one thread, seed 42, `PLACER_TOTAL_BUDGET=3300`, single benchmark at
a time). Each row is one deterministic-ish run of the placer —
deterministic under seed but with ~0.002 run-to-run jitter from
wall-clock-bound loops; see caveats.

| Benchmark | Proxy cost | Wall time | Overlaps |
|-----------|-----------:|----------:|---------:|
| ibm01     | 0.8107     | 1926 s    | 0        |
| ibm02     | 1.1002     | 1989 s    | 0        |
| ibm03     | 0.9912     | 1667 s    | 0        |
| ibm04     | 0.9889     | 2054 s    | 0        |
| ibm06     | 1.1826     | 2367 s    | 0        |
| ibm07     | 1.1277     | 2376 s    | 0        |
| ibm08     | 1.1132     | 2789 s    | 0        |
| ibm09     | 0.8238     | 2243 s    | 0        |
| ibm10     | 1.0989     | 3149 s    | 0        |
| ibm11     | 0.9133     | 2311 s    | 0        |
| ibm12     | 1.3199     | 3260 s    | 0        |
| ibm13     | 1.0010     | 2503 s    | 0        |
| ibm14     | 1.2675     | 3305 s    | 0        |
| ibm15     | 1.2291     | 3115 s    | 0        |
| ibm16     | 1.2024     | 3305 s    | 0        |
| ibm17     | 1.4535     | 3293 s    | 0        |
| ibm18     | 1.3689     | 3296 s    | 0        |
| **Mean**  | **1.1172** |           |          |

All 17 placements satisfy the overlap constraint (zero overlaps above
the 0.0040 threshold). ibm05 is not part of the 17-instance
competition suite, consistent with the list in `COMPETITION.md`.

Raw per-benchmark logs: `results_verified/ibm*.log`.
Tabular summary: `results_verified/SUMMARY.md`.

## File layout

```
submissions/vmallela_v2/
├── README.md                        This file
├── EXPERIMENTS.md                   Development log (120+ variants tried)
├── placer.py                        OptimalPlacer entry point; budget hard-capped to 3300 s
├── run.sh                           Locked-env launcher
├── run_verified_sweep.sh            Serial 17-benchmark driver
├── _softmacro.py                    Soft-macro coordinate descent
├── _fd_soft.py                      Force-directed soft attraction
├── _soft_lns.py                     Soft-macro large-neighborhood search
├── _per_net.py                      Per-net weighted-median HPWL step
├── _soft_surrogate_v2.py            MLP probe-ranking wrapper
├── _surrogate.py                    ProbeLogger + 2-layer MLP definition
├── _moves.py                        Hard-macro LNS destroy/repair
├── results_verified/                Raw logs + summary for the reported run
└── tests/
    ├── test_evaluator_equivalence.py
    └── EQUIVALENCE.md
```

`placer.py` imports `_load_plc`, `IncrementalEvaluator`,
`_push_apart`, `_legalize`, `_refine_toward_initial`,
`_coord_descent`, and `_cd_worker` from
`submissions/vmallela/placer.py`; both directories must be present.

## Reproducing the result

```bash
# All 17 benchmarks, serially, at the reported settings
./submissions/vmallela_v2/run.sh --all

# Single benchmark
./submissions/vmallela_v2/run.sh -b ibm01

# Override budget (still hard-capped at 3300 s internally)
PLACER_TOTAL_BUDGET=1800 ./submissions/vmallela_v2/run.sh -b ibm07
```

`run.sh` exports the following and invokes the `evaluate` CLI:

```
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
VECLIB_MAXIMUM_THREADS=1
NUMEXPR_NUM_THREADS=1
PYTHONHASHSEED=42
PLACER_TOTAL_BUDGET=${PLACER_TOTAL_BUDGET:-3300}
PLACER_PARALLEL_WORKERS=0
```

Expected per-benchmark result within ±0.002 of the table above.

## Caveats and limitations

1. **Run-to-run jitter ~0.002.** The pipeline uses
   `while time.time() - t0 < budget` loops in 13 call sites. The
   number of inner iterations completed in a given cycle depends on
   wall-clock speed and OS scheduling, so repeated runs from the
   same seed drift on the last digit of the cost. Locking BLAS
   thread counts (as `run.sh` does) tightens this but does not
   eliminate it. A refactor to iteration-count budgets would yield
   bit-reproducibility; it was out of scope for this submission.

2. **Hardware.** Reported numbers are from a 10-core Apple Silicon
   MacBook Pro. The competition harness runs on a 16-core AMD EPYC
   9655P with per-process CPU affinity; under the same 1-hour
   budget, that machine should reach equal or slightly better
   numbers because the time-budgeted loops will complete more
   iterations per unit wall-clock.

3. **Budget cap.** `OptimalPlacer._COMPETITION_CAP_SECONDS = 3300`
   enforces a hard ceiling of 3300 s on the placer's internal
   budget, regardless of `PLACER_TOTAL_BUDGET`. This leaves 300 s
   under the competition's 3600 s per-benchmark timeout for the
   validator and cost computation that the harness runs after
   `place()` returns.

4. **NG45 + OpenROAD flow.** The harness script
   `scripts/evaluate_with_orfs.py` exists but requires OpenROAD and
   OpenROAD-flow-scripts (or Docker). Neither was available on the
   measurement machine, so post-placement WNS/TNS/Area were not
   produced locally. Those metrics are computed by the competition
   harness for top-7 submissions.

5. **No per-benchmark specialization.** `benchmark.name` is passed
   only to `_load_plc(…)` to load the canonical input file. There
   are no conditional branches on benchmark name, no seed or budget
   tables keyed by benchmark. The same code path runs on every
   benchmark.

## Development log and exploration

`EXPERIMENTS.md` contains the full development log — approximately
120 variants, including many that underperformed and were discarded
(simulated annealing with uphill moves, Nesterov momentum on the
non-smooth objective, tabu search on softs, spectral initialization,
Langevin smoothing, and others). The log is preserved for
transparency about the search path, not for claim inflation: the
best single-number-per-benchmark across all exploration runs was
1.1533, slightly worse than the reported single-run 1.1172 from this
submission's pipeline under the locked environment.

## References and prior work

- Coordinate descent for placement: a long literature starting from
  TimberWolf (Sechen and Sangiovanni-Vincentelli, 1985).
- HPWL weighted-median optimum per net: Kahng et al., *VLSI Physical
  Design*; Tsay-Kuh ZERO formulation.
- Smooth analytical placement: RePlAce (Cheng et al., TCAD 2019),
  DREAMPlace (Lin et al., DAC 2019). These optimize smoothed HPWL +
  density and are orthogonal to the approach here; the choice between
  the two families trades global convergence for direct optimization
  of the exact objective.
- Reinforcement-learning macro placement: AlphaChip (Mirhoseini et
  al., *Nature* 2021) and the surrounding literature. The present
  work is a search-based baseline that does not use RL and does not
  rely on a learned value function beyond the small MLP surrogate for
  probe ranking.
