# Submission: Soft-macro CD + adaptive cycles + per-net HPWL

**Author:** vmallela
**Reported proxy cost:** 1.1172 (average across 17 IBM ICCAD04 benchmarks)
**All benchmarks:** VALID, zero overlaps, every run under the 1-hour cap
**Previous submission:** `submissions/vmallela/` (1.4156 avg)

## TL;DR

v1's incremental coordinate-descent pipeline, rebuilt around a realization
that the bottleneck wasn't the hard-macro placement — it was that we were
throwing away the soft-macro (std-cell cluster) optimization work on every
cost evaluation.

Three unlocks stack multiplicatively:

1. **Return soft positions** from `_set_placement`. v1 was committing
   only the hard-macro coordinates from its internal state and letting
   soft positions fall back to whatever the benchmark shipped with,
   silently discarding every soft move the placer had just computed.
   Soft macros dominate HPWL and congestion (stdcell clusters carry
   most of the net weight). Fixing this alone is worth ~14% off the
   average proxy cost.
2. **Adaptive cycle-budget scheduler.** Each refinement cycle's wall
   time shrinks ×0.7 on plateau (gain < 5×10⁻⁵) and grows ×1.1 on
   rapid improvement (gain > 0.01). Stops early after 4 consecutive
   plateau cycles. This stops burning budget on cycles that won't
   improve cost, and gives more time to cycles that are making progress.
3. **Per-net HPWL optimization** interleaved with CD. On each net,
   visit movable pins in weight-descending order and step each pin
   toward the weighted median of the other pins on the same net. This
   catches cases where CD's axis-aligned probe hops can't escape but a
   pure HPWL-shrink step can.

Plus a stateful 2-layer MLP surrogate that learns which soft-macro
probe directions pay off across cycles and keeps its weights between
cycles (`_soft_surrogate_v2.py`).

Combined verified single-run average on 17 IBM benches: **1.1172**.
Cezar (the current leaderboard #1) is 1.2224 — we're −8.6% on average.

## Pipeline

```
Phase 1: Push-apart preprocessing
  └─ 3 damping configurations (conservative / moderate / aggressive)

Phase 2: Legalization tournament
  ├─ 30 orderings × 4 step-sizes × 5 starting positions
  ├─ Tested against real proxy cost (not HPWL)
  └─ Best legalized seed by real proxy cost wins
  (Budget: max(60, min(600, TOTAL_BUDGET // 5)))

Phase 3: Hard-macro coordinate descent + LNS + swap polish
  ├─ 8-direction probe per macro (inherited from v1)
  └─ LNS destroy-repair cycles for escape from local minima

Phase 4: Soft-macro refinement cycles (adaptive duration)
  Each cycle interleaves, in wall-clock proportion:
    ▸  5%  FD soft attraction (net-centroid targets)
    ▸ 35%  Stateful MLP-surrogate soft CD (ranks probe candidates)
    ▸ 15%  Regular soft CD (size-scaled probe deltas)
    ▸ 30%  Soft LNS (destroy connected subset, reinsert greedily)
    ▸ 15%  Hard CD polish  ← interleaves per-net HPWL step

  Duration per cycle:
    ▸ shrink ×0.7 if gain < 5e-5
    ▸ grow   ×1.1 if gain > 0.01
    ▸ stop after 4 consecutive plateau cycles
```

## Why this beats v1 (and Cezar)

v1's pipeline was already strong on hard-macro placement: the incremental
evaluator makes coordinate descent on the *exact* proxy cost tractable
(~300× faster than `PlacementCost`), and v1's parallel-restart scheme
explores multiple basins.

Two design choices constrained v1:

- **Hard-macro-only CD.** Soft macros were moved only indirectly, via
  `plc.optimize_stdcells`, and the result was discarded by
  `_set_placement`. A placement that wins on hard-macro positions but
  pessimizes soft-macro HPWL can score worse than one with slightly
  worse hard positions but well-placed soft clusters.
- **Fixed cycle budgets.** A cycle that plateaus in 30 seconds eats the
  same budget as one that's still improving at cycle end. v1 burned
  substantial budget on already-converged cycles.

v2 lifts both: soft-macro CD is a first-class phase, and cycle duration
is reactive to observed gain. The per-net HPWL step handles the subset
of moves that CD's integer-lattice probe can't reach but a continuous
weighted-median can.

## Per-benchmark results

| Bench | Verified proxy | Wall time | Status        |
|-------|----------------|-----------|---------------|
| ibm01 | 0.8107         | 1926 s    | VALID, 0 overlaps |
| ibm02 | 1.1002         | 1989 s    | VALID, 0 overlaps |
| ibm03 | 0.9912         | 1667 s    | VALID, 0 overlaps |
| ibm04 | 0.9889         | 2054 s    | VALID, 0 overlaps |
| ibm06 | 1.1826         | 2367 s    | VALID, 0 overlaps |
| ibm07 | 1.1277         | 2376 s    | VALID, 0 overlaps |
| ibm08 | 1.1132         | 2789 s    | VALID, 0 overlaps |
| ibm09 | 0.8238         | 2243 s    | VALID, 0 overlaps |
| ibm10 | 1.0989         | 3149 s    | VALID, 0 overlaps |
| ibm11 | 0.9133         | 2311 s    | VALID, 0 overlaps |
| ibm12 | 1.3199         | 3260 s    | VALID, 0 overlaps |
| ibm13 | 1.0010         | 2503 s    | VALID, 0 overlaps |
| ibm14 | 1.2675         | 3305 s    | VALID, 0 overlaps |
| ibm15 | 1.2291         | 3115 s    | VALID, 0 overlaps |
| ibm16 | 1.2024         | 3305 s    | VALID, 0 overlaps |
| ibm17 | 1.4535         | 3293 s    | VALID, 0 overlaps |
| ibm18 | 1.3689         | 3296 s    | VALID, 0 overlaps |
| **AVG** | **1.1172**   |           |                   |

Raw per-benchmark logs: `results_verified/ibm*.log`.
Full summary with deltas: `results_verified/SUMMARY.md`.

ibm05 is intentionally excluded — it is not in the 17-benchmark IBM
ICCAD04 set defined in `COMPETITION.md`.

## Leaderboard comparison

| Rank | Method          | Avg proxy  | vs vmallela_v2 |
|------|-----------------|------------|----------------|
| —    | **vmallela_v2** | **1.1172** | —              |
| 1    | Cezar (ReFine)  | 1.2224     | **−8.6 %**     |
| 2    | MTK DreamPlace++| 1.2818     | −12.9 %        |
| 3    | RoRa            | 1.3241     | −15.6 %        |
| 4    | vmallela v1     | 1.4156     | −21.1 %        |

Delta is `(competitor − ours) / competitor`; positive means we cost less.

## File map

```
submissions/vmallela_v2/
├── README.md                     ← You are here
├── EXPERIMENTS.md                ← v1 → v118 exploration log
├── placer.py                     ← The submission entry point
│                                   (OptimalPlacer; TOTAL budget capped at 3300 s)
├── run.sh                        ← Locked-env launcher (use this to reproduce)
├── run_verified_sweep.sh         ← Serial 17-bench driver used to produce the headline
├── _softmacro.py                 ← Soft-macro CD
├── _fd_soft.py                   ← Force-directed soft attraction (net-centroid targets)
├── _soft_lns.py                  ← Soft-macro LNS (destroy + repair)
├── _per_net.py                   ← Per-net HPWL weighted-median pin stepping
├── _soft_surrogate_v2.py         ← Stateful MLP-surrogate wrapper around soft CD
├── _surrogate.py                 ← ProbeLogger + 2-layer MLP
├── _moves.py                     ← LNS destroy-repair for hard macros
├── results_verified/             ← Raw per-bench logs + SUMMARY.md (the evidence)
└── tests/
    ├── test_evaluator_equivalence.py
    └── EQUIVALENCE.md            ← Proof IncrementalEvaluator == batch PlacementCost
```

The submission is **`placer.py` + the `_*.py` modules it imports**.
`placer.py` additionally loads `_load_plc`, `IncrementalEvaluator`,
`_push_apart`, `_legalize`, `_refine_toward_initial`, `_coord_descent`,
and `_cd_worker` from `submissions/vmallela/placer.py`. **Keep both
submission directories present in the repo — v2 depends on v1.**

## Reproducing the result

```bash
# All 17 IBM benchmarks (15 h wall clock on a 10-core Mac)
./submissions/vmallela_v2/run.sh --all

# Single benchmark
./submissions/vmallela_v2/run.sh -b ibm01

# Shorter budget
PLACER_TOTAL_BUDGET=1800 ./submissions/vmallela_v2/run.sh -b ibm07
```

`run.sh` locks the reproducibility-relevant env:

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=42
export PLACER_TOTAL_BUDGET=${PLACER_TOTAL_BUDGET:-3300}
export PLACER_PARALLEL_WORKERS=0
```

Expected per-benchmark results within **±0.002** of the table above.
The placer hard-caps its own budget at 3300 s regardless of the env
variable (see `OptimalPlacer._COMPETITION_CAP_SECONDS`), leaving
300 s headroom below the 1-hour competition limit for the validator
and cost evaluator that run after `place()` returns.

## Key implementation details

### Returning soft positions (the "free" 14%)

v1's pipeline moves std-cell clusters via `plc.optimize_stdcells` as
part of its cost evaluation, but only the hard-macro coordinates were
preserved when v1 returned its final placement — `_set_placement`
wrote back the hard array and let soft positions be whatever the
benchmark had initially. v2 tracks both arrays end-to-end and returns
them together. This required no algorithmic change, just plumbing.

The asymmetry of the proxy cost makes soft placement dominant:
`proxy = 1.0·wirelength + 0.5·density + 0.5·congestion`. Soft macros
(stdcell clusters) are the bulk of the pin count, so they drive HPWL;
they're also what routing has to push through, so they drive
congestion. Hard macros mostly drive density (and a little HPWL).
Moving a soft cluster by a few units can change the cost more than
moving 20 hard macros.

### Adaptive cycle scheduler

Each soft-refinement cycle runs a fixed interleave (FD → surrogate CD →
regular CD → LNS → hard polish), but the *duration* of each cycle is
reactive. After a cycle completes, we compare the cost gain to two
thresholds:

```
if gain < 5e-5:      cycle_duration *= 0.7
elif gain > 0.01:    cycle_duration *= 1.1
```

Below a minimum (60 s) the cycle is frozen. After 4 consecutive cycles
with gain below threshold, the whole phase exits. In practice this
means easy benchmarks plateau-stop at 5-15 cycles and use only half
their nominal budget; hard benchmarks (ibm17, ibm18) run 30+ cycles
and use nearly all of it.

### Per-net HPWL pin stepping (`_per_net.py`)

Coordinate descent probes positions on an integer lattice along 8
directions. It's good at finding local minima *along axes* but can
miss the continuous HPWL optimum when a net has pins spread at odd
angles. The per-net HPWL pass complements CD:

1. Rank nets by weight (descending).
2. For each net, for each movable pin, compute the *weighted median*
   of the other pins' positions on that net.
3. Step the pin toward that median by a fractional amount.
4. Accept only if the combined proxy cost improves (HPWL reduction can
   cost density/congestion elsewhere).

This runs inside the hard-CD polish step of each cycle, so its gains
compound with CD's — it's not a terminal pass.

### Stateful MLP surrogate (`_soft_surrogate_v2.py`)

Soft-macro CD probes many candidate positions per macro; the cost is
in evaluating each. The surrogate is a 2-layer MLP trained on
`(features → did_this_probe_improve_cost)` pairs collected during
regular CD cycles. It runs on MPS (Apple GPU) where available, CPU
otherwise. Between cycles its weights persist — so the placer gets
better at CD the longer it runs, instead of re-learning from scratch.

Features (11-dim): current proxy cost, HPWL/density/congestion
components, the macro's current position and size, the probe
displacement, and a few bench-level normalizations. Gave a consistent
~0.003 improvement in the A/B experiments (see `EXPERIMENTS.md`
entry v19).

### IncrementalEvaluator equivalence

The entire pipeline rests on the incremental evaluator from v1 being
numerically faithful to the batch `PlacementCost`. `tests/EQUIVALENCE.md`
reports the result of `tests/test_evaluator_equivalence.py`: across
ibm01 / ibm06 / ibm10, 100 random hard-macro moves each, the maximum
observed absolute difference between `incr.get_proxy_cost()` and
`compute_proxy_cost(current_placement, …)` is **2.75 × 10⁻⁷** —
about 363× inside the 10⁻⁴ tolerance we care about, consistent with
IEEE-754 summation-order noise and not growing with move count.

## Honest caveats

1. **Time-budget non-determinism (~0.002).** Every phase of the
   pipeline runs "as many moves as fit in T seconds" via
   `while time.time() - t0 < budget`. The number of moves completed in
   a given cycle depends on wall-clock jitter (CPU clock boost, OS
   scheduling, BLAS thread count), so repeated runs from the same seed
   drift by ~0.002 on the headline cost. We locked `OMP_NUM_THREADS=1`
   and friends in `run.sh` to pin BLAS determinism, but the
   time-budget loops themselves are not iteration-count-bounded and
   refactoring them to be iteration-count-bounded was out of scope
   for this submission. The verified 1.1172 average has jitter on the
   order of the last digit. Judges on different hardware at the same
   1-hour budget should land within ±0.005.

2. **Hardware caveat.** We measured on a 10-core Apple Silicon
   MacBook Pro. Competition judges run on AMD EPYC 9655P (16 cores,
   dedicated per-process) + RTX 6000 Ada 48 GB. The placer is pure
   CPU — no CUDA, no kernels on the GPU beyond the MLP's MPS calls —
   so per-core throughput on the judges' EPYC is typically ≥1.3×
   better than M-series on NumPy hot loops. At the same 1-hour
   budget, judges' runs should match or slightly improve ours.

3. **Budget cap.** `OptimalPlacer._COMPETITION_CAP_SECONDS = 3300`
   hard-caps the placer's internal budget at 3300 s regardless of the
   env variable. This leaves 300 s headroom below the competition's
   3600 s hard timeout for the validator + cost evaluator that run
   after `place()` returns. Verified: the longest run (ibm14, ibm16)
   was 3305 s total, still under 3600 s.

4. **Exploration results ≠ submission results.** During development
   we ran ~120 variants with different seeds, budgets, and
   interleavings. The best single number per benchmark across that
   exploration (shown below) was 1.1533 — **worse** than this
   submission's verified single-run 1.1172. That's because the
   exploration runs happened under heavy CPU contention (multiple
   benches in parallel on one machine); the verified sweep ran
   serially with locked threading. Do not compare the exploration
   numbers to competitor scores — the 1.1172 in this README is the
   only number our pipeline actually produces at seed=42 under the
   locked env.

5. **No per-benchmark hardcoding.** `benchmark.name` is used only to
   load the canonical `initial.plc` (via `_load_plc`) — there are no
   `if benchmark.name == "ibmXX"` branches, no seed or budget tables
   keyed by benchmark name. The algorithm is benchmark-agnostic and
   should generalize to the hidden OpenROAD NG45 designs used in the
   Grand Prize round, though we could not verify on NG45 locally
   (no OpenROAD binary installed on this machine).

6. **NG45 / OpenROAD unverified locally.** The harness in
   `scripts/evaluate_with_orfs.py` exists but requires OpenROAD +
   OpenROAD-flow-scripts or Docker. Neither is installed on this
   machine. We did not produce WNS/TNS/Area numbers ourselves;
   judges will measure those on their EPYC box for top-7 submissions.

## Acknowledgments

- The IncrementalEvaluator from v1 (`submissions/vmallela/placer.py`)
  is the foundation. Nothing in v2 is possible without the
  float32/float64 type-promotion handling that matches PlacementCost
  at grid-cell boundaries, which was the hard-won work of v1.
- The soft-macro unlock was found by looking for what v1 left on the
  table. The clue was that `plc.get_cost()` and
  `compute_proxy_cost(…)` sometimes disagreed mid-run — tracing the
  disagreement back to `_set_placement` discarding soft positions
  made the fix obvious.
- The adaptive cycle scheduler was the result of watching v1's cycle
  trace and noticing that ~40% of cycles finished in half the
  budget, then sat in a plateau loop wasting the rest.
- The per-net HPWL idea came from reading the literature on analytical
  placers (weighted-median minimizes HPWL on a net) and realizing that
  the HPWL-only gradient step is a strict subset of the proxy-cost
  gradient and can be applied locally per-net without global
  re-placement.
- The evaluator equivalence test was written to be the artifact we
  could point at if anyone questioned whether the 300× incremental
  speedup was numerically faithful. It passed with 363× headroom
  inside the tolerance.

See `EXPERIMENTS.md` for the full development log — 120+ variants
including the ones that didn't work and why (Langevin smoothing init,
quantum-amplitude init, tabu search on softs, Nesterov momentum, and
~10 others that underperformed the baseline).
