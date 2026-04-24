# vmallela_v2 — verified macro placer

**Verified single-run average: 1.1172 across 17 IBM ICCAD04 benchmarks.**
Run: `./run.sh --all` at seed=42, budget=3300 s, `OMP_NUM_THREADS=1`.
Hardware: Apple Silicon MacBook Pro (10-core).
All 17 placements VALID, zero overlaps.
Run-to-run jitter ~0.002 due to time-budgeted iteration loops; not bit-deterministic but semantically reproducible.

## Leaderboard comparison

| Rank | Method            | Avg proxy  | Our delta |
|------|-------------------|------------|-----------|
| —    | **vmallela_v2**   | **1.1172** | —         |
| 1    | Cezar (ReFine)    | 1.2224     | **−8.6 %** |
| 2    | MTK DreamPlace++  | 1.2818     | −12.9 %   |
| 3    | RoRa              | 1.3241     | −15.6 %   |
| 4    | vmallela v1       | 1.4156     | −21.1 %   |

Delta = `(competitor − ours) / competitor`; positive means we cost less.

## Reproduction

```bash
./submissions/vmallela_v2/run.sh --all              # all 17 IBM benchmarks
./submissions/vmallela_v2/run.sh -b ibm01           # single benchmark
PLACER_TOTAL_BUDGET=1800 ./submissions/vmallela_v2/run.sh -b ibm07  # shorter budget
```

`run.sh` locks the reproducibility-relevant env vars:

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

### Expected per-benchmark proxy (± 0.002)

| Bench | Expected | Bench | Expected | Bench | Expected |
|-------|----------|-------|----------|-------|----------|
| ibm01 | 0.8107   | ibm08 | 1.1132   | ibm15 | 1.2291   |
| ibm02 | 1.1002   | ibm09 | 0.8238   | ibm16 | 1.2024   |
| ibm03 | 0.9912   | ibm10 | 1.0989   | ibm17 | 1.4535   |
| ibm04 | 0.9889   | ibm11 | 0.9133   | ibm18 | 1.3689   |
| ibm06 | 1.1826   | ibm12 | 1.3199   | **AVG** | **1.1172** |
| ibm07 | 1.1277   | ibm13 | 1.0010   |       |          |
|       |          | ibm14 | 1.2675   |       |          |

Full table with wall times and deltas: `results_verified/SUMMARY.md`.

ibm05 is intentionally excluded — it is not part of the 17-benchmark IBM ICCAD04 set defined in `COMPETITION.md`.

### Hardware caveat

Competition judges run on AMD EPYC 9655P (16 cores, dedicated per-process) + RTX 6000 Ada 48 GB. Our measurements are from a 10-core Apple Silicon MacBook Pro. Since the placer is pure CPU (NumPy / Torch-CPU) and a 16-core EPYC is typically ≥1.3× faster per core than M-series on these workloads, the judges' runs at the same 1-hour budget should produce equal or slightly better numbers.

## Algorithm summary

Incremental coordinate-descent pipeline built on top of `submissions/vmallela/placer.py`, with three unlocks over v1:

1. **Return soft-macro positions.** v1's `_set_placement` only committed hard-macro positions and silently discarded the optimized soft-macro (std-cell cluster) positions. Soft macros dominate HPWL and congestion — propagating their positions is worth ~14% off the average proxy cost.
2. **Adaptive cycle-budget scheduler.** Each refinement cycle's duration shrinks (×0.7) on plateau (gain < 5 × 10⁻⁵) and grows (×1.1) on rapid improvement (gain > 0.01). The placer stops early after 4 consecutive plateau cycles, which is why most benches finish well under their 3300 s budget.
3. **Per-net HPWL optimization** (weighted-median pin stepping). On each net, movable pins are visited in weight-descending order and stepped toward the weighted median of the other pins on that net. Interleaves with coordinate descent to escape CD local minima that HPWL-shrink would fix.

These compose with v1's CD infrastructure (`IncrementalEvaluator`, push-apart, legalize tournament) and a stateful MLP surrogate that ranks CD probe candidates. Combined result on 17 IBM benches: **1.1172 verified single-run average**.

### Pipeline

```
Phase 1  Push-apart preprocessing
          └─ 3 damping configs (conservative / moderate / aggressive)

Phase 2  Legalization tournament
          ├─ 30 orderings × 4 step-sizes × 5 starting positions
          └─ Best legalized result by real proxy cost wins
          (Budget: max(60, min(600, TOTAL_BUDGET // 5)))

Phase 3  Hard-macro coordinate descent + LNS + swap polish

Phase 4  Soft-macro refinement cycles (adaptive duration)
          Each cycle interleaves, in order, until cycle budget elapsed:
            5%   FD soft attraction (net-centroid targets)
            35%  Stateful MLP-surrogate soft CD (ProbeSurrogate on MPS)
            15%  Regular soft CD
            30%  Soft LNS (destroy connected subset, reinsert greedily)
            15%  Hard CD polish (also runs a per-net HPWL pass in-place)
```

Per-net HPWL optimization (`_per_net.per_net_optimize`) is **interleaved inside the soft-cycle hard-CD step**, not a separate terminal phase — it runs every cycle rather than once at the end.

### Code layout

```
submissions/vmallela_v2/
├── README.md                     ← This file
├── EXPERIMENTS.md                ← v1 → v118 exploration log
├── run.sh                        ← Locked-env launcher (use this)
├── run_verified_sweep.sh         ← Serial 17-bench driver used to produce the headline number
├── placer.py                     ← Entry point (OptimalPlacer; 3300 s hard cap)
├── _softmacro.py                 ← Soft-macro CD
├── _fd_soft.py                   ← Force-directed soft placement
├── _soft_lns.py                  ← Soft-macro LNS (destroy + repair)
├── _per_net.py                   ← Per-net HPWL weighted-median pin stepping
├── _soft_surrogate_v2.py         ← Stateful MLP surrogate wrapper
├── _surrogate.py                 ← ProbeLogger + 2-layer MLP
├── _moves.py                     ← LNS destroy-repair for hard macros
├── results_verified/             ← Per-bench logs + SUMMARY.md (the source of truth)
└── tests/
    ├── test_evaluator_equivalence.py
    └── EQUIVALENCE.md            ← Proof that IncrementalEvaluator agrees with batch PlacementCost to 2.75e-7
```

`placer.py` imports `_load_plc`, `IncrementalEvaluator`, `_push_apart`, `_legalize`, `_refine_toward_initial`, `_coord_descent`, `_cd_worker` from `submissions/vmallela/placer.py`. Keep both submission directories present in the repo — `vmallela_v2` depends on `vmallela`.

## Competition compliance

- All 17 reported wall times are under the 3600 s per-benchmark cap defined in `COMPETITION.md`. `placer.py` additionally hard-caps its own budget at 3300 s (300 s headroom for the validator + cost evaluator that run after `place()` returns).
- All placements VALID, zero overlaps.
- Dependencies: open-source only (`macro_place` package in this repo, NumPy, Torch). No proprietary tools.
- LICENSE: Apache 2.0 (repo root `LICENSE.md`).
- No per-benchmark hardcoding. `benchmark.name` is used only to load the canonical `initial.plc` file (via `_load_plc`) — there are no `if benchmark.name == "ibmXX"` branches, no seed/budget tables keyed by benchmark. The algorithm is benchmark-agnostic and should generalize to the hidden OpenROAD NG45 designs used in the Grand Prize round.

## Exploration results (not reproducible by judges)

During development we ran ~120 variants with different seeds, budgets, and interleaving schedules to understand the algorithm's sensitivity space. The best single number per benchmark across that exploration is:

| Bench | Exploration best | Seed / Budget |
|-------|------------------|---------------|
| ibm01 | 0.8147 | seed=42, 645 s  |
| ibm02 | 1.1444 | seed=42, 1881 s |
| ibm03 | 1.0374 | seed=1729, 1204 s |
| ibm04 | 1.0207 | seed=42, 644 s  |
| ibm06 | 1.2435 | seed=42, 1503 s |
| ibm07 | 1.1497 | seed=42, 1500 s |
| ibm08 | 1.1442 | seed=24680, 1194 s |
| ibm09 | 0.8558 | seed=42, 647 s  |
| ibm10 | 1.1344 | seed=42, 1796 s |
| ibm11 | 0.9569 | seed=42, 1505 s |
| ibm12 | 1.3406 | seed=42, 3000 s |
| ibm13 | 1.0620 | seed=42, 1494 s |
| ibm14 | 1.2788 | seed=42, 2002 s |
| ibm15 | 1.2559 | seed=8192, 2492 s |
| ibm16 | 1.2517 | seed=42, 1802 s |
| ibm17 | 1.4895 | seed=42, 3010 s |
| ibm18 | 1.4350 | seed=42, 1995 s |

**These are exploration results during development, not reproducible by judges running the submitted pipeline at seed=42.** They are included here only as an algorithmic reference — to show what the pipeline can reach under varied conditions — and should not be compared to competitor scores. The number that should be compared to competitor scores is the **1.1172** verified single-run average at the top of this README.

Interestingly, the verified single-run sweep beat every single exploration number (deltas −0.0040 to −0.0661) — the locked `OMP=1` single-thread env with no cross-bench CPU contention is a cleaner regime than the high-parallelism exploration runs.

Full development log: `EXPERIMENTS.md`.
