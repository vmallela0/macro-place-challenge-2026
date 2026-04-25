# Macro placement submission — vmallela_v4

A coordinate-descent macro placer that optimizes the exact ICCAD-style
proxy cost (`1.0·HPWL + 0.5·density + 0.5·congestion`) via local search,
using a custom incremental evaluator for fast per-move updates, with
low-temperature simulated annealing in the acceptance rule and an
aggressive escape-basin operator on plateau.

**Headline result.** Mean proxy cost **1.0186** across the 17 IBM
ICCAD-2004 benchmarks at the standard 1-hour-per-benchmark budget
(seed 42; this is the number to use). All 17 placements VALID, zero
overlaps.

**Multi-seed validation.** Re-running the same code across seeds 43 and
44 (51 total runs at 3300 s each, all VALID) yields:

| Configuration              | 17-bench mean | Δ vs `optimized` (v2) |
|----------------------------|--------------:|----------------------:|
| `optimized` v2 baseline    | 1.1172        | —                     |
| **v4 seed 42 (submitted)** | **1.0186**    | **-0.099 (-8.83%)**   |
| v4 seed 43                 | 1.0170        | -0.100 (-8.97%)       |
| v4 seed 44                 | 1.0196        | -0.098 (-8.74%)       |
| v4 min-of-3 best-of        | 1.0140        | -0.103 (-9.24%)       |

Per-seed variance is ~±0.005 per bench. The min-of-3 number is
informational — it shows the algorithm is robust to RNG; the submitted
score is the single seed-42 run.

## What's new versus the `optimized` branch

Four substantive changes; their commit hashes are on this branch:

1. **Incremental-evaluator speedup (7.7×, validated to ≤3.57e-7 vs
   TILOS PlacementCost)** — precomputed `macro_pins` reverse index
   eliminates two O(n_pins) linear scans in `move_macro`'s hot path
   (4.7×); smoothing rewritten as a cumsum-based 1-D box filter
   (1.4× more); top-k selection in density and congestion cost terms
   uses `np.partition` instead of full sort; routing primitives
   (`_route_net`, L-route, T-route, two/three-pin) rewritten to use
   numpy slice and fancy-index updates instead of per-cell Python
   for-loops (1.2× more).

2. **Simulated annealing in coordinate descent** —
   `_coord_descent`'s greedy accept replaced with a Metropolis rule at
   `T0 = 5×10⁻⁵`, cooling `T *= 0.9995` per accepted move. Keeps an
   explicit `best_pos` tracker so the SA walk cannot poison the return
   value. Greedy behaviour is preserved exactly when `PLACER_SA_T0` is
   unset / ≤ 0.

3. **Escape-basin on plateau** — when the soft-cycle loop hits its
   plateau-stop condition (4 consecutive cycles with gain < 5e-5), a
   large LNS destroy-repair (`n_destroy = 80`, congestion-biased seed
   selection) plus a big soft LNS fires to convert otherwise-idle
   budget into additional basin exploration. If either finds a ≥5e-5
   improvement, the plateau counter resets and normal cycles resume
   at 1.5× cycle length.

4. **Two state-leak bug fixes** — `per_net_optimize` and the
   soft-cycle hard-polish call were returning their cost improvement
   but dropping the returned `best_pos`, so their hard-macro moves
   were wiped by the next `sync_positions(best_pos)` call. Each
   accounted for ~0.005 on the mean by itself.

## Per-benchmark proxy cost (seed 42, the submitted run)

```
  ibm01  0.7803  █
  ibm02  0.9737  ████████████████
  ibm03  0.9254  ████████████
  ibm04  0.9345  █████████████
  ibm06  1.0755  █████████████████████
  ibm07  1.0432  ██████████████████
  ibm08  1.0550  ███████████████████
  ibm09  0.7785
  ibm10  0.9625  ███████████████
  ibm11  0.8191  ████
  ibm12  1.1764  █████████████████████████
  ibm13  0.8906  ██████████
  ibm14  1.1337  ██████████████████████
  ibm15  1.1029  ████████████████████
  ibm16  1.0771  █████████████████████
  ibm17  1.3012  ███████████████████████████████████
  ibm18  1.2865  ██████████████████████████████████

  range: 0.7785 – 1.3012       mean: 1.0186
```

All 17 benchmarks improved over v2; none regressed. Hardest benchmarks
(ibm12, ibm14, ibm16, ibm17) gained the most from the SA + escape-basin
combination in absolute terms (-0.125 to -0.152 each), because they
were budget-bound in v2 and the 7.7× evaluator speedup directly
translates to more completed LNS iterations within the same wall-clock.

## v4 vs v2 per-benchmark

| Benchmark | v2 (`optimized`) | v4 seed 42 | v4 seed 43 | v4 seed 44 | min-of-3 | Δ (min-3) |
|-----------|-----------------:|-----------:|-----------:|-----------:|---------:|----------:|
| ibm01 | 0.8107 | 0.7803 | 0.7754 | 0.7775 | 0.7754 | -0.035 |
| ibm02 | 1.1002 | 0.9737 | 0.9794 | 0.9594 | 0.9594 | -0.141 |
| ibm03 | 0.9912 | 0.9254 | 0.9115 | 0.9299 | 0.9115 | -0.080 |
| ibm04 | 0.9889 | 0.9345 | 0.9345 | 0.9272 | 0.9272 | -0.062 |
| ibm06 | 1.1826 | 1.0755 | 1.0768 | 1.0789 | 1.0755 | -0.107 |
| ibm07 | 1.1277 | 1.0432 | 1.0505 | 1.0431 | 1.0431 | -0.085 |
| ibm08 | 1.1132 | 1.0550 | 1.0497 | 1.0498 | 1.0497 | -0.064 |
| ibm09 | 0.8238 | 0.7785 | 0.7707 | 0.7882 | 0.7707 | -0.053 |
| ibm10 | 1.0989 | 0.9625 | 0.9726 | 0.9718 | 0.9625 | -0.136 |
| ibm11 | 0.9133 | 0.8191 | 0.8196 | 0.8185 | 0.8185 | -0.095 |
| ibm12 | 1.3199 | 1.1764 | 1.1724 | 1.1749 | 1.1724 | -0.148 |
| ibm13 | 1.0010 | 0.8906 | 0.8934 | 0.8931 | 0.8906 | -0.110 |
| ibm14 | 1.2675 | 1.1337 | 1.1264 | 1.1336 | 1.1264 | -0.141 |
| ibm15 | 1.2291 | 1.1029 | 1.1047 | 1.1049 | 1.1029 | -0.126 |
| ibm16 | 1.2024 | 1.0771 | 1.0758 | 1.0823 | 1.0758 | -0.127 |
| ibm17 | 1.4535 | 1.3012 | 1.2923 | 1.3052 | 1.2923 | -0.161 |
| ibm18 | 1.3689 | 1.2865 | 1.2841 | 1.2956 | 1.2841 | -0.085 |
| **Mean** | **1.1172** | **1.0186** | **1.0170** | **1.0196** | **1.0140** | **-0.103** |

## Reproduction

```bash
./submissions/vmallela_v2/run.sh --all       # all 17 IBM benchmarks
./submissions/vmallela_v2/run.sh -b ibm01    # single benchmark
```

`run.sh` exports the locked environment (seed 42, BLAS pinned to one
thread, 3300 s per-benchmark budget) and the v4-tuned operator settings
(`PLACER_SA_T0 = 5e-5`, `PLACER_ESC_HARD_DESTROY = 80`). These env var
values are also baked-in defaults inside `OptimalPlacer.__init__`, so
overriding them at the env level is optional; unset behaviour matches
the submitted-table values exactly.

Expected per-benchmark result within ±0.005 of the seed-42 column above
on different hardware (run-to-run jitter from 13 wall-clock-bounded
loops). Same seed + same hardware → bit-reproducible.

## Layout

```
submissions/vmallela_v2/
├── placer.py                        OptimalPlacer entry point
├── _softmacro.py                    Soft-macro coordinate descent
├── _fd_soft.py                      Force-directed soft attraction
├── _soft_lns.py                     Soft-macro LNS
├── _per_net.py                      Per-net weighted-median pin step
│                                    (with exp 0 hard-pos-leak fix)
├── _soft_surrogate_v2.py            MLP probe-ranking wrapper
├── _surrogate.py                    ProbeLogger + 2-layer MLP
├── _moves.py                        Hard LNS + congestion-biased seed
├── run.sh                           Locked-env launcher with v4 defaults
├── results_verified_v4/             ★ Submitted seed-42 logs (17, all VALID)
├── results_verified_v4_multi/       Multi-seed verification
│   ├── SUMMARY.md
│   ├── seed_43/                     17 logs, all VALID
│   └── seed_44/                     17 logs, all VALID
├── results_verified/                v2 baseline logs (for comparison)
└── tests/
    ├── test_evaluator_equivalence.py
    └── EQUIVALENCE.md

submissions/vmallela/                v1: shared IncrementalEvaluator
                                     (with v4 hot-path optimizations)

experiments/                         Round-by-round development log
├── SUMMARY.md                       Round 8 final-sweep summary
├── results.csv                      One row per experiment
└── logs/                            Per-experiment placer stdout
```

## Caveats

- Reported numbers are from a 10-core Apple Silicon MacBook Pro with
  BLAS pinned to a single thread. The competition harness (16-core
  AMD EPYC 9655P + RTX 6000 Ada) is faster per single-thread, so the
  judges should reach equal-or-slightly-better numbers under the same
  1-hour-per-benchmark budget.
- Tier-2 (OpenROAD / NG45) WNS / TNS / Area not measured locally; the
  upstream scoring pipeline handles this.
- Single-threaded CPU; the grader's 16 cores and GPU go unused.
- Per the competition rules, the submitted score is the single seed-42
  run (`results_verified_v4/`, mean 1.0186). The min-of-3 number
  (1.0140) is **informational** — it characterises the algorithm's
  variance under different RNG seeds and can be reproduced by setting
  `PLACER_SEED=43` or `PLACER_SEED=44` in `run.sh`.

Competition specification: [`COMPETITION.md`](COMPETITION.md).
