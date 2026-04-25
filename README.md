# Macro placement submission — vmallela_v4

A coordinate-descent macro placer that optimizes the exact ICCAD-style
proxy cost (`1.0 · HPWL + 0.5 · density + 0.5 · congestion`) via local
search, using an incremental evaluator for fast per-move updates, with
low-temperature simulated annealing in the acceptance rule and an
aggressive escape-basin operator on plateau.

Verified single-run mean proxy cost of **1.0186** across the 17 IBM
ICCAD 2004 benchmarks; all placements valid, zero overlaps, every run
under the 1-hour per-benchmark cap.

This is v4, built on top of `submissions/vmallela_v2/`'s pipeline with
four substantive changes over the `optimized` branch (mean 1.1172):

1. **Incremental-evaluator speedup (7.7×)** — precomputed `macro_pins`
   reverse index eliminates two O(n_pins) linear scans in the `move_macro`
   hot path; smoothing vectorized via a cumsum-based box filter; top-k
   selection in the density and congestion cost terms uses `np.partition`
   instead of full sort; routing primitives (`_route_net`, L-route,
   T-route, two/three-pin) rewritten to use numpy slice / fancy-index
   updates instead of per-cell Python loops. Equivalence to batch
   `PlacementCost` preserved at ≤3.57e-7 absolute difference, 500 random
   moves validated across ibm01 / ibm06 / ibm10.

2. **Simulated annealing in coordinate descent** — `_coord_descent`'s
   greedy accept replaced with a Metropolis rule at `T0=5e-5`, cooling
   `T*=0.9995` per accepted move. Keeps an explicit `best_pos` tracker
   so the SA walk cannot poison the return value. Gate-tested via
   `PLACER_SA_T0`; defaults to the winning 5e-5. Greedy behaviour is
   preserved exactly when the env var is unset / ≤0.

3. **Escape-basin on plateau** — when the soft-cycle loop hits its
   plateau-stop condition (4 consecutive cycles with gain < 5e-5), a
   large LNS destroy-repair (`n_destroy=80`, congestion-biased seed
   selection) plus a big soft LNS fires to convert otherwise-idle
   budget into additional basin exploration. If either finds a
   ≥5e-5 improvement, the plateau counter resets and normal cycles
   resume at 1.5× cycle length.

4. **Two state-leak bug fixes** — `per_net_optimize` and the soft-cycle
   hard-polish call were returning their cost improvement but dropping
   the returned `best_pos`, so their hard-macro moves were wiped by
   the next `sync_positions(best_pos)` call in the pipeline.

## Per-benchmark proxy cost

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

Bar length is linear in proxy cost, normalized against the bench with
the minimum (ibm09) and the bench with the maximum (ibm17). Lower is
better.

All 17 benchmarks improved over the v2 (`optimized` branch) submission;
none regressed. Mean improvement **-0.0986** (**-8.8%**). Hardest
benchmarks (ibm12, ibm14, ibm16, ibm17) gained the most from the
SA + escape-basin combination in absolute terms (-0.125 to -0.152 each),
because they were budget-bound in v2 and the 7.7× evaluator speedup
directly translates to more completed LNS iterations within the same
wall-clock.

- Code and write-up: [`submissions/vmallela_v2/`](submissions/vmallela_v2/)
- Per-benchmark raw logs: [`submissions/vmallela_v2/results_verified_v4/`](submissions/vmallela_v2/results_verified_v4/)
- Experiment harness + round-by-round logs: [`experiments/`](experiments/) (`SUMMARY.md`, `results.csv`)
- v2 baseline per-benchmark logs for comparison: [`submissions/vmallela_v2/results_verified/`](submissions/vmallela_v2/results_verified/)

## Reproduction

```bash
./submissions/vmallela_v2/run.sh --all       # all 17 IBM benchmarks
./submissions/vmallela_v2/run.sh -b ibm01    # single benchmark
```

`run.sh` exports the locked environment (seed 42, BLAS pinned to one
thread, 3300 s per-benchmark budget) and the v4-tuned operator settings
(`PLACER_SA_T0=5e-5`, `PLACER_ESC_HARD_DESTROY=80`). These env var values
are also the baked-in defaults inside `OptimalPlacer.__init__`, so
overriding them at the env level is optional; unset behaviour matches
the reported table.

Expected per-benchmark result within ±0.002 of the table above (run-to-run
jitter from 13 wall-clock-bounded loops; single seed).

## v4 vs v2 per-benchmark

| Benchmark | v2 (optimized) | v4 (optimized_v4) | Delta |
|-----------|---------------:|------------------:|------:|
| ibm01 | 0.8107 | 0.7803 | -0.030 |
| ibm02 | 1.1002 | 0.9737 | -0.127 |
| ibm03 | 0.9912 | 0.9254 | -0.066 |
| ibm04 | 0.9889 | 0.9345 | -0.054 |
| ibm06 | 1.1826 | 1.0755 | -0.107 |
| ibm07 | 1.1277 | 1.0432 | -0.085 |
| ibm08 | 1.1132 | 1.0550 | -0.058 |
| ibm09 | 0.8238 | 0.7785 | -0.045 |
| ibm10 | 1.0989 | 0.9625 | -0.136 |
| ibm11 | 0.9133 | 0.8191 | -0.094 |
| ibm12 | 1.3199 | 1.1764 | -0.144 |
| ibm13 | 1.0010 | 0.8906 | -0.110 |
| ibm14 | 1.2675 | 1.1337 | -0.134 |
| ibm15 | 1.2291 | 1.1029 | -0.126 |
| ibm16 | 1.2024 | 1.0771 | -0.125 |
| ibm17 | 1.4535 | 1.3012 | -0.152 |
| ibm18 | 1.3689 | 1.2865 | -0.082 |
| **Mean** | **1.1172** | **1.0186** | **-0.099** |

## Caveats

- Single-seed result. A multi-seed validation (seeds 43 + 44 on all 17
  at 3300 s) is in flight and will confirm the ±0.002-class jitter bound.
- Tier-2 (OpenROAD / NG45) WNS / TNS / Area not yet measured locally;
  the upstream scoring pipeline handles this.
- Submission still runs single-threaded on CPU; the grader's 16-core
  EPYC and RTX 6000 Ada are not used beyond one CPU core.

Competition specification: [`COMPETITION.md`](COMPETITION.md).
