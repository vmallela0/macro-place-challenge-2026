# Macro placement submissions — vmallela

Two submissions to the Partcl / HRT Macro Placement Challenge, both
search-based placers that optimize the exact ICCAD-style proxy cost

```
proxy = 1.0 · HPWL_norm + 0.5 · density_norm + 0.5 · congestion_norm
```

directly, without a smooth surrogate. The enabling component is an
incremental evaluator that updates the proxy cost on a per-move basis
in O(affected pins + affected grid cells), which is fast enough to
make coordinate descent on the non-differentiable objective practical
inside a 1-hour wall-clock budget per benchmark.

Analytical placers (DREAMPlace, RePlAce and variants) optimize a
smoothed HPWL + density objective and leave congestion for the
downstream router. The approach here trades that global convergence
guarantee for the ability to optimize the exact scoring function.

## Current submission — `vmallela_v2`

Coordinate descent with adaptive cycle scheduling and per-net
weighted-median HPWL stepping.

**Verified single-run result:** mean proxy cost 1.1172 across the 17
IBM ICCAD 2004 benchmarks, all placements valid with zero overlaps,
every benchmark completing under the 1-hour per-benchmark cap. Measured
on a 10-core Apple Silicon MacBook Pro under a locked environment
(BLAS / OMP threads pinned to one, seed 42, serial per-benchmark
execution).

- Code: [`submissions/vmallela_v2/placer.py`](submissions/vmallela_v2/placer.py)
- Write-up: [`submissions/vmallela_v2/README.md`](submissions/vmallela_v2/README.md)
- Per-benchmark table: [`submissions/vmallela_v2/results_verified/SUMMARY.md`](submissions/vmallela_v2/results_verified/SUMMARY.md)
- Raw logs: [`submissions/vmallela_v2/results_verified/`](submissions/vmallela_v2/results_verified/)
- Evaluator equivalence check: [`submissions/vmallela_v2/tests/EQUIVALENCE.md`](submissions/vmallela_v2/tests/EQUIVALENCE.md)
- Development log (~120 variants): [`submissions/vmallela_v2/EXPERIMENTS.md`](submissions/vmallela_v2/EXPERIMENTS.md)

### Differences from v1

v2 reuses v1's `IncrementalEvaluator`, push-apart pre-processing, and
legalization tournament. The changes are confined to the refinement
loop:

1. **Soft-macro positions are propagated through the return path.**
   v1's `_set_placement` wrote back only hard-macro coordinates,
   discarding the soft-macro (std-cell-cluster) positions computed
   during evaluation. v2 writes back both.
2. **Adaptive cycle scheduling.** Each refinement cycle's wall-clock
   duration is multiplicatively adjusted by the observed cost gain
   (shrink on plateau, grow on improvement), with early termination
   after repeated plateau. Replaces v1's fixed per-phase budget.
3. **Per-net weighted-median HPWL stepping** interleaved with
   coordinate descent. For each net, movable pins are stepped toward
   the 1-D weighted median of the other pins on that net — the
   classical Fermat-Weber optimum for HPWL — with acceptance gated
   on the full proxy cost. Complementary to CD: CD searches an
   integer-lattice neighborhood along 8 directions, whereas the
   weighted-median target is continuous and generally off-axis.

Full pipeline description and per-phase budgets in the v2 README.

### Reproduction

```bash
./submissions/vmallela_v2/run.sh --all       # all 17 IBM benchmarks
./submissions/vmallela_v2/run.sh -b ibm01    # single benchmark
```

`run.sh` fixes `OMP_NUM_THREADS=MKL_NUM_THREADS=...=1`,
`PYTHONHASHSEED=42`, and `PLACER_TOTAL_BUDGET=3300`. The placer
additionally hard-caps its own budget at 3300 s regardless of env
input, leaving 300 s under the 3600 s per-benchmark competition
timeout for the validator and cost evaluator that run after `place()`
returns. Expected per-benchmark results within ±0.002 of the verified
table (run-to-run jitter is discussed in the v2 README caveats).

## Previous submission — `vmallela`

Multi-restart coordinate descent with up to 15 parallel workers. The
initial submission; introduces the `IncrementalEvaluator` that v2
depends on. Reported mean proxy cost: 1.4156.

- Code: [`submissions/vmallela/placer.py`](submissions/vmallela/placer.py)
- Write-up: [`submissions/vmallela/README.md`](submissions/vmallela/README.md)
- Development log: [`submissions/vmallela/EXPERIMENTS.md`](submissions/vmallela/EXPERIMENTS.md)

## Repository layout

```
macro-place-challenge-2026/
├── README.md                        ← this file
├── COMPETITION.md                   ← competition specification
├── LICENSE.md                       ← Apache 2.0
├── macro_place/                     ← evaluation harness + proxy-cost implementation
├── scripts/                         ← utilities (NG45 / OpenROAD flow, benchmark converters, visualization)
├── external/MacroPlacement/         ← TILOS benchmark suite (git submodule)
└── submissions/
    ├── vmallela/                    ← v1: multi-restart parallel CD
    └── vmallela_v2/                 ← current: adaptive CD + per-net HPWL
```

## References

- Sechen and Sangiovanni-Vincentelli, *TimberWolf 3.2*, 1985 — the
  search-based placement lineage.
- Kahng, Lienig, Markov, Hu, *VLSI Physical Design*, Chapter 4 —
  weighted-median HPWL minimization per net.
- Lin et al., *DREAMPlace*, DAC 2019, and Cheng et al., *RePlAce*,
  TCAD 2019 — the analytical-placement family this work contrasts
  with on the differentiability question.
- Mirhoseini et al., *A graph placement methodology for fast chip
  design* (AlphaChip), Nature 2021 — the RL-based lineage; the
  present work is a non-RL search-based baseline.

Competition specification and evaluation rules: [`COMPETITION.md`](COMPETITION.md).
