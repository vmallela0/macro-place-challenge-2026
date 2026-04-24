# vmallela submissions

Submissions to the Partcl/HRT Macro Placement Challenge.

## Current: `vmallela_v2`

A coordinate-descent macro placer with an incremental proxy-cost evaluator.
Verified single-run average of **1.1172** across the 17 IBM ICCAD 2004
benchmarks, all placements valid with zero overlaps, every run under the
1-hour per-benchmark cap.

- Code: [`submissions/vmallela_v2/placer.py`](submissions/vmallela_v2/placer.py)
- Full write-up: [`submissions/vmallela_v2/README.md`](submissions/vmallela_v2/README.md)
- Per-benchmark table: [`submissions/vmallela_v2/results_verified/SUMMARY.md`](submissions/vmallela_v2/results_verified/SUMMARY.md)
- Evaluator equivalence check: [`submissions/vmallela_v2/tests/EQUIVALENCE.md`](submissions/vmallela_v2/tests/EQUIVALENCE.md)
- Development log: [`submissions/vmallela_v2/EXPERIMENTS.md`](submissions/vmallela_v2/EXPERIMENTS.md)

### Approach

The placer optimizes the exact ICCAD-style proxy cost
(`1.0 · HPWL + 0.5 · density + 0.5 · congestion`) directly via local
search, using an incremental evaluator to make per-move updates cheap
enough for coordinate descent to be practical inside a 1-hour budget.
v2 extends v1 by (i) propagating soft-macro (std-cell-cluster)
positions through the return path, (ii) replacing fixed per-phase
budgets with an adaptive cycle scheduler, and (iii) interleaving
classical per-net weighted-median HPWL stepping with the coordinate
descent. Full description and pipeline diagram in the submission
README.

### Reproduction

```bash
./submissions/vmallela_v2/run.sh --all    # all 17 IBM benchmarks (~15 h on 10 cores)
./submissions/vmallela_v2/run.sh -b ibm01 # single benchmark
```

`vmallela_v2/placer.py` imports a handful of functions from
`submissions/vmallela/placer.py` (the `IncrementalEvaluator`,
push-apart, and legalizer). Both directories must be present.

## Previous: `vmallela` (v1)

Multi-restart coordinate descent with parallel workers, single-file
implementation. Reported proxy-cost average: 1.4156.

- Code: [`submissions/vmallela/placer.py`](submissions/vmallela/placer.py)
- Write-up: [`submissions/vmallela/README.md`](submissions/vmallela/README.md)

Competition specification: [`COMPETITION.md`](COMPETITION.md).
