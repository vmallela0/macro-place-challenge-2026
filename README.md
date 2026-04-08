# vmallela submission

Submission to the Partcl/HRT Macro Placement Challenge.

Score: **1.4159** average across 17 IBM benchmarks (all valid, zero overlaps).

Code: [`submissions/vmallela/placer.py`](submissions/vmallela/placer.py)
Submission notes: [`submissions/vmallela/README.md`](submissions/vmallela/README.md)
Experiment log: [`submissions/vmallela/EXPERIMENTS.md`](submissions/vmallela/EXPERIMENTS.md)

Original competition README: [`COMPETITION.md`](COMPETITION.md)

## Method

Multi-restart coordinate descent on the real proxy cost (including congestion)
using a custom incremental evaluator with ~300x speedup over PlacementCost.
Pipeline: push-apart → multi-start legalization → coordinate descent + swap →
finite-difference gradient → 15 parallel CD restart workers.

## Reproduce

```bash
uv run evaluate submissions/vmallela/placer.py --benchmark ibm01
uv run evaluate submissions/vmallela/placer.py --all
```
