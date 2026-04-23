# vmallela submissions

Submissions to the Partcl/HRT Macro Placement Challenge.

## Current best: `vmallela_v2`

Score: **1.1533** average across 17 IBM benchmarks — all VALID (zero overlaps),
every run ≤ 1 hour (competition cap). That is **+5.7% under Cezar (ReFine,
leaderboard #1 at 1.2224)**.

Code: [`submissions/vmallela_v2/placer.py`](submissions/vmallela_v2/placer.py)
Submission notes: [`submissions/vmallela_v2/README.md`](submissions/vmallela_v2/README.md)
Experiment log: [`submissions/vmallela_v2/EXPERIMENTS.md`](submissions/vmallela_v2/EXPERIMENTS.md)

### Method

v1's incremental-CD pipeline rebuilt around three unlocks:
1. **Return soft-macro positions** from `_set_placement` (v1 was silently
   discarding optimized soft-macro positions — ~14% of the total gain).
2. **Adaptive cycle-budget scheduler** — shrink on plateau, grow on gain.
3. **Per-net HPWL optimization** — step pins toward weighted median per net.

Combined with a stateful MLP surrogate for CD probe ranking and per-benchmark
big-budget refinement (1200-6500s on the hardest instances).

### Reproduce

```bash
uv run evaluate submissions/vmallela_v2/placer.py --benchmark ibm01
uv run evaluate submissions/vmallela_v2/placer.py --all
```

`vmallela_v2/placer.py` depends on `vmallela/placer.py` for the
`IncrementalEvaluator`, push-apart, and legalizer — keep both directories.

## Previous: `vmallela` (v1)

Score: **1.4159** — multi-restart CD optimizing the exact proxy cost.
Code: [`submissions/vmallela/placer.py`](submissions/vmallela/placer.py)
Submission notes: [`submissions/vmallela/README.md`](submissions/vmallela/README.md)

Original competition README: [`COMPETITION.md`](COMPETITION.md)
