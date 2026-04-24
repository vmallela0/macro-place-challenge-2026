# vmallela submissions

Submissions to the Partcl/HRT Macro Placement Challenge.

## Current best: `vmallela_v2`

**Verified single-run average: 1.1172 across 17 IBM ICCAD04 benchmarks.**
All 17 VALID, zero overlaps, every run under the 1-hour cap. That is
**−8.6 % under Cezar (ReFine, leaderboard #1 at 1.2224)**.

- Code: [`submissions/vmallela_v2/placer.py`](submissions/vmallela_v2/placer.py)
- Submission notes: [`submissions/vmallela_v2/README.md`](submissions/vmallela_v2/README.md)
- Verified per-bench table: [`submissions/vmallela_v2/results_verified/SUMMARY.md`](submissions/vmallela_v2/results_verified/SUMMARY.md)
- Evaluator equivalence proof: [`submissions/vmallela_v2/tests/EQUIVALENCE.md`](submissions/vmallela_v2/tests/EQUIVALENCE.md)
- Experiment log: [`submissions/vmallela_v2/EXPERIMENTS.md`](submissions/vmallela_v2/EXPERIMENTS.md)

Measurement: `./submissions/vmallela_v2/run.sh --all` at seed=42,
`PLACER_TOTAL_BUDGET=3300`, `OMP_NUM_THREADS=1`, on a 10-core Apple Silicon
MacBook Pro. Run-to-run jitter ~0.002 (time-budgeted loops are
semantically — not bit- — reproducible).

### Method

v1's incremental-CD pipeline rebuilt around three unlocks:

1. **Return soft-macro positions** from `_set_placement` (v1 was silently discarding optimized soft-macro positions — ~14 % of the total gain).
2. **Adaptive cycle-budget scheduler** — shrink on plateau, grow on gain, stop after 4 consecutive plateau cycles.
3. **Per-net HPWL optimization** interleaved with CD — step movable pins toward the weighted median of other pins on the same net.

Plus a stateful MLP surrogate ranking CD probe candidates. No per-benchmark hardcoding — identical code path on every benchmark.

### Reproduce

```bash
./submissions/vmallela_v2/run.sh --all              # all 17 IBM benchmarks
./submissions/vmallela_v2/run.sh -b ibm01           # single benchmark
```

Expected per-bench results within ±0.002 of the table in `results_verified/SUMMARY.md`.
`vmallela_v2/placer.py` depends on `vmallela/placer.py` (for `IncrementalEvaluator`, push-apart, and legalizer) — keep both directories present in the repo.

## Previous: `vmallela` (v1)

Score: **1.4159** — multi-restart CD optimizing the exact proxy cost.
Code: [`submissions/vmallela/placer.py`](submissions/vmallela/placer.py)
Submission notes: [`submissions/vmallela/README.md`](submissions/vmallela/README.md)

Original competition README: [`COMPETITION.md`](COMPETITION.md)
