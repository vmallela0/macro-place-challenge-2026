# Macro placement submission — vmallela_v2

A coordinate-descent macro placer that optimizes the exact ICCAD-style
proxy cost (`1.0 · HPWL + 0.5 · density + 0.5 · congestion`) via local
search, using an incremental evaluator for fast per-move updates.

Verified single-run mean proxy cost of **1.1172** across the 17 IBM
ICCAD 2004 benchmarks; all placements valid, zero overlaps, every run
under the 1-hour per-benchmark cap.

- Code and write-up: [`submissions/vmallela_v2/`](submissions/vmallela_v2/)
- Per-benchmark table: [`submissions/vmallela_v2/results_verified/SUMMARY.md`](submissions/vmallela_v2/results_verified/SUMMARY.md)

## Reproduction

```bash
./submissions/vmallela_v2/run.sh --all       # all 17 IBM benchmarks
./submissions/vmallela_v2/run.sh -b ibm01    # single benchmark
```

Competition specification: [`COMPETITION.md`](COMPETITION.md).
