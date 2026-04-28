# Macro placement submission — vmallela_v2

A coordinate-descent macro placer that optimizes the exact ICCAD-style
proxy cost (`1.0 · HPWL + 0.5 · density + 0.5 · congestion`) via local
search, using an incremental evaluator for fast per-move updates.

Verified single-run mean proxy cost of **1.1172** across the 17 IBM
ICCAD 2004 benchmarks; all placements valid, zero overlaps, every run
under the 1-hour per-benchmark cap.

## Per-benchmark proxy cost

```
  ibm01  0.8107  █
  ibm02  1.1002  ███████████████████
  ibm03  0.9912  ████████████
  ibm04  0.9889  ████████████
  ibm06  1.1826  ████████████████████████
  ibm07  1.1277  ████████████████████
  ibm08  1.1132  ███████████████████
  ibm09  0.8238  ██
  ibm10  1.0989  ██████████████████
  ibm11  0.9133  ███████
  ibm12  1.3199  ████████████████████████████████
  ibm13  1.0010  █████████████
  ibm14  1.2675  █████████████████████████████
  ibm15  1.2291  ██████████████████████████
  ibm16  1.2024  █████████████████████████
  ibm17  1.4535  ████████████████████████████████████████
  ibm18  1.3689  ███████████████████████████████████

  range: 0.8107 – 1.4535       mean: 1.1172
```

Bar length is linear in proxy cost, normalized against the bench with
the minimum (ibm01) and the bench with the maximum (ibm17). Lower is
better. Density- and congestion-dominated benchmarks (ibm12, ibm14,
ibm17, ibm18) cluster at the top; lighter benchmarks (ibm01, ibm09,
ibm11) at the bottom.

- Code and write-up: [`submissions/vmallela_v2/`](submissions/vmallela_v2/)
- Per-benchmark table with wall times: [`submissions/vmallela_v2/results_verified/SUMMARY.md`](submissions/vmallela_v2/results_verified/SUMMARY.md)

## Reproduction

```bash
./submissions/vmallela_v2/run.sh --all       # all 17 IBM benchmarks
./submissions/vmallela_v2/run.sh -b ibm01    # single benchmark
```

Competition specification: [`COMPETITION.md`](COMPETITION.md).
