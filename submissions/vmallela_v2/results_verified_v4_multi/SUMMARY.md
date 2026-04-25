# v4 multi-seed verification — 17 benchmarks × 3 seeds × 3300 s

51 runs total, all VALID (zero overlaps), each at the per-benchmark
3300 s budget. Same code as `results_verified_v4/` (seed-42 single run);
only the seed differs.

## Per-bench results

| Bench | s42    | s43    | s44    | min-of-3 | Baseline | Δ vs base |
|-------|-------:|-------:|-------:|---------:|---------:|----------:|
| ibm01 | 0.7803 | 0.7754 | 0.7775 | 0.7754   | 0.8107   | -0.0353   |
| ibm02 | 0.9737 | 0.9794 | 0.9594 | 0.9594   | 1.1002   | -0.1408   |
| ibm03 | 0.9254 | 0.9115 | 0.9299 | 0.9115   | 0.9912   | -0.0797   |
| ibm04 | 0.9345 | 0.9345 | 0.9272 | 0.9272   | 0.9889   | -0.0617   |
| ibm06 | 1.0755 | 1.0768 | 1.0789 | 1.0755   | 1.1826   | -0.1071   |
| ibm07 | 1.0432 | 1.0505 | 1.0431 | 1.0431   | 1.1277   | -0.0846   |
| ibm08 | 1.0550 | 1.0497 | 1.0498 | 1.0497   | 1.1132   | -0.0635   |
| ibm09 | 0.7785 | 0.7707 | 0.7882 | 0.7707   | 0.8238   | -0.0531   |
| ibm10 | 0.9625 | 0.9726 | 0.9718 | 0.9625   | 1.0989   | -0.1364   |
| ibm11 | 0.8191 | 0.8196 | 0.8185 | 0.8185   | 0.9133   | -0.0948   |
| ibm12 | 1.1764 | 1.1724 | 1.1749 | 1.1724   | 1.3199   | -0.1475   |
| ibm13 | 0.8906 | 0.8934 | 0.8931 | 0.8906   | 1.0010   | -0.1104   |
| ibm14 | 1.1337 | 1.1264 | 1.1336 | 1.1264   | 1.2675   | -0.1411   |
| ibm15 | 1.1029 | 1.1047 | 1.1049 | 1.1029   | 1.2291   | -0.1262   |
| ibm16 | 1.0771 | 1.0758 | 1.0823 | 1.0758   | 1.2024   | -0.1266   |
| ibm17 | 1.3012 | 1.2923 | 1.3052 | 1.2923   | 1.4535   | -0.1612   |
| ibm18 | 1.2865 | 1.2841 | 1.2956 | 1.2841   | 1.3689   | -0.0848   |

## Means

| Configuration                        | Mean   | Δ vs baseline |
|--------------------------------------|-------:|--------------:|
| Baseline (`optimized` branch, committed)  | 1.1172 | —             |
| **Seed 42 alone** (the `results_verified_v4/` submission) | **1.0186** | -0.0986 (-8.83%) |
| Seed 43 alone                        | 1.0170 | -0.1002 (-8.97%) |
| Seed 44 alone                        | 1.0196 | -0.0976 (-8.74%) |
| **min-of-3 best-of**                 | **1.0140** | -0.1032 (-9.24%) |

## Notes

- Seed 43 vs seed 42: 9 wins (s43), 7 wins (s42), 1 tie. Per-bench
  variance σ ≈ 0.005-0.01 for most benches.
- Best-of-3 over single-seed: -0.0046 mean (~0.45%), confirming the
  algorithm is robust; per-seed jitter is the main source of variance.
- The reported submission number (in `run.sh` defaults +
  `results_verified_v4/`) is the single seed-42 run at 1.0186. The
  min-of-3 is *informational* — it shows the practical floor of the
  algorithm under different RNG seeds.
- Seed 43 alone (1.0170) is 0.0016 better than seed 42; submitting with
  `PLACER_SEED=43` is also defensible.

All 51 logs are in `seed_43/*.log` and `seed_44/*.log` (seed 42 lives
in the sibling `results_verified_v4/`).
