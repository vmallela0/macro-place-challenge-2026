# Round 2 grid plan

After round 1 (10 single-axis experiments) finishes, design round 2 around
winners. Below is the candidate grid; will trim based on round 1 signal.

## Axes (all env-gated on exp_extras-style "master grid" branch)

| Knob | Values | Notes |
|---|---|---|
| `PLACER_SA_T0` | unset, 0.0001, 0.0003, 0.001, 0.003 | unset = greedy |
| `PLACER_SA_COOLING` | 0.999, 0.9995, 0.9999 | only when T0 set |
| `PLACER_EXP7_LINESEARCH` | 0, 1 | per-net pin best-of-5 |
| `PLACER_ESC_HARD_DESTROY` | 10, 20, 40, 60 | escape hard LNS destroy |
| `PLACER_ESC_HARD_CAND` | 60, 120, 200 | escape hard LNS candidates |
| `PLACER_ESC_SOFT_DESTROY` | 15, 30, 60 | escape soft LNS destroy |
| `PLACER_PLATEAU_N` | 2, 4, 8 | plateau-cycle threshold |

## Round 2 sample grid (16-20 configs)

Picks: keep winners from round 1, vary everything else around them.

After round 1 results land, populate this section.

## Round 3: stack winners

Combine winning settings from each axis. ~10 final candidates, each at 3 seeds
(42/43/44), 550s. Best one becomes new baseline.

## Round 4: full-budget validation

Top 1-2 candidates from round 3 → full 3300s × 3 seeds × ibm01/ibm06/ibm10/ibm17.
This is the final commit-or-kill decision.
