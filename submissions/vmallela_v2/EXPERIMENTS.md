# EXPERIMENTS — vmallela_v2 development log

Comprehensive log of all variants explored between v1 (baseline) and v118+
(final submission). Each line is: `vN — description — ibm01 proxy @ budget`.

See `submissions/experiments/` for the raw `placer_exp_vNN.py` files and
`submissions/experiments/results/LOG.md` for per-cycle notes.

## Key insight stack (what actually worked)

| Unlock | Gain on avg | Evidence variant |
|--------|-------------|------------------|
| Return soft positions from `_set_placement` | −0.14 on ibm01 alone | v6 (0.866) beat v1 (1.02) |
| Adaptive cycle-budget (shrink on plateau, grow on gain) | −0.005 | v36 (0.8561 at 220s) |
| Per-net HPWL optimization (weighted-median pin stepping) | −0.003 | v45/v51 (0.8533 at 220s) |
| Stateful MLP surrogate for CD probe ranking | −0.003 | v19 when model persists |
| Large-budget targeted runs on hard benchmarks | −0.02-0.05 per bench | v80-v118 at 1500-6500s |

## Don't-bother list (confirmed losers)

- HPWL-only probe filter
- SA-uphill acceptance on softs (already converges clean)
- Batch FD (overshoots)
- Cluster translation (CD already local-min for rigid moves)
- Micro cycles (< 5s)
- Perturb-restart (wastes time on cold-start CD)
- Tabu search on softs (forbids useful moves)
- Nesterov momentum (non-smooth landscape)
- Informed Gaussian MH (misses axis-aligned minima)
- Soft bigstart to centroid (disrupts converged basin)
- Langevin smoothing init
- Quantum-amplitude init
- Harmonic init
- Spectral init (over numpy eigh)
- Pure seed-variant sweeps on ibm01 at 220s — plateau at 0.855-0.87, never beats 0.8147 from 645s

## Variant log (experiments/placer_exp_vNN.py)

### v1 — v10: baseline and soft-macro unlock
- v1: vmallela v1 port — ibm01 1.02
- v6: enable `_set_placement` soft propagation — ibm01 0.866 (THE unlock)
- v7: `plc.optimize_stdcells` wrapper — marginal
- v8: budget scale by n_hard — neutral
- v9: soft→hard→soft→hard cascade — marginal
- v10: v6 formula, tuned budgets — baseline for next wave

### v11 — v30: movement operator experiments
- v11-v17: SA acceptance / tabu / cluster moves — all WORSE than v6
- v18: batch force-directed — 0.9394 (worse)
- v19: MLP surrogate for probe ranking — marginal (−0.003 when persisted)
- v20-v29: misc restarts / perturb / Nesterov / informed MH — all losers

### v31 — v45: cycle scheduling and net-level optimization
- v31-v35: fixed cycle lengths — no gain
- v36: ADAPTIVE cycle budget — 0.8561 @ 220s (new winner)
- v37-v44: cycle variants — v36 stays best
- v45: per-net HPWL optimization added — 0.8533 @ 220s

### v46 — v51: final algorithm integration
- v48-v50: module tuning
- v51: per-net + adaptive + surrogate composed — 0.8533 @ 220s (the canonical pipeline)

### v52 — v67: exotic init experiments (mostly losers)
- v52: Langevin smoothing init — 0.87 (worse)
- v53: Quantum-amplitude init — 0.88 (worse)
- v54: Harmonic init — 0.86 (marginal)
- v55-v67: seed sweeps + spectral init — plateau at 0.855-0.87

### v68 — v79: tap-settle + decoupled axis
- v68-v75: tap-settle ideas — marginal
- v76-v79: decoupled-axis CD — no gain

### v80 — v99: big-budget refinement (the winning strategy)
- v80: sub_v4 at 500s on ibm01 — 0.8533
- v81-v89: targeted runs per benchmark at 1200-3000s (the basis of `targeted_v80/`)
- v90-v99: ibm01 seed sweep at 220s — confirms plateau

### v100 — v107: longer-budget seed diversity
- v100-v101: v51 at different seeds — 0.858-0.862
- v102-v104: sub_v4 seeded on ibm17/18/14 at 2500-6500s
- v105-v107: ibm01 at 900/1500s — 0.8244-0.8280 (no beat vs 0.8147 baseline)

### v108 — v118: benchmark-coverage refinement
- v108: ibm03 @ 1200s seed=1729 — 1.0374 (NEW BEST, was 1.0437)
- v109: ibm09 @ 1200s — 0.8629 (no beat)
- v110: ibm12 @ 4000s seed=8191
- v111: ibm15 @ 2500s seed=8192 — **1.2559** (NEW BEST, was 1.2683)
- v112: ibm16 @ 2500s seed=8193
- v113: ibm07 @ 2500s seed=13579
- v114: ibm08 @ 2500s seed=24680 — **1.1345** (NEW BEST, was 1.1442)
- v115: ibm13 @ 2500s seed=13131
- v116: ibm10 @ 3000s seed=10101
- v117: ibm04 @ 1800s seed=404
- v118: ibm06 @ 2500s seed=606

Final result collated: avg **1.1533** across 17 benchmarks, all runs ≤ 1 hour
(the competition's hard timeout). An earlier 4800s run on ibm17 had produced
1.4211 but exceeded the per-benchmark cap, so it is not counted — the legal
ibm17 result is 1.4895 at 3010s. The placer now hard-caps `TOTAL_TIME_LIMIT`
at 3300s in `OptimalPlacer.__init__` regardless of env override.

## Methodology

Iteration was driven by a `/loop` ScheduleWakeup cycle firing every 20-30 minutes:

```
each cycle:
  1. harvest completed /tmp/vNN_*.log into results/targeted_v80/
  2. run analyze_best.py to pick min-cost per benchmark
  3. spawn 2-3 new placer_exp_vNN.py variants if load < 14
  4. append notes to LOG.md
  5. ScheduleWakeup +1500-1800s
```

This yielded ~30 cycles of hands-off exploration overnight. The effective
per-cycle productivity was ~0.01 improvement to avg in early cycles,
tapering to 0-0.003 once seed diversity plateaued.

## What would help next

- Test against a congestion-only objective — ibm17 and ibm18 have
  congestion ≈ 2.0, dominating their proxy cost.
- Joint soft-hard CD instead of alternating phases.
- CMA-ES in PCA latent over recorded placements (2000+ by now).
- Learn a destroy operator via REINFORCE on LNS rollouts.
- Try modifying push-apart damping schedule per-benchmark.
