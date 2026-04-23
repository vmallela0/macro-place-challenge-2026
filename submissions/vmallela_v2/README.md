# Submission: vmallela_v2 — Soft-macro CD + adaptive cycles + per-net HPWL

**Author:** vmallela
**Reported proxy cost:** 1.1533 (average across 17 IBM ICCAD04 benchmarks)
**All benchmarks:** VALID, zero overlaps, every run ≤ 1 hour (competition cap)
**Previous submission:** `submissions/vmallela/` (1.4156 avg)

## TL;DR

Incremental-CD pipeline from v1, rebuilt around three unlocks:

1. **Return soft-macro positions.** v1's `_set_placement` was only
   committing hard-macro positions and silently discarding the optimized
   soft-macro (std-cell cluster) positions. Soft macros dominate HPWL and
   congestion — propagating their positions alone is worth ~14% off the
   average proxy cost.
2. **Adaptive cycle-budget scheduler.** Each refinement cycle's duration
   shrinks (×0.7) on plateau (gain < 5e-5) and grows (×1.1) on rapid
   improvement (gain > 0.01). The placer stops early if 4 consecutive
   cycles plateau.
3. **Per-net HPWL optimization.** On each net, visit movable pins in
   weight-descending order and step each pin toward the weighted median
   of the other pins on that net. Interleaves with coordinate descent to
   escape CD local minima that HPWL-shrink could fix.

These compose with v1's CD infrastructure (`IncrementalEvaluator`,
push-apart, legalize tournament) and MLP-surrogate-ranked soft CD probes
for a pipeline that averages **1.1533** across the 17 IBM benchmarks —
an 18.5% improvement over v1 and 5.7% under Cezar (ReFine) at 1.2224.

## Competition compliance

Per `COMPETITION.md`: each benchmark must complete within **1 hour**
end-to-end, placements must be valid (zero overlaps), and only
open-source placement tooling is permitted.

- **Runtime:** all 17 benchmarks finished in under 1 hour (max 3010s on
  ibm17). The placer internally caps its own budget at **3300 seconds**
  regardless of `PLACER_TOTAL_BUDGET` (see `OptimalPlacer._COMPETITION_CAP_SECONDS`)
  to leave headroom for the validator + cost evaluator that run after
  `place()` returns.
- **Validity:** every run returned zero overlaps (`VALID`).
- **Tooling:** uses only the competition's `macro_place` package and the
  code in this repo. No proprietary placers.

## Pipeline

```
Phase 1: Push-apart preprocessing
  └─ 3 damping configurations (conservative / moderate / aggressive)

Phase 2: Legalization tournament
  ├─ 30 orderings × 4 step-sizes × 5 starting positions
  └─ Best legalized result by real proxy cost wins
  (Budget: max(60, min(600, TOTAL_BUDGET // 5)))

Phase 3: Hard-macro coordinate descent + LNS + swap polish
  └─ As in v1 but with size-scaled 8-direction probe

Phase 4: Soft-macro refinement cycles (adaptive duration)
  ├─ 5% FD soft attraction (net-centroid targets)
  ├─ 35% Stateful MLP surrogate soft CD (ProbeLogger + 2-layer MLP on MPS)
  ├─ 15% Regular soft CD
  ├─ 30% Soft LNS (destroy connected subset, reinsert greedily)
  └─ 15% Hard CD polish

Phase 5: Per-net HPWL optimization
  └─ Weighted-median pin stepping per net
```

## Per-benchmark results (all ≤ 1 hour)

| Bench  | proxy  | vs SA   | vs RePlAce | Time (s) |
|--------|--------|---------|------------|----------|
| ibm01  | 0.8147 | +38.1 % | +18.3 %    |   645    |
| ibm02  | 1.1444 | +40.0 % | +37.7 %    |  1881    |
| ibm03  | 1.0374 | +40.4 % | +21.5 %    |  1204    |
| ibm04  | 1.0207 | +32.1 % | +21.6 %    |   644    |
| ibm06  | 1.2435 | +50.4 % | +23.2 %    |  1503    |
| ibm07  | 1.1497 | +43.2 % | +21.4 %    |  1500    |
| ibm08  | 1.1345 | +41.0 % | +20.6 %    |  2493    |
| ibm09  | 0.8558 | +38.3 % | +23.5 %    |   647    |
| ibm10  | 1.1344 | +46.3 % | +24.4 %    |  1796    |
| ibm11  | 0.9569 | +44.1 % | +18.7 %    |  1505    |
| ibm12  | 1.3406 | +52.6 % | +22.3 %    |  3000    |
| ibm13  | 1.0620 | +44.5 % | +20.5 %    |  1494    |
| ibm14  | 1.2788 | +43.8 % | +17.2 %    |  2002    |
| ibm15  | 1.2559 | +45.4 % | +17.2 %    |  2492    |
| ibm16  | 1.2517 | +44.0 % | +15.3 %    |  1802    |
| ibm17  | 1.4895 | +59.4 % |  +9.4 %    |  3010    |
| ibm18  | 1.4350 | +48.3 % | +19.0 %    |  1995    |
| **AVG**| **1.1533** | **+45.7 %** | **+20.9 %** |      |

Compared to public leaderboard:

| Rank | Method          | Avg proxy | vs ours (1.1533) |
|------|-----------------|-----------|------------------|
| —    | **vmallela_v2** | **1.1533**| —                |
| 1    | Cezar (ReFine)  | 1.2224    | +5.7 %           |
| 2    | MTK DreamPlace++| 1.2818    | +10.0 %          |
| 3    | RoRa            | 1.3241    | +12.9 %          |
| 4    | vmallela v1     | 1.4156    | +18.5 %          |

## File map

```
submissions/vmallela_v2/
├── README.md                    ← You are here
├── EXPERIMENTS.md               ← Log of v1→v118 variants and what worked
├── placer.py                    ← Entry point (OptimalPlacer, cap 3300s)
├── _softmacro.py                ← Soft-macro coordinate descent
├── _fd_soft.py                  ← Force-directed soft placement
├── _soft_lns.py                 ← Soft-macro LNS (destroy+repair)
├── _per_net.py                  ← Per-net HPWL weighted-median pin stepping
├── _soft_surrogate_v2.py        ← Stateful MLP surrogate wrapper
├── _surrogate.py                ← Probe logger + 2-layer MLP (ProbeSurrogate)
└── _moves.py                    ← LNS destroy-repair hard-macro phase
```

`placer.py` imports `_load_plc`, `IncrementalEvaluator`, `_push_apart`,
`_legalize`, `_refine_toward_initial`, `_coord_descent`, `_cd_worker`
from `submissions/vmallela/placer.py`. Keep both submission directories
present in the repo — `vmallela_v2` depends on `vmallela`.

## Reproducing

```bash
# single benchmark (default budget 3300 s, capped at 3300 s always)
uv run evaluate submissions/vmallela_v2/placer.py --benchmark ibm01

# all 17 benchmarks (default 3300 s each — still within the 1-hour cap)
uv run evaluate submissions/vmallela_v2/placer.py --all

# shorter budget override (cap ignored; placer will stop at the env value)
PLACER_TOTAL_BUDGET=1800 uv run evaluate submissions/vmallela_v2/placer.py -b ibm07

# attempting >3300s is silently clamped to 3300s (competition compliance)
PLACER_TOTAL_BUDGET=10000 uv run evaluate submissions/vmallela_v2/placer.py -b ibm17
```

Env vars:

- `PLACER_TOTAL_BUDGET` (default 3300, **capped at 3300**) — wall-clock budget
- `PLACER_SOFT_BUDGET` (default ≈ 60% of TOTAL) — portion spent in Phase 4
- `PLACER_PARALLEL_WORKERS` (default 0 = serial) — per-benchmark worker count
