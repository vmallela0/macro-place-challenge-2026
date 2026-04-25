# vmallela submission — Partcl/HRT Macro Placement Challenge

**Final submission: mean proxy cost 1.0186 across the 17 IBM ICCAD-2004
benchmarks** (all VALID, zero overlaps, 3300 s per-benchmark budget).
See branch [`optimized_v4`](https://github.com/vmallela0/macro-place-challenge-2026/tree/optimized_v4)
for the code, the verified per-bench logs, and the full write-up.

This `main` branch is intentionally minimal — it just routes you to the
right place. Each scoring run / iteration of the work lives on its own
branch.

## Branches

| Branch | Mean proxy | Description |
|--------|-----------:|-------------|
| `main` (this one) | — | Landing page only |
| [`optimized`](../../tree/optimized) | **1.1172** | v2 baseline. Coordinate-descent placer with custom incremental evaluator; the reference submission this work was built on. |
| [`optimized_v4`](../../tree/optimized_v4) | **1.0186** | ★ **Submission.** v2 + four substantive improvements: 7.7× evaluator speedup (pin-index reverse map, cumsum smoothing, np.partition top-k, vectorized routing primitives); low-T simulated annealing in CD; congestion-biased escape-basin on plateau; two state-leak bug fixes. Multi-seed verification (51 runs across seeds 42/43/44 — all VALID): per-seed means 1.0186 / 1.0170 / 1.0196; min-of-3 best-of 1.0140. |
| [`v5_cells_skip`](../../tree/v5_cells_skip) | — | Experimental: cells-unchanged skip optimization in `move_macro` (routing + smoothing are discrete in pin grid cells, so probes that don't cross cell boundaries can skip them). +8 % evaluator speed on CD-like workloads, equivalence to v4 verified at 3.57 × 10⁻⁷; **not** validated at full placer budget. |
| [`optimized_v3`](../../tree/optimized_v3) | — | Interim staging branch used during development. No submission value. |

## What's on `optimized_v4` (the submission)

Pulled from the branch's [`README.md`](../../blob/optimized_v4/README.md);
see that file for the full per-benchmark table.

| Configuration | 17-bench mean | Δ vs `optimized` (v2) |
|---------------|--------------:|----------------------:|
| `optimized` v2 baseline | 1.1172 | — |
| **v4 seed 42 (submitted)** | **1.0186** | **−0.099 (−8.83 %)** |
| v4 seed 43 | 1.0170 | −0.100 (−8.97 %) |
| v4 seed 44 | 1.0196 | −0.098 (−8.74 %) |
| v4 min-of-3 best-of | 1.0140 | −0.103 (−9.24 %) |

All 17 benchmarks improved over v2 in the seed-42 run; none regressed.
Per-seed jitter is ~±0.005 per bench (3-seed std). Submission number is
the canonical seed-42 run; the min-of-3 figure is informational.

## Reproducing the submission number

```bash
git clone https://github.com/vmallela0/macro-place-challenge-2026.git
cd macro-place-challenge-2026
git checkout optimized_v4
git submodule update --init external/MacroPlacement
uv sync

# Single benchmark:
./submissions/vmallela_v2/run.sh -b ibm01

# All 17 (≈ 15 hours serial):
./submissions/vmallela_v2/run.sh --all
```

`run.sh` exports the locked environment (seed 42, BLAS pinned to one
thread, 3300 s per-bench budget) and the v4-tuned operator settings.
The same env-var values are baked-in defaults inside the placer, so a
fresh run with no env overrides reproduces the submitted table.

Per-bench result expected within ±0.005 of the submitted seed-42 column
on different hardware (run-to-run wall-clock jitter); same seed + same
hardware → bit-reproducible.

## Where to look on `optimized_v4`

- [`submissions/vmallela_v2/`](../../tree/optimized_v4/submissions/vmallela_v2/) — placer entry point, operator modules, `run.sh`, README, layout
- [`submissions/vmallela_v2/results_verified_v4/`](../../tree/optimized_v4/submissions/vmallela_v2/results_verified_v4) — 17 seed-42 logs (the submitted run)
- [`submissions/vmallela_v2/results_verified_v4_multi/`](../../tree/optimized_v4/submissions/vmallela_v2/results_verified_v4_multi) — 34 seed-43/44 logs (the multi-seed verification)
- [`submissions/vmallela/`](../../tree/optimized_v4/submissions/vmallela) — shared `IncrementalEvaluator` and core helpers (with the v4 evaluator-speedup commits)
- [`experiments/`](../../tree/optimized_v4/experiments) — round-by-round development log (R1 – R8 grid sweep + Phase C multi-seed verification)
- [`HANDOFF.md`](../../blob/optimized_v4/HANDOFF.md) — handoff notes for the next session / reviewer
- [`README.md`](../../blob/optimized_v4/README.md) — full submission write-up with per-bench tables

## Repository layout

```
.
├── README.md                    (this file — landing page on main)
├── COMPETITION.md               original competition spec
├── benchmarks/                  IBM benchmarks (.pt files)
├── external/MacroPlacement      git submodule, TILOS PlacementCost
├── macro_place/                 evaluator API, official PlacementCost wrapper
├── submissions/
│   ├── vmallela_v2/             v2 + v4 code (current submission)
│   └── vmallela/                v1 + shared evaluator (used by v4)
└── scripts/                     visualisation tools
```

## Method (one paragraph)

Search-based placer optimising the **exact** ICCAD proxy cost
(1.0·HPWL + 0.5·density + 0.5·congestion) directly, no smoothing.
Built on a custom incremental evaluator that mirrors TILOS PlacementCost
to ≤3.57 × 10⁻⁷ but supports per-move updates in O(affected pins +
cells + nets) instead of O(all). Pipeline: push-apart preprocessing →
legalisation tournament (30 orderings × 4 step-sizes × 4 seeds) → hard
coordinate descent with size-scaled deltas and Metropolis acceptance
(SA T0 = 5e-5) → per-net weighted-median pin stepping → LNS
destroy-repair → adaptive-budget soft-macro cycle (force-directed
attraction / MLP-ranked CD / soft LNS / hard polish) with plateau-
triggered congestion-biased escape-basin (`n_destroy = 80`). Pure
Python + numpy; no GPU.

## Caveats

- Tier-2 (OpenROAD / NG45 WNS / TNS / Area) not measured locally; the
  upstream pipeline runs this for the top-7 by proxy.
- Single-threaded CPU; the grader's 16 cores and GPU go unused.
- Hardware drift: reported numbers are from a 10-core Apple Silicon
  MacBook Pro with BLAS pinned to one thread. The 16-core AMD EPYC
  9655P grader should reach equal or slightly better numbers under the
  same 1-hour budget (faster per single-thread → more iterations
  completed inside wall-clock-bounded loops).

Original competition README: [`COMPETITION.md`](COMPETITION.md).
