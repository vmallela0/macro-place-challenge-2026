# Handoff — vmallela_v4 macro-placement submission

This document is for the next session / next reviewer to pick up the work
quickly. Everything important is on the `optimized_v4` branch.

## TL;DR

**Submission number to report: 1.0186** (mean across 17 IBM benchmarks,
seed 42, 3300 s budget per bench, all 17 VALID with zero overlaps).

This is what we should put in the Google form. The min-of-3 best-of
across seeds 42/43/44 gives 1.0140, but per the rules we submit one
algorithm and report what that algorithm produces at its default seed.

## Where everything lives

| What | Where | On remote? |
|---|---|---|
| Submission code | `submissions/vmallela_v2/` (uses `submissions/vmallela/` for shared evaluator) | ✅ pushed |
| Submission run.sh | `submissions/vmallela_v2/run.sh` (winning env vars baked in) | ✅ pushed |
| Submission result logs (seed 42) | `submissions/vmallela_v2/results_verified_v4/` (17 logs) | ✅ pushed |
| Multi-seed verification (seeds 43, 44) | `submissions/vmallela_v2/results_verified_v4_multi/` (34 more logs + SUMMARY.md) | ✅ pushed |
| Repository root README | `README.md` (has full per-bench table + multi-seed) | ✅ pushed |
| Experiment harness + dev logs | `experiments/` | ✅ pushed |
| v5 cells-skip optimization (8% extra eval speed, untested at full budget) | branch `v5_cells_skip` | needs push |
| v2 baseline submission (for comparison) | branch `optimized` (`submissions/vmallela_v2/results_verified/`) | already on remote |

## Branches

```
optimized       v2 submission, mean 1.1172 (the prior baseline)
optimized_v3    interim — used for harness + stash work, no submission value
optimized_v4    ★ THIS IS THE SUBMISSION, mean 1.0186
v5_cells_skip   experimental: cells-unchanged skip in evaluator, +8% eval speed
                bit-equivalence to v4 confirmed, but NOT tested at full placer budget
```

## How to reproduce the 1.0186 number

```bash
git clone https://github.com/vmallela0/macro-place-challenge-2026.git
cd macro-place-challenge-2026
git checkout optimized_v4
git submodule update --init external/MacroPlacement
uv sync

./submissions/vmallela_v2/run.sh -b ibm01     # single bench
./submissions/vmallela_v2/run.sh --all        # all 17, ~15 hours serial
```

Expected per-bench numbers within ±0.005 of those in `README.md`'s
"v4 vs v2 per-benchmark" table.

## What's verified vs not

- ✅ All 51 multi-seed runs (3 × 17) at 3300 s each, every one VALID
- ✅ Evaluator equivalence to TILOS PlacementCost: ≤3.57e-7 abs diff
  over 30 random moves on ibm01 (`tests/test_evaluator_equivalence.py`)
- ✅ Cells-skip equivalence: bit-identical to v4 (3.57e-7) at the
  evaluator level
- ❌ Tier-2 OpenROAD / NG45 WNS / TNS / Area: NOT measured locally
  (no OpenROAD setup); the upstream pipeline runs this for top 7 by
  proxy
- ❌ Cells-skip placer-level effect: NOT measured at full budget; only
  evaluator microbenchmark (8% per-probe speedup on CD-like workloads)
- ❌ Bit-reproducibility across hardware: NO — we use wall-clock-bounded
  loops in 13 places, so iteration count varies by CPU speed. ±0.005
  jitter per bench is expected on different hardware

## Submission form fields (when filing the Google form)

| Field | Value |
|---|---|
| Team Name | `vmallela` |
| Method Name | `Incremental CD+LNS+SA` |
| Avg proxy cost | **1.0186** |
| Avg runtime | **3290 s / bench** (≈ 54.8 min) |
| WNS / TNS / Area NG45 | leave blank — Tier-2 evaluation pending |
| GitHub repo | `https://github.com/vmallela0/macro-place-challenge-2026` |
| Branch to evaluate | `optimized_v4` |
| Shared with judges | ✓ partclxhrtmacroplace@gmail.com, will@partcl.com |
| Open-source | ✓ |

## Known levers we did not pursue

In rough order of remaining EV:

1. **Multi-worker portfolio (existing infra at `placer.py:170`)** —
   The grader has 16 cores; we use 1. Forking 4-8 perturbed-seed
   workers and taking min would near-guarantee the algorithm's
   min-of-N number across seeds. Code path exists but is gated off
   (`PLACER_PARALLEL_WORKERS=0`). Realistic gain: -0.005 to -0.015.
2. **v5 cells-skip placer-level test** — committed on `v5_cells_skip`
   branch, evaluator-microbenchmark says +8%; haven't measured at full
   3300 s placer budget. Could shave -0.002 to -0.008.
3. **Iteration-count budgets** (replacing the 13 `time.time()` loops
   with iteration counters) — gives bit-reproducibility across
   hardware. Cosmetic; doesn't change cost.
4. **Numba JIT or Cython** of the routing primitives — could be 5-10×
   on those alone, would bring evaluator to ~0.15 ms/probe. Adds a
   build dep.
5. **Analytical warm-start** with a "stay-close-to-init" regulariser
   added to LSE-HPWL. Earlier exploration showed the unregularised
   version clusters too aggressively for the legaliser; this fix is
   what DREAMPlace-style flows use. ~150 lines.

## What I'd do first if continuing

1. **Push the v5 branch**: `git push origin v5_cells_skip` so it's not
   stranded locally.
2. **R9 sweep of v5**: full 17-bench × 3 seeds × 3300 s with cells-skip
   enabled to confirm or kill the 8% evaluator speedup actually moves
   the placer-level mean.
3. **Multi-worker portfolio** if you have a few days — likely the
   biggest remaining lever for getting under 1.0.

## Current task state at handoff

Active: `#11 Submit privately via Google Form` (still pending), `#32
B: push for sub-1.0` (still pending — v5 staged, not run; multi-worker
not started).

Closed: all evaluator speedups, bug audits, single-seed sweep,
multi-seed validation.

## A few gotchas for the next session

- `pgrep -f "macro_place.evaluate"` will match shell scripts that
  contain that string in their command-line. Use a more precise pattern
  (e.g. `pgrep -f "/python -m macro_place.evaluate"`) when writing
  waiters.
- `uv run` reinstalls the editable `macro-place` package when the
  resolved path differs (e.g. running from a worktree replaces
  `.venv/lib/.../macro_place` with the worktree version). Calling the
  venv's `.venv/bin/python` directly with `PYTHONPATH=.` and
  `python -m macro_place.evaluate ...` avoids this.
- `*.log` is in `.gitignore` at the repo root; use `git add -f` to
  include result-log files in commits.

—

Final state: pushed and ready. README has full per-bench tables. The
1.0186 number is reproducible. Ready to file the form.
