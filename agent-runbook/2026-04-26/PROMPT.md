# Agent prompt — execute on the 64-thread / 258 GB machine, 2026-04-26

You are an autonomous coding agent running on a fresh 258 GB / 64-thread Linux
machine. Your job is to run a multi-branch placer sweep and tell me which
branch is the new winner. Treat this entire file as your task.

## What you're doing in one sentence

Run 6 placer-branch variants across all 17 IBM benchmarks at seed 42 / 3300 s
budget, compare each to the `optimized_v4` baseline (mean 1.0186), and report
the winner. **Do not touch `main` or `optimized_v4`. Do not push anything to
remotes without confirming with me.**

## Context you need

- The submitted v4 placer (`optimized_v4` branch) gets mean 1.0186 across 17
  benches. We want to beat it.
- Yesterday I built 5 v5 branches off `optimized_v4`, each with one
  experimental change, plus `v5_combined` stacking all 4 changes. Tooling
  scripts and a runbook are on `v5_combined`.
- Yesterday's sanity ibm01 runs were killed early due to memory pressure on
  the laptop. You're starting fresh on a much bigger machine — full 17-bench
  sweep is the goal.

The 6 variants:

| Branch | Activate | What it changes |
|---|---|---|
| `v5_cells_skip_v4` | (default on) | Evaluator skips routing recompute when no pin's grid cell changed |
| `v5_escape_v2` | `PLACER_ESC_K_REGIONS=4` | Escape-basin LNS uses 4 dispersed congestion-weighted seeds |
| `v5_surrogate_struct` | `PLACER_SURR_STRUCTURED=1` | Soft surrogate uses 24 structured candidates instead of 20 random |
| `v5_warmstart` | `PLACER_WARMSTART=1` | LSE-HPWL+repulsion analytical init added to legalize tournament |
| `v5_combined` | all three above | Stack of cells-skip + escape-v2 + surrogate-struct + warmstart |
| `v5_combined_pp` | combined + `PLACER_PARALLEL_WORKERS=4` | Combined plus 4 perturbed-seed parallel restarts per benchmark (uses the multi-worker code path that was off in v4) |

`v5_combined_pp` is reusing `v5_combined`'s worktree, only the env differs.

## Step-by-step execution

### 1. Set up

```bash
git clone https://github.com/vmallela0/macro-place-challenge-2026.git
cd macro-place-challenge-2026
git fetch --all
git checkout v5_combined           # the runbook lives here
git submodule update --init external/MacroPlacement
uv sync                            # installs deps; takes ~1-2 min

# Sanity that the v5 branches arrived from origin:
for b in v5_cells_skip_v4 v5_escape_v2 v5_surrogate_struct v5_warmstart v5_combined; do
  git rev-parse --verify "origin/$b" >/dev/null && echo "  $b ok" || echo "  $b MISSING";
done
```

If any branch is MISSING, **stop and tell me** before doing anything else.

### 2. Create worktrees (one-time)

```bash
bash experiments/v5_setup_worktrees.sh
```

This creates 5 worktrees in `/tmp/wt_v5_*`, each with the submodule
symlinked to the main checkout's `external/MacroPlacement` (so worktree
size stays small). Verify all 5 exist before moving on:

```bash
ls -d /tmp/wt_v5_* | wc -l    # expect 5
```

### 3. Pre-flight sanity (15-20 min)

Before burning 3 hours on the full sweep, run **one** ibm01 at 600s on
`v5_combined` to confirm the branch produces VALID output and looks sane.

```bash
PLACER_TOTAL_BUDGET=600 PLACER_ESC_K_REGIONS=4 PLACER_SURR_STRUCTURED=1 PLACER_WARMSTART=1 \
  bash experiments/run_in_worktree.sh v5_combined_quick /tmp/wt_v5_combined ibm01 42 600 \
    PLACER_ESC_K_REGIONS=4 PLACER_SURR_STRUCTURED=1 PLACER_WARMSTART=1
```

Expected: `proxy=0.7XXX (...) VALID` line at the end. If you get INVALID,
or no `proxy=` line, or a Python traceback — **stop and tell me with the
log**. The most likely failure modes are:

- `_warmstart.py` import failure → check `submissions/vmallela_v2/_warmstart.py` exists
- env var not propagating → confirm `run_in_worktree.sh` exports the EXTRAS

If sanity is good (VALID, cost ~0.78), proceed.

### 4. Full sweep

```bash
# 12 concurrent benchmarks × ~1-5 cores each. Target: 60 of 64 cores.
# Total wall: ~3 hours for 6 branches × 17 benches = 102 runs.
nohup bash experiments/v5_full_sweep.sh 42 3300 12 > /tmp/v5_sweep.out 2>&1 &
echo "sweep launched, pid=$!"
```

Monitor periodically:

```bash
bash experiments/v5_status.sh                                         # quick status
tail -50 /tmp/v5_sweep.out                                            # launcher log
wc -l experiments/results.csv                                         # rows accumulating
ls experiments/logs/v5_*/ | head                                      # logs landing
pgrep -f "macro_place.evaluate" | wc -l                               # alive placers
```

Each placer should complete in ~55-65 minutes on a single core. With 12
concurrent and 6 branches × 17 benches = 102 runs, total wall is roughly
`102 / 12 × 60min ≈ 8.5 hours` if things are perfectly serial. You should
see most branches done in 6-9 hours.

If it looks stuck (no new CSV rows for >90 min, OR `pgrep` returns 0 with
incomplete CSV), **stop and tell me** with the latest `/tmp/v5_sweep.out`
tail and `experiments/results.csv` row count.

### 5. Analyze results

Once the sweep completes (or completes 5/6 branches and you want a
preliminary read):

```bash
python experiments/v5_analyze.py
```

This prints:
- Per-bench v4-vs-v5 deltas for each of the 6 branches
- Per-branch summary: n_done, mean_cost, mean_delta, win/loss counts
  (threshold: 5e-3)

### 6. Decide the winner and report

Pick the branch with **all 17 VALID and lowest mean cost**. Report to me:

1. Per-branch summary table from `v5_analyze.py`
2. The winning branch's per-bench deltas (highlight any bench that
   regressed >5e-3 vs v4)
3. The winner's mean cost, and how it compares to v4's 1.0186
4. Wall-clock time the sweep took
5. Any branch that produced INVALID — and what happened in its log

Do **not** push anything to `optimized` or `optimized_v4`. Do **not** create
new branches or do follow-up experiments without checking with me. Your
output is a report; I'll decide what to merge.

If the winner beats v4 by ≥0.005 on the mean and has all 17 VALID, mention
that we should consider it the new submission candidate (`optimized_v5`).

### 7. Optional: multi-seed verification of the winner

If the winning branch is clearly better and you have time, re-run the
winner across seeds 43 and 44 to confirm the gain isn't seed luck:

```bash
# replace v5_combined with whichever branch won
nohup bash experiments/run_in_worktree.sh \
  v5_winner_s43 /tmp/wt_v5_combined "$BENCH" 43 3300 \
  PLACER_ESC_K_REGIONS=4 PLACER_SURR_STRUCTURED=1 PLACER_WARMSTART=1 \
  > /tmp/v5_s43.out 2>&1 &
```

(You'd want a loop over benches and the 2 seeds.) Do this **only if** the
winner is clearly best and you have ≥6 hours left. Otherwise just report
the seed-42 result.

## Things that can go wrong

- **`uv sync` fails on Linux**: check if uv is installed; `pip install uv`
  then retry.
- **submodule clone slow / fails**: it's GitHub-hosted, ~50 MB. Network
  retry usually fixes it.
- **Worktree creation fails because `/tmp/wt_v5_*` exists**: the setup
  script skips existing worktrees. To force-rebuild: `git worktree remove
  --force /tmp/wt_v5_<name>` then re-run.
- **`run_in_worktree.sh` complains about missing `.venv/bin/python`**:
  verify `uv sync` actually completed in the main repo dir. The harness
  uses MAIN_REPO's venv directly.
- **A placer hangs** (no new log lines for >5 min): check load + memory.
  If genuinely stuck, kill that one process; the rest of the sweep
  continues. Note which bench/branch hung.
- **Out of memory**: 258 GB should be plenty for 12 concurrent placers
  (each ~2 GB) plus 4 workers each (each ~1 GB). If OOM somehow happens,
  reduce `max_parallel` arg to `v5_full_sweep.sh`.

## What success looks like

- All 102 runs (6 branches × 17 benches) complete with VALID
- `python experiments/v5_analyze.py` shows at least one branch with mean
  delta < -0.005 vs v4
- You have a clear winner to recommend

## What to NOT do

- Do not push to `main`, `optimized`, or `optimized_v4`.
- Do not modify `submissions/vmallela_v2/run.sh` — that's the v4 submission
  config; it stays untouched until I OK a v5 promotion.
- Do not delete any v5 branches or worktrees.
- Do not start new experiments beyond what's listed here.
- Do not push the results CSV to remote unless I ask. The CSV is local
  artifact; share it inline in your report.

## Quick reference — the v5 branches

```
v5_cells_skip_v4    f16a006   cells-skip evaluator
v5_escape_v2        d7a3d63   multi-region dispersed-seed escape
v5_surrogate_struct b2095a5   structured candidate set for soft surrogate
v5_warmstart        fbf9879   LSE-HPWL + repulsion analytical init
v5_combined         ff9ebe3+  all 4 + tooling + this runbook
```

All branches are off `optimized_v4` (commit `f53fda8`).

## Quick reference — env vars

| Var | Default | Effect |
|---|---|---|
| `PLACER_TOTAL_BUDGET` | 3300 | per-bench wall budget (s) |
| `PLACER_SEED` | 42 | RNG seed |
| `PLACER_PARALLEL_WORKERS` | 0 | 0 = no parallel; >0 = N perturbed restarts |
| `PLACER_ESC_K_REGIONS` | 1 | 1 = v4 single-region escape; ≥2 = v5 multi-region |
| `PLACER_ESC_HOTSPOT_FRAC` | 0.05 | top-fraction of cells considered "hot" |
| `PLACER_ESC_MIN_DIST_FRAC` | 0.25 | min seed separation as fraction of canvas diag |
| `PLACER_SURR_STRUCTURED` | 0 | 1 = structured 24-cand soft surrogate |
| `PLACER_WARMSTART` | 0 | 1 = enable analytical warmstart |
| `PLACER_WARMSTART_BUDGET` | 60 | warmstart wall budget (s) |
| `PLACER_SA_T0` | 5e-5 | CD simulated-annealing T0 (v4 baked-in) |
| `PLACER_ESC_HARD_DESTROY` | 80 | escape hard-LNS destroy size (v4 baked-in) |

That's it. Execute step-by-step. Report back when done.
