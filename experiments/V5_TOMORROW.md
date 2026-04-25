# v5 sweep on the 64-thread machine — runbook

This is the recipe for taking the 5 v5 branches built today and full-sweeping
them on the 258 GB / 64-thread machine.

## What's on the branches

Each is a single-feature experiment off `optimized_v4`. Bit-equivalent to v4
when its env gate is unset/0; activated by an env var.

| Branch | Env to activate | What it changes |
|---|---|---|
| `v5_cells_skip_v4` | (none — always on) | IncrementalEvaluator skips routing recompute when no pin's grid cell changed (~24% of CD probes); 7-8% per-probe speedup |
| `v5_escape_v2` | `PLACER_ESC_K_REGIONS=4` | Escape-basin LNS picks 4 dispersed seeds (continuous blockage-in-hotspot score, farthest-point), each with `n_destroy/4` BFS budget |
| `v5_surrogate_struct` | `PLACER_SURR_STRUCTURED=1` | Soft surrogate uses 24 structured candidates (8 dirs × 3 mags) instead of 20 random radial |
| `v5_warmstart` | `PLACER_WARMSTART=1` | LSE-HPWL + soft-repulsion analytical init added to legalize tournament; budget `PLACER_WARMSTART_BUDGET=60s` |
| `v5_combined` | all three above | Stack of cells-skip + escape-v2 + surrogate-struct + warmstart |
| `v5_combined_pp` | combined + `PLACER_PARALLEL_WORKERS=4` | Combined plus 4 perturbed-seed parallel restarts per benchmark |

## One-time setup on the new machine

```bash
git clone https://github.com/vmallela0/macro-place-challenge-2026.git
cd macro-place-challenge-2026
git fetch --all
git checkout optimized_v4
git submodule update --init external/MacroPlacement
uv sync
git fetch origin v5_cells_skip_v4 v5_escape_v2 v5_surrogate_struct v5_warmstart v5_combined 2>/dev/null || true
# (branches haven't been pushed; if running locally the worktree script
#  will create them from local refs automatically)

bash experiments/v5_setup_worktrees.sh    # creates 5 worktrees in /tmp with submodule symlinked
```

Each worktree is ~50 MB (submodule symlinked, not cloned). 5 worktrees = 250 MB. Plus the main repo's submodule (~3.5 GB). Total ~4 GB.

## Run the sweep

```bash
# Recommended: 12 concurrent benchmarks × ~5 cores each (1 main + 4 workers
# for the pp variant) = 60 of 64 cores. About 2.5-3 hours total.
bash experiments/v5_full_sweep.sh 42 3300 12

# Aggressive: 16 concurrent × 1-5 cores = 16-80 cores. Some contention;
# may wallclock-extend per-bench beyond 1 hour.
bash experiments/v5_full_sweep.sh 42 3300 16
```

Each branch runs all 17 benches at seed 42 budget 3300. CSV updates in
`experiments/results.csv`. Logs in `experiments/logs/<branch>/<bench>_s42_b3300.log`.

## Inspect results

```bash
python experiments/v5_analyze.py
```

Output: per-bench v4-vs-v5 deltas + per-branch mean delta + win/loss counts
(threshold: 5e-3). Anything in the `↓` (win) column is reproducibly better.

## Decision after the sweep

Pick the branch with:
- All 17 VALID
- Lowest mean cost
- Mean delta vs v4 < -0.005

Cherry-pick its commit(s) onto a new `optimized_v5` branch off `optimized_v4`,
update `submissions/vmallela_v2/run.sh` to bake in the env vars, regenerate
`results_verified_v5/` logs, push.

## Memory & disk notes

- Each ibm01-ibm18 placer process at peak: ~1.5-2 GB resident. 16 concurrent ≈ 32 GB. 64 GB headroom on the 258 GB box.
- Each pp worker process within a bench: ~1 GB. Add to the above when pp variant runs.
- Logs: ~80 KB per bench × 17 × 6 branches = ~8 MB total. Negligible.
- /tmp on the new machine: 5 worktrees × 50 MB symlinked = 250 MB. Confirm `df /tmp` has at least 1 GB free.

## Sanity checks before launching the full sweep

```bash
# 1. All 5 worktrees exist and have the right placer code
for d in /tmp/wt_v5_cells /tmp/wt_v5_escape /tmp/wt_v5_surr /tmp/wt_v5_warm /tmp/wt_v5_combined; do
  echo "=== $d ===";
  ls -la "$d/external/MacroPlacement/Testcases/ICCAD04/ibm01" | head -2
done

# 2. Quick ibm01 sanity at 600s on v5_combined to confirm it produces VALID
PLACER_TOTAL_BUDGET=600 PLACER_ESC_K_REGIONS=4 PLACER_SURR_STRUCTURED=1 PLACER_WARMSTART=1 \
  bash experiments/run_in_worktree.sh v5_combined_quick /tmp/wt_v5_combined ibm01 42 600
# Expect VALID, cost in [0.78, 0.82].

# 3. CSV writeable
test -w experiments/results.csv && echo "csv ok" || touch experiments/results.csv
```

## What the today-machine ibm01 sanity runs showed

Today's machine is 10-core / 16 GB. We launched 5 ibm01 sanity runs in
parallel; with 5 placers contending, each got ~0.5 effective cores. Results
land in `experiments/logs/v5_*/ibm01_s42_b3300.log` and the CSV. Look at
those before launching the full sweep — they tell you which branch
plausibly works and which has a regression bug.

If any branch's ibm01 came back NOT VALID, do not run that branch in the full
sweep until the bug is fixed. The fix likely lives in the env-gate path —
search for `PLACER_*` in the relevant module.
