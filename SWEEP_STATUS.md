# Overnight v6 sweep — status (2nd run, with determinism fixes + viz)

**Started:** 2026-04-28 **18:28:54 PDT** (kickoff cancelled, ran early)
**Sweep PID:** 23242
**Caffeinate PID:** 23269
**Results dir:** `/tmp/v6_overnight_20260428_182854/`
**ETA done:** ~03:30 AM (~9 hours, 17 benches × ~32 min)

**Previous run (1st sweep):** completed; results at
`/tmp/v6_overnight_20260428_013503/`. 1.0196 mean over 17 IBM benches.
Superseded by this run, which has the determinism fixes (BLAS pin via
threadpoolctl + cuDNN deterministic) and per-bench placement plots.

## What's running (or about to run)

```
Kickoff wrapper (PID 22100) is asleep until 19:10:00 PDT, then it spawns:

  17 benches × PLACER_TOTAL_BUDGET=1800s × 8-worker portfolio (1 GPU + 7 CPU)
    + consensus warm-start (graft, refine 120s)
    + per-bench placement plot to assets/v6_<bench>.png + run-dir
    per-bench hard timeout: 2700s
    expected per-bench wall-clock: ~32 min
    expected total sweep: ~9 hours → completes ~04:30 AM next day
```

NEW vs the 1st run:
  - Determinism fixes (threadpoolctl runtime BLAS pin, cuDNN deterministic,
    CUDA RNG seed, env setdefault at module top) so same code on grader
    won't see the v2-style 27% reproducibility gap.
  - Per-bench static placement plot (red rectangles for hard macros,
    blue dots for soft cluster centroids, canvas border, title with
    proxy/wl/den/cong/overlaps). Same style as assets/ibm01_v4.png.

## How to check progress

```bash
# Until 19:10, the kickoff wrapper is just sleeping:
cat /tmp/v6_delayed.log
ps -p 22100   # kickoff wrapper

# After 19:10, the actual sweep starts. Find its dir:
ls -td /tmp/v6_overnight_* | head -1

# Then:
tail -f /tmp/v6_overnight_<TIMESTAMP>/sweep.log   # high-level
cat /tmp/v6_overnight_<TIMESTAMP>/results.csv      # CSV (live)
ls /tmp/v6_overnight_<TIMESTAMP>/*.png             # generated plots
```

## What happens when it finishes

1. The sweep loops through `ibm01..ibm18` (skipping ibm05) serially.
2. After all 17 land, the script runs `scripts/v6_results_to_readme.py`
   which patches `submissions/vmallela_v6/README.md` with a per-bench
   table comparing v6 to v4 seed-42.
3. The README change is **NOT auto-committed** — review it, then:
   ```bash
   git diff submissions/vmallela_v6/README.md
   git add submissions/vmallela_v6/README.md
   git commit -m "v6-gpu: per-bench results from overnight sweep"
   git push origin v6-gpu
   ```

## If something goes wrong

- Sweep PID dies: `ps -p 32680` returns nothing → start it again with
  `nohup ./scripts/v6_overnight_sweep.sh > /tmp/v6_sweep_main.log 2>&1 &`
- A bench hangs past 2700s: the script kills it via the inline timeout
  emulation; the next bench resumes automatically.
- Mac sleeps: shouldn't (caffeinate active). If caffeinate died, the
  sweep stalled — kill it (`kill 32680`), restart caffeinate
  (`caffeinate -i -m -s &`), restart sweep.

## Chained 3300s/worker re-run

After the 1800s sweep finishes (PID 32680), a chained sweep
automatically starts at 3300s/worker (matching v4's submitted budget
exactly).

- **Chain PID:** 39250 (started 2026-04-28 09:34)
- **Chain log:** `/tmp/v6_chain.log`
- **3300s sweep wall-clock:** ~17 hours (17 × 60 min/bench)
- **Expected 3300s sweep done:** ~2026-04-29 04:30 AM PDT

The 3300s results dir will have a new timestamp under `/tmp/v6_overnight_*`.
The 3300s sweep auto-updates the v6 README on completion (overwrites
the partial 1800s table). Does **not** auto-commit — review with
`git diff submissions/vmallela_v6/README.md` first.

## Cleanup once both sweeps are done

- Kill caffeinate: `kill 26116`
- Archive logs: `mv /tmp/v6_overnight_* ~/Desktop/`
- Delete this file: `rm SWEEP_STATUS.md`
