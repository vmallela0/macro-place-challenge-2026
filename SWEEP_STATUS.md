# Overnight v6 sweep — status

**Started:** 2026-04-28 01:35:03 PDT
**Sweep PID:** 32680
**Caffeinate PID:** 26116 (keeps Mac awake)
**Results dir:** `/tmp/v6_overnight_20260428_013503/`

## What's running

```
17 benches × PLACER_TOTAL_BUDGET=1800s × 8-worker portfolio (1 GPU + 7 CPU)
  + consensus warm-start (graft, refine 120s)
  per-bench hard timeout: 2700s
  expected per-bench wall-clock: ~32 min
  expected total: ~9 hours → completes ~10:35 AM
```

## How to check progress

```bash
# Top-level log (one line per bench start/end + summary):
tail -f /tmp/v6_overnight_20260428_013503/sweep.log

# Live CSV (appended after each bench finishes):
cat /tmp/v6_overnight_20260428_013503/results.csv

# Detailed per-bench log:
tail -f /tmp/v6_overnight_20260428_013503/ibm01.log

# Verify sweep still alive:
ps -p 32680
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
