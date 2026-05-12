# zeus workflow — running claude + experiments durably

Goal: SSH drop, claude crash, or laptop death should never kill an in-flight
experiment, and the agent should be re-attachable from any fresh SSH.

## TL;DR — quick recovery after a session drop

```bash
ssh vedu@optimeshr640
cd ~/vmallela/personal/macro-place-challenge-2026
cat ~/zeus_runs/AUTOPILOT_STATUS.md   # what state are experiments in?
bash scripts/zeus_claude.sh           # bring back claude (tmux-wrapped)
# inside the new claude, if you want the prior conversation:
#   claude --resume 92576a0d-dd43-4e0f-b0c9-9eca15f2c6dd
```

## One-time setup (already done on `optimeshr640`)

- `tmux 3.4` is installed.
- `scripts/zeus_claude.sh` — launches/attaches claude inside tmux.
- `scripts/zeus_run_detached.sh` — launches any command fully detached
  (setsid + nohup + disown, reparented to PID 1).
- `scripts/zeus_status.sh` — checks state of detached runs.
- `scripts/zeus_autopilot.sh` — watches a detached clean_ab, decides the
  winning arm, and launches the full-17 sweep automatically.

## Recipe — start a session

```bash
ssh vedu@optimeshr640
cd ~/vmallela/personal/macro-place-challenge-2026
bash scripts/zeus_claude.sh        # attaches to tmux 'zeus' or creates it
```

If you detach (`Ctrl-B D`) and SSH drops, claude keeps running.

## Recipe — recover a session

```bash
ssh vedu@optimeshr640
cd ~/vmallela/personal/macro-place-challenge-2026
bash scripts/zeus_claude.sh        # finds existing session, re-attaches
```

If you need a totally fresh claude:

```bash
bash scripts/zeus_claude.sh new    # kills old tmux session, starts fresh
```

## Recipe — start a long-running experiment durably

```bash
bash scripts/zeus_run_detached.sh <name> <command...>
```

Example (the current clean-A/B):

```bash
ARMS="base rudy rudy_hmc" WORKER_BUDGET=2300 \
  BENCHES="ibm06 ibm12 ibm15" \
  bash scripts/zeus_run_detached.sh clean_ab \
    bash scripts/zeus_clean_ab.sh
```

The run is reparented to PID 1 — survives SSH drop, claude exit, laptop close.

## Recipe — check on detached runs

```bash
bash scripts/zeus_status.sh                # all runs
bash scripts/zeus_status.sh clean_ab       # one specific run
bash scripts/zeus_status.sh tail clean_ab  # tail -f the log
```

Run directories live in `~/zeus_runs/<name>-<timestamp>/` with a
`latest_<name>` symlink for convenience.

## Autonomous pipeline (currently running)

`scripts/zeus_autopilot.sh` is launched via `zeus_run_detached.sh` so its
parent is PID 1 — it survives SSH drops, claude exits, and laptop deaths.
It runs the following pipeline without intervention:

1. Poll the detached clean_ab run until its PID exits.
2. Parse `results.csv`, compute mean Δ per arm.
3. Pick the winning arm (lowest mean Δ; tie-break = simpler arm).
   - Arms: `base` (0.9975 config, cong-OFF), `rudy` (+ differentiable
     RUDY routing demand), `rudy_hmc` (+ subspace HMC escape).
4. If best mean Δ ≤ 0.005: launch `zeus_full17.sh` detached with that
   arm's env vars. Otherwise: write "needs human review" to STATUS.md.
5. Poll full-17 until done; write final mean and per-bench table to STATUS.md.

Inspect at any time:

```bash
cat ~/zeus_runs/AUTOPILOT_STATUS.md            # current pipeline state
bash scripts/zeus_status.sh                    # all detached runs
bash scripts/zeus_status.sh tail autopilot     # tail -f autopilot log
bash scripts/zeus_status.sh tail clean_ab      # tail -f clean_ab log
bash scripts/zeus_status.sh tail full17        # tail -f full17 log
```

## What survives what

| event                       | clean_ab | autopilot | full17 | claude (this session) |
|-----------------------------|:--------:|:---------:|:------:|:---------------------:|
| SSH connection drops        | ✓        | ✓         | ✓      | ✗ (dies on SIGHUP)    |
| this claude process exits   | ✓        | ✓         | ✓      | n/a                   |
| laptop closes               | ✓        | ✓         | ✓      | ✗                     |
| `kill -9` on init           | ✗        | ✗         | ✗      | ✗                     |
| `reboot`                    | ✗        | ✗         | ✗      | ✗                     |

To make a FUTURE claude session ALSO survive SSH drops, launch it via
`bash scripts/zeus_claude.sh` instead of plain `claude`. This wraps it in
tmux so the controlling pty no longer matters. The current session was
launched outside tmux and cannot be retroactively reparented without
`reptyr` (needs sudo to install).
