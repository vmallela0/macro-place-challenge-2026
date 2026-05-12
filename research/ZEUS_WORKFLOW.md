# zeus workflow — running claude + experiments durably

Goal: SSH drop, claude crash, or laptop death should never kill an in-flight
experiment, and the agent should be re-attachable from any fresh SSH.

## One-time setup (already done on `optimeshr640`)

- `tmux 3.4` is installed.
- `scripts/zeus_claude.sh` — launches/attaches claude inside tmux.
- `scripts/zeus_run_detached.sh` — launches any command fully detached
  (setsid + nohup + disown, reparented to PID 1).
- `scripts/zeus_status.sh` — checks state of detached runs.

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

## When the current clean-A/B finishes

CSV at `~/zeus_runs/latest_clean_ab/` and `/tmp/zeus_clean_ab_*/results.csv`.

Arms:
- `base`     : production 0.9975 config (electrostatic-norm, cong-OFF)
- `rudy`     : same + differentiable RUDY routing demand (cong-ON)
- `rudy_hmc` : same as rudy + subspace HMC escape (K=6, T=16 trajectories)

Decision rule (`task #3`):

- If `rudy` arm mean Δ ≤ `base` mean Δ on the 3 benches → keep RUDY.
- If `rudy_hmc` mean Δ < `rudy` mean Δ → HMC also helps; use combined.
- Whichever wins, launch:
  ```bash
  bash scripts/zeus_run_detached.sh full17 bash scripts/zeus_full17.sh
  ```
  with appropriate `PLACER_V7_HESSIAN_RUDY`/`HMC_*` env vars.

If both lose: read `/tmp/zeus_clean_ab_*/{base,rudy,rudy_hmc}_*.log`,
look for the `[v7] hessian: ...` lines to debug.
