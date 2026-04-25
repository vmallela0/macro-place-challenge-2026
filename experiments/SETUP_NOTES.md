# v5 sweep — setup notes (2026-04-25)

For future sessions to reproduce / analyze the v5 sweep results in
`experiments/results.csv` and `experiments/logs/`.

## Box

- **Hardware**: 16 vCPU (8 physical Zen 5 cores + SMT siblings), 60 GiB RAM,
  9.7 GB root disk
- **OS**: Linux 6.1.0-44-cloud-amd64 / Debian 12
- **Python**: 3.11.2
- **Torch**: 2.11.0 + cpu (CPU-only — full CUDA torch wouldn't fit on the disk;
  installed via `--index-url https://download.pytorch.org/whl/cpu`)

## Sweep parameters

- **Seed**: 42
- **Budget**: 3300 s per benchmark (matching v4 protocol)
- **Parallelism**: 8-way (1 placer per physical Zen 5 core, BLAS pinned to 1
  thread via `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1` in the
  harness). 9th worker would land on an SMT sibling and contend with an
  existing placer for execution units.
- **Total runs**: 5 branches × 17 IBM benches = 85 runs at seed 42
- **Wall ETA**: ~10 h
- **Started**: 2026-04-25T18:02 UTC, in tmux session `sweep`
- **Variant skipped**: `v5_combined_pp` (4 perturbed-seed parallel restarts per
  bench, 5x resource cost) — would oversubscribe a 16-vCPU box

## v5 branches under sweep

| Branch | Activated by | Worktree |
|--------|--------------|----------|
| `v5_cells_skip_v4` | (default on after merge) | `/tmp/wt_v5_cells` |
| `v5_escape_v2` | `PLACER_ESC_K_REGIONS=4` | `/tmp/wt_v5_escape` |
| `v5_surrogate_struct` | `PLACER_SURR_STRUCTURED=1` | `/tmp/wt_v5_surr` |
| `v5_warmstart` | `PLACER_WARMSTART=1` | `/tmp/wt_v5_warm` |
| `v5_combined` | all four above | `/tmp/wt_v5_combined` |
| `v5_cluster` (local only) | combined + `PLACER_ESC_CLUSTER_FRAC=0.3` | `/tmp/wt_v5_cluster` |

Worktrees use a symlinked `external/MacroPlacement` to avoid cloning the
~3.5 GB submodule per worktree. Each worktree is 18 MB.

## v4 reference

- `optimized_v4` seed 42, 3300 s, 17 benches, **on a 10-core Apple Silicon
  MacBook Pro** with BLAS pinned to one thread → mean **1.0186**
- Same seed 43 → 1.0170, seed 44 → 1.0196, min-of-3 oracle → 1.0140

**The v4 reference numbers are not from this Linux box.** Apparent v5 wins
include both algorithmic gain and per-thread hardware speedup (Linux faster
than Apple Silicon → more SA/CD iterations fit in 3300 s budget). Phase A
of `v5_post_sweep.sh` runs `optimized_v4` on this box at seed 42 to
disentangle the two contributions.

## Reproducing on a fresh box

```bash
# 1. Clone, fetch all branches
git clone https://github.com/vmallela0/macro-place-challenge-2026.git
cd macro-place-challenge-2026
git fetch --all
git submodule update --init external/MacroPlacement

# 2. CPU-only torch (manual, bypasses CUDA-pinning uv.lock)
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cpu "torch>=2.0.0"
uv pip install --python .venv/bin/python "numpy>=1.20.0" "matplotlib>=3.5.0" "tqdm>=4.65.0" "absl-py>=1.0.0"
uv pip install --python .venv/bin/python -e . --no-deps

# 3. Worktrees (use experiments/v5_setup_worktrees.sh from v5_combined branch).
#    First switch main checkout off v5_combined or worktree creation will fail.
git checkout landing_page    # or any non-v5_combined branch
bash /tmp/wt_v5_combined/experiments/v5_setup_worktrees.sh
# Then manually fix the v5_combined worktree (the script fails for the
# branch already checked out — see HANDOFF below).

# 4. Fix run_in_worktree.sh in each worktree (it has a Mac path baked in).
sed -i 's|/Users/vmallela/Desktop/challenges/macro-place-challenge-2026|/home/<user>/macro-place-challenge-2026|' \
  /tmp/wt_*/experiments/run_in_worktree.sh

# 5. Drop v5_combined_pp from the sweep on small boxes
sed -i '/v5_combined_pp/d' /tmp/wt_v5_combined/experiments/v5_full_sweep.sh

# 6. Launch in tmux
tmux new-session -d -s sweep \
  "cd /tmp/wt_v5_combined && bash experiments/v5_full_sweep.sh 42 3300 8 2>&1 | tee /tmp/v5_sweep.out"
```

## Artifacts being captured

- `experiments/results.csv` — one row per run: exp_id, branch@SHA, bench, seed,
  budget_s, final_cost, hpwl, density, congestion, wall_clock_s, timestamp,
  notes (env vars used). Source of truth.
- `experiments/logs/<exp_id>/<bench>_s<seed>_b<budget>.log` — full placer
  stdout per run (~50 KB at full budget).
- `experiments/launcher.log` — main sweep launcher output (snapshot of
  `/tmp/v5_sweep.out` at last commit).
- `experiments/launcher_per_task/` — per-task harness output snapshots.

`/tmp` files are ephemeral on this box — `experiments/snapshot_logs.sh` is
called at every BRANCH_DONE to copy them into the repo.

## Known gotchas

1. The runbook's `v5_setup_worktrees.sh` will silently `rm -rf
   external/MacroPlacement` in the **main repo** if `v5_combined` is the
   currently-checked-out branch (the cd into the worktree fails but bash keeps
   running). Always switch main to a different branch first, or hand-create
   the `v5_combined` worktree.
2. `run_in_worktree.sh` has the original author's macOS path hardcoded as
   `MAIN_REPO`. Must be patched per-worktree to your Linux home.
3. The full `uv sync` will pull CUDA torch + dependencies (~3 GB). On a
   small-disk box, install CPU torch manually (above) and skip `uv sync`.
4. With 9.7 GB root disk, watch headroom: project ~4 GB + .venv ~600 MB +
   submodule ~3.5 GB + worktrees ~108 MB + logs ~5 MB → ~200 MB headroom.
