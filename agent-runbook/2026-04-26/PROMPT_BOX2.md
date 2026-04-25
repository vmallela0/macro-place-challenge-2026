# Agent prompt — parallel v5 work on a second machine, 2026-04-25

You are a fresh Claude Code agent on a more powerful Linux box. A first box
is already running the main v5 sweep (5 branches × 17 benches × seed 42 ×
3300 s, 8-way parallel) — **do not duplicate that work**. Your job is to
run the *complementary* experiments that the first box can't fit, push
results to a dedicated branch, and report back.

## What's already in flight on the other box (do NOT redo)

- `v5_cells_skip_v4` (✅ done, mean 1.0046)
- `v5_escape_v2` (in flight)
- `v5_surrogate_struct` (in flight)
- `v5_warmstart` (in flight)
- `v5_combined` (queued)

All at seed 42, 3300 s budget. Results land in `experiments/results.csv` on
the `landing_page` branch.

## Your job: 3 parallel tracks

### Track 1: `v5_combined_pp` full sweep (needs ≥32 cores)
This is the 4-perturbed-seed-restart variant we skipped on the small box.
17 benches × seed 42 × 3300 s × 4 worker subprocesses per bench.
Net: ~5x compute per bench.

### Track 2: `v5_cluster` full sweep
Experimental branch (already pushed to origin) that wires
`cluster_translate_phase` (orphan code in `_moves.py` since v2) into the
escape phase. Tests whether correlated multi-macro moves help on the hard
benches (ibm12/17/18). 17 benches × seed 42 × 3300 s.

### Track 3: env-tune stacking on top of v5_combined / v5_cluster
After tracks 1+2 complete, run `experiments/v5_envtune.sh` on the hardest
3 benches (ibm12, 17, 18) to test env-only stacking experiments
(PLATEAU_N=2, ESC_HARD_DESTROY=120, EXP7_LINESEARCH=1, etc.) — chasing the
sub-1.0 mean.

Resource scheduling depends on your box. With ≥32 physical cores, run
tracks 1 + 2 in parallel. With 16-32, run them sequentially.

## Setup (one-time)

```bash
# 1. Clone & fetch
git clone https://github.com/vmallela0/macro-place-challenge-2026.git
cd macro-place-challenge-2026
git fetch --all
git submodule update --init external/MacroPlacement

# 2. Read what's already known about the box / sweep / gotchas
cat experiments/SETUP_NOTES.md
# → especially the 'Known gotchas' section (worktree script self-clobber,
#   Mac-path in run_in_worktree.sh, CPU-torch install workaround if disk
#   is small)

# 3. Python env. If disk > 20 GB, use uv sync; otherwise CPU-only torch:
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Default path (full deps including CUDA torch, ~5 GB):
uv sync

# Tight-disk path (CPU-only torch, ~600 MB):
# uv venv --python 3.11 .venv
# uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cpu "torch>=2.0.0"
# uv pip install --python .venv/bin/python "numpy>=1.20.0" "matplotlib>=3.5.0" "tqdm>=4.65.0" "absl-py>=1.0.0"
# uv pip install --python .venv/bin/python -e . --no-deps

# 4. Sanity: imports work, benchmarks present
.venv/bin/python -c "import torch, macro_place; print('ok', torch.__version__)"
ls external/MacroPlacement/Testcases/ICCAD04/ibm01/   # should show ibm01.pb.txt etc

# 5. Switch off landing_page to avoid worktree collision when v5_combined
#    needs its own worktree
git checkout landing_page    # the experiments/* scripts live here
```

## Create the worktrees you need

```bash
# v5_combined and v5_cluster both off the same parent SHA but different
# branches. Use the v5_combined branch (the runbook lives there) as the
# template, then add v5_cluster.

# Use the existing setup script for v5_combined family (creates 5
# worktrees but you only need v5_combined for Track 1):
bash experiments/v5_setup_worktrees.sh
# CRITICAL: this script does `rm -rf external/MacroPlacement` in the main
# repo if v5_combined is currently checked out. The script's `cd` fails
# silently and the rm runs in the wrong directory. Make sure your main
# checkout is on landing_page (step 5 above) before running.

# v5_cluster (separate worktree, off the v5_cluster branch — already on
# origin from the first box):
git worktree add /tmp/wt_v5_cluster v5_cluster
rm -rf /tmp/wt_v5_cluster/external/MacroPlacement
ln -s "$(pwd)/external/MacroPlacement" /tmp/wt_v5_cluster/external/MacroPlacement

# Patch the Mac path in every worktree's harness (the script has the
# original author's macOS path baked in):
HOME_REPO="$(pwd)"
sed -i "s|/Users/vmallela/Desktop/challenges/macro-place-challenge-2026|$HOME_REPO|" \
  /tmp/wt_*/experiments/run_in_worktree.sh
```

## Use a dedicated results branch (avoid CSV merge conflicts)

The other box pushes results.csv to `landing_page`. To avoid append-merge
conflicts, push your results to a sibling branch `v5_box2_work`:

```bash
git checkout -b v5_box2_work landing_page
# this is the branch you'll commit + push results from
```

I'll merge `v5_box2_work` → `landing_page` from the first box once your
runs complete.

## Track 1: launch v5_combined_pp sweep

```bash
# v5_combined_pp = v5_combined env vars + PLACER_PARALLEL_WORKERS=4.
# Uses /tmp/wt_v5_combined as worktree. 4 worker subprocesses per bench
# = ~5x compute. With (max_parallel)+(workers per bench) products, each
# bench effectively uses 5 cores.
#
# Throttle max_parallel so total cores = (max_parallel * 5) ≤ physical cores.
# Examples:
#   16 physical cores: max_parallel=3 (15 cores busy)
#   32 physical cores: max_parallel=6 (30 cores)
#   64 physical cores: max_parallel=12 (60 cores)

PHYSICAL_CORES=$(lscpu | awk '/^Core\(s\) per socket:/{c=$NF} /^Socket\(s\):/{s=$NF} END{print c*s}')
MAX_PARALLEL_PP=$(( PHYSICAL_CORES / 5 ))   # leave a slot for OS
echo "PHYSICAL_CORES=$PHYSICAL_CORES → MAX_PARALLEL_PP=$MAX_PARALLEL_PP"

# Launch in tmux session 'pp':
tmux new-session -d -s pp \
  "for B in ibm01 ibm02 ibm03 ibm04 ibm06 ibm07 ibm08 ibm09 ibm10 ibm11 \
           ibm12 ibm13 ibm14 ibm15 ibm16 ibm17 ibm18; do
     bash /tmp/wt_v5_combined/experiments/run_in_worktree.sh \
       v5_combined_pp_box2 /tmp/wt_v5_combined \"\$B\" 42 3300 \
       PLACER_ESC_K_REGIONS=4 PLACER_SURR_STRUCTURED=1 PLACER_WARMSTART=1 \
       PLACER_PARALLEL_WORKERS=4 \
       > /tmp/pp_\${B}.out 2>&1 &
     while [ \$(jobs -rp | wc -l) -ge $MAX_PARALLEL_PP ]; do sleep 5; done
   done; wait; echo 'pp DONE'"
```

## Track 2: launch v5_cluster sweep (in parallel if cores allow)

```bash
# v5_cluster = v5_combined env + PLACER_ESC_CLUSTER_FRAC=0.3.
# Single-threaded, so max_parallel = physical cores - whatever Track 1 uses.
MAX_PARALLEL_CL=$(( PHYSICAL_CORES - MAX_PARALLEL_PP * 5 ))
[ "$MAX_PARALLEL_CL" -lt 1 ] && MAX_PARALLEL_CL=1
echo "MAX_PARALLEL_CL=$MAX_PARALLEL_CL"

tmux new-session -d -s cl \
  "for B in ibm01 ibm02 ibm03 ibm04 ibm06 ibm07 ibm08 ibm09 ibm10 ibm11 \
           ibm12 ibm13 ibm14 ibm15 ibm16 ibm17 ibm18; do
     bash /tmp/wt_v5_cluster/experiments/run_in_worktree.sh \
       v5_cluster_box2 /tmp/wt_v5_cluster \"\$B\" 42 3300 \
       PLACER_ESC_K_REGIONS=4 PLACER_SURR_STRUCTURED=1 PLACER_WARMSTART=1 \
       PLACER_ESC_CLUSTER_FRAC=0.3 \
       > /tmp/cl_\${B}.out 2>&1 &
     while [ \$(jobs -rp | wc -l) -ge $MAX_PARALLEL_CL ]; do sleep 5; done
   done; wait; echo 'cluster DONE'"
```

## Track 3: env-tune (after tracks 1 + 2)

```bash
bash experiments/v5_envtune.sh v5_combined hard 1800 8
# Replace v5_combined with v5_cluster if Track 2 won.
# Tests 6 env tweaks × 3 hard benches × 1800s × 8-way ≈ 30 min wall.
```

## Push results back to origin

After each track completes (or every ~30 min, whichever first):

```bash
git add experiments/results.csv experiments/launcher_per_task experiments/last_status.txt
git commit -m "v5_box2: <track> results"
git push origin v5_box2_work
```

**Do NOT push to:**
- `main` / `master` — never
- `landing_page` — first box owns this branch (would conflict)
- `v5_cluster` — first box's experimental branch; only push results to v5_box2_work

## Reporting

Once your tracks complete (or every ~2 h), summarize:
- Per-track: what completed, what mean cost, vs v4 mean (1.0186), vs the
  other box's v5_combined mean (when known)
- Anything that broke (INVALID, OOM, divergence)

Keep `experiments/results.csv` and `last_status.txt` current on
`v5_box2_work` so the first box can pull and merge into the analysis.

## What success looks like

- `v5_combined_pp` and `v5_cluster` both have 17/17 VALID rows on
  `v5_box2_work`
- At least one of them has mean cost < `v5_combined`'s mean (which the
  first box will produce around 03:46 UTC)
- `v5_envtune` row(s) point to which env stacking buys the most on hard
  benches

## What NOT to do

- Don't touch `main` / `master`
- Don't push to `landing_page` (first box owns it)
- Don't re-run any of the 5 main-sweep branches that the first box is
  already doing at seed 42, 3300 s
- Don't modify `submissions/vmallela_v2/run.sh` or `placer.py` — those are
  the canonical submission code; experimental changes go in env vars or in
  experiment-specific branches
- Don't skip the `landing_page` checkout step before running
  `v5_setup_worktrees.sh` — it WILL silently delete the main repo's
  `external/MacroPlacement` if you're on `v5_combined` when you run it

That's it. Set up, launch, push, report.
