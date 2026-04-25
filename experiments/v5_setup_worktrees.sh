#!/bin/bash
# experiments/v5_setup_worktrees.sh
# Set up the v5_* worktrees in /tmp with symlinked submodule.
# Run this once on each new machine (e.g. the 64-thread sweep box) before
# launching v5_full_sweep.sh.

set -u
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SUBMODULE="$REPO_ROOT/external/MacroPlacement"

if [ ! -d "$SUBMODULE/Testcases" ]; then
  echo "[setup] initializing main submodule (one-time)"
  cd "$REPO_ROOT" && git submodule update --init external/MacroPlacement
fi

declare -a BRANCHES=(
  "v5_cells_skip_v4:/tmp/wt_v5_cells"
  "v5_escape_v2:/tmp/wt_v5_escape"
  "v5_surrogate_struct:/tmp/wt_v5_surr"
  "v5_warmstart:/tmp/wt_v5_warm"
  "v5_combined:/tmp/wt_v5_combined"
)

for entry in "${BRANCHES[@]}"; do
  IFS=':' read -r branch wt <<< "$entry"
  if [ -d "$wt" ]; then
    echo "[setup] $wt already exists; skipping (remove first to rebuild)"
    continue
  fi
  echo "[setup] creating worktree $wt for branch $branch"
  cd "$REPO_ROOT" && git worktree add --no-checkout "$wt" "$branch"
  cd "$wt"
  git checkout "$branch" -- .
  rm -rf external/MacroPlacement
  mkdir -p external
  ln -s "$SUBMODULE" external/MacroPlacement
  echo "[setup]   ok: $(du -sh "$wt" | awk '{print $1}')"
done

echo "[setup] all worktrees ready. Run experiments/v5_full_sweep.sh next."
