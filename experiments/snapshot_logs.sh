#!/bin/bash
# experiments/snapshot_logs.sh — copy ephemeral /tmp launcher logs into repo.
# Called at every BRANCH_DONE event so artifacts survive reboots and are
# committable to git.
set -u
REPO=/home/vedu_mallela/macro-place-challenge-2026
mkdir -p "$REPO/experiments/launcher_per_task"

# Main launcher log (sweep + post-sweep)
[ -f /tmp/v5_sweep.out ] && cp /tmp/v5_sweep.out "$REPO/experiments/launcher.log"
[ -f /tmp/v5_post.out ] && cp /tmp/v5_post.out "$REPO/experiments/launcher_post.log"

# Per-task harness outputs (one per launched run)
for f in /tmp/v5_sweep_*.out /tmp/post_*.out /tmp/envtune_*.out /tmp/winner_*.out; do
  [ -f "$f" ] && cp "$f" "$REPO/experiments/launcher_per_task/" 2>/dev/null
done

# Sweep status snapshot
bash "$REPO/experiments/sweep_status.sh" > "$REPO/experiments/last_status.txt" 2>&1

echo "[snapshot] $(date -u +%Y-%m-%dT%H:%M:%SZ) — copied launcher + per-task logs"
