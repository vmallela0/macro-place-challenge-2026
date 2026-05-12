#!/bin/bash
# zeus_status — check on detached runs launched via zeus_run_detached.sh.
#
# Usage:
#   bash scripts/zeus_status.sh              # status of all known runs
#   bash scripts/zeus_status.sh <name>       # status of one named run
#   bash scripts/zeus_status.sh tail <name>  # tail -f the run log

set -u

ROOT="${ZEUS_RUN_DIR:-$HOME/zeus_runs}"

if [ "${1:-}" = "tail" ]; then
    NAME="${2:-}"
    if [ -z "$NAME" ]; then echo "Usage: $0 tail <name>" >&2; exit 2; fi
    LATEST="$ROOT/latest_$NAME"
    if [ ! -e "$LATEST" ]; then echo "no run named '$NAME'" >&2; exit 1; fi
    exec tail -f "$LATEST/run.log"
fi

show_one() {
    local dir="$1"
    local pidfile="$dir/pid"
    local log="$dir/run.log"
    if [ ! -f "$pidfile" ]; then echo "no pid file"; return; fi
    local pid
    pid=$(cat "$pidfile")
    local state
    if kill -0 "$pid" 2>/dev/null; then state="RUNNING"; else state="EXITED"; fi
    local size
    size=$(du -h "$log" 2>/dev/null | awk '{print $1}')
    local started
    started=$(stat -c '%y' "$pidfile" 2>/dev/null | cut -d. -f1)
    echo "  pid:     $pid ($state)"
    echo "  started: $started"
    echo "  dir:     $dir"
    echo "  log:     $log ($size)"
    echo "  tail:"
    tail -5 "$log" 2>/dev/null | sed 's/^/    /'
}

if [ -n "${1:-}" ]; then
    NAME="$1"
    LATEST="$ROOT/latest_$NAME"
    if [ ! -e "$LATEST" ]; then echo "no run named '$NAME'" >&2; exit 1; fi
    echo "== $NAME =="
    show_one "$(readlink -f "$LATEST")"
    exit 0
fi

if [ ! -d "$ROOT" ]; then
    echo "no runs (no $ROOT dir yet)"
    exit 0
fi
for dir in "$ROOT"/*/; do
    [ -d "$dir" ] || continue
    name=$(basename "$dir")
    [[ "$name" == latest_* ]] && continue
    echo "== $name =="
    show_one "$dir"
    echo
done
