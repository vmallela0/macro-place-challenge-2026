#!/bin/bash
# zeus_run_detached — run any command fully detached so it survives:
#   - claude dying
#   - SSH dropping
#   - the laptop closing
#
# Uses setsid + nohup + redirect-all + disown. The launched process
# is reparented to init (PID 1) and writes everything to a log dir.
#
# Usage:
#   bash scripts/zeus_run_detached.sh <run_name> <command...>
#
# Example:
#   bash scripts/zeus_run_detached.sh clean_ab bash scripts/zeus_clean_ab.sh
#
# After launch the script prints:
#   - PID file path
#   - log path
#   - one-liner status check
#
# To inspect later:
#   bash scripts/zeus_status.sh                # checks every known run
#   bash scripts/zeus_status.sh <run_name>     # checks one run

set -u

if [ $# -lt 2 ]; then
    echo "Usage: $0 <run_name> <command...>" >&2
    exit 2
fi

NAME="$1"; shift
REPO="$(cd "$(dirname "$0")/.." && pwd)"
RUN_DIR="${ZEUS_RUN_DIR:-$HOME/zeus_runs}/$NAME-$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"

LOG="$RUN_DIR/run.log"
PIDFILE="$RUN_DIR/pid"
CMDFILE="$RUN_DIR/cmd"
echo "$@" > "$CMDFILE"
echo "cwd=$REPO" >> "$CMDFILE"

cd "$REPO"

# setsid → new session (immune to SIGHUP from parent tty)
# nohup  → ignore SIGHUP (belt-and-suspenders)
# &      → background
# stdin /dev/null, stdout/stderr → log
# disown → remove from shell's job table so shell exit doesn't reap
setsid nohup "$@" </dev/null >"$LOG" 2>&1 &
PID=$!
disown "$PID" 2>/dev/null || true
echo "$PID" > "$PIDFILE"

# Symlink ~/zeus_runs/latest_<name> for easy access.
LATEST="${ZEUS_RUN_DIR:-$HOME/zeus_runs}/latest_$NAME"
ln -sfn "$RUN_DIR" "$LATEST"

echo "launched detached:"
echo "  name:    $NAME"
echo "  pid:     $PID"
echo "  run dir: $RUN_DIR"
echo "  log:     $LOG"
echo "  latest:  $LATEST"
echo
echo "check status:"
echo "  bash $REPO/scripts/zeus_status.sh $NAME"
echo
echo "tail log:"
echo "  tail -f $LOG"
