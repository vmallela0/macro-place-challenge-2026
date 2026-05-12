#!/bin/bash
# zeus_claude — persistent claude launcher.
#
# Wraps `claude` in a tmux session so an SSH drop / closed laptop /
# network blip cannot kill the agent. On reconnect, the user just
# runs this script again and re-attaches to the same session.
#
# Usage:
#   bash scripts/zeus_claude.sh                # attach if exists, else create
#   bash scripts/zeus_claude.sh new            # force a NEW session (kill old first)
#   bash scripts/zeus_claude.sh kill           # kill the session and exit
#   bash scripts/zeus_claude.sh status         # show whether session exists
#
# Once attached, detach with Ctrl-B then D (do NOT close the terminal
# normally — Ctrl-D kills the inner shell). After detach, claude keeps
# running in the background; reattach with this same script.

set -u
SESSION="${ZEUS_TMUX_SESSION:-zeus}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
CLAUDE_CMD="${ZEUS_CLAUDE_CMD:-claude --dangerously-skip-permissions}"

cmd="${1:-attach}"

if ! command -v tmux >/dev/null 2>&1; then
    echo "ERROR: tmux not installed. apt install tmux" >&2
    exit 1
fi

session_exists() { tmux has-session -t "$SESSION" 2>/dev/null; }

case "$cmd" in
    status)
        if session_exists; then
            echo "tmux session '$SESSION' is RUNNING."
            tmux list-windows -t "$SESSION"
        else
            echo "tmux session '$SESSION' is NOT running."
        fi
        ;;
    kill)
        if session_exists; then
            tmux kill-session -t "$SESSION"
            echo "killed tmux session '$SESSION'."
        else
            echo "no session '$SESSION' to kill."
        fi
        ;;
    new)
        if session_exists; then
            tmux kill-session -t "$SESSION"
            echo "killed old session, creating fresh."
        fi
        cd "$REPO"
        # detached create
        tmux new-session -d -s "$SESSION" -c "$REPO" "$CLAUDE_CMD"
        echo "created tmux session '$SESSION' running: $CLAUDE_CMD"
        echo "attaching now (detach with Ctrl-B D)..."
        exec tmux attach -t "$SESSION"
        ;;
    attach|"")
        if session_exists; then
            echo "attaching to existing tmux session '$SESSION'"
            echo "(detach with Ctrl-B D to leave claude running in background)"
            exec tmux attach -t "$SESSION"
        else
            echo "no existing session — creating '$SESSION' with: $CLAUDE_CMD"
            cd "$REPO"
            tmux new-session -d -s "$SESSION" -c "$REPO" "$CLAUDE_CMD"
            sleep 1
            exec tmux attach -t "$SESSION"
        fi
        ;;
    *)
        echo "Usage: $0 {attach|new|kill|status}" >&2
        exit 2
        ;;
esac
