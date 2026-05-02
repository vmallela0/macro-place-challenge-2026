"""Tiny RUNLOG.md appender. Used by every v8 phase + scaffolding.

One function: log(component, event, details). Atomic single-line append
with UTC timestamp. Safe to call from worker processes (each open+write+close).
"""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path

_RUNLOG_PATH = Path(__file__).resolve().parent / "RUNLOG.md"


def log(component: str, event: str, details: str = "") -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    line = f"[{ts}] {component} {event}"
    if details:
        line += f" {details}"
    line += "\n"
    with _RUNLOG_PATH.open("a") as f:
        f.write(line)
