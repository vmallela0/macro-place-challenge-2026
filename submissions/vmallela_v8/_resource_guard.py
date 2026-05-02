"""Resource guard: hard wall-time timeout + memory soft check.

Usage:
    with phase_guard("phase_a", max_wall_seconds=3500, max_memory_gb=18):
        ... do work ...

Wall-time: SIGALRM-based on Linux. The block is interrupted with
TimeoutError when the deadline passes. Falls back to "no-op" on platforms
without SIGALRM.

Memory: best-effort pre-check via psutil if available. Logs a warning to
RUNLOG if available memory < max_memory_gb at entry; does not enforce
mid-run (Python doesn't have a clean way to enforce this).
"""
from __future__ import annotations
import contextlib
import signal
from typing import Iterator

from _runlog import log


@contextlib.contextmanager
def phase_guard(
    name: str,
    max_wall_seconds: int,
    max_memory_gb: float | None = None,
) -> Iterator[None]:
    log("guard", f"enter {name}",
        f"max_wall={max_wall_seconds}s max_mem={max_memory_gb}GB")

    if max_memory_gb is not None:
        try:
            import psutil
            avail_gb = psutil.virtual_memory().available / (1024 ** 3)
            if avail_gb < max_memory_gb:
                log("guard", "low_memory_warning",
                    f"avail={avail_gb:.1f}GB < threshold={max_memory_gb}GB")
        except ImportError:
            pass

    have_alarm = hasattr(signal, "SIGALRM")
    prev_handler = None
    if have_alarm:
        def _handler(signum, frame):
            raise TimeoutError(
                f"phase {name} exceeded {max_wall_seconds}s wall-time")
        prev_handler = signal.signal(signal.SIGALRM, _handler)
        signal.alarm(max_wall_seconds)

    try:
        yield
    except TimeoutError as e:
        log("guard", f"timeout {name}", str(e))
        raise
    except Exception as e:
        log("guard", f"error {name}", f"{type(e).__name__}: {e}")
        raise
    finally:
        if have_alarm:
            signal.alarm(0)
            if prev_handler is not None:
                signal.signal(signal.SIGALRM, prev_handler)
        log("guard", f"exit {name}", "")
