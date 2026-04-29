"""Determinism regression test (cross-hardware portability).

Runs v6 placer twice on ibm01 with the same seed and a short budget.
Asserts that the two runs produce proxy costs within 0.01 of each other.

What we are NOT testing
-----------------------
Bit-reproducibility. v4's pipeline has 13 wall-clock-bound loops that
complete a slightly different number of iterations each run (OS scheduler
jitter). That residual jitter is documented at ~±0.005 per benchmark in
v2's HANDOFF.md. Achieving bit-reproducibility would require replacing
wall-clock budgets with iteration-count budgets — a substantial refactor
(T4.2 in the plan). Out of scope for this test.

What we ARE testing
-------------------
Gross non-determinism that would indicate STRUCTURAL bugs:

  - Multi-thread BLAS reduction order (OpenBLAS / MKL on grader)
  - cuDNN non-deterministic algorithm selection
  - Dict iteration order (PYTHONHASHSEED unset)
  - CUDA RNG unseeded (torch.manual_seed only seeds CPU)

When any of these fire, the gap grows from ±0.005 (wall-clock jitter) to
±0.02 - 0.30 (basin-divergence). v2 saw a 0.30 gap on the grader. A
passing test here (gap < 0.01 between two M-Pro runs with different
multi-thread settings) is evidence we've defended against the structural
classes of non-determinism that caused v2's gap.

We use a small budget (60s/worker × 2 workers ~= 1 min total) so the
test runs in CI-friendly time.
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def _run_once(seed=42, budget=60, n_workers=2):
    """Invoke the placer EXACTLY as the maintainer would (`uv run evaluate
    <path>` semantics). Captures stdout, returns the proxy_cost line."""
    env = os.environ.copy()
    # Deliberately DO NOT set OMP_NUM_THREADS etc. — we want to test that
    # the placer self-applies the locked env from inside placer.py.
    for key_to_remove in [
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "PYTHONHASHSEED",
    ]:
        env.pop(key_to_remove, None)
    env["PLACER_TOTAL_BUDGET"] = str(budget)
    env["PLACER_V6_WORKERS"] = str(n_workers)
    env["PLACER_V6_GPU_WORKERS"] = "0"   # skip GPU for fast determinism test
    env["PLACER_V6_CONSENSUS"] = "0"     # skip consensus too (we only need
                                          # to verify worker determinism)

    cmd = [str(ROOT / ".venv/bin/python"), "-m", "macro_place.evaluate",
           str(ROOT / "submissions/vmallela_v6/placer.py"),
           "--benchmark", "ibm01"]
    out = subprocess.run(cmd, env=env, cwd=str(ROOT),
                         capture_output=True, text=True, timeout=600)
    if out.returncode != 0:
        print("STDOUT:", out.stdout[-2000:])
        print("STDERR:", out.stderr[-2000:])
        raise RuntimeError(f"evaluate exited with {out.returncode}")
    # Pull the final 'proxy=...' line.
    proxy_line = None
    for line in out.stdout.splitlines():
        if line.lstrip().startswith("proxy="):
            proxy_line = line.strip()
    if proxy_line is None:
        print("STDOUT (no proxy line):", out.stdout[-2000:])
        raise RuntimeError("no proxy=... line found")
    return proxy_line


def _parse_proxy(line):
    """Extract proxy_cost from a line like 'proxy=0.9170  (wl=...)  VALID  [...]'."""
    import re
    m = re.search(r"proxy=([0-9]+\.[0-9]+)", line)
    return float(m.group(1)) if m else float("nan")


def test_two_runs_same_seed_within_jitter_budget():
    """Run v6 twice, identical seed/config, assert proxy cost gap < 0.01.

    Bar: 0.01. Documented residual wall-clock jitter is ±0.005 per
    benchmark; we double that for safety. A gap > 0.01 indicates
    structural non-determinism (multi-thread BLAS, cuDNN, dict order)
    has slipped through our defenses.
    """
    print("  Running v6 placer twice (60s/worker, 2 workers, same seed)...")
    line1 = _run_once(seed=42, budget=60, n_workers=2)
    print(f"  run 1: {line1}")
    line2 = _run_once(seed=42, budget=60, n_workers=2)
    print(f"  run 2: {line2}")

    p1 = _parse_proxy(line1)
    p2 = _parse_proxy(line2)
    gap = abs(p1 - p2)
    print(f"  gap = |{p1:.4f} - {p2:.4f}| = {gap:.4f}")

    assert gap < 0.01, (
        f"DETERMINISM REGRESSION — gap {gap:.4f} exceeds the 0.01 jitter\n"
        f"budget. Inherent wall-clock jitter is ±0.005 per benchmark; a\n"
        f"larger gap indicates structural non-determinism (multi-thread\n"
        f"BLAS, cuDNN algorithm selection, or unseeded CUDA RNG) has\n"
        f"escaped our defenses. On the grader (16-core EPYC), the gap\n"
        f"can grow to 0.10-0.30 (which is what hit v2). Investigate via:\n"
        f"  - check threadpoolctl.threadpool_info() reports 1 thread\n"
        f"  - check torch.backends.cudnn.deterministic == True\n"
        f"  - check PYTHONHASHSEED set in subprocess env")
    print(f"  ✓ within jitter budget ({gap:.4f} < 0.01)")


if __name__ == "__main__":
    test_two_runs_same_seed_within_jitter_budget()
    print("ALL OK — placer non-determinism is within the jitter budget.")
