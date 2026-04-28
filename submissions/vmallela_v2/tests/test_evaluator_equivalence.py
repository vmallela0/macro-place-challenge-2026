"""Verify vmallela's IncrementalEvaluator agrees numerically with the batch
TILOS PlacementCost evaluator (via macro_place.objective.compute_proxy_cost).

Methodology:
  1. For each benchmark: load; build IncrementalEvaluator from the initial plc.
  2. Assert incr.get_proxy_cost() matches compute_proxy_cost(initial_placement, ..)
     at the initial state.
  3. Apply 100 random hard-macro moves (legal = inside canvas; overlap-free not
     required — we test evaluator equivalence, not placement validity).
     After each move, compute both evaluators' proxy costs and record the diff.
  4. Report max absolute and relative diffs per benchmark.
"""
import os
import sys
import time
import random
import importlib.util
from pathlib import Path

# Force single-threaded BLAS so this test does not steal CPU from the main sweep.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost

# Load vmallela v1 placer to access IncrementalEvaluator
_vmallela_path = ROOT / "submissions" / "vmallela" / "placer.py"
_spec = importlib.util.spec_from_file_location("_vmallela_placer", str(_vmallela_path))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
IncrementalEvaluator = _mod.IncrementalEvaluator

TESTCASE_ROOT = ROOT / "external" / "MacroPlacement" / "Testcases" / "ICCAD04"


def _test_one(name: str, n_moves: int = 100, abs_tol: float = 1e-4):
    """Run equivalence check on one benchmark. Returns dict with metrics."""
    t_start = time.time()
    print(f"\n=== {name} ===", flush=True)

    bench_dir = TESTCASE_ROOT / name
    benchmark, plc = load_benchmark_from_dir(str(bench_dir))
    n_hard = benchmark.num_hard_macros
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    print(f"  loaded: {n_hard} hard macros, canvas {cw:.1f}x{ch:.1f}", flush=True)

    # --- Initial-state equivalence check ---------------------------------
    # Build IncrementalEvaluator first — it mutates plc via _set_placement.
    incr = IncrementalEvaluator(plc, benchmark)
    incr_cost_initial = incr.get_proxy_cost()

    # compute_proxy_cost also calls _set_placement, so it re-syncs plc.
    # Pass the same placement tensor incr was built from.
    placement = benchmark.macro_positions.clone()
    batch_initial = compute_proxy_cost(placement, benchmark, plc)
    batch_cost_initial = float(batch_initial["proxy_cost"])

    init_diff = abs(batch_cost_initial - incr_cost_initial)
    init_rel = init_diff / max(abs(batch_cost_initial), 1e-9)
    print(f"  initial: batch={batch_cost_initial:.8f} incr={incr_cost_initial:.8f} "
          f"abs_diff={init_diff:.2e} rel_diff={init_rel:.2e}", flush=True)

    # Re-sync incr after compute_proxy_cost mutated plc (safety — though incr
    # holds its own state, plc state for subsequent batch calls is what matters).
    # Actually we'll sync plc back to what incr thinks before each batch eval
    # below — see inside the loop.

    # --- Random-move loop ------------------------------------------------
    # Movable hard macros only (soft macros are 0-size and handled elsewhere).
    movable = [i for i in range(n_hard) if not bool(benchmark.macro_fixed[i])]
    if len(movable) == 0:
        return dict(bench=name, status="NO_MOVABLE_MACROS")
    print(f"  {len(movable)} movable hard macros", flush=True)

    rng = random.Random(42)
    max_abs = init_diff
    max_rel = init_rel
    worst = None
    diverged_move = None

    # Record initial placement so we can rebuild tensor after each move
    current_placement = placement.clone()

    for i in range(n_moves):
        # Random movable macro
        m = rng.choice(movable)
        # Current position
        cx = float(incr.macro_pos[m, 0])
        cy = float(incr.macro_pos[m, 1])
        w = float(incr.macro_w[m])
        h = float(incr.macro_h[m])
        # Random small displacement, keep on-canvas (allow overlap, we only test eval)
        step_x = rng.uniform(-0.2 * cw, 0.2 * cw)
        step_y = rng.uniform(-0.2 * ch, 0.2 * ch)
        nx = max(0.0, min(cw - w, cx + step_x))
        ny = max(0.0, min(ch - h, cy + step_y))

        # Apply to incr
        incr.move_macro(m, nx, ny)
        incr_cost = incr.get_proxy_cost()

        # Apply to placement tensor
        current_placement[m, 0] = nx
        current_placement[m, 1] = ny

        # Batch cost (this mutates plc to match current_placement, which is fine)
        batch = compute_proxy_cost(current_placement, benchmark, plc)
        batch_cost = float(batch["proxy_cost"])

        diff = abs(batch_cost - incr_cost)
        rel = diff / max(abs(batch_cost), 1e-9)
        if diff > max_abs:
            max_abs = diff
            max_rel = rel
            worst = dict(move=i, macro=m, new_x=nx, new_y=ny,
                        batch=batch_cost, incr=incr_cost, abs_diff=diff, rel_diff=rel)
        if diff > abs_tol and diverged_move is None:
            diverged_move = dict(move=i, macro=m, new_x=nx, new_y=ny,
                                batch=batch_cost, incr=incr_cost, abs_diff=diff)
            print(f"  DIVERGENCE at move {i}: macro={m} new=({nx:.2f},{ny:.2f}) "
                  f"batch={batch_cost:.8f} incr={incr_cost:.8f} diff={diff:.2e}",
                  flush=True)
            break

        if (i + 1) % 25 == 0:
            elapsed = time.time() - t_start
            print(f"  move {i + 1}/{n_moves}: max_abs_diff={max_abs:.2e} "
                  f"max_rel_diff={max_rel:.2e} t={elapsed:.1f}s", flush=True)

    elapsed = time.time() - t_start
    return dict(
        bench=name,
        status="DIVERGED" if diverged_move else "OK",
        n_moves=i + 1,
        init_abs_diff=init_diff,
        init_rel_diff=init_rel,
        max_abs_diff=max_abs,
        max_rel_diff=max_rel,
        worst=worst,
        diverged=diverged_move,
        elapsed_s=elapsed,
    )


def main():
    results = []
    for name in ["ibm01", "ibm06", "ibm10"]:
        try:
            r = _test_one(name, n_moves=100, abs_tol=1e-4)
        except Exception as e:
            import traceback
            r = dict(bench=name, status="EXCEPTION", error=str(e),
                     traceback=traceback.format_exc())
        results.append(r)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for r in results:
        print(f"  {r['bench']}: {r['status']}", end="")
        if r["status"] == "OK":
            print(f"  max_abs={r['max_abs_diff']:.2e}  max_rel={r['max_rel_diff']:.2e}  "
                  f"moves={r['n_moves']}  t={r['elapsed_s']:.1f}s")
        elif r["status"] == "DIVERGED":
            d = r["diverged"]
            print(f"  at move {d['move']} macro={d['macro']} "
                  f"batch={d['batch']:.6f} incr={d['incr']:.6f} diff={d['abs_diff']:.2e}")
        else:
            print(f"  {r.get('error', '')}")

    # Pass if ALL benches OK and all max_abs_diff < 1e-4
    all_ok = all(r["status"] == "OK" and r["max_abs_diff"] < 1e-4 for r in results)
    print(f"\nOverall: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
