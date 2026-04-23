"""exp_v66: per-macro binary line search — each soft picks own best delta."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util
_s = importlib.util.spec_from_file_location("_v36", str(Path(__file__).resolve().parent / "placer_exp_v36.py"))
_m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)

import time
import numpy as np


def binary_search_soft(pos_np, benchmark, incr_eval, max_time, verbose=False):
    """For each soft, probe exponentially-spaced deltas; take best in best direction."""
    n_hard = benchmark.num_hard_macros
    n_total = benchmark.macro_positions.shape[0]
    cw, ch = float(benchmark.canvas_width), float(benchmark.canvas_height)
    macro_fixed = benchmark.macro_fixed.numpy() if hasattr(benchmark.macro_fixed, 'numpy') else benchmark.macro_fixed
    soft_movable = [n_hard + i for i in range(n_total - n_hard) if not macro_fixed[n_hard + i]]
    if not soft_movable: return pos_np, incr_eval.get_proxy_cost()
    net_count = np.array([len(incr_eval.macro_nets[i]) for i in range(n_total)])
    soft_sorted = sorted(soft_movable, key=lambda i: -net_count[i])

    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1),
            (0.7, 0.7), (-0.7, 0.7), (0.7, -0.7), (-0.7, -0.7)]
    # Log-spaced deltas
    deltas = [0.03, 0.08, 0.2, 0.5, 1.2, 3.0]

    current_cost = incr_eval.get_proxy_cost()
    best_cost = current_cost
    t0 = time.time()
    n_moves = 0
    while time.time() - t0 < max_time:
        improved = False
        for i in soft_sorted:
            if time.time() - t0 > max_time: break
            ox = float(incr_eval.macro_pos[i, 0])
            oy = float(incr_eval.macro_pos[i, 1])
            best_c = current_cost
            best_nx, best_ny = None, None
            # Probe all (dir, delta) combos
            for dx, dy in dirs:
                for d in deltas:
                    nx = float(max(0, min(cw, ox + d * dx)))
                    ny = float(max(0, min(ch, oy + d * dy)))
                    if abs(nx - ox) < 1e-4 and abs(ny - oy) < 1e-4: continue
                    c = incr_eval.move_macro(i, nx, ny)
                    if c < best_c:
                        best_c = c
                        best_nx, best_ny = nx, ny
                    incr_eval.undo_move()
            if best_nx is not None:
                incr_eval.move_macro(i, best_nx, best_ny)
                current_cost = best_c
                if best_c < best_cost: best_cost = best_c
                n_moves += 1
                improved = True
        if not improved: break

    if verbose:
        print(f"    binary: {n_moves} moves, cost {best_cost:.6f}")
    return pos_np, best_cost


import _softmacro
_softmacro.soft_macro_cd = binary_search_soft
_m.soft_macro_cd = binary_search_soft


class OptimalPlacer(_m.OptimalPlacer):
    TOTAL_BUDGET = 220
    LEGALIZE_BUDGET = 75
