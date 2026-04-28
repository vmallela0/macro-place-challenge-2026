"""exp_v63: soft visit order by total-net-weight descending."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util
_s = importlib.util.spec_from_file_location("_v36", str(Path(__file__).resolve().parent / "placer_exp_v36.py"))
_m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)

import time
import numpy as np


def net_weight_soft(pos_np, benchmark, incr_eval, max_time, verbose=False):
    n_hard = benchmark.num_hard_macros
    n_total = benchmark.macro_positions.shape[0]
    cw, ch = float(benchmark.canvas_width), float(benchmark.canvas_height)
    macro_fixed = benchmark.macro_fixed.numpy() if hasattr(benchmark.macro_fixed, 'numpy') else benchmark.macro_fixed
    soft_movable = [n_hard + i for i in range(n_total - n_hard) if not macro_fixed[n_hard + i]]
    if not soft_movable:
        return pos_np, incr_eval.get_proxy_cost()

    # Sort by SUM of net_weights this macro participates in
    total_weight = np.zeros(n_total)
    for m in range(n_total):
        for nid in incr_eval.macro_nets[m]:
            total_weight[m] += float(incr_eval.net_weight[nid])
    soft_sorted = sorted(soft_movable, key=lambda i: -total_weight[i])

    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1),
            (0.7, 0.7), (-0.7, 0.7), (0.7, -0.7), (-0.7, -0.7)]
    deltas = [1.0, 0.5, 0.25, 0.12, 0.06]

    current_cost = incr_eval.get_proxy_cost()
    best_cost = current_cost
    t0 = time.time()
    n_moves = 0
    while time.time() - t0 < max_time:
        improved = False
        for delta in deltas:
            if time.time() - t0 > max_time: break
            for i in soft_sorted:
                if time.time() - t0 > max_time: break
                ox = float(incr_eval.macro_pos[i, 0])
                oy = float(incr_eval.macro_pos[i, 1])
                best_c = current_cost
                best_nx, best_ny = None, None
                for dx, dy in dirs:
                    nx = float(max(0, min(cw, ox + delta * dx)))
                    ny = float(max(0, min(ch, oy + delta * dy)))
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
        print(f"    net_weight_order: {n_moves} moves, cost {best_cost:.6f}")
    return pos_np, best_cost


import _softmacro
_softmacro.soft_macro_cd = net_weight_soft
_m.soft_macro_cd = net_weight_soft


class OptimalPlacer(_m.OptimalPlacer):
    TOTAL_BUDGET = 220
    LEGALIZE_BUDGET = 75
