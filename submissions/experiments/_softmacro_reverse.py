"""Soft-macro CD with REVERSE macro ordering (idea A1 control).

Diff from _softmacro.py: sort softs by ASCENDING net_count instead of
descending. Tests whether the 'most-connected first' heuristic actually
helps — if reverse tied or beat the default, the heuristic is a wash.
"""
import time
import numpy as np


def soft_macro_cd_reverse(pos_np, benchmark, incr_eval, max_time, verbose=False):
    n_hard = benchmark.num_hard_macros
    n_total = benchmark.macro_positions.shape[0]
    cw, ch = float(benchmark.canvas_width), float(benchmark.canvas_height)
    incr_eval.sync_positions(pos_np)

    macro_fixed = benchmark.macro_fixed.numpy() if hasattr(benchmark.macro_fixed, 'numpy') else benchmark.macro_fixed
    soft_movable = [n_hard + i for i in range(n_total - n_hard) if not macro_fixed[n_hard + i]]
    if not soft_movable:
        return pos_np, incr_eval.get_proxy_cost()

    net_count = np.array([len(incr_eval.macro_nets[i]) for i in range(n_total)])
    soft_sorted = sorted(soft_movable, key=lambda i: +net_count[i])  # ascending

    sizes = np.zeros((n_total, 2), dtype=np.float64)
    if hasattr(benchmark.macro_sizes, 'numpy'):
        s = benchmark.macro_sizes.numpy()
        sizes[:s.shape[0]] = s

    current_cost = incr_eval.get_proxy_cost()
    best_cost = current_cost

    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1),
            (0.7, 0.7), (-0.7, 0.7), (0.7, -0.7), (-0.7, -0.7)]
    deltas = [1.0, 0.5, 0.25, 0.12, 0.06, 0.03]

    t0 = time.time()
    pass_num = 0
    n_moves = 0
    while time.time() - t0 < max_time:
        pass_num += 1
        pass_improved = False
        for delta in deltas:
            if time.time() - t0 > max_time:
                break
            for i in soft_sorted:
                if time.time() - t0 > max_time:
                    break
                ox = float(incr_eval.macro_pos[i, 0])
                oy = float(incr_eval.macro_pos[i, 1])
                hw = sizes[i, 0] / 2
                hh = sizes[i, 1] / 2
                best_dir_cost = current_cost
                best_nx = best_ny = None
                for dx, dy in dirs:
                    nx = float(np.clip(ox + delta * dx, hw, cw - hw))
                    ny = float(np.clip(oy + delta * dy, hh, ch - hh))
                    if abs(nx - ox) < 0.001 and abs(ny - oy) < 0.001:
                        continue
                    c = incr_eval.move_macro(i, nx, ny)
                    if c < best_dir_cost:
                        best_dir_cost = c
                        best_nx = nx
                        best_ny = ny
                    incr_eval.undo_move()
                if best_nx is not None:
                    incr_eval.move_macro(i, best_nx, best_ny)
                    current_cost = best_dir_cost
                    if best_dir_cost < best_cost:
                        best_cost = best_dir_cost
                    pass_improved = True
                    n_moves += 1
        if not pass_improved:
            break

    if verbose:
        print(f"    soft_macro_cd_reverse: {pass_num} passes, {n_moves} moves, cost {current_cost:.6f}")
    return pos_np, best_cost
