"""Tabu soft CD. Short-term memory prevents re-moving recently moved macros,
forces exploration of less-touched subsets of softs.
"""
import time
import numpy as np
from collections import deque


def tabu_soft_cd(pos_np, benchmark, incr_eval, max_time, tabu_tenure=40,
                 verbose=False):
    n_hard = benchmark.num_hard_macros
    n_total = benchmark.macro_positions.shape[0]
    cw, ch = float(benchmark.canvas_width), float(benchmark.canvas_height)
    macro_fixed = benchmark.macro_fixed.numpy() if hasattr(benchmark.macro_fixed, 'numpy') else benchmark.macro_fixed
    soft_movable = [n_hard + i for i in range(n_total - n_hard) if not macro_fixed[n_hard + i]]
    if not soft_movable:
        return pos_np, incr_eval.get_proxy_cost()

    net_count = np.array([len(incr_eval.macro_nets[i]) for i in range(n_total)])
    soft_sorted = sorted(soft_movable, key=lambda i: -net_count[i])

    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1),
            (0.7, 0.7), (-0.7, 0.7), (0.7, -0.7), (-0.7, -0.7)]
    deltas = [1.0, 0.5, 0.25, 0.12, 0.06]

    current_cost = incr_eval.get_proxy_cost()
    best_cost = current_cost
    tabu = deque(maxlen=tabu_tenure)
    tabu_set = set()

    t0 = time.time()
    n_moves = 0
    n_aspirated = 0
    pass_num = 0

    while time.time() - t0 < max_time:
        pass_num += 1
        improved = False
        for delta in deltas:
            if time.time() - t0 > max_time:
                break
            for i in soft_sorted:
                if time.time() - t0 > max_time:
                    break
                is_tabu = i in tabu_set
                ox = float(incr_eval.macro_pos[i, 0])
                oy = float(incr_eval.macro_pos[i, 1])
                best_dir_c = current_cost
                best_nx, best_ny = None, None
                for dx, dy in dirs:
                    nx = float(max(0, min(cw, ox + delta * dx)))
                    ny = float(max(0, min(ch, oy + delta * dy)))
                    if abs(nx - ox) < 1e-4 and abs(ny - oy) < 1e-4:
                        continue
                    c = incr_eval.move_macro(i, nx, ny)
                    if c < best_dir_c:
                        best_dir_c = c
                        best_nx, best_ny = nx, ny
                    incr_eval.undo_move()
                if best_nx is None:
                    continue
                # Accept if: improves AND (not tabu OR aspiration: best ever)
                if is_tabu:
                    if best_dir_c < best_cost - 1e-8:
                        n_aspirated += 1
                    else:
                        continue
                incr_eval.move_macro(i, best_nx, best_ny)
                current_cost = best_dir_c
                if best_dir_c < best_cost:
                    best_cost = best_dir_c
                # Push to tabu
                if len(tabu) == tabu_tenure:
                    old = tabu.popleft()
                    tabu_set.discard(old)
                tabu.append(i)
                tabu_set.add(i)
                n_moves += 1
                improved = True
        if not improved:
            # Clear tabu to allow another pass
            tabu.clear()
            tabu_set.clear()
            if pass_num > 2:
                break

    if verbose:
        print(f"    tabu_soft: {pass_num} passes, {n_moves} moves, "
              f"{n_aspirated} aspirated, cost {best_cost:.6f}")
    return pos_np, best_cost
