"""Nesterov-style momentum for soft macro CD.

Each macro has a per-macro velocity. When accepting a move, record the
displacement as velocity. On next visit, first PROBE a step in the
velocity direction (look-ahead), then fall back to 8-direction search
if that doesn't help.

Hypothesis: correlated moves (net centroids shifting consistently)
accumulate momentum. Look-ahead captures that momentum without waste.
"""
import time
import numpy as np


def nesterov_soft_cd(pos_np, benchmark, incr_eval, max_time,
                     momentum=0.6, verbose=False):
    n_hard = benchmark.num_hard_macros
    n_total = benchmark.macro_positions.shape[0]
    cw, ch = float(benchmark.canvas_width), float(benchmark.canvas_height)
    macro_fixed = benchmark.macro_fixed.numpy() if hasattr(benchmark.macro_fixed, 'numpy') else benchmark.macro_fixed
    soft_movable = [n_hard + i for i in range(n_total - n_hard) if not macro_fixed[n_hard + i]]
    if not soft_movable:
        return pos_np, incr_eval.get_proxy_cost()

    net_count = np.array([len(incr_eval.macro_nets[i]) for i in range(n_total)])
    soft_sorted = sorted(soft_movable, key=lambda i: -net_count[i])

    # Per-macro velocity
    velocity = {m: (0.0, 0.0) for m in soft_movable}

    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1),
            (0.7, 0.7), (-0.7, 0.7), (0.7, -0.7), (-0.7, -0.7)]
    deltas = [1.0, 0.5, 0.25, 0.12]

    current_cost = incr_eval.get_proxy_cost()
    best_cost = current_cost
    t0 = time.time()
    n_moves = 0
    n_momentum_hits = 0
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
                ox = float(incr_eval.macro_pos[i, 0])
                oy = float(incr_eval.macro_pos[i, 1])
                vx, vy = velocity[i]

                best_dir_c = current_cost
                best_nx, best_ny = None, None

                # Momentum look-ahead first
                if abs(vx) > 1e-5 or abs(vy) > 1e-5:
                    # Scale velocity to roughly delta magnitude
                    vmag = (vx * vx + vy * vy) ** 0.5
                    s = delta / max(vmag, 1e-6)
                    pnx = float(max(0, min(cw, ox + s * vx * momentum)))
                    pny = float(max(0, min(ch, oy + s * vy * momentum)))
                    if abs(pnx - ox) > 1e-4 or abs(pny - oy) > 1e-4:
                        c = incr_eval.move_macro(i, pnx, pny)
                        if c < best_dir_c:
                            best_dir_c = c
                            best_nx, best_ny = pnx, pny
                            n_momentum_hits += 1
                        incr_eval.undo_move()

                # Try 8 directions
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

                if best_nx is not None:
                    # Accept; update velocity
                    incr_eval.move_macro(i, best_nx, best_ny)
                    new_vx = best_nx - ox
                    new_vy = best_ny - oy
                    velocity[i] = (momentum * vx + new_vx, momentum * vy + new_vy)
                    current_cost = best_dir_c
                    if best_dir_c < best_cost:
                        best_cost = best_dir_c
                    n_moves += 1
                    improved = True
                else:
                    # Decay velocity
                    velocity[i] = (vx * 0.5, vy * 0.5)
        if not improved:
            break

    if verbose:
        print(f"    nesterov: {pass_num} passes, {n_moves} moves, "
              f"{n_momentum_hits} momentum, cost {best_cost:.6f}")
    return pos_np, best_cost
