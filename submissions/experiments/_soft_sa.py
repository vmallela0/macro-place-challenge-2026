"""Simulated-annealing soft CD.

Accept improving moves always; accept worsening moves with probability
exp(-ΔC / T). Exponential cooling schedule. Combined with CD-style probe
(all 8 directions, best valid).

Different from our prior SA failure (EXPERIMENTS.md): that was on HARD
macros with uniform proposals. This is on SOFT macros (no overlap check)
with CD-style proposals (8 axes x 6 deltas). Much less wasted work.
"""
import time
import math
import numpy as np


def soft_cd_sa(pos_np, benchmark, incr_eval, max_time,
               t_start=0.02, t_end=1e-4, verbose=False):
    """SA over soft macros with CD-style proposals."""
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
    best_pos = incr_eval.macro_pos.copy()

    rng = np.random.RandomState(424242)
    t0 = time.time()
    n_moves = 0
    n_uphill_accepted = 0
    pass_num = 0

    while time.time() - t0 < max_time:
        pass_num += 1
        progress = min(1.0, (time.time() - t0) / max_time)
        T = t_start * (t_end / t_start) ** progress  # exponential cool

        improved = False
        for delta in deltas:
            if time.time() - t0 > max_time:
                break
            for i in soft_sorted:
                if time.time() - t0 > max_time:
                    break
                ox = float(incr_eval.macro_pos[i, 0])
                oy = float(incr_eval.macro_pos[i, 1])

                # Try all 8 directions; for each, SA accept decision.
                # Accept the "best" by SA criterion (lowest |energy| / T after accept)
                best_dir_c = current_cost
                best_nx, best_ny = None, None
                for dx, dy in dirs:
                    nx = float(max(0, min(cw, ox + delta * dx)))
                    ny = float(max(0, min(ch, oy + delta * dy)))
                    if abs(nx - ox) < 1e-4 and abs(ny - oy) < 1e-4:
                        continue
                    c = incr_eval.move_macro(i, nx, ny)
                    # SA accept
                    dC = c - current_cost
                    if dC < 0:
                        # always accept
                        if c < best_dir_c:
                            best_dir_c = c
                            best_nx, best_ny = nx, ny
                    else:
                        # accept with probability exp(-dC/T)
                        if rng.random() < math.exp(-dC / max(T, 1e-9)) and c < best_dir_c + 0.005:
                            best_dir_c = c
                            best_nx, best_ny = nx, ny
                    incr_eval.undo_move()

                if best_nx is not None:
                    incr_eval.move_macro(i, best_nx, best_ny)
                    dC = best_dir_c - current_cost
                    current_cost = best_dir_c
                    if best_dir_c < best_cost:
                        best_cost = best_dir_c
                        best_pos = incr_eval.macro_pos.copy()
                    else:
                        if dC > 0:
                            n_uphill_accepted += 1
                    n_moves += 1
                    improved = True
        if not improved:
            break

    if verbose:
        print(f"    sa_soft: {pass_num} passes, {n_moves} moves, "
              f"{n_uphill_accepted} uphill, best {best_cost:.6f} "
              f"(end {current_cost:.6f})")

    # Restore best if current is worse
    if best_cost < current_cost - 1e-7:
        # Snap softs to best_pos by moving each
        for i in soft_movable:
            if (best_pos[i, 0] != incr_eval.macro_pos[i, 0] or
                best_pos[i, 1] != incr_eval.macro_pos[i, 1]):
                incr_eval.move_macro(i, float(best_pos[i, 0]), float(best_pos[i, 1]))

    return pos_np, best_cost
