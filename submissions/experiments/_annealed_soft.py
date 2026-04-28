"""Soft CD with annealed delta schedule — not per-call but ACROSS calls.

Schedule: early calls use coarse deltas only; later calls use fine deltas.
Implemented as a phase counter that persists across calls.
"""
import time
import numpy as np

_phase = [0]  # 0..3 => [coarse, mid, fine, ultra-fine]

DELTAS_BY_PHASE = {
    0: [2.0, 1.5, 1.0],         # coarse
    1: [1.5, 1.0, 0.5, 0.25],
    2: [0.5, 0.25, 0.12],       # fine
    3: [0.25, 0.12, 0.06, 0.03],  # ultra-fine
}

def reset_phase():
    _phase[0] = 0

def soft_cd_annealed(pos_np, benchmark, incr_eval, max_time, verbose=False):
    phase = min(_phase[0], 3)
    _phase[0] += 1
    deltas = DELTAS_BY_PHASE[phase]

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
    current_cost = incr_eval.get_proxy_cost()
    best_cost = current_cost
    t0 = time.time()
    n_moves = 0
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
                best_c = current_cost
                best_nx, best_ny = None, None
                for dx, dy in dirs:
                    nx = float(max(0, min(cw, ox + delta * dx)))
                    ny = float(max(0, min(ch, oy + delta * dy)))
                    if abs(nx - ox) < 1e-4 and abs(ny - oy) < 1e-4:
                        continue
                    c = incr_eval.move_macro(i, nx, ny)
                    if c < best_c:
                        best_c = c
                        best_nx, best_ny = nx, ny
                    incr_eval.undo_move()
                if best_nx is not None:
                    incr_eval.move_macro(i, best_nx, best_ny)
                    current_cost = best_c
                    if best_c < best_cost:
                        best_cost = best_c
                    n_moves += 1
                    improved = True
        if not improved:
            break

    if verbose:
        print(f"    annealed_phase{phase}: {pass_num} passes, {n_moves} moves, "
              f"deltas={deltas}, cost {best_cost:.6f}")
    return pos_np, best_cost
