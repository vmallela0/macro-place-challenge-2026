"""Fast soft-macro CD variants.

Variant A — fast_soft_cd_4dir: only 4 cardinal directions per probe (halves
probes vs the 8-direction version).

Variant B — soft_cd_wl_only: predict delta wirelength analytically; skip
congestion update until accept. For soft macros, HPWL delta is fast to
compute (per-net bbox change).

Variant C — soft_cd_surrogate: use an MLP surrogate to rank candidates;
verify only top-k with real eval.
"""
import time
import numpy as np


def fast_soft_cd_4dir(pos_np, benchmark, incr_eval, max_time, verbose=False):
    """Halved-direction soft CD. Same best-of logic, 4 cardinal only."""
    n_hard = benchmark.num_hard_macros
    n_total = benchmark.macro_positions.shape[0]
    cw, ch = float(benchmark.canvas_width), float(benchmark.canvas_height)
    macro_fixed = benchmark.macro_fixed.numpy() if hasattr(benchmark.macro_fixed, 'numpy') else benchmark.macro_fixed
    soft_movable = [n_hard + i for i in range(n_total - n_hard) if not macro_fixed[n_hard + i]]
    if not soft_movable:
        return pos_np, incr_eval.get_proxy_cost()

    net_count = np.array([len(incr_eval.macro_nets[i]) for i in range(n_total)])
    soft_sorted = sorted(soft_movable, key=lambda i: -net_count[i])

    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    deltas = [1.0, 0.5, 0.25, 0.12, 0.06]

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
                if best_nx is not None:
                    incr_eval.move_macro(i, best_nx, best_ny)
                    current_cost = best_dir_c
                    if best_dir_c < best_cost:
                        best_cost = best_dir_c
                    n_moves += 1
                    improved = True
        if not improved:
            break

    if verbose:
        print(f"    fast_4dir: {pass_num} passes, {n_moves} moves, cost {best_cost:.6f}")
    return pos_np, best_cost


def soft_cd_per_macro_grid(pos_np, benchmark, incr_eval, max_time,
                           grid_step=0.5, grid_radius=3, verbose=False):
    """For each soft, probe a local grid (2*radius+1)^2 around it.

    With incremental eval ~50ms/probe, this is 49 probes × 50ms = 2.5s/macro,
    but the dense grid coverage finds better moves. Best-of-grid beats
    8-direction when the cost landscape is non-axis-aligned.
    """
    n_hard = benchmark.num_hard_macros
    n_total = benchmark.macro_positions.shape[0]
    cw, ch = float(benchmark.canvas_width), float(benchmark.canvas_height)
    macro_fixed = benchmark.macro_fixed.numpy() if hasattr(benchmark.macro_fixed, 'numpy') else benchmark.macro_fixed
    soft_movable = [n_hard + i for i in range(n_total - n_hard) if not macro_fixed[n_hard + i]]
    if not soft_movable:
        return pos_np, incr_eval.get_proxy_cost()

    net_count = np.array([len(incr_eval.macro_nets[i]) for i in range(n_total)])
    soft_sorted = sorted(soft_movable, key=lambda i: -net_count[i])

    # Offset grid (excluding center)
    offs = []
    for dx in range(-grid_radius, grid_radius + 1):
        for dy in range(-grid_radius, grid_radius + 1):
            if dx == 0 and dy == 0:
                continue
            offs.append((dx * grid_step, dy * grid_step))

    current_cost = incr_eval.get_proxy_cost()
    best_cost = current_cost
    t0 = time.time()
    n_moves = 0

    for i in soft_sorted:
        if time.time() - t0 > max_time:
            break
        ox = float(incr_eval.macro_pos[i, 0])
        oy = float(incr_eval.macro_pos[i, 1])
        best_c = current_cost
        best_nx, best_ny = None, None
        for ddx, ddy in offs:
            nx = ox + ddx
            ny = oy + ddy
            if nx < 0 or nx > cw or ny < 0 or ny > ch:
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

    if verbose:
        print(f"    grid_cd: {n_moves} moves, cost {best_cost:.6f}")
    return pos_np, best_cost
