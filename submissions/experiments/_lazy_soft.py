"""Lazy soft CD — only visit softs whose net neighbors moved recently.

After the first pass (visits all), subsequent passes visit only softs
connected (via nets) to a macro moved in the previous pass. This is
active-set CD — standard for sparse problems.
"""
import time
import numpy as np


def lazy_soft_cd(pos_np, benchmark, incr_eval, max_time, verbose=False):
    n_hard = benchmark.num_hard_macros
    n_total = benchmark.macro_positions.shape[0]
    cw, ch = float(benchmark.canvas_width), float(benchmark.canvas_height)
    macro_fixed = benchmark.macro_fixed.numpy() if hasattr(benchmark.macro_fixed, 'numpy') else benchmark.macro_fixed
    soft_movable = [n_hard + i for i in range(n_total - n_hard) if not macro_fixed[n_hard + i]]
    if not soft_movable:
        return pos_np, incr_eval.get_proxy_cost()

    # Neighbor index: for each macro, which softs are in a net with it?
    soft_neighbors = {m: set() for m in range(n_total)}
    for nid in range(incr_eval.n_nets):
        macros_in_net = incr_eval.net_macros[nid]
        softs_in_net = [m for m in macros_in_net if m in set(soft_movable)]
        for a in macros_in_net:
            for s in softs_in_net:
                if a != s:
                    soft_neighbors[a].add(s)

    net_count = np.array([len(incr_eval.macro_nets[i]) for i in range(n_total)])
    soft_sorted = sorted(soft_movable, key=lambda i: -net_count[i])

    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1),
            (0.7, 0.7), (-0.7, 0.7), (0.7, -0.7), (-0.7, -0.7)]
    deltas = [1.0, 0.5, 0.25, 0.12, 0.06]

    current_cost = incr_eval.get_proxy_cost()
    best_cost = current_cost
    t0 = time.time()
    n_moves = 0
    pass_num = 0

    # First pass: visit all
    active = set(soft_movable)

    while time.time() - t0 < max_time:
        pass_num += 1
        if not active:
            break
        next_active = set()
        order = sorted(active, key=lambda i: -net_count[i])

        for delta in deltas:
            if time.time() - t0 > max_time:
                break
            for i in order:
                if i not in active:
                    continue
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
                    # Add neighbors to next active set
                    for nbr in soft_neighbors.get(i, set()):
                        next_active.add(nbr)

        active = next_active
        if not active:
            break

    if verbose:
        print(f"    lazy_soft: {pass_num} passes, {n_moves} moves, cost {best_cost:.6f}")
    return pos_np, best_cost
