"""Fast soft-macro (stdcell cluster) co-optimization.

Ideas we already know don't work:
  - Full force-directed via plc.optimize_stdcells (too slow: minutes per call)

New approach: use IncrementalEvaluator's move_macro (which works for soft
macros too — the only difference is soft macros don't add macro-blockage).
Run CD with a coarser delta schedule over soft macros only, after hard-macro
CD converges. Soft macros are ~4x more numerous than hard, so this is a
real compute cost — we keep it cheap with a shorter delta schedule.
"""
import time
import numpy as np


def soft_macro_cd(pos_np, benchmark, incr_eval, max_time, verbose=False):
    """Coord descent on soft macros only, using incremental evaluator.

    pos_np is just the hard-macro positions; soft positions live inside
    incr_eval.macro_pos. We update incr_eval.macro_pos for soft macros,
    then call move_macro to update costs.
    """
    n_hard = benchmark.num_hard_macros
    n_total = benchmark.macro_positions.shape[0]
    cw, ch = float(benchmark.canvas_width), float(benchmark.canvas_height)

    # Sync hard positions first
    incr_eval.sync_positions(pos_np)

    # Soft macro indices (exclude fixed)
    macro_fixed = benchmark.macro_fixed.numpy() if hasattr(benchmark.macro_fixed, 'numpy') else benchmark.macro_fixed
    soft_movable = [n_hard + i for i in range(n_total - n_hard) if not macro_fixed[n_hard + i]]
    if not soft_movable:
        if verbose:
            print(f"    soft_macro_cd: no movable softs")
        return pos_np, incr_eval.get_proxy_cost()

    # Connectivity-sort softs (by # nets)
    net_count = np.array([len(incr_eval.macro_nets[i]) for i in range(n_total)])
    soft_sorted = sorted(soft_movable, key=lambda i: -net_count[i])

    # Soft macros in this benchmark are zero-size (stdcell cluster centroids).
    # They don't create overlap constraints — the evaluator treats them as
    # points. So no overlap check needed.
    # Canvas clipping: use 0 as half-size.
    sizes = np.zeros((n_total, 2), dtype=np.float64)
    if hasattr(benchmark.macro_sizes, 'numpy'):
        s = benchmark.macro_sizes.numpy()
        sizes[:s.shape[0]] = s

    current_cost = incr_eval.get_proxy_cost()
    best_cost = current_cost
    best_pos = incr_eval.macro_pos.copy()

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
                best_nx = None
                best_ny = None
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
                        best_pos = incr_eval.macro_pos.copy()
                    pass_improved = True
                    n_moves += 1
        if not pass_improved:
            break

    # Restore incr_eval to best_pos (soft macros)
    # We only moved softs, so hard positions are unchanged. Easiest: sync softs.
    # But IncrementalEvaluator.sync_positions only takes hard positions.
    # So leave the current incr_eval state as-is (reflects latest moves).
    # If the latest weren't best, the caller will re-sync.
    if verbose:
        print(f"    soft_macro_cd: {pass_num} passes, {n_moves} moves, "
              f"cost {current_cost:.6f} (best {best_cost:.6f})")
    # Return hard positions unchanged; best_cost reflects improved state.
    return pos_np, best_cost
