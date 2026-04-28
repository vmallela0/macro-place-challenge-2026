"""Density-aware soft CD: prefer moves into LOW-density grid cells.

Adds a density-bias to the probe candidate generation. Builds a per-cell
density map each pass, and generates more candidates aimed at low-density
cells. Uses full incremental eval for accept decision so ground-truth
cost is optimized.
"""
import time
import numpy as np


def density_aware_soft_cd(pos_np, benchmark, incr_eval, max_time,
                           n_targets_per_macro=8, verbose=False):
    n_hard = benchmark.num_hard_macros
    n_total = benchmark.macro_positions.shape[0]
    cw, ch = float(benchmark.canvas_width), float(benchmark.canvas_height)
    macro_fixed = benchmark.macro_fixed.numpy() if hasattr(benchmark.macro_fixed, 'numpy') else benchmark.macro_fixed
    soft_movable = [n_hard + i for i in range(n_total - n_hard) if not macro_fixed[n_hard + i]]
    if not soft_movable:
        return pos_np, incr_eval.get_proxy_cost()

    grid_w = incr_eval.grid_width
    grid_h = incr_eval.grid_height
    grid_col = incr_eval.grid_col
    grid_row = incr_eval.grid_row

    net_count = np.array([len(incr_eval.macro_nets[i]) for i in range(n_total)])
    soft_sorted = sorted(soft_movable, key=lambda i: -net_count[i])

    rng = np.random.RandomState(31415)
    current_cost = incr_eval.get_proxy_cost()
    best_cost = current_cost
    t0 = time.time()
    n_moves = 0

    while time.time() - t0 < max_time:
        # Compute cell density
        density = np.asarray(incr_eval.grid_density).reshape(grid_row, grid_col)
        # Sample cells inversely proportional to density (low-density targets)
        flat_density = density.flatten()
        weights = 1.0 / (1.0 + flat_density)  # low density -> high weight
        weights = weights / weights.sum()

        improved = False
        for i in soft_sorted:
            if time.time() - t0 > max_time:
                break
            ox = float(incr_eval.macro_pos[i, 0])
            oy = float(incr_eval.macro_pos[i, 1])

            # Generate candidates: mix of near-current (incremental) + low-density cells
            cands = []
            # 4 local small moves
            for delta in [0.25, 0.5, 1.0]:
                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    cands.append((ox + delta * dx, oy + delta * dy))
            # N_targets_per_macro targets in low-density cells
            cell_idxs = rng.choice(len(flat_density), n_targets_per_macro, p=weights,
                                   replace=False)
            for ci in cell_idxs:
                row = ci // grid_col
                col = ci % grid_col
                jitter_x = rng.uniform(0, grid_w)
                jitter_y = rng.uniform(0, grid_h)
                cands.append((col * grid_w + jitter_x, row * grid_h + jitter_y))

            best_c = current_cost
            best_nx, best_ny = None, None
            for cx, cy in cands:
                cx = float(max(0, min(cw, cx)))
                cy = float(max(0, min(ch, cy)))
                if abs(cx - ox) < 1e-4 and abs(cy - oy) < 1e-4:
                    continue
                c = incr_eval.move_macro(i, cx, cy)
                if c < best_c:
                    best_c = c
                    best_nx, best_ny = cx, cy
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
        print(f"    density_aware: {n_moves} moves, cost {best_cost:.6f}")
    return pos_np, best_cost
