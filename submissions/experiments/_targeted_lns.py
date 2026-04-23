"""Targeted LNS: destroy macros in the worst-congestion region.

Random LNS (_moves.py) picks a random seed. Targeted LNS identifies the
grid cell with the highest *smoothed* routing congestion and destroys
macros in or around that cell. The intuition: those are the macros that
are contributing most to the cost ceiling; shuffling them is most likely
to reduce proxy cost.
"""
import time
import numpy as np


def targeted_lns_phase(pos_np, benchmark, incr_eval, max_time,
                       n_destroy=5, n_candidates=40, verbose=False):
    """LNS variant that targets worst-congestion regions for destruction."""
    n_hard = benchmark.num_hard_macros
    sizes = benchmark.macro_sizes[:n_hard].numpy().astype(np.float64)
    cw, ch = float(benchmark.canvas_width), float(benchmark.canvas_height)
    gap = 0.05
    half_w = sizes[:, 0] / 2
    half_h = sizes[:, 1] / 2
    sep_x = (sizes[:, 0:1] + sizes[:, 0:1].T) / 2
    sep_y = (sizes[:, 1:2] + sizes[:, 1:2].T) / 2
    movable = benchmark.get_movable_mask()[:n_hard].numpy()

    grid_w = incr_eval.grid_width
    grid_h = incr_eval.grid_height
    grid_col = incr_eval.grid_col
    grid_row = incr_eval.grid_row

    pos = pos_np.copy()
    incr_eval.sync_positions(pos)
    current_cost = incr_eval.get_proxy_cost()
    best_cost = current_cost
    best_pos = pos.copy()

    rng = np.random.RandomState(54321)
    t0 = time.time()
    n_iter = 0
    n_accepted = 0

    while time.time() - t0 < max_time:
        n_iter += 1

        # Find top-K hotspot cells by combined V+H routing congestion.
        combined = incr_eval.V_routing_smooth + incr_eval.H_routing_smooth
        # Pick a cell biased toward worst — take one of top 10 randomly for diversity
        topk = np.argsort(combined)[-10:]
        cell_idx = int(rng.choice(topk))
        row = cell_idx // grid_col
        col = cell_idx % grid_col
        cell_cx = (col + 0.5) * grid_w
        cell_cy = (row + 0.5) * grid_h

        # Find movable hard macros whose bounding box overlaps the hotspot
        # neighborhood (3x3 cells around hotspot).
        radius_x = 1.5 * grid_w
        radius_y = 1.5 * grid_h
        dists = []
        for m in range(n_hard):
            if not movable[m]:
                continue
            dx = pos[m, 0] - cell_cx
            dy = pos[m, 1] - cell_cy
            # Weighted L∞
            d = max(abs(dx) / radius_x, abs(dy) / radius_y)
            dists.append((d, m))
        dists.sort(key=lambda x: x[0])
        subset = [m for d, m in dists[:n_destroy] if d < 4.0]
        if len(subset) < 2:
            continue

        # Reinsert in random order
        rng.shuffle(subset)
        move_trail = []
        moved_any = False

        for m in subset:
            old_x, old_y = float(pos[m, 0]), float(pos[m, 1])
            # Candidate positions: biased toward net-centroid + random spread
            cx_sum, cy_sum, cnt = 0.0, 0.0, 0
            for nid in incr_eval.macro_nets[m]:
                for m2 in incr_eval.net_macros[nid]:
                    if 0 <= m2 < n_hard and m2 != m:
                        cx_sum += pos[m2, 0]
                        cy_sum += pos[m2, 1]
                        cnt += 1
            cands = []
            if cnt > 0:
                cx_c = cx_sum / cnt
                cy_c = cy_sum / cnt
                # Net-centroid jittered
                for _ in range(8):
                    cands.append((cx_c + rng.uniform(-1.5, 1.5),
                                  cy_c + rng.uniform(-1.5, 1.5)))
            # Random uniform
            for _ in range(n_candidates - 8):
                cands.append((rng.uniform(half_w[m], cw - half_w[m]),
                              rng.uniform(half_h[m], ch - half_h[m])))

            best_cx, best_cy = old_x, old_y
            best_c = current_cost
            kept_old = True
            for cx, cy in cands:
                cx = float(np.clip(cx, half_w[m], cw - half_w[m]))
                cy = float(np.clip(cy, half_h[m], ch - half_h[m]))
                ddx = np.abs(cx - pos[:, 0])
                ddy = np.abs(cy - pos[:, 1])
                ov = (ddx < sep_x[m] + gap) & (ddy < sep_y[m] + gap)
                ov[m] = False
                if ov.any():
                    continue
                c = incr_eval.move_macro(m, cx, cy)
                if c < best_c:
                    best_c = c
                    best_cx = cx
                    best_cy = cy
                    kept_old = False
                incr_eval.undo_move()

            if not kept_old:
                incr_eval.move_macro(m, best_cx, best_cy)
                move_trail.append((m, old_x, old_y))
                pos[m, 0] = best_cx
                pos[m, 1] = best_cy
                current_cost = best_c
                moved_any = True

        if not moved_any:
            continue

        final_cost = incr_eval.get_proxy_cost()
        if final_cost < best_cost - 1e-7:
            best_cost = final_cost
            best_pos = pos.copy()
            n_accepted += 1
        else:
            for m, ox, oy in reversed(move_trail):
                incr_eval.move_macro(m, ox, oy)
                pos[m, 0] = ox
                pos[m, 1] = oy
            current_cost = best_cost

    if verbose:
        print(f"    targeted_lns: {n_iter} iters, {n_accepted} accepted, "
              f"final {best_cost:.6f}")

    incr_eval.sync_positions(best_pos)
    return best_pos, best_cost
